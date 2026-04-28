from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .face_adapter_runtime import FaceAdapterRuntime

logger = logging.getLogger(__name__)


class FaceAdapterFaceReenactment(FaceAdapterRuntime):
    """FaceAdapter reenactment postprocessor.

    This mode keeps the source identity while transferring target pose/expression,
    and returns a crop suitable for bbox-based placement in the caller.
    """

    def swap_batch(self, source_aligned_batch: list[torch.Tensor], target_aligned_batch: list[torch.Tensor]) -> list[torch.Tensor | None]:
        if len(source_aligned_batch) != len(target_aligned_batch):
            raise ValueError("source_aligned_batch and target_aligned_batch must have the same length")

        count = len(source_aligned_batch)
        results: list[torch.Tensor | None] = [None] * count
        if count == 0:
            return results
        self._reset_failure_reasons(count)

        try:
            with torch.no_grad():
                prepared: list[dict[str, object]] = []
                valid_indices: list[int] = []

                for idx, (source_aligned, target_aligned) in enumerate(zip(source_aligned_batch, target_aligned_batch)):
                    try:
                        source_t = source_aligned
                        target_t = target_aligned
                        if target_t.min() < -0.1:
                            target_t = (target_t + 1.0) / 2.0
                        if source_t.min() < -0.1:
                            source_t = (source_t + 1.0) / 2.0

                        target_t = target_t.clamp(0.0, 1.0)
                        source_t = source_t.clamp(0.0, 1.0)
                        
                        _, h, w = source_t.shape

                        source_pil = self._tensor_to_pil(source_t)
                        target_pil = self._tensor_to_pil(target_t)

                        try:
                            src = self._prepare_aligned_face(source_pil, role="source")
                        except Exception as source_exc:
                            self._set_failure_reason(idx, "source_no_face")
                            logger.error("FaceAdapter reenactment failed for batch item %d (source): %s", idx, source_exc)
                            continue

                        try:
                            tar = self._prepare_aligned_face(target_pil, role="target")
                        except Exception as target_exc:
                            self._set_failure_reason(idx, "target_no_face")
                            logger.error("FaceAdapter reenactment failed for batch item %d (target): %s", idx, target_exc)
                            continue

                        # --- 1. PROPER 3DMM RECOMBINATION ---
                        src_d3d_coeff = self.net_d3dfr(src["image_256"])
                        tar_d3d_coeff = self.net_d3dfr(tar["image_256"])
                        
                        # Use SOURCE as the absolute base
                        recon_d3d_coeff = src_d3d_coeff.clone()
                        
                        # Only inject TARGET Expression and Pose
                        recon_d3d_coeff[:, 80:144] = tar_d3d_coeff[:, 80:144]
                        recon_d3d_coeff[:, 224:227] = tar_d3d_coeff[:, 224:227]
                        
                        recon_pts68 = self.bfm_facemodel.get_lm68(recon_d3d_coeff)

                        # Draw landmarks using SOURCE pupils to completely avoid spatial tearing
                        im_pts70 = self._draw_pts70_batch(
                            recon_pts68,
                            src_d3d_coeff[:, 257:],  
                            src["warp_mat_256"],     
                            self.test_image_size,
                            return_pt=True,
                        ).to(src["image_512"])

                        # --- 2. DYNAMIC BACKGROUND MASKING (net_seg_res18) ---
                        # We use the Adapting Area Predictor which looks at BOTH the source image
                        # and the new landmarks to dynamically expand the mask for the open mouth.
                        mask_input = torch.cat([src["image_512"], im_pts70], dim=1)
                        face_masks_src = (self.net_seg_res18(mask_input) > 0.5).float()
                        
                        # Composite the Condition Image
                        controlnet_image = (im_pts70 * face_masks_src + src["image_512"] * (1 - face_masks_src)).to(dtype=self.weight_dtype)

                        # Soft feathering for blending
                        face_masks_src_pad = F.pad(face_masks_src, (16, 16, 16, 16), "constant", 0)
                        blend_mask = F.max_pool2d(face_masks_src_pad, kernel_size=17, stride=1, padding=8)
                        blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
                        blend_mask = blend_mask[:, :, 16:528, 16:528]

                        # --- 3. EXTRACT SOURCE IDENTITY & ATTRIBUTES ---
                        faceid = self.net_arcface(F.interpolate(src["image_256"], [128, 128], mode="bilinear", align_corners=False))
                        prompt_embeds = self.net_id2token(faceid).to(dtype=self.weight_dtype)

                        src_last_hidden = self.net_vision_encoder(src["clip"]).last_hidden_state
                        control_prompt_embeds = self.net_image2token(src_last_hidden).to(dtype=self.weight_dtype)

                        prepared.append(
                            {
                                "prompt": prompt_embeds,
                                "control_prompt": control_prompt_embeds,
                                "control_image": controlnet_image,
                                "blend_mask": blend_mask,
                                "warp_mat_512": src["warp_mat_512"], 
                                "target_pil": source_pil,            
                                "target_hw": (h, w),
                                "target_device": source_aligned.device,
                            }
                        )
                        valid_indices.append(idx)
                    except Exception as item_exc:
                        self._set_failure_reason(idx, "other_error")
                        logger.error("FaceAdapter reenactment failed for batch item %d: %s", idx, item_exc)

                if not prepared:
                    return results

                prompt_embeds = torch.cat([item["prompt"] for item in prepared], dim=0)
                control_prompt_embeds = torch.cat([item["control_prompt"] for item in prepared], dim=0)
                control_images = torch.cat([item["control_image"] for item in prepared], dim=0)

                batch_size = prompt_embeds.shape[0]

                # Robust negative prompt to suppress base model watermarks/artifacts
                neg_prompt_text = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, letters, watermark, signature, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
                
                text_inputs = self.pipe.tokenizer(
                    neg_prompt_text,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                neg_prompt_embeds = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
                neg_prompt_embeds = neg_prompt_embeds.to(dtype=self.weight_dtype)

                negative_prompt_embeds = neg_prompt_embeds.expand(batch_size, -1, -1)
                control_negative_prompt_embeds = neg_prompt_embeds.expand(batch_size, -1, -1)

                self._set_seed(self.seed)
                generator = torch.Generator(device=self.device).manual_seed(self.seed)

                gen_pils = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    controlnet_prompt_embeds=control_prompt_embeds,
                    controlnet_negative_prompt_embeds=control_negative_prompt_embeds,
                    image=control_images,
                    num_inference_steps=self.inference_steps,
                    generator=generator,
                    guidance_scale=self.guidance_scale,
                    controlnet_conditioning_scale=1.0,
                ).images

                for out_idx, item in enumerate(prepared):
                    gen_np = np.array(gen_pils[out_idx].convert("RGB"))
                    blend_mask_np = item["blend_mask"][0, 0].cpu().numpy()[:, :, np.newaxis]
                    
                    composite_512 = gen_np

                    h, w = item["target_hw"]
                    orig_crop_np = np.array(item["target_pil"]) 
                    
                    inv_face = cv2.warpAffine(
                        composite_512,
                        item["warp_mat_512"],
                        (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                    inv_mask = cv2.warpAffine(
                        blend_mask_np.squeeze(),
                        item["warp_mat_512"],
                        (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )[:, :, np.newaxis]

                    final_np = (inv_face * inv_mask + orig_crop_np * (1.0 - inv_mask)).astype(np.uint8)
                    final_tensor = torch.from_numpy(final_np.transpose(2, 0, 1)).float().div(255.0)
                    results[valid_indices[out_idx]] = final_tensor.to(item["target_device"])

                return results
        except Exception as exc:
            logger.error("FaceAdapter batched reenactment failed: %s", exc)
            return results

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> torch.Tensor | None:
        return self.swap_batch([source_aligned], [target_aligned])[0]