from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .face_adapter_runtime import FaceAdapterRuntime

logger = logging.getLogger(__name__)


class FaceAdapterFaceSwap(FaceAdapterRuntime):
    """FaceAdapter-backed face swapper for aligned KFAAR face crops.

    Inputs are aligned face tensors in CHW format and range [-1, 1] or [0, 1].
    Output is a swapped aligned target tensor in [0, 1] or None on failure.
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
                        _, h, w = target_t.shape

                        source_pil = self._tensor_to_pil(source_t)
                        target_pil = self._tensor_to_pil(target_t)

                        try:
                            src = self._prepare_aligned_face(source_pil, role="source")
                        except Exception as source_exc:
                            self._set_failure_reason(idx, "source_no_face")
                            logger.error("FaceAdapter swap failed for batch item %d (source): %s", idx, source_exc)
                            continue

                        try:
                            tar = self._prepare_aligned_face(target_pil, role="target")
                        except Exception as target_exc:
                            self._set_failure_reason(idx, "target_no_face")
                            logger.error("FaceAdapter swap failed for batch item %d (target): %s", idx, target_exc)
                            continue

                        src_d3d_coeff = self.net_d3dfr(src["image_256"])
                        gt_d3d_coeff = self.net_d3dfr(tar["image_256"])
                        gt_d3d_coeff[:, 0:80] = src_d3d_coeff[:, 0:80]
                        gt_pts68 = self.bfm_facemodel.get_lm68(gt_d3d_coeff)

                        im_pts70 = self._draw_pts70_batch(
                            gt_pts68,
                            gt_d3d_coeff[:, 257:],
                            tar["warp_mat_256"],
                            self.test_image_size,
                            return_pt=True,
                        ).to(tar["image_512"])

                        face_masks_tar, blend_mask = self._build_swap_mask(tar["image_512"])
                        controlnet_image_swap = (im_pts70 * face_masks_tar + tar["image_512"] * (1 - face_masks_tar)).to(dtype=self.weight_dtype)

                        faceid = self.net_arcface(F.interpolate(src["image_256"], [128, 128], mode="bilinear", align_corners=False))
                        encoder_hidden_states_src = self.net_id2token(faceid).to(dtype=self.weight_dtype)

                        tar_last_hidden = self.net_vision_encoder(tar["clip"]).last_hidden_state
                        controlnet_encoder_hidden_states_tar = self.net_image2token(tar_last_hidden).to(dtype=self.weight_dtype)

                        prepared.append(
                            {
                                "prompt": encoder_hidden_states_src,
                                "control_prompt": controlnet_encoder_hidden_states_tar,
                                "control_image": controlnet_image_swap,
                                "blend_mask": blend_mask,
                                "tar_image_512": tar["image_512"],
                                "tar_pil_512": tar["pil_512"],
                                "warp_mat_512": tar["warp_mat_512"],
                                "target_pil": target_pil,
                                "target_hw": (h, w),
                                "target_device": target_aligned.device,
                            }
                        )
                        valid_indices.append(idx)
                    except Exception as item_exc:
                        self._set_failure_reason(idx, "other_error")
                        logger.error("FaceAdapter swap failed for batch item %d: %s", idx, item_exc)

                if not prepared:
                    return results

                prompt_embeds = torch.cat([item["prompt"] for item in prepared], dim=0)
                control_prompt_embeds = torch.cat([item["control_prompt"] for item in prepared], dim=0)
                control_images = torch.cat([item["control_image"] for item in prepared], dim=0)

                batch_size = prompt_embeds.shape[0]
                negative_prompt_embeds = self.empty_prompt_token.expand(batch_size, -1, -1)
                control_negative_prompt_embeds = self.empty_prompt_token.expand(batch_size, -1, -1)

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

                    object_mask_np = np.zeros_like(blend_mask_np)
                    coco_results = self.coco_seg_model(item["tar_pil_512"], classes=[39, 41, 67, 78, 79], verbose=False)
                    if len(coco_results) > 0 and coco_results[0].masks is not None:
                        mask_data = coco_results[0].masks.data.cpu().numpy()
                        if mask_data.size > 0:
                            combined_coco = np.max(mask_data, axis=0)
                            resized_coco = cv2.resize(combined_coco, (512, 512), interpolation=cv2.INTER_LINEAR)
                            object_mask_np = (resized_coco > 0.5).astype(np.float32)[:, :, np.newaxis]

                    safe_blend_mask = blend_mask_np * (1.0 - object_mask_np)
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
                        safe_blend_mask.squeeze(),
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
            logger.error("FaceAdapter batched swap failed: %s", exc)
            return results
