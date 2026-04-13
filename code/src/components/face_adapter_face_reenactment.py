from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .diffusion_swapper import DiffusionFaceSwapper

logger = logging.getLogger(__name__)


class FaceAdapterFaceReenactment(DiffusionFaceSwapper):
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

                        src = self._prepare_aligned_face(source_pil)
                        tar = self._prepare_aligned_face(target_pil)

                        src_d3d_coeff = self.net_d3dfr(src["image_256"])
                        tar_d3d_coeff = self.net_d3dfr(tar["image_256"])
                        tar_d3d_coeff[:, 0:80] = src_d3d_coeff[:, 0:80]
                        tar_pts68 = self.bfm_facemodel.get_lm68(tar_d3d_coeff)

                        im_pts70 = self._draw_pts70_batch(
                            tar_pts68,
                            tar_d3d_coeff[:, 257:],
                            tar["warp_mat_256"],
                            self.test_image_size,
                            return_pt=True,
                        ).to(tar["image_512"])

                        face_masks_tar, blend_mask = self._build_swap_mask(tar["image_512"])
                        controlnet_image = (im_pts70 * face_masks_tar + src["image_512"] * (1 - face_masks_tar)).to(dtype=self.weight_dtype)

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
                                "base_image_512": src["image_512"],
                                "warp_mat_512": tar["warp_mat_512"],
                                "target_pil": target_pil,
                                "target_hw": (h, w),
                                "target_device": target_aligned.device,
                            }
                        )
                        valid_indices.append(idx)
                    except Exception as item_exc:
                        logger.error("FaceAdapter reenactment failed for batch item %d: %s", idx, item_exc)

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
                    base_512_np = (
                        item["base_image_512"][0].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5
                    ).clip(0, 255).astype(np.uint8)
                    blend_mask_np = item["blend_mask"][0, 0].cpu().numpy()[:, :, np.newaxis]

                    composite_512 = (gen_np * blend_mask_np + base_512_np * (1.0 - blend_mask_np)).astype(np.uint8)

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
