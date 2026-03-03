from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F


class SimSwapFaceSwapper:
    """Lightweight wrapper to run SimSwap for single-image swapping.

    Uses the StyleGAN-generated face as identity and swaps it into the target frame.
    """

    def __init__(
        self,
        simswap_root: Path,
        checkpoints_dir: Path,
        name: str,
        which_epoch: str,
        arcface_ckpt: Path,
        crop_size: int = 224,
        device: torch.device | str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SimSwap requires CUDA but torch.cuda.is_available() is False")

        # Normalize device; SimSwap expects an explicit CUDA index
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        # Ensure SimSwap code is on import path
        simswap_root = Path(simswap_root).resolve()
        if str(simswap_root) not in sys.path:
            sys.path.insert(0, str(simswap_root))

        from models.fs_model import fsModel  # type: ignore

        # Build a minimal opt namespace expected by fsModel.initialize
        opt = SimpleNamespace(
            isTrain=False,
            resize_or_crop="none",
            crop_size=int(crop_size),
            Arc_path=str(arcface_ckpt),
            checkpoints_dir=str(checkpoints_dir),
            name=name,
            which_epoch=str(which_epoch),
            gpu_ids=[self.device.index or 0],
            verbose=False,
            load_pretrain="",
            gan_mode="hinge",
            lambda_feat=0.0,
            lambda_rec=0.0,
            no_ganFeat_loss=True,
            no_vgg_loss=True,
        )

        # fsModel currently hardcodes cuda:0; set the default device accordingly
        torch.cuda.set_device(self.device.index or 0)
        self.model = fsModel()
        self.model.initialize(opt)
        self.model.eval()

        # Precompute mean/std for ArcFace normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std
        self.crop_size = int(crop_size)

    def swap(self, source_img: torch.Tensor, target_img: torch.Tensor) -> Optional[torch.Tensor]:
        """Swap the identity from source_img into target_img.
        
        Closely mirrors the official SimSwap example while adapting for 
        dynamic aspect ratios (via padding) and pipeline tensor ranges [-1, 1].
        """
        try:
            # --- PIPELINE ADAPTATION: Range Fix ---
            src = source_img.unsqueeze(0).to(self.device)
            tgt = target_img.unsqueeze(0).to(self.device)
            
            target_was_minus_one = tgt.min() < 0.0
            
            if src.min() < 0.0: src = (src + 1.0) / 2.0
            if target_was_minus_one: tgt = (tgt + 1.0) / 2.0
            
            src = src.clamp(0.0, 1.0)
            tgt = tgt.clamp(0.0, 1.0)

            # --- PIPELINE ADAPTATION: Aspect Ratio Padding ---
            # Source Padding
            _, _, h_src, w_src = src.shape
            pad_h_src = max(h_src, w_src) - h_src
            pad_w_src = max(h_src, w_src) - w_src
            src_pad = F.pad(src, (pad_w_src // 2, pad_w_src - pad_w_src // 2, 
                                  pad_h_src // 2, pad_h_src - pad_h_src // 2))

            # Target Padding
            _, _, h_tgt, w_tgt = tgt.shape
            pad_h_tgt = max(h_tgt, w_tgt) - h_tgt
            pad_w_tgt = max(h_tgt, w_tgt) - w_tgt
            
            tgt_left = pad_w_tgt // 2
            tgt_right = pad_w_tgt - tgt_left
            tgt_top = pad_h_tgt // 2
            tgt_bottom = pad_h_tgt - tgt_top
            
            tgt_pad = F.pad(tgt, (tgt_left, tgt_right, tgt_top, tgt_bottom))

            # --- SIMSWAP ORIGINAL LOGIC MAPPING ---
            
            # 1. Prepare Target Image (Corresponds to 'img_b' -> 'img_att')
            # SimSwap applies only ToTensor() for attributes
            img_att = F.interpolate(tgt_pad, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)

            # 2. Prepare Source Image (Corresponds to 'img_a' -> 'img_id')
            # SimSwap applies ToTensor() + ImageNet Normalization for the identity image
            src_crop = F.interpolate(src_pad, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
            img_id = (src_crop - self._mean) / self._std

            # 3. Create Latent ID
            # SimSwap interpolates the normalized img_id down to 112x112 for netArc
            img_id_downsample = F.interpolate(img_id, size=(112, 112), mode="bilinear", align_corners=False)
            latend_id = self.model.netArc(img_id_downsample)
            
            # The original example moves the tensor to CPU to use numpy.linalg.norm. 
            # We use the pure PyTorch equivalent to keep it on the GPU, avoiding a massive pipeline bottleneck.
            latend_id = F.normalize(latend_id, p=2, dim=1)

            # 4. Forward Pass (Exactly as written in the original example)
            with torch.no_grad():
                img_fake = self.model(img_id, img_att, latend_id, latend_id, True)

# --- PIPELINE ADAPTATION: Revert Padding and Range ---
            
            # Resize back to the padded square dimension of the target
            max_tgt_dim = max(h_tgt, w_tgt)
            swapped_square = F.interpolate(img_fake, size=(max_tgt_dim, max_tgt_dim), mode="bilinear", align_corners=False)

            # Slice out the padding to restore original exact dimensions
            swapped_restored = swapped_square[..., tgt_top : max_tgt_dim - tgt_bottom, 
                                                   tgt_left : max_tgt_dim - tgt_right]

            # FIX: Always return [0, 1]. Do not reverse back to [-1, 1].
            # KfaarPipeline expects det_input and the saved image to be [0, 1].
            return swapped_restored.squeeze(0).clamp(0.0, 1.0)
            
        except Exception as e:
            import logging
            logging.error(f"FaceSwap failed: {e}")
            return None