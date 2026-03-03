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
        
        Universally handles any aspect ratio by padding to a perfect square, 
        and dynamically adapts to [-1, 1] or [0, 1] input tensor ranges.
        """
        try:
            # 1. DYNAMIC RANGE FIX: SimSwap strictly expects [0.0, 1.0]
            src = source_img.unsqueeze(0).to(self.device)
            tgt = target_img.unsqueeze(0).to(self.device)
            
            target_was_minus_one = tgt.min() < 0.0
            
            if src.min() < 0.0: src = (src + 1.0) / 2.0
            if target_was_minus_one: tgt = (tgt + 1.0) / 2.0
            
            src = src.clamp(0.0, 1.0)
            tgt = tgt.clamp(0.0, 1.0)

            # 2. UNIVERSAL ASPECT RATIO PADDING
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

            # 3. RESIZE SQUARES TO CROP_SIZE
            src_crop = F.interpolate(src_pad, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
            tgt_crop = F.interpolate(tgt_pad, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)

            # 4. ARCFACE EMBEDDING
            src_arc = F.interpolate(src_crop, size=(112, 112), mode="bilinear", align_corners=False)
            src_arc = (src_arc - self._mean) / self._std
            latent_id = self.model.netArc(src_arc)  # type: ignore[attr-defined]
            latent_id = latent_id / (torch.norm(latent_id, dim=1, keepdim=True) + 1e-6)

            # 5. FORWARD PASS
            with torch.no_grad():
                swapped = self.model(src_crop, tgt_crop, latent_id, latent_id, True)

            # 6. REVERSE ASPECT RATIO FIX (UN-PAD)
            # Resize back to the padded square dimension of the target
            max_tgt_dim = max(h_tgt, w_tgt)
            swapped_square = F.interpolate(swapped, size=(max_tgt_dim, max_tgt_dim), mode="bilinear", align_corners=False)

            # Slice out the padding to restore original exact dimensions
            swapped_restored = swapped_square[..., tgt_top : max_tgt_dim - tgt_bottom, 
                                                   tgt_left : max_tgt_dim - tgt_right]

            # 7. REVERSE RANGE FIX
            if target_was_minus_one:
                swapped_restored = (swapped_restored * 2.0) - 1.0

            # Clamp to the original requested limits based on input
            out_min = -1.0 if target_was_minus_one else 0.0
            return swapped_restored.squeeze(0).clamp(out_min, 1.0)
            
        except Exception as e:
            import logging
            logging.error(f"FaceSwap failed: {e}")
            return None
