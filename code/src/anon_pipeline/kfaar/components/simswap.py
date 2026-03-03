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

        Automatically handles mapping standard pipeline tensors [-1, 1] 
        to SimSwap's expected [0, 1] range and back.
        """
        try:
            # 1. FIX THE TENSOR RANGES: [-1.0, 1.0] -> [0.0, 1.0]
            src = (source_img.unsqueeze(0).to(self.device) + 1.0) / 2.0
            tgt = (target_img.unsqueeze(0).to(self.device) + 1.0) / 2.0
            
            # Clamp to guarantee no StyleGAN outliers ruin the normalization
            src = src.clamp(0.0, 1.0)
            tgt = tgt.clamp(0.0, 1.0)

            # Resize to SimSwap crop size
            src_crop = F.interpolate(src, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
            tgt_crop = F.interpolate(tgt, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)

            # ArcFace embedding from source
            src_arc = F.interpolate(src_crop, size=(112, 112), mode="bilinear", align_corners=False)
            src_arc = (src_arc - self._mean) / self._std
            latent_id = self.model.netArc(src_arc)  # type: ignore[attr-defined]
            latent_id = latent_id / (torch.norm(latent_id, dim=1, keepdim=True) + 1e-6)

            with torch.no_grad():
                swapped = self.model(src_crop, tgt_crop, latent_id, latent_id, True)

            # Resize back to target resolution
            swapped = F.interpolate(swapped, size=target_img.shape[-2:], mode="bilinear", align_corners=False)
            
            # 2. CONVERT BACK TO PIPELINE RANGE: [0.0, 1.0] -> [-1.0, 1.0]
            swapped = (swapped * 2.0) - 1.0
            
            return swapped.squeeze(0).clamp(-1.0, 1.0)
            
        except Exception as e:
            print(f"FaceSwap failed: {e}")
            return None
