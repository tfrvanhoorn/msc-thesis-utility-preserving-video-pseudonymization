from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F


class SimSwapFaceSwapper:
    """Lightweight wrapper to run SimSwap for single-image swapping.

    This version bypasses SimSwap's native InsightFace detector and assumes the 
    inputs (source and target) are already aligned by the SKPG pipeline. 
    It dynamically pads non-square inputs to preserve aspect ratios.
    """

    def __init__(
        self,
        simswap_root: Path,
        checkpoints_dir: Path,
        name: str,
        which_epoch: str,
        arcface_ckpt: Path,
        crop_size: int = 512,
        device: torch.device | str = "cuda:0",
        **kwargs  # Catches legacy SKPG args
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SimSwap requires CUDA but torch.cuda.is_available() is False")

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
            
        simswap_root = Path(simswap_root).resolve()
        if str(simswap_root) not in sys.path:
            sys.path.insert(0, str(simswap_root))

        from models.models import create_model  # type: ignore

        # --- 512 MODEL AUTO-CONFIGURE ---
        # If 512 is requested, force the model to look for the high-res weights
        if int(crop_size) == 512:
            name = '512'
            which_epoch = '550000'

        opt = SimpleNamespace(
            # --- Core / BaseOptions ---
            isTrain=False,
            model='fs',                            
            name=name,                             
            checkpoints_dir=str(checkpoints_dir),  
            which_epoch=str(which_epoch),          
            gpu_ids=[self.device.index or 0],      
            verbose=False,
            resize_or_crop="none",
            load_pretrain="",
            gan_mode="hinge",
            lambda_feat=0.0,
            lambda_rec=0.0,
            no_ganFeat_loss=True,
            no_vgg_loss=True,
            
            # --- TestOptions Defaults ---
            ntest=float("inf"),
            results_dir='./results/',
            aspect_ratio=1.0,
            phase='test',
            how_many=50,
            cluster_path='features_clustered_010.npy',
            use_encoded_image=False,
            export_onnx=None,
            engine=None,
            onnx=None,
            Arc_path=str(arcface_ckpt),            
            pic_a_path='G:/swap_data/ID/elon-musk-hero-image.jpeg',
            pic_b_path='./demo_file/multi_people.jpg',
            pic_specific_path='./crop_224/zrf.jpg',
            multisepcific_dir='./demo_file/multispecific',
            video_path='G:/swap_data/video/HSB_Demo_Trim.mp4',
            temp_path='./temp_results',
            output_path='./output/',
            id_thres=0.03,
            no_simswaplogo=False,
            use_mask=False,
            crop_size=int(crop_size)               
        )

        torch.cuda.set_device(self.device.index or 0)
        self.model = create_model(opt)
        self.model.eval()
        # Freeze SimSwap weights but keep graph for input grads
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.crop_size = int(crop_size)

        # Precompute mean/std for ArcFace Identity Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _pad_and_resize(self, tensor: torch.Tensor, target_size: int) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int]]:
        """Pads a [C, H, W] tensor to square, then resizes to target_size."""
        _, h, w = tensor.shape
        original_shape = (h, w)
        max_dim = max(h, w)
        
        # Calculate padding to make it a square (left, right, top, bottom)
        pad_w = max_dim - w
        pad_h = max_dim - h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        padded = F.pad(tensor, padding, mode='constant', value=0.0)
        resized = F.interpolate(padded.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        
        return resized, padding, original_shape

    def _restore_shape(self, tensor: torch.Tensor, padding: tuple[int, int, int, int], original_shape: tuple[int, int]) -> torch.Tensor:
        """Restores a [C, target_size, target_size] tensor back to original_shape by resizing and unpadding."""
        h, w = original_shape
        max_dim = max(h, w)
        
        # Resize back to the square max_dim
        restored_square = F.interpolate(tensor.unsqueeze(0), size=(max_dim, max_dim), mode='bilinear', align_corners=False).squeeze(0)
        
        # Unpad (slice out the padded borders)
        pad_left, pad_right, pad_top, pad_bottom = padding
        restored = restored_square[:, pad_top : max_dim - pad_bottom, pad_left : max_dim - pad_right]
        return restored

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> Optional[torch.Tensor]:
        """Swaps faces between two ALREADY ALIGNED [C, H, W] tensors of arbitrary aspect ratios.
        
        Expects tensors to be in [-1, 1] or [0, 1] range. Returns a [0, 1] tensor.
        """
        try:
            # 1. Range Fix: Ensure tensors are strictly [0, 1]
            if source_aligned.min() < 0.0: source_aligned = (source_aligned + 1.0) / 2.0
            if target_aligned.min() < 0.0: target_aligned = (target_aligned + 1.0) / 2.0
            src = source_aligned.clamp(0.0, 1.0).to(self.device)
            tgt = target_aligned.clamp(0.0, 1.0).to(self.device)

            # 2. Pad to square and resize to 224x224 safely
            img_id_224, _, _ = self._pad_and_resize(src, self.crop_size)
            img_att_224, tgt_pad, tgt_shape = self._pad_and_resize(tgt, self.crop_size)

            # 3. Normalize Source for ArcFace (Identity Extraction)
            img_id_norm = (img_id_224.unsqueeze(0) - self.mean) / self.std
            
            # Extract Latent ID
            img_id_downsample = F.interpolate(img_id_norm, size=(112, 112), mode="bilinear", align_corners=False)
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            # 4. Forward Pass through GAN
            img_att_unsqueeze = img_att_224.unsqueeze(0)
            img_fake = self.model(img_id_norm, img_att_unsqueeze, latend_id, latend_id, True)

            # 5. Restore original target dimensions and return
            swapped_restored = self._restore_shape(img_fake.squeeze(0), tgt_pad, tgt_shape)

            return swapped_restored.clamp(0.0, 1.0)

        except Exception as e:
            import logging
            logging.error(f"FaceSwap failed: {e}")
            return None