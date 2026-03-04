from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class SimSwapFaceSwapper:
    """Wrapper to run SimSwap for single-image swapping using affine alignment."""

    def __init__(
        self,
        simswap_root: Path,
        checkpoints_dir: Path,
        name: str,
        which_epoch: str,
        arcface_ckpt: Path,
        parsing_ckpt: Optional[Path] = None,
        detector_name: str = "antelopev2",
        detector_root: Optional[Path] = None,
        crop_size: int = 224,
        use_mask: bool = True,
        device: torch.device | str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SimSwap requires CUDA.")

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
            
        simswap_root = Path(simswap_root).resolve()
        if str(simswap_root) not in sys.path:
            sys.path.insert(0, str(simswap_root))

        from models.fs_model import fsModel  # type: ignore
        from insightface_func.face_detect_crop_single import Face_detect_crop # type: ignore
        from util.norm import SpecificNorm # type: ignore
        
        # 1. Initialize SimSwap Model
        opt = SimpleNamespace(
            isTrain=False, resize_or_crop="none", crop_size=int(crop_size),
            Arc_path=str(arcface_ckpt), checkpoints_dir=str(checkpoints_dir),
            name=name, which_epoch=str(which_epoch), gpu_ids=[self.device.index or 0],
            verbose=False, load_pretrain="", gan_mode="hinge",
            lambda_feat=0.0, lambda_rec=0.0, no_ganFeat_loss=True, no_vgg_loss=True,
        )
        torch.cuda.set_device(self.device.index or 0)
        self.model = fsModel()
        self.model.initialize(opt)
        self.model.eval()

        # Detector root (defaults to SimSwap insightface models directory)
        if detector_root is None:
            actual_detector_root = str(simswap_root / "insightface_func" / "models")
        else:
            actual_detector_root = str(Path(detector_root).resolve())

        # 2. Initialize Detector/Aligner
        self.app = Face_detect_crop(name=detector_name, root=actual_detector_root)
        # Using 'None' mode for generic arbitrary images
        self.app.prepare(ctx_id=self.device.index or 0, det_thresh=0.6, det_size=(640,640), mode='None')
        
        # 3. Initialize Parsing Model (for masking)
        self.use_mask = use_mask
        self.net = None
        if self.use_mask and parsing_ckpt is not None:
            from parsing_model.model import BiSeNet # type: ignore
            self.net = BiSeNet(n_classes=19)
            self.net.to(self.device)
            self.net.load_state_dict(torch.load(str(parsing_ckpt), map_location=self.device))
            self.net.eval()
            
        self.spNorm = SpecificNorm()
        self.crop_size = int(crop_size)
        
        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Dummy logoclass to satisfy reverse2wholeimage arguments (disabling logo anyway)
        class DummyLogo:
            def apply_frames(self, x): return x
        self.logoclass = DummyLogo()

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts [C, H, W] [0, 1] tensor to [H, W, C] [0, 255] BGR numpy array."""
        rgb_np = (tensor.detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    def _bgr_to_tensor(self, bgr_np: np.ndarray, target_device: torch.device) -> torch.Tensor:
        """Converts [H, W, C] [0, 255] BGR numpy array to [C, H, W] [0, 1] tensor."""
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        return tensor.to(target_device)

    def swap(self, source_img: torch.Tensor, target_img: torch.Tensor) -> Optional[torch.Tensor]:
        from util.reverse2original import reverse2wholeimage # type: ignore
        
        # --- 1. Prepare Images ---
        # Ensure tensors are [0, 1]
        if source_img.min() < 0.0: source_img = (source_img + 1.0) / 2.0
        if target_img.min() < 0.0: target_img = (target_img + 1.0) / 2.0
        source_img = source_img.clamp(0.0, 1.0)
        target_img = target_img.clamp(0.0, 1.0)

        src_bgr = self._tensor_to_bgr(source_img)
        tgt_bgr = self._tensor_to_bgr(target_img)

        # --- 2. Identity Extraction (Source) ---
        img_a_align_crop_list, _ = self.app.get(src_bgr, self.crop_size)
        if not img_a_align_crop_list:
            # Fallback: if source is already an aligned StyleGAN output, use it directly
            src_crop = cv2.resize(src_bgr, (self.crop_size, self.crop_size))
        else:
            src_crop = img_a_align_crop_list[0]

        img_a_pil = Image.fromarray(cv2.cvtColor(src_crop, cv2.COLOR_BGR2RGB))
        img_id = self.transformer_Arcface(img_a_pil).unsqueeze(0).to(self.device)

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        # --- 3. Target Extraction and Swap ---
        img_b_align_crop_list, b_mat_list = self.app.get(tgt_bgr, self.crop_size)
        if not img_b_align_crop_list:
            return None # No face found in target frame

        swap_result_list = []
        b_align_crop_tensor_list = []

        for b_align_crop in img_b_align_crop_list:
            crop_rgb = cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
            b_align_crop_tensor = torch.from_numpy(crop_rgb).float().div(255)
            b_align_crop_tensor = b_align_crop_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            swap_result = self.model(None, b_align_crop_tensor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tensor_list.append(b_align_crop_tensor)

        # --- 4. Reverse Mapping ---
        # reverse2wholeimage forces a save to disk. We use a temp file to bridge it back to memory.
        # (In the future, editing reverse2wholeimage to just return the image is faster for training).
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        reverse2wholeimage(
            b_align_crop_tensor_list, swap_result_list, b_mat_list, self.crop_size,
            tgt_bgr, self.logoclass, tmp_path, no_simswaplogo=True,
            pasring_model=self.net, use_mask=self.use_mask, norm=self.spNorm
        )

        result_bgr = cv2.imread(tmp_path)
        os.remove(tmp_path)

        if result_bgr is None:
            return None

        return self._bgr_to_tensor(result_bgr, target_img.device)

        # except Exception as e:
        #     import logging
        #     logging.error(f"FaceSwap failed: {e}")
        #     return None