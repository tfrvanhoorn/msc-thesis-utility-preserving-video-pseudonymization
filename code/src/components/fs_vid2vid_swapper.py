from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FsVid2VidFaceSwapper:
    """KFAAR wrapper around imaginaire fs-vid2vid for frame-wise inference.

    Inputs are expected in CHW format, in [0, 1] (or [-1, 1], which is auto-normalized).
    The source acts as the few-shot identity reference and the target provides driving landmarks.
    """

    def __init__(
        self,
        imaginaire_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        shape_predictor_path: str | Path | None = None,
        detector_upsample: int = 0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("fs-vid2vid swapper requires CUDA.")

        self.device = torch.device("cuda")
        self.imaginaire_root = Path(imaginaire_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.shape_predictor_path = Path(shape_predictor_path).resolve() if shape_predictor_path else None
        self.detector_upsample = max(int(detector_upsample), 0)

        if not self.config_path.exists():
            raise FileNotFoundError(f"fs-vid2vid config not found: {self.config_path}")
        if self.shape_predictor_path is None or not self.shape_predictor_path.exists():
            raise FileNotFoundError(
                "dlib shape predictor path is required for fs-vid2vid and must exist."
            )

        if str(self.imaginaire_root) not in sys.path:
            sys.path.insert(0, str(self.imaginaire_root))

        from imaginaire.config import Config
        from imaginaire.utils.io import get_checkpoint as imaginaire_get_checkpoint
        from imaginaire.utils.trainer import (
            get_model_optimizer_and_scheduler,
            get_trainer,
            set_random_seed,
        )
        from imaginaire.utils.visualization.face import connect_face_keypoints

        try:
            import dlib
        except Exception as exc:
            raise ImportError("dlib is required for fs-vid2vid swapping") from exc

        self._connect_face_keypoints = connect_face_keypoints
        self._dlib = dlib

        set_random_seed(seed, by_rank=False)
        cfg = Config(str(self.config_path))
        self.cfg = cfg

        net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg, seed=seed)
        self.trainer = get_trainer(
            cfg,
            net_G,
            net_D,
            opt_G,
            opt_D,
            sch_G,
            sch_D,
            None,
            None,
        )

        ckpt = self._resolve_checkpoint(imaginaire_get_checkpoint)
        self.trainer.load_checkpoint(cfg, str(ckpt))
        self.trainer.reset()

        self._detector = self._dlib.get_frontal_face_detector()
        self._predictor = self._dlib.shape_predictor(str(self.shape_predictor_path))

        h_str, w_str = [x.strip() for x in str(cfg.data.output_h_w).split(",")]
        self.output_h = int(h_str)
        self.output_w = int(w_str)

    def _resolve_checkpoint(self, imaginaire_get_checkpoint) -> Path:
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"fs-vid2vid checkpoint not found: {self.checkpoint_path}")
            return self.checkpoint_path

        pretrained = getattr(self.cfg, "pretrained_weight", "")
        if not pretrained:
            raise FileNotFoundError(
                "No checkpoint path provided and config has no pretrained_weight set."
            )

        default_ckpt = self.config_path.with_suffix("")
        default_ckpt = default_ckpt.parent / f"{default_ckpt.name}-{pretrained}.pt"
        ckpt_path = imaginaire_get_checkpoint(str(default_ckpt), pretrained)
        return Path(ckpt_path).resolve()

    def reset_sequence(self) -> None:
        self.trainer.reset()

    @staticmethod
    def _to_unit_range(img: torch.Tensor) -> torch.Tensor:
        out = img.detach().float()
        if out.min().item() < 0.0:
            out = out.add(1.0).div(2.0)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
        x = img.detach().cpu().clamp(0.0, 1.0)
        return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def _extract_landmarks68(self, image_chw: torch.Tensor) -> Optional[np.ndarray]:
        image_np = self._to_uint8_hwc(image_chw)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        dets = self._detector(gray, self.detector_upsample)
        if len(dets) == 0:
            return None
        rect = max(dets, key=lambda d: d.width() * d.height())
        shape = self._predictor(gray, rect)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.float32)
        if pts.shape[0] < 68:
            return None
        return pts[:68]

    def _build_label_map(self, keypoints: np.ndarray) -> torch.Tensor:
        # imaginaire expects keypoints as [T, K, 2] for face sketch rendering.
        labels = self._connect_face_keypoints(
            self.output_h,
            self.output_w,
            None,
            None,
            None,
            None,
            False,
            self.cfg.data,
            keypoints[np.newaxis, :, :],
        )
        label = torch.from_numpy(labels[0]).permute(2, 0, 1).float()
        return label

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        img = self._to_unit_range(img)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.output_h, self.output_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return img

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            source = self._prepare_image(source_aligned.to(self.device))
            target = self._prepare_image(target_aligned.to(self.device))

            src_lm = self._extract_landmarks68(source)
            tgt_lm = self._extract_landmarks68(target)
            if src_lm is None or tgt_lm is None:
                return None

            src_label = self._build_label_map(src_lm).to(self.device)
            tgt_label = self._build_label_map(tgt_lm).to(self.device)

            data = {
                "images": target.unsqueeze(0).unsqueeze(0),
                "label": tgt_label.unsqueeze(0).unsqueeze(0),
                "few_shot_images": source.unsqueeze(0).unsqueeze(0),
                "few_shot_label": src_label.unsqueeze(0).unsqueeze(0),
            }

            data = self.trainer.start_of_iteration(data, current_iteration=-1)
            out = self.trainer.test_single(data, output_dir=None)
            fake = out["fake_images"][0].detach()
            if fake.min().item() < 0.0:
                fake = fake.add(1.0).div(2.0)
            return fake.clamp(0.0, 1.0)
        except Exception as exc:
            logger.error("fs-vid2vid swap failed: %s", exc)
            return None

    def swap_batch(
        self,
        source_aligned_batch: list[torch.Tensor],
        target_aligned_batch: list[torch.Tensor],
    ) -> list[Optional[torch.Tensor]]:
        if len(source_aligned_batch) != len(target_aligned_batch):
            raise ValueError("source_aligned_batch and target_aligned_batch must have same length")
        outputs: list[Optional[torch.Tensor]] = []
        for src, tgt in zip(source_aligned_batch, target_aligned_batch):
            outputs.append(self.swap(src, tgt))
        return outputs
