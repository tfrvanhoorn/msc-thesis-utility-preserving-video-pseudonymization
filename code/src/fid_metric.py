from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:  # pragma: no cover - dependency is optional at import time
    FrechetInceptionDistance = None


class FidEvaluator:
    def __init__(
        self,
        *,
        device: torch.device,
        feature: int = 2048,
        normalize: bool = True,
        feature_extractor_weights_path: Path | None = None,
        antialias: bool = True,
    ) -> None:
        self.device = device
        self.real_frames = 0
        self.generated_frames = 0

        if FrechetInceptionDistance is None:
            raise ImportError(
                "FID metric requested but package 'torchmetrics[image]' or 'torch-fidelity' is not installed"
            )

        try:
            self._metric = FrechetInceptionDistance(
                feature=feature,
                normalize=normalize,
                feature_extractor_weights_path=feature_extractor_weights_path,
                antialias=antialias,
            ).to(device)
            self._metric = self._metric.set_dtype(torch.float64)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "FID metric requested but package 'torchmetrics[image]' or 'torch-fidelity' is not installed"
            ) from exc

    def update_real(self, frames: torch.Tensor) -> None:
        batch = self._prepare_frames(frames)
        if batch.numel() == 0:
            return
        self._metric.update(batch, real=True)
        self.real_frames += int(batch.shape[0])

    def update_generated(self, frames: torch.Tensor) -> None:
        batch = self._prepare_frames(frames)
        if batch.numel() == 0:
            return
        self._metric.update(batch, real=False)
        self.generated_frames += int(batch.shape[0])

    def compute(self) -> float | None:
        if self.real_frames == 0 or self.generated_frames == 0:
            return None
        value = self._metric.compute()
        return float(value.item())

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError("Expected frames as TCHW or NCHW tensors")

        batch = frames.detach().to(self.device)
        if batch.dtype != torch.float32:
            batch = batch.float()
        batch = batch.clamp(0.0, 1.0)
        if batch.shape[0] == 0:
            return batch
        if int(batch.shape[2]) != 299 or int(batch.shape[3]) != 299:
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        return batch
