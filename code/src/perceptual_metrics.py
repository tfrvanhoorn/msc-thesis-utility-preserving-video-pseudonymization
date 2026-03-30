from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips
except ImportError:  # pragma: no cover - dependency is optional at import time
    lpips = None

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - dependency is optional at import time
    structural_similarity = None


class PerceptualEvaluator:
    def __init__(
        self,
        *,
        device: torch.device,
        compute_lpips: bool,
        compute_ssim: bool,
        lpips_net: str = "alex",
        lpips_cache_dir: Path | None = None,
    ) -> None:
        self.device = device
        self.compute_lpips = bool(compute_lpips)
        self.compute_ssim = bool(compute_ssim)
        self.lpips_net = lpips_net
        self._lpips_model: torch.nn.Module | None = None

        if lpips_cache_dir is not None:
            cache_str = str(lpips_cache_dir)
            os.environ["TORCH_HOME"] = cache_str
            torch.hub.set_dir(cache_str)

        if self.compute_lpips and lpips is None:
            raise ImportError("LPIPS metric requested but package 'lpips' is not installed")
        if self.compute_ssim and structural_similarity is None:
            raise ImportError("SSIM metric requested but package 'scikit-image' is not installed")

    def prepare_video_pair(self, input_frames: torch.Tensor, output_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_frames.dim() != 4 or output_frames.dim() != 4:
            raise ValueError("Expected input/output frames as TCHW tensors")

        pair_count = min(int(input_frames.shape[0]), int(output_frames.shape[0]))
        input_trimmed = input_frames[:pair_count]
        output_trimmed = output_frames[:pair_count]

        if pair_count == 0:
            return input_trimmed, output_trimmed

        in_h, in_w = int(input_trimmed.shape[2]), int(input_trimmed.shape[3])
        out_h, out_w = int(output_trimmed.shape[2]), int(output_trimmed.shape[3])
        if (in_h, in_w) != (out_h, out_w):
            output_trimmed = F.interpolate(
                output_trimmed,
                size=(in_h, in_w),
                mode="bilinear",
                align_corners=False,
            )
        return input_trimmed, output_trimmed

    def compute_frame_pair(self, input_frame: torch.Tensor, output_frame: torch.Tensor) -> tuple[float | None, float | None]:
        if input_frame.dim() != 3 or output_frame.dim() != 3:
            raise ValueError("Expected per-frame tensors in CHW format")

        lpips_value: float | None = None
        ssim_value: float | None = None

        if self.compute_lpips:
            model = self._get_lpips_model()
            input_batch = input_frame.unsqueeze(0).to(self.device).clamp(0.0, 1.0)
            output_batch = output_frame.unsqueeze(0).to(self.device).clamp(0.0, 1.0)
            # LPIPS expects normalized tensors in [-1, 1].
            input_batch = input_batch * 2.0 - 1.0
            output_batch = output_batch * 2.0 - 1.0
            with torch.no_grad():
                lpips_value = float(model(input_batch, output_batch).item())

        if self.compute_ssim:
            ssim_value = self._compute_ssim_value(input_frame, output_frame)

        return lpips_value, ssim_value

    def _get_lpips_model(self) -> torch.nn.Module:
        if self._lpips_model is None:
            if lpips is None:
                raise ImportError("LPIPS metric requested but package 'lpips' is not installed")
            self._lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    @staticmethod
    def _compute_ssim_value(input_frame: torch.Tensor, output_frame: torch.Tensor) -> float | None:
        if structural_similarity is None:
            raise ImportError("SSIM metric requested but package 'scikit-image' is not installed")

        input_np = PerceptualEvaluator._to_hwc_numpy(input_frame)
        output_np = PerceptualEvaluator._to_hwc_numpy(output_frame)
        min_dim = min(int(input_np.shape[0]), int(input_np.shape[1]))
        if min_dim < 3:
            return None

        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return None

        value = structural_similarity(
            input_np,
            output_np,
            channel_axis=2,
            data_range=1.0,
            win_size=win_size,
        )
        return float(value)

    @staticmethod
    def _to_hwc_numpy(frame: torch.Tensor) -> np.ndarray:
        chw = frame.detach().to("cpu")
        if chw.dtype != torch.float32:
            chw = chw.float()
        chw = chw.clamp(0.0, 1.0)
        return chw.permute(1, 2, 0).numpy()