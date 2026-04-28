from __future__ import annotations

try:
    from typing import Protocol
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol

import torch

from .detector import Detection


class FaceAligner(Protocol):
    def align(self, image: torch.Tensor, detection: Detection) -> torch.Tensor:
        ...

    def align_batch(self, images: list[torch.Tensor], detections: list[Detection | None]) -> list[torch.Tensor]:
        ...


class MTCNNAligner:
    def __init__(self, output_size: int = 160) -> None:
        self.output_size = output_size

    def align(self, image: torch.Tensor, detection: Detection) -> torch.Tensor:
        img = self._to_tensor(image)
        bbox = detection.bbox.to(img.device)

        x1, y1, x2, y2 = bbox.round().long()
        h, w = img.shape[1], img.shape[2]
        x1 = x1.clamp(0, w)
        x2 = x2.clamp(0, w)
        y1 = y1.clamp(0, h)
        y2 = y2.clamp(0, h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clamping: {(x1, y1, x2, y2)}")

        face = img[:, y1:y2, x1:x2].unsqueeze(0)
        face = torch.nn.functional.interpolate(face, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        face = face.squeeze(0).contiguous()
        return face

    def align_batch(self, images: list[torch.Tensor], detections: list[Detection | None]) -> list[torch.Tensor]:
        if len(images) != len(detections):
            raise ValueError("images and detections must have equal length")

        aligned_faces: list[torch.Tensor] = []
        for image, detection in zip(images, detections):
            if detection is None:
                aligned_faces.append(torch.empty(0))
                continue
            aligned_faces.append(self.align(image, detection))
        return aligned_faces

    @staticmethod
    def _to_tensor(image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            img = image
        else:
            raise TypeError(f"Expected torch.Tensor image, got {type(image)}")

        if img.dim() == 3 and img.shape[0] == 3:
            pass
        elif img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected image shape (3,H,W) or (H,W,3), got {tuple(img.shape)}")

        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.0 or img.min() < 0.0:
            img = img / 255.0
        return img
