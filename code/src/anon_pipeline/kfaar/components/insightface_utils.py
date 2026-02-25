from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch


logger = logging.getLogger(__name__)


def load_insightface_swapper(
    model_path: Path,
    analyzer_name: str = "buffalo_l",
    det_size: Tuple[int, int] = (640, 640),
    device: torch.device | str = "cpu",
) -> tuple[object | None, object | None]:
    """Load InsightFace FaceAnalysis and inswapper model.

    Returns (swapper, analyzer) or (None, None) on failure.
    """

    try:
        import insightface
        from insightface.app import FaceAnalysis

        ctx_device = torch.device(device)
        ctx_id = 0 if ctx_device.type == "cuda" else -1

        analyzer = FaceAnalysis(name=analyzer_name)
        analyzer.prepare(ctx_id=ctx_id, det_size=det_size)

        download_flag = not Path(model_path).exists()
        swapper = insightface.model_zoo.get_model(
            str(model_path), download=download_flag, download_zip=download_flag
        )
        return swapper, analyzer
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize InsightFace swapper/analyzer: %s", exc)
        return None, None
