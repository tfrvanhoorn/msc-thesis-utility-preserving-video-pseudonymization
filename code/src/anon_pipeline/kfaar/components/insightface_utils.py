from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Tuple

import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)


def load_insightface_swapper(
    model_path: Path,
    analyzer_name: str = "buffalo_l",
    det_size: Tuple[int, int] = (640, 640),
    device: torch.device | str = "cuda",
) -> tuple[object | None, object | None]:
    """Load InsightFace FaceAnalysis and inswapper model.

    Returns (swapper, analyzer) or (None, None) on failure.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to import insightface: %s", exc)
        raise

    # Prefer CUDA, allow CPU fallback
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_providers = ort.get_available_providers()
    logger.debug(
        "InsightFace init: torch.cuda.is_available=%s, ort=%s, ort providers=%s",
        torch.cuda.is_available(),
        ort.__version__,
        ort_providers,
    )

    # Analyzer
    try:
        analyzer = FaceAnalysis(name=analyzer_name, providers=providers)
    except TypeError:
        analyzer = FaceAnalysis(name=analyzer_name)
    analyzer.prepare(ctx_id=0, det_size=det_size)

    # Swapper
    try:
        swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)
    except TypeError:
        swapper = insightface.model_zoo.get_model(str(model_path))
    return swapper, analyzer
