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

    Returns (swapper, analyzer) or raises if CUDA is not used.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to import insightface: %s", exc)
        raise

    # Basic CUDA checks
    ctx_device = torch.device(device)
    if ctx_device.type != "cuda":
        msg = f"InsightFace swapper/analyzer require CUDA, got device={ctx_device}"
        logger.error(msg)
        raise RuntimeError(msg)
    if not torch.cuda.is_available():
        msg = "CUDA requested for InsightFace but torch reports no CUDA device"
        logger.error(msg)
        raise RuntimeError(msg)

    providers = ["CUDAExecutionProvider"]
    ort_providers = ort.get_available_providers()
    logger.debug(
        "InsightFace init: torch.cuda.is_available=%s, ort=%s, ort providers=%s",
        torch.cuda.is_available(),
        ort.__version__,
        ort_providers,
    )
    if "CUDAExecutionProvider" not in ort_providers:
        msg = f"onnxruntime CUDAExecutionProvider not available (providers={ort_providers})"
        logger.error(msg)
        raise RuntimeError(msg)

    # Analyzer
    try:
        analyzer = FaceAnalysis(name=analyzer_name, providers=providers)
    except TypeError:
        analyzer = FaceAnalysis(name=analyzer_name)
    analyzer.prepare(ctx_id=0, det_size=det_size)
    analyzer_providers = getattr(analyzer, "providers", None)

    # Fallback: inspect loaded model sessions for provider info
    def _analyzer_has_cuda(app: object) -> tuple[bool, list[str]]:
        provs: list[str] = []
        for model in getattr(app, "models", {}).values():
            sess = getattr(model, "session", None)
            if sess and hasattr(sess, "get_providers"):
                p = list(sess.get_providers())
                provs.extend(p)
        return ("CUDAExecutionProvider" in provs), provs

    has_cuda, model_provs = _analyzer_has_cuda(analyzer)
    if analyzer_providers:
        has_cuda = has_cuda or ("CUDAExecutionProvider" in analyzer_providers)
        model_provs = analyzer_providers if not model_provs else model_provs

    if not has_cuda:
        msg = (
            f"InsightFace analyzer initialized without CUDA; "
            f"providers={analyzer_providers or model_provs}, "
            f"ort_available={ort_providers}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Swapper
    try:
        swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)
    except TypeError:
        swapper = insightface.model_zoo.get_model(str(model_path))
    swapper_providers = None
    if hasattr(swapper, "providers"):
        swapper_providers = swapper.providers
    elif hasattr(swapper, "session") and hasattr(swapper.session, "get_providers"):
        swapper_providers = swapper.session.get_providers()
    if not swapper_providers or "CUDAExecutionProvider" not in swapper_providers:
        msg = (
            f"InsightFace swapper initialized without CUDA; "
            f"providers={swapper_providers}, ort_available={ort_providers}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return swapper, analyzer
