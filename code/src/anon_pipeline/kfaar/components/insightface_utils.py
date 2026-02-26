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
    device: torch.device | str = "cuda",
) -> tuple[object | None, object | None]:
    """Load InsightFace FaceAnalysis and inswapper model.

    Returns (swapper, analyzer) or (None, None) on failure.
    """

    try:
        import insightface
        from insightface.app import FaceAnalysis

        ctx_device = torch.device(device)
        if ctx_device.type != "cuda":
            logger.error("InsightFace swapper/analyzer require CUDA, got device=%s", ctx_device)
            raise RuntimeError(f"InsightFace swapper/analyzer require CUDA, got device={ctx_device}")
        if not torch.cuda.is_available():
            logger.error("CUDA requested for InsightFace but torch reports no CUDA device")
            raise RuntimeError("CUDA requested for InsightFace but torch reports no CUDA device")

        ctx_id = 0
        providers = ["CUDAExecutionProvider"]

        try:
            analyzer = FaceAnalysis(name=analyzer_name, providers=providers)
        except TypeError:
            # Older insightface may not support providers kwarg; fall back
            analyzer = FaceAnalysis(name=analyzer_name)
        analyzer.prepare(ctx_id=ctx_id, det_size=det_size)

        analyzer_providers = getattr(analyzer, "providers", providers)
        if "CUDAExecutionProvider" not in analyzer_providers:
            logger.error("InsightFace analyzer initialized without CUDA; providers=%s", analyzer_providers)
            raise RuntimeError(f"InsightFace analyzer initialized without CUDA; providers={analyzer_providers}")

        download_flag = not Path(model_path).exists()
        try:
            swapper = insightface.model_zoo.get_model(
                str(model_path), download=download_flag, download_zip=download_flag, providers=providers
            )
        except TypeError:
            # Older insightface may not support providers kwarg; fall back
            swapper = insightface.model_zoo.get_model(
                str(model_path), download=download_flag, download_zip=download_flag
            )
        swapper_providers = providers
        if hasattr(swapper, "providers"):
            swapper_providers = getattr(swapper, "providers")
        elif hasattr(swapper, "session") and hasattr(swapper.session, "get_providers"):
            swapper_providers = swapper.session.get_providers()

        if "CUDAExecutionProvider" not in swapper_providers:
            logger.error("InsightFace swapper initialized without CUDA; providers=%s", swapper_providers)
            raise RuntimeError(f"InsightFace swapper initialized without CUDA; providers={swapper_providers}")

        return swapper, analyzer
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize InsightFace swapper/analyzer: %s", exc)
        return None, None
