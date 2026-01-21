from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar.components.detector import MTCNNDetector
from anon_pipeline.kfaar.components.embedding import FacenetEmbedder
from anon_pipeline.kfaar.components.projector import ProjectorMLP
from anon_pipeline.kfaar.components import load_stylegan2

# Adjustable knobs for quick experimentation
NUM_IDENTITIES: int = 2
SAMPLES_PER_IDENTITY: int = 1
KEY_DIM: int = 128
# Fixed key tensor (shared across all samples) to keep the pipeline deterministic/trainable
GLOBAL_KEY = torch.randn(KEY_DIM)
RUN_BACKPROP_CHECK: bool = True

# Optional StyleGAN2 checkpoint (set to an existing .pkl to enable mapping/generation)
STYLEGAN_CKPT = PROJECT_ROOT / "src" / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl"
STYLEGAN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Point to a folder with face images (CelebA layout by default)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "celeba" / "img_align_celeba"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _load_images(limit: int) -> List[Path]:
    if not DATA_ROOT.exists():
        return []
    images = sorted(DATA_ROOT.glob("*.jpg"))
    return images[:limit]


def run_facenet_smoke() -> None:
    """Smoke test for detection + embedding without pytest."""
    _configure_logging()
    max_images = NUM_IDENTITIES * SAMPLES_PER_IDENTITY
    sample_paths = _load_images(max_images)
    if not sample_paths:
        logging.warning("No images found under %s; skipping smoke run", DATA_ROOT)
        return

    detector = MTCNNDetector(image_size=160, margin=0, score_threshold=0.4, max_faces=1)
    embedder = FacenetEmbedder(pretrained="vggface2")
    projector = ProjectorMLP(key_dim=KEY_DIM, output_dim=embedder.embedding_size)

    stylegan = None
    if STYLEGAN_CKPT.exists():
        try:
            stylegan = load_stylegan2(ckpt_path=STYLEGAN_CKPT, device=STYLEGAN_DEVICE)
            if RUN_BACKPROP_CHECK and hasattr(stylegan, "_G"):
                stylegan._G.requires_grad_(True)
            logging.info(
                "Loaded StyleGAN2: ckpt=%s z_dim=%d w_dim=%d device=%s",
                STYLEGAN_CKPT,
                stylegan.z_dim,
                stylegan.w_dim,
                STYLEGAN_DEVICE,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("StyleGAN2 load failed (%s); skipping mapper/generator step", exc)
            stylegan = None
    else:
        logging.info("StyleGAN2 checkpoint not found at %s; skipping mapper/generator step", STYLEGAN_CKPT)

    total_detections = 0
    total_embeddings = 0

    for path in sample_paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        logging.info("Processing image: %s", path.name)

        detections = detector.detect(image)
        logging.info("Detections found: %d", len(detections))
        if not detections:
            raise RuntimeError(f"Expected at least one detection for {path.name}")

        aligned = [d.aligned for d in detections if d.aligned is not None]
        if not aligned:
            raise RuntimeError(f"No aligned faces returned for {path.name}")

        embeddings = embedder.embed(aligned, source_paths=[path] * len(aligned))
        if RUN_BACKPROP_CHECK:
            embeddings = embedder.embed(aligned, source_paths=[path] * len(aligned), with_grad=True)
        logging.info("Embeddings shape (torch): %s", tuple(embeddings.shape))
        if embeddings.shape[0] != len(aligned):
            raise RuntimeError("Mismatch between aligned faces and embeddings")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Non-finite values in embeddings")

        key = GLOBAL_KEY.to(embeddings.device).expand(embeddings.shape[0], -1)
        logging.info("Using fixed global key tensor shape: %s", tuple(key.shape))
        projected = projector.project(embeddings, key)
        logging.info("Projected embeddings shape: %s (torch)", tuple(projected.shape))
        if projected.shape[0] != embeddings.shape[0]:
            raise RuntimeError("Mismatch between embeddings and projected outputs")
        if not torch.isfinite(projected).all():
            raise RuntimeError("Non-finite values in projected embeddings")

        if stylegan is not None:
            z = torch.randn(projected.shape[0], stylegan.z_dim, device=STYLEGAN_DEVICE, requires_grad=True)
            w = stylegan.map(z)
            logging.info("StyleGAN2 mapping: z shape=%s w shape=%s", tuple(z.shape), tuple(w.shape))
            images = stylegan.synthesize(w, noise_mode="const")
            logging.info("StyleGAN2 synthesis output shape: %s", tuple(images.shape))
            if not torch.isfinite(images).all():
                raise RuntimeError("Non-finite values in synthesized images")

            # Re-embed synthesized images with gradients for downstream loss computation
            images_01 = images.clamp(-1, 1).add(1).div(2.0)
            synth_faces = [img for img in images_01]
            synth_embeddings = embedder.embed(synth_faces, with_grad=RUN_BACKPROP_CHECK)
            logging.info("Synth embeddings shape (torch, grad): %s | requires_grad=%s", tuple(synth_embeddings.shape), synth_embeddings.requires_grad)
            if not torch.isfinite(synth_embeddings).all():
                raise RuntimeError("Non-finite values in synthesized embeddings")

            if RUN_BACKPROP_CHECK:
                # Minimal dummy loss to exercise gradients end-to-end
                loss = projected.pow(2).mean() + synth_embeddings.pow(2).mean()
                projector.zero_grad(set_to_none=True)
                if hasattr(stylegan, "_G"):
                    stylegan._G.zero_grad(set_to_none=True)
                if hasattr(embedder, "model"):
                    embedder.model.zero_grad(set_to_none=True)
                loss.backward()
                logging.info(
                    "Backprop check completed | loss=%.6f | proj_grad=%s | stylegan_grad=%s | embedder_grad=%s",
                    loss.item(),
                    any(p.grad is not None for p in projector.parameters()),
                    any(p.grad is not None for p in stylegan._G.parameters()) if hasattr(stylegan, "_G") else False,
                    any(p.grad is not None for p in embedder.model.parameters()) if hasattr(embedder, "model") else False,
                )

        total_detections += len(detections)
        total_embeddings += embeddings.shape[0]

    logging.info(
        "Completed facenet smoke test: images=%d detections=%d embeddings=%d",
        len(sample_paths),
        total_detections,
        total_embeddings,
    )


if __name__ == "__main__":
    run_facenet_smoke()
