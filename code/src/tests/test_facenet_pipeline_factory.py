from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar.config import DataConfig, DetectorConfig, EmbeddingConfig, PipelineConfig, ProjectorConfig, SeedConfig
from anon_pipeline.kfaar.pipeline.factory import build_kfaar_pipeline
from anon_pipeline.kfaar.components import load_stylegan2
from anon_pipeline.shared.data.loaders import iter_samples

# Adjustable knobs
NUM_IDENTITIES: int = 2
SAMPLES_PER_IDENTITY: int = 2
KEY_DIM: int = 128

# Paths
DATA_ROOT = PROJECT_ROOT / "data" / "celeba"
STYLEGAN_CKPT = PROJECT_ROOT / "src" / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl"
TARGET_DEVICE = torch.device("cpu")
# TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _build_config() -> PipelineConfig:
    data_cfg = DataConfig(
        dataset_path=DATA_ROOT,
        dataset_type="celeba",
        options={
            # Only cap per-identity; let the loader stream until the batch is filled.
            "max_per_identity": SAMPLES_PER_IDENTITY,
        },
    )
    detector_cfg = DetectorConfig(score_threshold=0.4, image_size=160, margin=0, min_face_size=20, max_faces=1, device=str(TARGET_DEVICE))
    embedding_cfg = EmbeddingConfig(method="facenet", pretrained="vggface2", device=str(TARGET_DEVICE))
    seed_cfg = SeedConfig(secret_key="dummy")
    projector_cfg = ProjectorConfig(key_dim=KEY_DIM)
    return PipelineConfig(data=data_cfg, detector=detector_cfg, embedding=embedding_cfg, seed=seed_cfg, projector=projector_cfg)


def _load_batch(data_cfg: DataConfig) -> tuple[list[np.ndarray], torch.Tensor]:
    target = NUM_IDENTITIES * SAMPLES_PER_IDENTITY
    counts: dict[str, int] = {}
    images: list[np.ndarray] = []
    labels: list[int] = []

    for sample in iter_samples(data_cfg):
        if counts.get(sample.identity, 0) >= SAMPLES_PER_IDENTITY:
            continue
        if len(counts) >= NUM_IDENTITIES and sample.identity not in counts:
            continue

        try:
            arr = np.asarray(Image.open(sample.path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load %s (%s); skipping", sample.path, exc)
            continue

        images.append(arr)
        labels.append(int(sample.identity))
        counts[sample.identity] = counts.get(sample.identity, 0) + 1

        if len(images) >= target:
            break

    return images, torch.tensor(labels, dtype=torch.long)


def verify_parameter_changes(pipeline, images, labels, key1, key2):
    # 1. Capture snapshots of parameters before training
    # We use .clone() to ensure we have a static copy of the values
    original_projector_params = {n: p.clone() for n, p in pipeline.projector.named_parameters()}
    original_embedder_params = {n: p.clone() for n, p in pipeline.embedder.model.named_parameters()}
    original_stylegan_params = {n: p.clone() for n, p in pipeline.stylegan._G.named_parameters()}

    # 2. Run a training step
    pipeline.hpvg_train_step(images, labels, key1, key2)

    # 3. Check Projector (SHOULD change)
    projector_changed = False
    for name, param in pipeline.projector.named_parameters():
        if not torch.equal(original_projector_params[name], param):
            projector_changed = True
            break
    
    # 4. Check Embedder (SHOULD NOT change)
    embedder_changed = False
    for name, param in pipeline.embedder.model.named_parameters():
        if not torch.equal(original_embedder_params[name], param):
            embedder_changed = True
    
    # 5. Check StyleGAN (SHOULD NOT change)
    stylegan_changed = False
    for name, param in pipeline.stylegan._G.named_parameters():
        if not torch.equal(original_stylegan_params[name], param):
            stylegan_changed = True

    # Report results
    logging.info(f"Projector parameters changed: {projector_changed}")
    logging.info(f"Embedder parameters changed: {embedder_changed}")
    logging.info(f"StyleGAN parameters changed: {stylegan_changed}")
    
    if projector_changed and not embedder_changed and not stylegan_changed:
        logging.info("SUCCESS: Only Projector parameters were updated.")
    else:
        logging.error("FAILURE: Parameter freeze logic is incorrect.")


def run_pipeline_training_step_smoke() -> None:
    _configure_logging()

    if not STYLEGAN_CKPT.exists():
        logging.warning("StyleGAN2 checkpoint missing at %s; skipping training smoke test", STYLEGAN_CKPT)
        return

    cfg = _build_config()
    images, labels = _load_batch(cfg.data)
    if len(images) < NUM_IDENTITIES * SAMPLES_PER_IDENTITY:
        logging.warning("Insufficient samples for the requested batch; skipping test")
        return

    try:
        stylegan = load_stylegan2(ckpt_path=STYLEGAN_CKPT, device=TARGET_DEVICE)
    except Exception as exc:  # noqa: BLE001
        logging.warning("StyleGAN2 load failed (%s); skipping test", exc)
        return

    pipeline = build_kfaar_pipeline(cfg, stylegan=stylegan, device=TARGET_DEVICE)

    key1 = torch.randn(KEY_DIM, device=TARGET_DEVICE)
    key2 = torch.randn(KEY_DIM, device=TARGET_DEVICE)

    loss = pipeline.hpvg_train_step(images, labels, key1, key2)
    logging.info("Training step completed | loss=%.6f | batch=%d", loss.item(), len(images))

    verify_parameter_changes(pipeline, images, labels, key1, key2)


if __name__ == "__main__":
    run_pipeline_training_step_smoke()
