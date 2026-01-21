from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar import KfaarTrainer  # noqa: E402
from anon_pipeline.kfaar.config import (  # noqa: E402
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from anon_pipeline.kfaar.pipeline.factory import build_kfaar_pipeline  # noqa: E402
from anon_pipeline.kfaar.components import load_stylegan2  # noqa: E402
from anon_pipeline.shared.data.splits import build_train_test_loaders  # noqa: E402


NUM_IDENTITIES = 8
SAMPLES_PER_IDENTITY = 2
BATCH_IDENTITIES = 2
SAMPLES_PER_IDENTITY_IN_BATCH = 2
TRAIN_FRACTION = 0.5
SPLIT_SEED = 42

STYLEGAN_CKPT = PROJECT_ROOT / "src" / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl"
TARGET_DEVICE = torch.device("cpu")


def _build_config() -> PipelineConfig:
    data_cfg = DataConfig(
        dataset_path=PROJECT_ROOT / "data" / "celeba",
        dataset_type="celeba",
        options={
            "max_per_identity": SAMPLES_PER_IDENTITY,
        },
    )
    detector_cfg = DetectorConfig(score_threshold=0.4, image_size=256, margin=0, min_face_size=20, max_faces=1, device=str(TARGET_DEVICE))
    embedding_cfg = EmbeddingConfig(method="facenet", pretrained="vggface2", device=str(TARGET_DEVICE))
    seed_cfg = SeedConfig(secret_key="dummy")
    projector_cfg = ProjectorConfig(key_dim=128)
    return PipelineConfig(data=data_cfg, detector=detector_cfg, embedding=embedding_cfg, seed=seed_cfg, projector=projector_cfg)


def test_kfaar_trainer_identity_batches(tmp_path: Path) -> None:
    if not STYLEGAN_CKPT.exists():
        logging.warning("StyleGAN2 checkpoint missing; skipping integration test")
        return

    cfg = _build_config()
    split, train_loader, val_loader = build_train_test_loaders(
        cfg.data,
        train_fraction=TRAIN_FRACTION,
        seed=SPLIT_SEED,
        max_identities=NUM_IDENTITIES,
        batch_size=BATCH_IDENTITIES * SAMPLES_PER_IDENTITY_IN_BATCH,
        shuffle_train=False,
        shuffle_test=False,
    )

    if not split.train or not split.test:
        logging.warning("Insufficient identities to create train/val split; skipping")
        return

    stylegan = load_stylegan2(ckpt_path=STYLEGAN_CKPT, device=TARGET_DEVICE)
    pipeline = build_kfaar_pipeline(cfg, stylegan=stylegan, device=TARGET_DEVICE)

    call_counts = {"steps": 0}
    original_step = pipeline.hpvg_train_step

    def counting_step(*args, **kwargs):
        call_counts["steps"] += 1
        return original_step(*args, **kwargs)

    pipeline.hpvg_train_step = counting_step  # type: ignore[assignment]

    trainer = KfaarTrainer(
        pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        key_dim=cfg.projector.key_dim,
        batch_identities=BATCH_IDENTITIES,
        samples_per_identity=SAMPLES_PER_IDENTITY_IN_BATCH,
        checkpoint_dir=tmp_path,
        device=TARGET_DEVICE,
        train_identities=split.train,
        val_identities=split.test,
    )

    trainer.train()
    logging.info("Train steps executed: %s", call_counts["steps"])
    ckpt_path = tmp_path / "kfaar_projector_epoch_1.pt"
    logging.info("Checkpoint exists: %s", ckpt_path.exists())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    temp_dir = Path(tempfile.mkdtemp(prefix="kfaar_trainer_"))
    logging.info("Running direct test with temp dir %s", temp_dir)
    test_kfaar_trainer_identity_batches(temp_dir)
