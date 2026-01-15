"""Quick KFAAR embedding smoke test over CelebA.

Configure DATASET_ROOT to your CelebA root (containing img_align_celeba and identity_CelebA.txt).
Adjust NUM_IDENTITIES / MAX_SAMPLES_PER_IDENTITY for shorter runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator
import sys
import torch
import numpy as np
from PIL import Image

# Ensure local src is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from anon_pipeline.kfaar.config import (
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    ExperimentConfig,
    FeatureSelectorConfig,
    SeedConfig,
)
from anon_pipeline.kfaar import KfaarPipeline, build_kfaar_pipeline
from anon_pipeline.kfaar import ProjectorMLP
from anon_pipeline.kfaar.models import load_stylegan2
from anon_pipeline.shared.data.loaders import CelebADataset
from anon_pipeline.shared.data.io import load_image

TUNE_SCORE_THRESHOLD = 0.5  # lower threshold to catch more faces

# Tweak these for your run
DATASET_ROOT = Path("data/celeba")
NUM_IDENTITIES: int | None = 2  # set None to traverse all identities
MAX_SAMPLES_PER_IDENTITY: int | None = 2  # set None for no per-identity cap
MAX_SAMPLES_GLOBAL: int | None = None  # optional global cap
KEY_BITS: int = 128  # projector key length
TEST_MAPPER: bool = True  # toggles StyleGAN2 mapper smoke test
PREFER_CUDA_FOR_GENERATOR: bool = True  # try to keep StyleGAN2 on GPU when available

# Detector/embedding defaults; adjust providers/ctx if you have GPU
_DETECTOR = DetectorConfig(
    score_threshold=TUNE_SCORE_THRESHOLD,
    image_size=160,
    margin=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
_EMBEDDING = EmbeddingConfig(
    method="facenet",
    pretrained="vggface2",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
_SEED = SeedConfig(secret_key="dummy")  # unused by KFAAR but required

logger = logging.getLogger("kfaar_test")


def _to_uint8_hwc(array: np.ndarray) -> np.ndarray:
    """Convert arbitrary image-like array to uint8 HWC for saving."""
    arr = array
    if arr.ndim == 4:  # assume NCHW, take first dim removal handled elsewhere
        raise ValueError("Unexpected 4D array passed to _to_uint8_hwc")
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.moveaxis(arr, 0, -1)
    # Heuristically scale StyleGAN outputs in [-1, 1] to [0, 255]
    if arr.dtype != np.uint8:
        if arr.min() >= -1.5 and arr.max() <= 1.5:
            arr = (arr * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_image(array: np.ndarray, path: Path) -> None:
    img = _to_uint8_hwc(array)
    Image.fromarray(img).save(path)


def iter_celeba_samples(max_identities: int | None, max_per_identity: int | None, max_samples: int | None) -> Iterator[tuple[str, Path]]:
    dataset = CelebADataset(
        root=DATASET_ROOT,
        identities=None,
        max_per_identity=max_per_identity,
        max_samples=max_samples,
    )
    seen_ids: set[str] = set()
    for sample in dataset:
        if max_identities is not None and len(seen_ids) >= max_identities and sample.identity not in seen_ids:
            continue
        seen_ids.add(sample.identity)
        yield sample.identity, sample.path


def build_pipeline(stylegan=None) -> KfaarPipeline:
    data_cfg = DataConfig(dataset_path=DATASET_ROOT, dataset_type="celeba", options={})
    config = ExperimentConfig(
        data=data_cfg,
        detector=_DETECTOR,
        embedding=_EMBEDDING,
        seed=_SEED,
    )
    return build_kfaar_pipeline(config, stylegan=stylegan)


def load_mapper(generator_device: torch.device):
    """Load StyleGAN2 generator if available, logging a helpful hint otherwise."""
    try:
        sg2 = load_stylegan2(device=generator_device)
        logger.info(
            "Loaded StyleGAN2 generator | z_dim=%d w_dim=%d device=%s",
            sg2.z_dim,
            sg2.w_dim,
            generator_device,
        )
        return sg2
    except Exception as exc:  # pragma: no cover - optional dependency
        hint = "Ensure dnnlib and torch_utils live under src/ and ffhq.pkl is present."
        logger.warning("Skipping mapper test: %s (%s)", exc, hint)
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting KFAAR embedding test | dataset=%s", DATASET_ROOT)

    cuda_available = torch.cuda.is_available()
    if PREFER_CUDA_FOR_GENERATOR and not cuda_available:
        logger.warning("CUDA requested for generator but torch.cuda.is_available() is False")
    generator_device = torch.device("cuda") if (PREFER_CUDA_FOR_GENERATOR and cuda_available) else torch.device("cpu")
    device = generator_device  # keep projector tensors on same device as generator to avoid transfers
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    logger.info(
        "Init | device=%s | cuda_available=%s | cuda_device=%s | torch_cuda=%s",
        device,
        cuda_available,
        cuda_name,
        torch.version.cuda,
    )
    mapper_enabled = TEST_MAPPER
    logger.info("Init | loading StyleGAN2 (mapper=%s)", mapper_enabled)
    sg2 = load_mapper(generator_device) if mapper_enabled else None
    mapper_enabled = mapper_enabled and sg2 is not None
    logger.info("Init | building pipeline")
    pipeline = build_pipeline(stylegan=sg2)
    logger.info("Init | building projector")
    projector = ProjectorMLP(key_dim=KEY_BITS).to(device).eval()
    logger.info("Init | done; starting sample loop")
    processed = 0
    found_embeddings = 0
    projected = 0
    mapped = 0

    for identity, image_path in iter_celeba_samples(NUM_IDENTITIES, MAX_SAMPLES_PER_IDENTITY, MAX_SAMPLES_GLOBAL):
        image = load_image(image_path)
        # Create per-identity output directory to inspect results
        sample_dir = TEST_OUTPUT_DIR / f"{identity}_{image_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        result = pipeline.process_image(image, source_path=image_path)
        has_emb = result.embeddings is not None and getattr(result.embeddings, "size", 0) > 0
        if has_emb:
            found_embeddings += 1
            emb = result.embeddings
            logger.info("OK | id=%s file=%s embeddings_shape=%s", identity, image_path.name, emb.shape)
            _save_image(image, sample_dir / "input_rgb.png")
            for i, face in enumerate(result.aligned_faces):
                _save_image(face, sample_dir / f"aligned_{i}.png")
            # Take first embedding and run projector with a random binary key
            z = torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)
            key = torch.randint(0, 2, (1, KEY_BITS), device=device, dtype=torch.float32)
            z_proj = None
            with torch.no_grad():
                z_proj = projector(z, key)
            projected += 1
            logger.info("PROJECTOR | id=%s file=%s z_proj_shape=%s", identity, image_path.name, tuple(z_proj.shape))
            if mapper_enabled and sg2 is not None:
                z_input = z_proj if z_proj is not None else z
                sg2_device = next(sg2._G.parameters()).device  # type: ignore[attr-defined]
                if z_input.device != sg2_device:
                    z_input = z_input.to(sg2_device)
                with torch.no_grad():
                    w = sg2.map(z_input, truncation_psi=1.0)
                    img = sg2.synthesize(w, noise_mode="const")
                mapped += 1
                logger.info(
                    "MAPPER | id=%s file=%s w_shape=%s img_shape=%s",
                    identity,
                    image_path.name,
                    tuple(w.shape),
                    tuple(img.shape),
                )
                gen = img.detach().cpu().numpy()
                for j, g in enumerate(gen):
                    _save_image(g, sample_dir / f"generated_{j}.png")
            if result.generated_images is not None:
                logger.info(
                    "PIPELINE_GEN | id=%s file=%s w_shape=%s img_shape=%s",
                    identity,
                    image_path.name,
                    tuple(result.w_latents.shape) if result.w_latents is not None else None,
                    tuple(result.generated_images.shape),
                )
                for j, g in enumerate(result.generated_images):
                    _save_image(g, sample_dir / f"pipeline_generated_{j}.png")
        else:
            logger.warning("NO_EMB | id=%s file=%s detections=%d", identity, image_path.name, len(result.detections))
        processed += 1
    logger.info(
        "Done | processed=%d with_embeddings=%d projected=%d mapped=%d",
        processed,
        found_embeddings,
        projected,
        mapped,
    )


if __name__ == "__main__":
    main()
