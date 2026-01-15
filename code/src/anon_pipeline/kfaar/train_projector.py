from __future__ import annotations

"""
Projector training loop for KFAAR using post-StyleGAN ArcFace re-embeddings.

Pipeline per sample:
  real image -> detect -> align -> ArcFace embed (real)
  real emb + random key -> Projector -> z' -> StyleGAN mapping/synthesis
  synth image -> detect -> align -> ArcFace embed (virtual)
  losses on embeddings (anonymity/synchronism/diversity/differentiation)

NOTE: ArcFace + detection/alignment are non-differentiable (ONNX/NumPy). Gradients
stop at the re-embedding stage, so projector updates rely on surrogate behavior
only. To train with gradients flowing through embeddings, replace the embedder
with a differentiable model.
"""

import argparse
import itertools
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

import numpy as np
import torch
from PIL import Image

# Ensure src/ is on sys.path when run as a script (no package context)
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anon_pipeline.kfaar.components.projector import ProjectorMLP
from anon_pipeline.kfaar.losses import (
    anonymity_loss,
    differentiation_loss,
    diversity_loss,
    synchronism_loss,
    total_hpvg_loss,
)
from anon_pipeline.kfaar.pipeline.factory import build_kfaar_pipeline
from anon_pipeline.kfaar.config import DataConfig, ExperimentConfig, DetectorConfig, EmbeddingConfig, SeedConfig
from anon_pipeline.shared.data.io import load_image
from anon_pipeline.shared.data.loaders import ImageSample, iter_samples
from anon_pipeline.kfaar.models import load_stylegan2

logger = logging.getLogger(__name__)
STYLEGAN_DEFAULT = Path(__file__).parent / "models" / "ffhq.pkl"


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_identities: int
    samples_per_identity: int
    steps_per_epoch: int
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    lambda_ano: float
    lambda_syn: float
    lambda_div: float
    lambda_dif: float
    margin: float
    key_dim: int
    hidden_dims: Sequence[int]
    dropout: float
    truncation_psi: float
    det_threshold: float
    device: str
    stylegan_ckpt: Path | None
    max_identities: int | None
    max_per_identity: int | None
    allowed_identities: Set[str] | None
    out_dir: Path
    save_every_epochs: int
    resume: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KFAAR projector with ArcFace re-embeddings")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/celeba"))
    parser.add_argument("--dataset-type", type=str, default="celeba", choices=["celeba", "image_folder"])
    parser.add_argument("--max-per-identity", type=int, default=None)
    parser.add_argument("--stylegan-ckpt", type=Path, default=STYLEGAN_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--det-threshold", type=float, default=0.5, help="Detection score threshold")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-identities", type=int, default=4)
    parser.add_argument("--samples-per-identity", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-ano", type=float, default=0.4)
    parser.add_argument("--lambda-syn", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument("--lambda-dif", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=(1024, 512))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--train-identities", type=str, nargs="*", default=None, help="Whitelist of identity ids for training")
    parser.add_argument("--train-identities-file", type=Path, default=None, help="Path to a file with one identity id per line for training")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/kfaar_projector"))
    parser.add_argument("--save-every-epochs", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    allowed_ids: Set[str] | None = None
    if args.train_identities:
        allowed_ids = {str(x) for x in args.train_identities}
    if args.train_identities_file:
        file_ids = load_identities_file(args.train_identities_file)
        allowed_ids = set(file_ids) if allowed_ids is None else allowed_ids.union(file_ids)

    return TrainingConfig(
        num_epochs=args.epochs,
        batch_identities=args.batch_identities,
        samples_per_identity=args.samples_per_identity,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        lambda_ano=args.lambda_ano,
        lambda_syn=args.lambda_syn,
        lambda_div=args.lambda_div,
        lambda_dif=args.lambda_dif,
        margin=args.margin,
        key_dim=args.key_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        truncation_psi=args.truncation_psi,
        det_threshold=args.det_threshold,
        device=args.device,
        stylegan_ckpt=args.stylegan_ckpt,
        max_identities=args.max_identities,
        max_per_identity=args.max_per_identity,
        allowed_identities=allowed_ids,
        out_dir=args.out_dir,
        save_every_epochs=max(1, args.save_every_epochs),
        resume=args.resume,
    )


def load_identities_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Identities file not found: {path}")
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            ident = line.strip()
            if ident:
                ids.append(ident)
    return ids


def build_identity_index(
    config: ExperimentConfig,
    max_identities: int | None,
    max_per_identity: int | None,
    allowed_identities: Set[str] | None,
) -> Dict[str, List[Path]]:
    bucket: Dict[str, List[Path]] = defaultdict(list)
    for sample in iter_samples(config.data):
        if allowed_identities is not None and sample.identity not in allowed_identities:
            continue
        if max_identities is not None and sample.identity not in bucket and len(bucket) >= max_identities:
            continue  # respect cap on distinct identities but keep filling existing ones
        paths = bucket[sample.identity]
        if max_per_identity is not None and len(paths) >= max_per_identity:
            continue
        paths.append(sample.path)
    logger.info("Indexed %s identities", len(bucket))
    return {k: v for k, v in bucket.items() if len(v) >= 1}


def save_checkpoint(out_dir: Path, name: str, projector: ProjectorMLP, optimizer: torch.optim.Optimizer, epoch: int, step: int, train_cfg: TrainingConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name
    torch.save(
        {
            "projector_state": projector.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "train_config": asdict(train_cfg),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(path: Path, projector: ProjectorMLP, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    projector.load_state_dict(payload["projector_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    epoch = int(payload.get("epoch", -1))
    step = int(payload.get("step", -1))
    return epoch, step


def sample_batch_identities(identity_index: Mapping[str, Sequence[Path]], batch_identities: int, samples_per_identity: int) -> Dict[str, List[Path]]:
    eligible = [k for k, v in identity_index.items() if len(v) >= samples_per_identity]
    if len(eligible) < batch_identities:
        raise ValueError(f"Not enough identities with >= {samples_per_identity} samples (found {len(eligible)})")
    chosen = random.sample(eligible, batch_identities)
    batch: Dict[str, List[Path]] = {}
    for ident in chosen:
        paths = identity_index[ident]
        if len(paths) == samples_per_identity:
            batch[ident] = list(paths)
        else:
            batch[ident] = random.sample(paths, samples_per_identity)
    return batch


def embed_face_from_image(image: np.ndarray, detector, aligner, embedder, is_bgr: bool = False) -> np.ndarray | None:
    # Detector/aligner expect RGB; flip back if the caller provided BGR
    if is_bgr:
        image = image[..., ::-1].copy()

    detections = detector.detect(image)
    if not detections:
        return None
    aligned = aligner.align(image, detections[0])
    emb = embedder.embed([aligned])
    if emb.size == 0:
        return None
    return emb[0]


def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    # image: CHW float in [-1, 1]
    img = image.detach().clamp(-1.0, 1.0)
    img = (img * 127.5 + 127.5).clamp(0, 255)
    img = img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return img


def resize_to_max_dim(image: np.ndarray, max_dim: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_size = (int(w * scale), int(h * scale))
    return np.array(Image.fromarray(image).resize(new_size, Image.BILINEAR))


def compute_losses(
    real_by_id: Dict[str, List[torch.Tensor]],
    synth_by_id: Dict[str, List[torch.Tensor]],
    margin: float,
    lambda_ano: float,
    lambda_syn: float,
    lambda_div: float,
    lambda_dif: float,
) -> Dict[str, torch.Tensor]:
    any_tensor = None
    for tensors in real_by_id.values():
        if tensors:
            any_tensor = tensors[0]
            break
    if any_tensor is None:
        for tensors in synth_by_id.values():
            if tensors:
                any_tensor = tensors[0]
                break
    if any_tensor is None:
        raise ValueError("Empty embeddings passed to compute_losses")
    device = any_tensor.device

    ano_terms: List[torch.Tensor] = []
    syn_terms: List[torch.Tensor] = []
    div_terms: List[torch.Tensor] = []
    dif_terms: List[torch.Tensor] = []

    identities = list(real_by_id.keys())

    for ident in identities:
        real_list = real_by_id.get(ident, [])
        synth_list = synth_by_id.get(ident, [])
        paired = zip(real_list, synth_list)
        for r, s in paired:
            ano_terms.append(anonymity_loss(r.unsqueeze(0), s.unsqueeze(0), margin=margin))

        if len(synth_list) >= 2:
            for a, b in itertools.combinations(synth_list, 2):
                syn_terms.append(synchronism_loss(a.unsqueeze(0), b.unsqueeze(0), margin=margin))
                div_terms.append(diversity_loss(a.unsqueeze(0), b.unsqueeze(0), margin=margin))

    # differentiation across identities
    for id_a, id_b in itertools.combinations(identities, 2):
        synth_a = synth_by_id.get(id_a, [])
        synth_b = synth_by_id.get(id_b, [])
        if not synth_a or not synth_b:
            continue
        dif_terms.append(
            differentiation_loss(
                synth_a[0].unsqueeze(0),
                synth_b[0].unsqueeze(0),
                margin=margin,
            )
        )

    losses: Dict[str, torch.Tensor] = {}
    losses["anonymity"] = torch.stack(ano_terms).mean() if ano_terms else torch.zeros((), device=device)
    losses["synchronism"] = torch.stack(syn_terms).mean() if syn_terms else torch.zeros((), device=device)
    losses["diversity"] = torch.stack(div_terms).mean() if div_terms else torch.zeros((), device=device)
    losses["differentiation"] = torch.stack(dif_terms).mean() if dif_terms else torch.zeros((), device=device)

    losses["total"] = total_hpvg_loss(
        losses["anonymity"],
        losses["synchronism"],
        losses["diversity"],
        losses["differentiation"],
        lambda_ano=lambda_ano,
        lambda_syn=lambda_syn,
        lambda_div=lambda_div,
        lambda_dif=lambda_dif,
    )
    return losses


def train_projector(exp_config: ExperimentConfig, train_cfg: TrainingConfig) -> None:
    device = torch.device(train_cfg.device)

    logger.info("Loading StyleGAN2 checkpoint=%s", train_cfg.stylegan_ckpt or "<default ffhq.pkl>")
    stylegan = load_stylegan2(ckpt_path=train_cfg.stylegan_ckpt, device=device)
    logger.info(
        "Loaded StyleGAN2 | z_dim=%d w_dim=%d device=%s",
        stylegan.z_dim,
        stylegan.w_dim,
        device,
    )

    pipeline = build_kfaar_pipeline(exp_config, stylegan=stylegan)
    detector = pipeline.detector
    aligner = pipeline.aligner
    embedder = pipeline.embedder

    projector = ProjectorMLP(
        key_dim=train_cfg.key_dim,
        output_dim=stylegan.z_dim,
        hidden_dims=tuple(train_cfg.hidden_dims),
        dropout=train_cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        projector.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )

    start_epoch = 0
    if train_cfg.resume is not None:
        ckpt_epoch, ckpt_step = load_checkpoint(train_cfg.resume, projector, optimizer, device)
        start_epoch = ckpt_epoch + 1
        logger.info("Resumed from %s at epoch=%s step=%s", train_cfg.resume, ckpt_epoch, ckpt_step)

    identity_index = build_identity_index(
        exp_config,
        train_cfg.max_identities,
        train_cfg.max_per_identity,
        train_cfg.allowed_identities,
    )
    logger.info(
        "Training start: epochs=%s steps/epoch=%s batch_ids=%s samples/id=%s",
        train_cfg.num_epochs,
        train_cfg.steps_per_epoch,
        train_cfg.batch_identities,
        train_cfg.samples_per_identity,
    )

    debug_dir = train_cfg.out_dir / "debug_synth"
    debug_saved = 0

    last_step = -1
    for epoch in range(start_epoch, train_cfg.num_epochs):
        epoch_losses = defaultdict(list)
        last_step = -1
        for step in range(train_cfg.steps_per_epoch):
            last_step = step
            try:
                batch_paths = sample_batch_identities(
                    identity_index,
                    train_cfg.batch_identities,
                    train_cfg.samples_per_identity,
                )
            except ValueError as err:
                logger.warning("Skipping step: %s", err)
                continue

            real_by_id: Dict[str, List[torch.Tensor]] = defaultdict(list)
            synth_by_id: Dict[str, List[torch.Tensor]] = defaultdict(list)

            for ident, paths in batch_paths.items():
                for path in paths:
                    image = load_image(path)
                    real_emb = embed_face_from_image(image, detector, aligner, embedder, is_bgr=False)
                    if real_emb is None:
                        logger.debug("No detection for real image %s", path)
                        continue
                    real_t = torch.from_numpy(real_emb).to(device=device, dtype=torch.float32)

                    key = torch.randn(train_cfg.key_dim, device=device)
                    z_prime = projector.project(real_t, key)
                    w = stylegan.map(z_prime, truncation_psi=train_cfg.truncation_psi)
                    synth_images = stylegan.synthesize(w, noise_mode="const")

                    synth_emb = None
                    for idx_img, img_tensor in enumerate(synth_images):
                        synth_np = image_tensor_to_uint8(img_tensor)
                        synth_np = resize_to_max_dim(synth_np, max_dim=640)

                        if debug_saved < 10:
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            debug_path = debug_dir / f"epoch{epoch+1}_step{step+1}_{idx_img}.png"
                            Image.fromarray(synth_np).save(debug_path)
                            debug_saved += 1

                        synth_emb = embed_face_from_image(synth_np, detector, aligner, embedder, is_bgr=False)
                        if synth_emb is not None:
                            break

                    if synth_emb is None:
                        logger.debug("No detection on synthetic image for %s", path)
                        continue

                    synth_t = torch.from_numpy(synth_emb).to(device=device, dtype=torch.float32)

                    real_by_id[ident].append(real_t)
                    synth_by_id[ident].append(synth_t)

            if not synth_by_id:
                logger.debug("No valid synthetic embeddings in step %s", step)
                continue

            losses = compute_losses(
                real_by_id,
                synth_by_id,
                margin=train_cfg.margin,
                lambda_ano=train_cfg.lambda_ano,
                lambda_syn=train_cfg.lambda_syn,
                lambda_div=train_cfg.lambda_div,
                lambda_dif=train_cfg.lambda_dif,
            )

            if not losses["total"].requires_grad:
                logger.debug("Skipping backward: total loss is not connected to gradients (non-differentiable path)")
                continue

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for name, value in losses.items():
                epoch_losses[name].append(float(value.detach().cpu()))

            if step % 10 == 0:
                logger.info(
                    "Epoch %d Step %d: total=%.4f ano=%.4f syn=%.4f div=%.4f dif=%.4f",
                    epoch + 1,
                    step + 1,
                    losses["total"].item(),
                    losses["anonymity"].item(),
                    losses["synchronism"].item(),
                    losses["diversity"].item(),
                    losses["differentiation"].item(),
                )

        if epoch_losses:
            logger.info(
                "Epoch %d done: total=%.4f ano=%.4f syn=%.4f div=%.4f dif=%.4f",
                epoch + 1,
                np.mean(epoch_losses.get("total", [0.0])),
                np.mean(epoch_losses.get("anonymity", [0.0])),
                np.mean(epoch_losses.get("synchronism", [0.0])),
                np.mean(epoch_losses.get("diversity", [0.0])),
                np.mean(epoch_losses.get("differentiation", [0.0])),
            )

        # Save per-epoch checkpoint
        if (epoch + 1) % train_cfg.save_every_epochs == 0:
            path = save_checkpoint(train_cfg.out_dir, f"projector_epoch{epoch + 1}.pt", projector, optimizer, epoch, last_step, train_cfg)
            save_checkpoint(train_cfg.out_dir, "projector_latest.pt", projector, optimizer, epoch, last_step, train_cfg)
            logger.info("Saved checkpoint: %s", path)

    # Final checkpoint after all epochs
    final_path = save_checkpoint(train_cfg.out_dir, "projector_final.pt", projector, optimizer, train_cfg.num_epochs - 1, last_step, train_cfg)
    logger.info("Saved final projector weights: %s", final_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data = DataConfig(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        options={"max_per_identity": args.max_per_identity} if args.max_per_identity is not None else {},
    )
    # Detector/embedder configs from CLI
    detector = DetectorConfig(score_threshold=args.det_threshold)
    embedding = EmbeddingConfig()
    seed = SeedConfig(secret_key="change-me")
    exp_config = ExperimentConfig(
        data=data,
        detector=detector,
        embedding=embedding,
        seed=seed,
    )

    train_cfg = build_training_config(args)
    train_projector(exp_config, train_cfg)


if __name__ == "__main__":
    main()
