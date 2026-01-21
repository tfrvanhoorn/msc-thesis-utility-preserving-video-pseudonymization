from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pipeline import KfaarPipeline


class KfaarTrainer:
    """Lightweight trainer for the KFAAR pipeline projector."""

    def __init__(
        self,
        pipeline: KfaarPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int = 10,
        key_dim: int = 128,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
        batch_identities: int | None = None,
        samples_per_identity: int | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device = "cuda",
        train_identities: Sequence[Any] | None = None,
        val_identities: Sequence[Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.epochs = epochs
        self.key_dim = key_dim
        self.margin = margin
        self.lambda_ano = lambda_ano
        self.lambda_syn = lambda_syn
        self.lambda_div = lambda_div
        self.lambda_dif = lambda_dif
        self.batch_identities = batch_identities
        self.samples_per_identity = samples_per_identity
        self._pending_by_identity: dict[Any, list[np.ndarray]] = defaultdict(list)
        self.train_identities = list(train_identities) if train_identities else None
        self.val_identities = list(val_identities) if val_identities else None
        self.eval_history: list[dict[str, float]] = []

        default_ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else default_ckpt_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Ensure pipeline modules are on the target device
        if hasattr(self.pipeline, "projector"):
            self.pipeline.projector.to(self.device)
        if hasattr(self.pipeline, "embedder") and hasattr(self.pipeline.embedder, "to"):
            self.pipeline.embedder.to(self.device)
        if hasattr(self.pipeline, "stylegan") and self.pipeline.stylegan is not None and hasattr(self.pipeline.stylegan, "to"):
            self.pipeline.stylegan.to(self.device)
        self.pipeline.device = self.device

    def train(self) -> None:
        logging.info("Starting KFAAR training for %s epochs", self.epochs)

        for epoch in range(self.epochs):
            self.pipeline.projector.train()
            epoch_loss = 0.0
            step_count = 0

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in progress:
                images, labels = self._extract_batch(batch)

                for sub_images, sub_labels in self._iter_effective_batches(images, labels):
                    key_1 = torch.randn(self.key_dim, device=self.device)
                    key_2 = torch.randn(self.key_dim, device=self.device)

                    loss = self.pipeline.hpvg_train_step(
                        sub_images,
                        sub_labels,
                        key_1,
                        key_2,
                        margin=self.margin,
                        lambda_ano=self.lambda_ano,
                        lambda_syn=self.lambda_syn,
                        lambda_div=self.lambda_div,
                        lambda_dif=self.lambda_dif,
                    )

                    loss_value = float(loss.item())
                    epoch_loss += loss_value
                    step_count += 1
                    progress.set_postfix({"loss": f"{loss_value:.4f}"})

            # Drop any partial batches that could not form a full identity-based batch
            self._pending_by_identity.clear()

            avg_loss = epoch_loss / max(1, step_count)
            logging.info("Epoch %s complete | avg loss=%.6f", epoch + 1, avg_loss)

            val_metrics: dict[str, float] | None = None
            if self.val_loader is not None:
                val_metrics = self.evaluate(epoch=epoch + 1)
                logging.info("Validation avg loss=%.6f", val_metrics["total"])

            self.save_checkpoint(epoch, avg_loss, val_metrics)

    def save_checkpoint(self, epoch: int, loss: float, val_metrics: dict[str, float] | None = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{timestamp}_kfaar_projector_epoch_{epoch + 1}"
        path = self.checkpoint_dir / f"{base}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.pipeline.projector.state_dict(),
                "optimizer_state_dict": self.pipeline.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        logging.info("Checkpoint saved to %s", path)

        json_path = self.checkpoint_dir / f"{base}.json"
        payload = {
            "epoch": epoch + 1,
            "timestamp": timestamp,
            "train_loss_total": loss,
            "val_loss_total": val_metrics["total"] if val_metrics else None,
            "val_loss_components": val_metrics,
            "config": {
                "epochs": self.epochs,
                "key_dim": self.key_dim,
                "margin": self.margin,
                "lambda_ano": self.lambda_ano,
                "lambda_syn": self.lambda_syn,
                "lambda_div": self.lambda_div,
                "lambda_dif": self.lambda_dif,
                "batch_identities": self.batch_identities,
                "samples_per_identity": self.samples_per_identity,
                "device": str(self.device),
                "train_identities": self.train_identities,
                "val_identities": self.val_identities,
            },
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Checkpoint metadata saved to %s", json_path)
        return path

    def evaluate(self, epoch: int | None = None) -> dict[str, float]:
        if self.val_loader is None:
            return {
                "total": 0.0,
                "ano": 0.0,
                "syn": 0.0,
                "div": 0.0,
                "dif": 0.0,
            }

        self.pipeline.projector.eval()
        total_loss = 0.0
        total_ano = 0.0
        total_syn = 0.0
        total_div = 0.0
        total_dif = 0.0
        steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = self._extract_batch(batch)
                for sub_images, sub_labels in self._iter_effective_batches(images, labels):
                    key_1 = torch.randn(self.key_dim, device=self.device)
                    key_2 = torch.randn(self.key_dim, device=self.device)
                    ano, syn, div, dif, loss = self.pipeline.hpvg_loss_components(
                        sub_images,
                        sub_labels,
                        key_1,
                        key_2,
                        margin=self.margin,
                        lambda_ano=self.lambda_ano,
                        lambda_syn=self.lambda_syn,
                        lambda_div=self.lambda_div,
                        lambda_dif=self.lambda_dif,
                    )
                    total_ano += float(ano.item())
                    total_syn += float(syn.item())
                    total_div += float(div.item())
                    total_dif += float(dif.item())
                    total_loss += float(loss.item())
                    steps += 1

        self.pipeline.projector.train()
        denom = max(1, steps)
        avg_total = total_loss / denom
        avg_ano = total_ano / denom
        avg_syn = total_syn / denom
        avg_div = total_div / denom
        avg_dif = total_dif / denom

        logging.info(
            "Validation epoch=%s | total=%.6f ano=%.6f syn=%.6f div=%.6f dif=%.6f",
            epoch if epoch is not None else "?",
            avg_total,
            avg_ano,
            avg_syn,
            avg_div,
            avg_dif,
        )

        self.eval_history.append(
            {
                "epoch": float(epoch) if epoch is not None else float(len(self.eval_history) + 1),
                "total": avg_total,
                "ano": avg_ano,
                "syn": avg_syn,
                "div": avg_div,
                "dif": avg_dif,
            }
        )

        return {
            "total": avg_total,
            "ano": avg_ano,
            "syn": avg_syn,
            "div": avg_div,
            "dif": avg_dif,
        }

    def _extract_batch(self, batch: Any) -> tuple[list[np.ndarray], torch.Tensor]:
        images: Any
        labels: torch.Tensor | None = None

        if isinstance(batch, dict):
            images = batch.get("image")
            if images is None:
                images = batch.get("images")

            labels = batch.get("label")
            if labels is None:
                labels = batch.get("labels")
        elif isinstance(batch, Sequence) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise TypeError("Unsupported batch format for KfaarTrainer")

        if labels is None:
            raise ValueError("Batch is missing labels for training")

        labels = labels.to(self.device)
        image_list = self._to_numpy_list(images)
        return image_list, labels

    def _iter_effective_batches(self, images: list[np.ndarray], labels: torch.Tensor) -> Sequence[tuple[list[np.ndarray], torch.Tensor]]:
        # If no identity-based batching is requested, pass through the incoming batch
        if not self.batch_identities or not self.samples_per_identity:
            return [(images, labels)]

        if len(images) != int(labels.numel()):
            raise ValueError("Mismatch between number of images and labels for batching")

        # Queue the samples by identity
        for img, label_value in zip(images, labels.view(-1).tolist()):
            self._pending_by_identity[label_value].append(img)

        ready_batches: list[tuple[list[np.ndarray], torch.Tensor]] = []
        while True:
            eligible = [ident for ident, imgs in self._pending_by_identity.items() if len(imgs) >= self.samples_per_identity]
            if len(eligible) < self.batch_identities:
                break

            chosen = eligible[: self.batch_identities]
            batch_images: list[np.ndarray] = []
            batch_labels: list[Any] = []

            for ident in chosen:
                imgs = self._pending_by_identity[ident]
                take = imgs[: self.samples_per_identity]
                del imgs[: self.samples_per_identity]
                batch_images.extend(take)
                batch_labels.extend([ident] * self.samples_per_identity)
                if not imgs:
                    self._pending_by_identity.pop(ident, None)

            ready_batches.append((batch_images, torch.tensor(batch_labels, device=self.device)))

        return ready_batches

    def _to_numpy_list(self, images: Any) -> list[np.ndarray]:
        if isinstance(images, np.ndarray):
            return [images]
        if torch.is_tensor(images):
            if images.dim() == 4:
                return [self._tensor_to_numpy(img) for img in images]
            return [self._tensor_to_numpy(images)]
        if isinstance(images, Sequence):
            return [self._tensor_to_numpy(img) if torch.is_tensor(img) else np.array(img) for img in images]
        return [np.array(images)]

    @staticmethod
    def _tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
        arr = img.detach().cpu()
        if arr.dim() == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        return arr.numpy()
