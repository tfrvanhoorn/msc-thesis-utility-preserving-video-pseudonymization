from __future__ import annotations

import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence
import sys

import psutil
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
        lambda_temp: float = 0.0,
        batch_identities: int | None = None,
        samples_per_identity: int | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device = "cuda",
        train_identities: Sequence[Any] | None = None,
        val_identities: Sequence[Any] | None = None,
        start_epoch: int = 0,
        save_generated_faces: bool = False,
        save_generated_dir: str | Path | None = None,
        save_generated_mode: str = "detected",
        save_generated_max_per_epoch: int | None = None,
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
        self.lambda_temp = lambda_temp
        self.batch_identities = batch_identities
        self.samples_per_identity = samples_per_identity
        self.start_epoch = start_epoch
        self._memory_log_interval = 100
        self._proc = psutil.Process()
        self._interval_stats = {"discarded_batches": 0, "input_no_det": 0, "gen_no_det": 0}
        self.train_identities = list(train_identities) if train_identities else None
        self.val_identities = list(val_identities) if val_identities else None
        self.eval_history: list[dict[str, float]] = []
        self.save_generated_faces = save_generated_faces
        self.save_generated_dir = Path(save_generated_dir) if save_generated_dir else None
        self.save_generated_mode = save_generated_mode
        self.save_generated_max_per_epoch = save_generated_max_per_epoch

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

        if self.save_generated_faces and hasattr(self.pipeline, "configure_saving"):
            target_dir = self.save_generated_dir if self.save_generated_dir is not None else (self.checkpoint_dir / "generated_faces")
            self.pipeline.configure_saving(
                target_dir,
                mode=self.save_generated_mode,
                max_per_epoch=self.save_generated_max_per_epoch,
            )

    def train(self) -> None:
        logging.info("Starting KFAAR training for %s epochs", self.epochs)

        for epoch in range(self.start_epoch, self.epochs):
            self.pipeline.projector.train()
            epoch_loss = 0.0
            step_count = 0
            self._reset_interval_stats()

            if self.save_generated_faces and hasattr(self.pipeline, "begin_epoch"):
                self.pipeline.begin_epoch(epoch + 1)

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", file=sys.stdout)
            for batch in progress:
                frames, labels, seq_lens = self._extract_batch(batch)

                for sub_frames, sub_labels, sub_seq_lens in self._iter_effective_batches(frames, labels, seq_lens):
                    key_1 = torch.randn(self.key_dim, device=self.device)
                    key_2 = torch.randn(self.key_dim, device=self.device)

                    loss = self.pipeline.hpvg_train_step(
                        sub_frames,
                        sub_labels,
                        sub_seq_lens,
                        key_1,
                        key_2,
                        margin=self.margin,
                        lambda_ano=self.lambda_ano,
                        lambda_syn=self.lambda_syn,
                        lambda_div=self.lambda_div,
                        lambda_dif=self.lambda_dif,
                        lambda_temp=self.lambda_temp,
                    )

                    loss_value = float(loss.item())
                    epoch_loss += loss_value
                    step_count += 1
                    progress.set_postfix({"loss": f"{loss_value:.4f}"})

                    self._pull_interval_stats()

                    if step_count % self._memory_log_interval == 0:
                        self._log_interval_stats(epoch + 1, step_count)
                        self._log_memory(f"train epoch {epoch + 1} step {step_count}")

            avg_loss = epoch_loss / max(1, step_count)
            logging.info("Epoch %s complete | avg loss=%.6f", epoch + 1, avg_loss)
            self._log_memory(f"train epoch {epoch + 1} end")
            sys.stdout.flush()

            if step_count % self._memory_log_interval:
                self._log_interval_stats(epoch + 1, step_count)

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

        saving_was_active = False
        if self.save_generated_faces and hasattr(self.pipeline, "disable_saving"):
            saving_was_active = bool(getattr(self.pipeline, "_saving_active", False))
            self.pipeline.disable_saving()

        self.pipeline.projector.eval()
        total_loss = 0.0
        total_ano = 0.0
        total_syn = 0.0
        total_div = 0.0
        total_dif = 0.0
        steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                frames, labels, seq_lens = self._extract_batch(batch)
                for sub_frames, sub_labels, sub_seq_lens in self._iter_effective_batches(frames, labels, seq_lens):
                    key_1 = torch.randn(self.key_dim, device=self.device)
                    key_2 = torch.randn(self.key_dim, device=self.device)
                    comps = self.pipeline.hpvg_loss_components(
                        sub_frames,
                        sub_labels,
                        sub_seq_lens,
                        key_1,
                        key_2,
                        margin=self.margin,
                        lambda_ano=self.lambda_ano,
                        lambda_syn=self.lambda_syn,
                        lambda_div=self.lambda_div,
                        lambda_dif=self.lambda_dif,
                        lambda_temp=self.lambda_temp,
                    )

                    if comps is None:
                        continue

                    ano, syn, div, dif, loss = comps
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

        if self.save_generated_faces and saving_was_active and hasattr(self.pipeline, "enable_saving"):
            self.pipeline.enable_saving()

        return {
            "total": avg_total,
            "ano": avg_ano,
            "syn": avg_syn,
            "div": avg_div,
            "dif": avg_dif,
        }

    def _extract_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames: Any
        labels: torch.Tensor | None = None
        seq_lens: Any = None

        if isinstance(batch, dict):
            frames = batch.get("frames")
            labels = batch.get("label")
            if labels is None:
                labels = batch.get("labels")
            seq_lens = batch.get("seq_lens")
        elif isinstance(batch, Sequence) and len(batch) >= 2:
            frames, labels = batch[0], batch[1]
            if len(batch) >= 3:
                seq_lens = batch[2]
        else:
            raise TypeError("Unsupported batch format for KfaarTrainer")

        if labels is None:
            raise ValueError("Batch is missing labels for training")
        if frames is None:
            raise ValueError("Batch is missing frames/images for training")

        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        frame_tensor = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
        if frame_tensor.dim() == 4:
            frame_tensor = frame_tensor.unsqueeze(1)
        if frame_tensor.dim() != 5:
            raise ValueError(f"Expected frames with 5 dimensions (B,Seq,C,H,W), got {tuple(frame_tensor.shape)}")

        seq_len_tensor = torch.as_tensor(seq_lens, dtype=torch.long) if seq_lens is not None else torch.full(
            (frame_tensor.shape[0],), frame_tensor.shape[1], dtype=torch.long
        )

        return frame_tensor, labels_tensor, seq_len_tensor

    def _iter_effective_batches(
        self, frames: torch.Tensor, labels: torch.Tensor, seq_lens: torch.Tensor
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Batches are expected to arrive pre-grouped by the identity batching
        # dataset. Simply move tensors to the target device.
        return [(frames.to(self.device), labels.to(self.device), seq_lens.to(self.device))]

    def _log_memory(self, tag: str) -> None:
        rss_gb = self._proc.memory_info().rss / (1024 ** 3)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            logging.info("%s | CPU RSS=%.2f GB | CUDA alloc=%.2f GB reserved=%.2f GB", tag, rss_gb, alloc, reserved)
        else:
            logging.info("%s | CPU RSS=%.2f GB", tag, rss_gb)

    def _pull_interval_stats(self) -> None:
        self._interval_stats["discarded_batches"] += int(self.pipeline.stats.get("discarded_batches", 0))
        self._interval_stats["input_no_det"] += int(self.pipeline.stats.get("input_no_det", 0))
        self._interval_stats["gen_no_det"] += int(self.pipeline.stats.get("gen_no_det", 0))
        self.pipeline.stats["discarded_batches"] = 0
        self.pipeline.stats["input_no_det"] = 0
        self.pipeline.stats["gen_no_det"] = 0

    def _log_interval_stats(self, epoch: int, step_count: int) -> None:
        if not any(self._interval_stats.values()):
            return
        rss_gb, alloc_gb, reserved_gb = self._current_memory()
        logging.info(
            "Interval summary | epoch=%s step=%s | discarded_batches=%d | input_no_det_frames=%d | gen_no_det_frames=%d | CPU RSS=%.2f GB | CUDA alloc=%.2f GB reserved=%.2f GB",
            epoch,
            step_count,
            self._interval_stats["discarded_batches"],
            self._interval_stats["input_no_det"],
            self._interval_stats["gen_no_det"],
            rss_gb,
            alloc_gb,
            reserved_gb,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._reset_interval_stats()

    def _reset_interval_stats(self) -> None:
        self._interval_stats = {"discarded_batches": 0, "input_no_det": 0, "gen_no_det": 0}

    def _current_memory(self) -> tuple[float, float, float]:
        rss_gb = self._proc.memory_info().rss / (1024 ** 3)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        else:
            alloc = 0.0
            reserved = 0.0
        return rss_gb, alloc, reserved


