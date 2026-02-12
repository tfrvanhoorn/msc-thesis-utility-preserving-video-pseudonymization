from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict
import numpy as np
import torch
from torchvision import utils as vutils
from ..components import EmbeddingModel, FaceDetector, ProjectorMLP, ProjectorLSTM
from ..components.alignment import FaceAligner
from ..components.detector import Detection
from ..components import StyleGAN2Generator
from ..losses import (
    anonymity_loss,
    synchronism_loss,
    diversity_loss,
    differentiation_loss,
    total_hpvg_loss,
)

logger = logging.getLogger(__name__)

@dataclass
class KfaarResult:
    detections: Sequence[Detection]
    aligned_faces: Sequence[torch.Tensor]
    real_embeddings: torch.Tensor
    projected_z: torch.Tensor
    virtual_embeddings: torch.Tensor
    valid_mask: torch.Tensor # Mask where both input AND gen-output had faces

class KfaarPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: EmbeddingModel,
        projector: ProjectorMLP,
        stylegan: Optional[StyleGAN2Generator] = None,
        device: Optional[torch.device] = None,
        *,
        save_dir: Optional[Path] = None,
        save_mode: str = "detected",
        save_max_per_epoch: Optional[int] = None,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.projector = projector
        self.stylegan = stylegan
        self.device = device or next(projector.parameters()).device
        self._projector_is_lstm = isinstance(projector, ProjectorLSTM)
        
        # Optimizer Setup
        self.optimizer = torch.optim.Adam(self.projector.parameters(), lr=1e-4)
        
        self.stats: Dict[str, int] = {"input_no_det": 0, "gen_no_det": 0, "discarded_batches": 0}

        # Transfer and Freeze
        if hasattr(self.embedder, "to"): self.embedder.to(self.device)
        if self.stylegan is not None: self.stylegan.to(self.device)
        self.projector.to(self.device)

        # Generated face saving configuration
        self._save_enabled = save_dir is not None
        self._saving_active = self._save_enabled
        self._save_dir = Path(save_dir) if save_dir is not None else None
        self._save_mode = save_mode
        self._save_max_per_epoch = save_max_per_epoch
        self._saved_this_epoch = 0
        self._current_epoch = 0
        if self._save_enabled:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, frames: torch.Tensor, key: torch.Tensor) -> KfaarResult:
        device = self.device
        frames_t = self._to_sequence_tensor(frames, device=device)
        seq_len = frames_t.shape[0]

        # 1. Input Processing
        input_mask = []
        aligned_faces = []
        real_emb = torch.zeros(seq_len, 512, device=device)
        
        for frame in frames_t:
            dets = self.detector.detect(frame)
            if dets:
                top = max(dets, key=lambda d: d.score)
                aligned = self.aligner.align(frame, top).to(device)
                aligned_faces.append(aligned)
                input_mask.append(True)
            else:
                aligned_faces.append(torch.empty(0))
                input_mask.append(False)
                self.stats["input_no_det"] += 1

        if any(input_mask):
            valid_idx = [i for i, m in enumerate(input_mask) if m]
            embs = self.embedder.embed([aligned_faces[i] for i in valid_idx], with_grad=False)
            for e, idx in zip(embs, valid_idx):
                real_emb[idx] = e

        # 2. Projection
        key_t = key.to(device)
        if self._projector_is_lstm:
            k_in = key_t.view(1, 1, -1).expand(1, seq_len, -1)
            projected_z = self.projector.project(real_emb.unsqueeze(0), k_in)[0]
        else:
            projected_z = self.projector.project(real_emb, key_t.expand(seq_len, -1))

        # 3. Virtual Processing
        v_embeddings = torch.zeros(seq_len, 512, device=device)
        gen_mask = [False] * seq_len
        
        images = None
        if self.stylegan is not None:
            w = self.stylegan.map(projected_z)
            images = self.stylegan.synthesize(w, noise_mode="const")
            for i in range(seq_len):
                img = images[i].clamp(-1, 1).add(1).div(2.0)
                dets = self.detector.detect(img)
                if dets:
                    top = max(dets, key=lambda d: d.score)
                    aligned = self.aligner.align(img, top).to(device)
                    # Carry grad from projected_z through StyleGAN to Embedder
                    v_embeddings[i] = self.embedder.embed([aligned], with_grad=True)[0]
                    gen_mask[i] = True
                else:
                    # Maintain grad path but keep value 0 to signal failure
                    v_embeddings[i] = projected_z[i].sum() * 0.0
                    self.stats["gen_no_det"] += 1

            self._maybe_save_generated(images, gen_mask)

        valid_mask = torch.tensor([i and g for i, g in zip(input_mask, gen_mask)], device=device)

        return KfaarResult(
            detections=[], # Simplified for training speed
            aligned_faces=aligned_faces,
            real_embeddings=real_emb.detach(),
            projected_z=projected_z,
            virtual_embeddings=v_embeddings,
            valid_mask=valid_mask
        )

    def hpvg_train_step(self, frames, labels, seq_lens, key_1, key_2, **kwargs) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        
        # components will return None if criteria not met
        comps = self.hpvg_loss_components(frames, labels, seq_lens, key_1, key_2, **kwargs)
        
        if comps is None:
            self.stats["discarded_batches"] += 1
            return torch.tensor(0.0, device=self.device)

        ano, syn, div, dif, total = comps
        
        # Only backprop if total is linked to a grad_fn
        if total.requires_grad:
            total.backward()
            self.optimizer.step()
            return total
        
        return torch.tensor(0.0, device=self.device)

    def hpvg_loss_components(
        self, frames, labels, seq_lens, key_1, key_2,
        margin=0.5, lambda_ano=0.4, lambda_syn=1.0, lambda_div=1.0, lambda_dif=1.0, lambda_temp=0.0
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        device = self.device
        batch_size = frames.shape[0]
        seq_lens_list = [int(x) for x in (seq_lens.tolist() if torch.is_tensor(seq_lens) else seq_lens)]

        all_real, all_v1, all_v2, all_labels = [], [], [], []

        for b in range(batch_size):
            res1 = self.forward(frames[b, :seq_lens_list[b]], key_1)
            res2 = self.forward(frames[b, :seq_lens_list[b]], key_2)
            
            mask = res1.valid_mask & res2.valid_mask
            if mask.any():
                all_real.append(res1.real_embeddings[mask])
                all_v1.append(res1.virtual_embeddings[mask])
                all_v2.append(res2.virtual_embeddings[mask])
                all_labels.extend([labels[b].item()] * int(mask.sum()))

        # --- VALIDATION GATE ---
        if not all_labels:
            return None

        # Requirement: At least 2 identities AND each has at least 2 samples
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        valid_identities = unique_labels[counts >= 2]

        if len(valid_identities) < 2:
            return None

        # Filter tensors to only include identities with >= 2 samples
        lab_t = torch.tensor(all_labels, device=device)
        keep_mask = torch.zeros_like(lab_t, dtype=torch.bool)
        for vid in valid_identities:
            keep_mask |= (lab_t == int(vid))

        real_c = torch.cat(all_real)[keep_mask]
        v1_c = torch.cat(all_v1)[keep_mask]
        v2_c = torch.cat(all_v2)[keep_mask]
        lab_final = lab_t[keep_mask]

        # Compute Losses
        ano = anonymity_loss(real_c, v1_c, margin=margin)
        div = diversity_loss(v1_c, v2_c, margin=margin)
        syn = synchronism_loss(v1_c, v1_c, margin=margin)
        dif = differentiation_loss(v1_c, v1_c, margin=margin)

        total = (lambda_ano * ano + lambda_syn * syn + 
                 lambda_div * div + lambda_dif * dif)

        return ano, syn, div, dif, total

    @staticmethod
    def _to_sequence_tensor(frames, device):
        t = frames if torch.is_tensor(frames) else torch.from_numpy(frames)
        return t.float().to(device)

    def configure_saving(self, save_dir: Path, *, mode: str = "detected", max_per_epoch: Optional[int] = None) -> None:
        """Enable saving synthesized faces to disk with a per-epoch cap."""
        self._save_enabled = True
        self._saving_active = True
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._save_mode = mode
        self._save_max_per_epoch = max_per_epoch
        self._saved_this_epoch = 0

    def begin_epoch(self, epoch: int) -> None:
        """Reset save counters for a new epoch and prepare the epoch directory."""
        self._current_epoch = epoch
        self._saved_this_epoch = 0
        if self._save_enabled and self._save_dir is not None:
            (self._save_dir / f"epoch_{epoch:03d}").mkdir(parents=True, exist_ok=True)
        self._saving_active = self._save_enabled

    def disable_saving(self) -> None:
        self._saving_active = False

    def enable_saving(self) -> None:
        if self._save_enabled:
            self._saving_active = True

    def _maybe_save_generated(self, images: Optional[torch.Tensor], gen_mask: Sequence[bool]) -> None:
        if not self._save_enabled or not self._saving_active:
            return
        if images is None:
            return
        if self._save_max_per_epoch is not None and self._saved_this_epoch >= self._save_max_per_epoch:
            return

        mode = self._save_mode
        save_dir = self._save_dir if self._save_dir is not None else None
        if save_dir is None:
            return

        epoch_dir = save_dir / f"epoch_{self._current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx in range(images.shape[0]):
                if self._save_max_per_epoch is not None and self._saved_this_epoch >= self._save_max_per_epoch:
                    break

                detected = bool(gen_mask[idx])
                if mode == "detected" and not detected:
                    continue
                if mode == "undetected" and detected:
                    continue

                filename = (
                    f"epoch{self._current_epoch:03d}_sample{self._saved_this_epoch:06d}_"
                    f"f{idx:03d}_{'det' if detected else 'undet'}.png"
                )
                vutils.save_image(images[idx].detach().cpu().clamp(0.0, 1.0), epoch_dir / filename)
                self._saved_this_epoch += 1