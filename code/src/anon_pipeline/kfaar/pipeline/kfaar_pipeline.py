from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..components import EmbeddingModel, FaceDetector, ProjectorMLP, ProjectorLSTM
from ..components.alignment import FaceAligner
from ..components.detector import Detection
from ..components import StyleGAN2Generator
from ..losses import (
    anonymity_loss,
    synchronism_loss,
    diversity_loss,
    differentiation_loss,
    temporal_smoothness_loss,
    total_hpvg_loss,
)


logger = logging.getLogger(__name__)


@dataclass
class KfaarResult:
    detections: Sequence[Detection]
    aligned_faces: Sequence[torch.Tensor]
    real_embeddings: torch.Tensor
    projected_z: torch.Tensor
    w_latents: Optional[torch.Tensor] = None
    generated_images: Optional[torch.Tensor] = None
    virtual_embeddings: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None


class KfaarPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: EmbeddingModel,
        projector: ProjectorMLP,
        stylegan: Optional[StyleGAN2Generator] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.projector = projector
        self.stylegan = stylegan
        self.device = device or next(projector.parameters()).device
        self._projector_is_lstm = isinstance(projector, ProjectorLSTM)
        self.optimizer = torch.optim.Adam(
            self.projector.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999)
        )
        self.stats: dict[str, int] = {
            "input_no_det": 0,
            "gen_no_det": 0,
            "discarded_batches": 0,
        }

        # Freeze other components when they expose parameters
        if hasattr(self.embedder, "parameters"):
            for param in self.embedder.parameters():
                param.requires_grad = False

        if self.stylegan is not None and hasattr(self.stylegan, "parameters"):
            for param in self.stylegan.parameters():
                param.requires_grad = False

        # Place modules on the target device when possible
        if hasattr(self.embedder, "to"):
            self.embedder.to(self.device)
        if self.stylegan is not None and hasattr(self.stylegan, "to"):
            self.stylegan.to(self.device)
        self.projector.to(self.device)

    def forward(self, frames: torch.Tensor | np.ndarray, key: torch.Tensor, source_path: Path | None = None, seq_len: int | None = None) -> KfaarResult:
        device = self.device
        frames_t = self._to_sequence_tensor(frames, device=device)
        if seq_len is not None:
            frames_t = frames_t[:seq_len]
        seq_len = frames_t.shape[0]

        detections: list[Detection | None] = []
        aligned_by_idx: list[tuple[int, torch.Tensor]] = []
        aligned_faces: list[torch.Tensor] = []
        valid_mask: list[bool] = []

        for idx in range(seq_len):
            frame = frames_t[idx]
            dets = self.detector.detect(frame)
            if dets:
                top_det = max(dets, key=lambda d: d.score)
                detections.append(top_det)
                aligned_face = self.aligner.align(frame, top_det).to(device)
                aligned_by_idx.append((idx, aligned_face))
                aligned_faces.append(aligned_face)
                valid_mask.append(True)
            else:
                detections.append(None)
                aligned_faces.append(torch.empty(0, device=device))
                valid_mask.append(False)
                self.stats["input_no_det"] = self.stats.get("input_no_det", 0) + 1

        embed_size = getattr(self.embedder, "embedding_size", None) or 512
        real_embeddings = torch.zeros(seq_len, embed_size, device=device)

        if aligned_by_idx:
            align_only = [item[1] for item in aligned_by_idx]
            source_paths = [source_path] * len(align_only) if source_path is not None else None
            emb = self.embedder.embed(align_only, source_paths=source_paths, with_grad=False).to(device)
            for emb_val, (idx, _) in zip(emb, aligned_by_idx):
                real_embeddings[idx] = emb_val

        key_t = key.to(device)
        projected_z: torch.Tensor

        if self._projector_is_lstm:
            if key_t.dim() == 1:
                key_t = key_t.unsqueeze(0)
            if key_t.dim() == 2:
                key_t = key_t.unsqueeze(1).expand(-1, seq_len, -1)
            elif key_t.dim() == 3 and key_t.shape[1] != seq_len:
                key_t = key_t.expand(-1, seq_len, -1)

            proj_in = real_embeddings.unsqueeze(0)
            projected_seq = self.projector.project(proj_in, key_t)
            projected_z = projected_seq[0]
        else:
            projected_list: list[torch.Tensor] = []
            for idx in range(seq_len):
                z_in = real_embeddings[idx]
                projected = self.projector.project(z_in, key_t)
                projected_list.append(projected.squeeze(0) if projected.dim() > 1 else projected)
            projected_z = torch.stack(projected_list, dim=0) if projected_list else torch.zeros(seq_len, embed_size, device=device)

        w_latents: torch.Tensor | None = None
        generated_images: torch.Tensor | None = None
        virtual_embeddings: torch.Tensor | None = None

        if self.stylegan is not None and projected_z.numel() > 0:
            w_list: list[torch.Tensor] = []
            gen_images: list[torch.Tensor] = []
            virtual_list: list[torch.Tensor] = []

            for latent in projected_z:
                w = self.stylegan.map(latent.unsqueeze(0))
                use_fp32 = device.type == "cpu"
                images = self.stylegan.synthesize(w, noise_mode="const", force_fp32=use_fp32)

                w_list.append(w[0])
                gen_img = images[0]
                gen_images.append(gen_img)

                img_01 = gen_img.clamp(-1, 1).add(1).div(2.0)
                dets = self.detector.detect(img_01)
                if dets:
                    top_det = max(dets, key=lambda d: d.score)
                    synth_aligned = self.aligner.align(img_01, top_det).to(device)
                    virt = self.embedder.embed([synth_aligned], with_grad=True).to(device)[0]
                else:
                    virt = torch.zeros(embed_size, device=device)
                    self.stats["gen_no_det"] = self.stats.get("gen_no_det", 0) + 1
                virtual_list.append(virt)

            w_latents = torch.stack(w_list, dim=0) if w_list else None
            generated_images = torch.stack(gen_images, dim=0) if gen_images else None
            virtual_embeddings = torch.stack(virtual_list, dim=0) if virtual_list else None
        else:
            virtual_embeddings = torch.zeros(seq_len, embed_size, device=device)

        return KfaarResult(
            detections=detections,
            aligned_faces=aligned_faces,
            real_embeddings=real_embeddings.detach(),
            projected_z=projected_z,
            w_latents=w_latents,
            generated_images=generated_images,
            virtual_embeddings=virtual_embeddings,
            valid_mask=torch.tensor(valid_mask, device=device, dtype=torch.bool),
        )

    @staticmethod
    def _to_sequence_tensor(frames: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
        if isinstance(frames, torch.Tensor):
            tensor = frames
        else:
            tensor = torch.from_numpy(np.asarray(frames))

        if tensor.dim() == 3:
            if tensor.shape[0] == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Expected image with 3 channels, got {tuple(tensor.shape)}")
        elif tensor.dim() == 4:
            if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected frames with shape (Seq,C,H,W) or (C,H,W), got {tuple(tensor.shape)}")

        tensor = tensor.float()
        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        return tensor.to(device)

    def hpvg_train_step(
        self,
        frames: torch.Tensor,
        labels: torch.Tensor,
        seq_lens: torch.Tensor | list[int] | None,
        key_1: torch.Tensor,
        key_2: torch.Tensor,
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
        lambda_temp: float = 0.0,
    ) -> torch.Tensor:
        """Compute loss, apply gradients, and return total loss."""
        loss = self.hpvg_loss(
            frames,
            labels,
            seq_lens,
            key_1,
            key_2,
            margin=margin,
            lambda_ano=lambda_ano,
            lambda_syn=lambda_syn,
            lambda_div=lambda_div,
            lambda_dif=lambda_dif,
            lambda_temp=lambda_temp,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        return loss

    def hpvg_loss(
        self,
        frames: torch.Tensor,
        labels: torch.Tensor,
        seq_lens: torch.Tensor | list[int] | None,
        key_1: torch.Tensor,
        key_2: torch.Tensor,
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
        lambda_temp: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute the HPVG loss without performing an optimizer step.
        Useful for validation or any gradient-free evaluation.
        """
        *_, total = self.hpvg_loss_components(
            frames,
            labels,
            seq_lens,
            key_1,
            key_2,
            margin=margin,
            lambda_ano=lambda_ano,
            lambda_syn=lambda_syn,
            lambda_div=lambda_div,
            lambda_dif=lambda_dif,
            lambda_temp=lambda_temp,
        )
        return total

    def hpvg_loss_components(
        self,
        frames: torch.Tensor,
        labels: torch.Tensor,
        seq_lens: torch.Tensor | list[int] | None,
        key_1: torch.Tensor,
        key_2: torch.Tensor,
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
        lambda_temp: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        device = self.device

        if frames.dim() == 4:
            frames = frames.unsqueeze(1)

        if frames.dim() != 5:
            raise ValueError(f"Expected frames with shape (B,Seq,C,H,W), got {tuple(frames.shape)}")

        batch_size, max_seq, _, _, _ = frames.shape
        if seq_lens is None:
            seq_lens = [max_seq] * batch_size
        seq_lens_list = [int(x) for x in (seq_lens.tolist() if torch.is_tensor(seq_lens) else seq_lens)]

        outs_k1 = []
        outs_k2 = []
        for idx in range(batch_size):
            seq_len = seq_lens_list[idx]
            sample_frames = frames[idx, :seq_len]
            outs_k1.append(self.forward(sample_frames, key_1, seq_len=seq_len))
            outs_k2.append(self.forward(sample_frames, key_2, seq_len=seq_len))

        real_feats: list[torch.Tensor] = []
        virt_k1_feats: list[torch.Tensor] = []
        virt_k2_feats: list[torch.Tensor] = []
        filtered_labels: list[int] = []

        for idx in range(batch_size):
            out1 = outs_k1[idx]
            out2 = outs_k2[idx]

            mask1 = out1.valid_mask if out1.valid_mask is not None else torch.ones(out1.real_embeddings.shape[0], device=device, dtype=torch.bool)
            mask2 = out2.valid_mask if out2.valid_mask is not None else torch.ones(out2.real_embeddings.shape[0], device=device, dtype=torch.bool)
            joint_mask = mask1 & mask2
            if joint_mask.dim() == 0:
                joint_mask = joint_mask.unsqueeze(0)

            if joint_mask.any():
                real_feats.append(out1.real_embeddings[joint_mask])
                virt_k1_feats.append(out1.virtual_embeddings[joint_mask])
                virt_k2_feats.append(out2.virtual_embeddings[joint_mask])
                filtered_labels.extend([int(labels[idx].item())] * int(joint_mask.sum().item()))

        if not filtered_labels or sum(t.shape[0] for t in real_feats) < 2:
            self.stats["discarded_batches"] = self.stats.get("discarded_batches", 0) + 1
            penalty_val = 10.0
            param_anchor = next(iter(self.projector.parameters()), None)
            if param_anchor is None:
                penalty = torch.tensor(penalty_val, device=device, requires_grad=True)
            else:
                penalty = param_anchor.sum() * 0.0 + torch.tensor(penalty_val, device=device)
            return penalty, penalty, penalty, penalty, penalty

        real_concat = torch.cat(real_feats, dim=0)
        virt_k1_concat = torch.cat(virt_k1_feats, dim=0)
        virt_k2_concat = torch.cat(virt_k2_feats, dim=0)
        labels_tensor = torch.tensor(filtered_labels, device=device, dtype=torch.long)

        ano = anonymity_loss(real_feat=real_concat, virtual_feat=virt_k1_concat, margin=margin)
        div = diversity_loss(virtual_feat_k1=virt_k1_concat, virtual_feat_k2=virt_k2_concat, margin=margin)

        syn_loss_val = torch.tensor(0.0, device=device)
        dif_loss_val = torch.tensor(0.0, device=device)
        syn_count = 0
        dif_count = 0

        num_valid = real_concat.shape[0]
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                f_i = virt_k1_concat[i : i + 1]
                f_j = virt_k1_concat[j : j + 1]
                if labels_tensor[i] == labels_tensor[j]:
                    syn_loss_val = syn_loss_val + synchronism_loss(f_i, f_j, margin=margin)
                    syn_count += 1
                else:
                    dif_loss_val = dif_loss_val + differentiation_loss(f_i, f_j, margin=margin)
                    dif_count += 1

        if syn_count > 0:
            syn_loss_val = syn_loss_val / syn_count
        if dif_count > 0:
            dif_loss_val = dif_loss_val / dif_count

        temp_loss = torch.tensor(0.0, device=device)
        if lambda_temp > 0.0 and self._projector_is_lstm:
            seq_virtuals: list[torch.Tensor] = []
            for out1 in outs_k1:
                vm = out1.valid_mask if out1.valid_mask is not None else torch.ones(out1.virtual_embeddings.shape[0], device=device, dtype=torch.bool)
                seq_feats = out1.virtual_embeddings[vm]
                if seq_feats.dim() == 1:
                    seq_feats = seq_feats.unsqueeze(0)
                seq_virtuals.append(seq_feats)
            if len(seq_virtuals) >= 2:
                temp_loss = temporal_smoothness_loss(seq_virtuals, reduction="mean")

        total = total_hpvg_loss(
            ano,
            syn_loss_val,
            div,
            dif_loss_val,
            temp_loss if lambda_temp > 0.0 and self._projector_is_lstm else None,
            lambda_ano,
            lambda_syn,
            lambda_div,
            lambda_dif,
            lambda_temp,
        )

        return ano, syn_loss_val, div, dif_loss_val, total
