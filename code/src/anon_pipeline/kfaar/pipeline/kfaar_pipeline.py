from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..components import EmbeddingModel, FaceDetector, ProjectorMLP
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
    w_latents: Optional[torch.Tensor] = None
    generated_images: Optional[torch.Tensor] = None
    virtual_embeddings: Optional[torch.Tensor] = None


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
        self.optimizer = torch.optim.Adam(
            self.projector.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999)
        )

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

    def forward(self, image: torch.Tensor | np.ndarray, key: torch.Tensor, source_path: Path | None = None) -> KfaarResult:
        device = self.device

        image_t = self._to_tensor(image, device=device)

        detections = self.detector.detect(image_t)
        if detections:
            detections = [max(detections, key=lambda d: d.score)]  # keep highest-score face only
        if not detections:
            logger.debug("No detections returned; skipping embedding stage")
            empty = torch.empty(0, 0, device=device)
            return KfaarResult(detections, [], empty, empty)

        aligned = [self.aligner.align(image_t, det).to(device) for det in detections]
        source_paths = [source_path] * len(aligned) if source_path is not None else None

        real_embeddings = self.embedder.embed(aligned, source_paths=source_paths, with_grad=False).to(device)

        key = key.to(device)
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if key.shape[0] == 1 and real_embeddings.shape[0] > 1:
            key = key.expand(real_embeddings.shape[0], -1)

        projected = self.projector.project(real_embeddings.detach(), key)

        w_latents = None
        generated_images = None
        virtual_embeddings = None

        if self.stylegan is not None and projected is not None and projected.numel() > 0:
            z = projected.to(device).float()
            w = self.stylegan.map(z)

            use_fp32 = device.type == "cpu"
            images = self.stylegan.synthesize(w, noise_mode="const", force_fp32=use_fp32)

            w_latents = w
            generated_images = images

            images_01 = images.clamp(-1, 1).add(1).div(2.0)
            synth_aligned: list[torch.Tensor] = []
            for img in images_01:
                dets = self.detector.detect(img)
                if not dets:
                    continue
                top_det = max(dets, key=lambda d: d.score)
                synth_aligned.append(self.aligner.align(img, top_det).to(device))

            if synth_aligned:
                virtual_embeddings = self.embedder.embed(synth_aligned, with_grad=True).to(device)
            else:
                virtual_embeddings = torch.empty(0, device=device)

        return KfaarResult(
            detections=detections,
            aligned_faces=aligned,
            real_embeddings=real_embeddings.detach(),
            projected_z=projected,
            w_latents=w_latents,
            generated_images=generated_images,
            virtual_embeddings=virtual_embeddings,
        )

    @staticmethod
    def _to_tensor(image: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            img = image
        else:
            img = torch.from_numpy(image)

        if img.dim() == 3 and img.shape[0] == 3:
            pass
        elif img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected image shape (3,H,W) or (H,W,3), got {tuple(img.shape)}")

        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.0 or img.min() < 0.0:
            img = img / 255.0
        return img.to(device)

    def hpvg_train_step(
        self,
        images: list[np.ndarray | torch.Tensor],      # Batch of images
        labels: torch.Tensor,          # Identity labels for images
        key_1: torch.Tensor,           # First set of keys
        key_2: torch.Tensor,           # Second set of keys
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
    ) -> torch.Tensor:
        """Compute loss, apply gradients, and return total loss."""
        loss = self.hpvg_loss(
            images,
            labels,
            key_1,
            key_2,
            margin=margin,
            lambda_ano=lambda_ano,
            lambda_syn=lambda_syn,
            lambda_div=lambda_div,
            lambda_dif=lambda_dif,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        return loss

    def hpvg_loss(
        self,
        images: list[np.ndarray | torch.Tensor],
        labels: torch.Tensor,
        key_1: torch.Tensor,
        key_2: torch.Tensor,
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the HPVG loss without performing an optimizer step.
        Useful for validation or any gradient-free evaluation.
        """
        *_, total = self.hpvg_loss_components(
            images,
            labels,
            key_1,
            key_2,
            margin=margin,
            lambda_ano=lambda_ano,
            lambda_syn=lambda_syn,
            lambda_div=lambda_div,
            lambda_dif=lambda_dif,
        )
        return total

    def hpvg_loss_components(
        self,
        images: list[np.ndarray | torch.Tensor],
        labels: torch.Tensor,
        key_1: torch.Tensor,
        key_2: torch.Tensor,
        *,
        margin: float = 0.5,
        lambda_ano: float = 0.4,
        lambda_syn: float = 1.0,
        lambda_div: float = 1.0,
        lambda_dif: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. Run forward passes for both keys
        # Note: Ensure your self.forward logic now picks the 'best' single face
        outs_k1 = [self.forward(img, key_1) for img in images]
        outs_k2 = [self.forward(img, key_2) for img in images]

        # 2. Find indices where BOTH keys resulted in successful face detection
        valid_indices = []
        for i in range(len(images)):
            # Check if embeddings exist and contain data
            k1_valid = outs_k1[i].virtual_embeddings is not None and outs_k1[i].virtual_embeddings.numel() > 0
            k2_valid = outs_k2[i].virtual_embeddings is not None and outs_k2[i].virtual_embeddings.numel() > 0
            if k1_valid and k2_valid:
                valid_indices.append(i)

        # 3. Guard against empty/too small batches
        if len(valid_indices) < 2:
            logger.warning(f"Batch discarded: only {len(valid_indices)} frames had valid faces for both keys.")
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, zero, zero, zero, zero

        # 4. Extract tensors using only the valid indices
        # We use torch.cat to safely handle [1, 512] tensors into an [N, 512] batch
        real_feats = torch.cat([outs_k1[i].real_embeddings for i in valid_indices], dim=0)
        virt_k1_feats = torch.cat([outs_k1[i].virtual_embeddings for i in valid_indices], dim=0)
        virt_k2_feats = torch.cat([outs_k2[i].virtual_embeddings for i in valid_indices], dim=0)
        
        # Filter labels to match the new batch size
        filtered_labels = labels[valid_indices]

        # 5. Compute standard losses (these are usually averaged internally)
        ano = anonymity_loss(real_feat=real_feats, virtual_feat=virt_k1_feats, margin=margin)
        div = diversity_loss(virtual_feat_k1=virt_k1_feats, virtual_feat_k2=virt_k2_feats, margin=margin)

        # 6. Compute Identity-based losses (Synchronism and Differentiation)
        syn_loss_val = torch.tensor(0.0, device=self.device)
        dif_loss_val = torch.tensor(0.0, device=self.device)
        
        syn_count = 0
        dif_count = 0
        
        num_valid = len(valid_indices)
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                # Slice to keep dimensions [1, 512] for loss functions
                f_i = virt_k1_feats[i : i + 1]
                f_j = virt_k1_feats[j : j + 1]
                
                # Check labels using the filtered label tensor
                if filtered_labels[i] == filtered_labels[j]:
                    syn_loss_val = syn_loss_val + synchronism_loss(f_i, f_j, margin=margin)
                    syn_count += 1
                else:
                    dif_loss_val = dif_loss_val + differentiation_loss(f_i, f_j, margin=margin)
                    dif_count += 1

        # --- NORMALIZATION STEP ---
        # Normalize the pairwise sums by the number of pairs to get the mean
        if syn_count > 0:
            syn_loss_val = syn_loss_val / syn_count
        
        if dif_count > 0:
            dif_loss_val = dif_loss_val / dif_count

        # 7. Total weighted loss
        total = total_hpvg_loss(
            ano, syn_loss_val, div, dif_loss_val,
            lambda_ano, lambda_syn, lambda_div, lambda_dif,
        )

        return ano, syn_loss_val, div, dif_loss_val, total
