from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict
import numpy as np
import torch
from torchvision import utils as vutils, io as tvio
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None
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
    gen_mask: torch.Tensor   # Mask where generated output had a detected face

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
        truncation_psi: float = 0.5,
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
        self._save_dir_images = self._save_dir / "images" if self._save_dir is not None else None
        self._save_dir_videos = self._save_dir / "videos" if self._save_dir is not None else None
        self._save_mode = save_mode
        self._save_max_per_epoch = save_max_per_epoch
        self._save_videos = False
        self._saved_this_epoch = 0
        self._current_epoch = 0
        self.truncation_psi = truncation_psi
        if self._save_enabled:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            if self._save_dir_images is not None:
                self._save_dir_images.mkdir(parents=True, exist_ok=True)
            if self._save_dir_videos is not None:
                self._save_dir_videos.mkdir(parents=True, exist_ok=True)

    def forward(
        self,
        frames: torch.Tensor,
        key: torch.Tensor,
        *,
        sample_label: Optional[int] = None,
        key_tag: Optional[str] = None,
        batch_index: Optional[int] = None,
    ) -> KfaarResult:
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
            w = self.stylegan.map(projected_z, truncation_psi=self.truncation_psi)
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

            self._maybe_save_generated(
                images,
                gen_mask,
                sample_label=sample_label,
                key_tag=key_tag,
                batch_index=batch_index,
                input_frames=frames_t,
            )

        input_mask_t = torch.tensor(input_mask, device=device, dtype=torch.bool)
        gen_mask_t = torch.tensor(gen_mask, device=device, dtype=torch.bool)
        valid_mask = input_mask_t & gen_mask_t

        return KfaarResult(
            detections=[], # Simplified for training speed
            aligned_faces=aligned_faces,
            real_embeddings=real_emb.detach(),
            projected_z=projected_z,
            virtual_embeddings=v_embeddings,
            valid_mask=valid_mask,
            gen_mask=gen_mask_t,
        )

    def hpvg_train_step(self, frames, labels, seq_lens, key_1, key_2, batch_index: Optional[int] = None, **kwargs) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        
        # components will return None if criteria not met
        comps = self.hpvg_loss_components(frames, labels, seq_lens, key_1, key_2, batch_index=batch_index, **kwargs)
        
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
        margin=0.5, lambda_ano=0.4, lambda_syn=1.0, lambda_div=1.0, lambda_dif=1.0, lambda_temp=0.0,
        batch_index: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        device = self.device
        batch_size = frames.shape[0]
        seq_lens_list = [int(x) for x in (seq_lens.tolist() if torch.is_tensor(seq_lens) else seq_lens)]

        all_real, all_v1, all_v2, all_labels = [], [], [], []
        proj_norm_terms = []
        nondet_faces = 0
        det_failures = 0

        for b in range(batch_size):
            label_int = int(labels[b].item()) if torch.is_tensor(labels) else int(labels[b])
            res1 = self.forward(
                frames[b, :seq_lens_list[b]],
                key_1,
                sample_label=label_int,
                key_tag="k1",
                batch_index=batch_index,
            )
            res2 = self.forward(
                frames[b, :seq_lens_list[b]],
                key_2,
                sample_label=label_int,
                key_tag="k2",
                batch_index=batch_index,
            )

            proj_norm_terms.append(res1.projected_z.pow(2).mean())
            proj_norm_terms.append(res2.projected_z.pow(2).mean())
            nondet_faces += int((~res1.gen_mask).sum().item() + (~res2.gen_mask).sum().item())
            
            mask = res1.valid_mask & res2.valid_mask
            
            # Track frames where we could not form a valid pair (input or gen missing)
            det_failures += int((~mask).sum().item())

            if mask.any():
                all_real.append(res1.real_embeddings[mask])
                all_v1.append(res1.virtual_embeddings[mask])
                all_v2.append(res2.virtual_embeddings[mask])
                all_labels.extend([label_int] * int(mask.sum()))

        proj_norm = torch.stack(proj_norm_terms).mean() if proj_norm_terms else torch.tensor(0.0, device=device)

        # --- VALIDATION GATE ---
        if not all_labels:
            ano = syn = div = dif = torch.tensor(0.0, device=device)
            penalty_missing_pairs = proj_norm * (1.0 + 0.5 * float(det_failures))
            penalty_nondet = proj_norm * (2.0 * float(nondet_faces))
            total = penalty_missing_pairs + penalty_nondet
            return ano, syn, div, dif, total

        # Requirement: At least 2 identities AND each has at least 2 samples
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        valid_identities = unique_labels[counts >= 2]

        # If the criteria for Syn/Dif aren't met, we return the penalty to guide the model
        if len(valid_identities) < 2:
            ano = syn = div = dif = torch.tensor(0.0, device=device)
            penalty_missing_pairs = proj_norm * (1.0 + 0.5 * float(det_failures))
            penalty_nondet = proj_norm * (2.0 * float(nondet_faces))
            total = penalty_missing_pairs + penalty_nondet
            return ano, syn, div, dif, total

        # Filter tensors for Synchronism/Differentiation
        lab_t = torch.tensor(all_labels, device=device)
        keep_mask = torch.zeros_like(lab_t, dtype=torch.bool)
        for vid in valid_identities:
            keep_mask |= (lab_t == int(vid))

        v1_c = torch.cat(all_v1)[keep_mask]
        v2_c = torch.cat(all_v2)[keep_mask]
        real_c = torch.cat(all_real)[keep_mask]
        lab_final = lab_t[keep_mask]

        # 1. Sample-wise Losses
        ano = anonymity_loss(real_c, v1_c, v2_c, margin=margin)
        div = diversity_loss(v1_c, v2_c, margin=margin)
        syn = synchronism_loss(v1_c, v2_c, lab_final, margin=margin)
        dif = differentiation_loss(v1_c, v2_c, lab_final, margin=margin)

        penalty_missing_pairs = proj_norm * (0.1 * float(det_failures))
        penalty_nondet = proj_norm * (2.0 * float(nondet_faces))

        total = (lambda_ano * ano + lambda_syn * syn + 
            lambda_div * div + lambda_dif * dif + penalty_nondet + penalty_missing_pairs)

        return ano, syn, div, dif, total

    @staticmethod
    def _to_sequence_tensor(frames, device):
        t = frames if torch.is_tensor(frames) else torch.from_numpy(frames)
        return t.float().to(device)

    def configure_saving(
        self,
        save_dir: Path,
        *,
        mode: str = "detected",
        max_per_epoch: Optional[int] = None,
        save_videos: bool = False,
    ) -> None:
        """Enable saving synthesized faces to disk with a per-epoch cap."""
        self._save_enabled = True
        self._saving_active = True
        self._save_dir = Path(save_dir)
        self._save_dir_images = self._save_dir / "images"
        self._save_dir_videos = self._save_dir / "videos"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir_images.mkdir(parents=True, exist_ok=True)
        if save_videos:
            self._save_dir_videos.mkdir(parents=True, exist_ok=True)
        self._save_mode = mode
        self._save_max_per_epoch = max_per_epoch
        self._save_videos = bool(save_videos)
        self._saved_this_epoch = 0

    def begin_epoch(self, epoch: int) -> None:
        """Reset save counters for a new epoch and prepare the epoch directory."""
        self._current_epoch = epoch
        self._saved_this_epoch = 0
        if self._save_enabled:
            if self._save_dir_images is not None:
                (self._save_dir_images / f"epoch_{epoch:03d}").mkdir(parents=True, exist_ok=True)
            if self._save_videos and self._save_dir_videos is not None:
                (self._save_dir_videos / f"epoch_{epoch:03d}").mkdir(parents=True, exist_ok=True)
        self._saving_active = self._save_enabled

    def disable_saving(self) -> None:
        self._saving_active = False

    def enable_saving(self) -> None:
        if self._save_enabled:
            self._saving_active = True

    def _maybe_save_generated(
        self,
        images: Optional[torch.Tensor],
        gen_mask: Sequence[bool],
        *,
        sample_label: Optional[int],
        key_tag: Optional[str],
        batch_index: Optional[int],
        input_frames: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._save_enabled or not self._saving_active:
            return
        if images is None:
            return
        if self._save_max_per_epoch is not None and self._saved_this_epoch >= self._save_max_per_epoch:
            return

        mode = self._save_mode
        save_images_dir = self._save_dir_images if self._save_dir_images is not None else None
        save_videos_dir = self._save_dir_videos if self._save_videos and self._save_dir_videos is not None else None
        if save_images_dir is None and save_videos_dir is None:
            return
        epoch_image_dir = save_images_dir / f"epoch_{self._current_epoch:03d}" if save_images_dir is not None else None
        epoch_video_dir = save_videos_dir / f"epoch_{self._current_epoch:03d}" if save_videos_dir is not None else None
        if epoch_image_dir is not None:
            epoch_image_dir.mkdir(parents=True, exist_ok=True)
        if epoch_video_dir is not None:
            epoch_video_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            video_written = False
            base_video = None
            for idx in range(images.shape[0]):
                if self._save_max_per_epoch is not None and self._saved_this_epoch >= self._save_max_per_epoch:
                    break

                detected = bool(gen_mask[idx])
                if mode == "detected" and not detected:
                    continue
                if mode == "undetected" and detected:
                    continue

                label_val = int(sample_label) if sample_label is not None else None
                label_part = f"id{label_val:06d}" if label_val is not None else "idunknown"
                key_part = f"key{key_tag}" if key_tag else "key"
                batch_part = f"{int(batch_index):06d}" if batch_index is not None else "000000"
                sample_part = f"{self._saved_this_epoch:06d}"
                status_part = "det" if detected else "undet"
                base = f"{label_part}_key{key_part}_batch{batch_part}_sample{sample_part}_{status_part}"
                if base_video is None:
                    base_video = base

                sample_id = self._saved_this_epoch
                if input_frames is not None:
                    input_img = input_frames[idx].detach().cpu()
                    if input_img.min() < 0.0 or input_img.max() > 1.0:
                        input_img = input_img.add(1).div(2.0)
                    input_img = input_img.clamp(0.0, 1.0)
                    if epoch_image_dir is not None:
                        vutils.save_image(input_img, epoch_image_dir / f"{base}_input.png")

                gen_img = images[idx].detach().cpu().add(1).div(2.0).clamp(0.0, 1.0)
                if epoch_image_dir is not None:
                    vutils.save_image(gen_img, epoch_image_dir / f"{base}_gen.png")

                if (not video_written) and epoch_video_dir is not None and input_frames is not None:
                    # Attempt torchvision (requires PyAV), then fallback to imageio; otherwise skip with warning.
                    inp_frames = input_frames.detach().cpu()
                    if inp_frames.min() < 0.0 or inp_frames.max() > 1.0:
                        inp_frames = inp_frames.add(1).div(2.0)
                    inp_frames = inp_frames.clamp(0.0, 1.0)
                    inp_vid = (inp_frames.permute(0, 2, 3, 1) * 255).byte()

                    gen_vid_frames = images.detach().cpu()
                    gen_vid_frames = gen_vid_frames.add(1).div(2.0).clamp(0.0, 1.0)
                    gen_vid = (gen_vid_frames.permute(0, 2, 3, 1) * 255).byte()

                    written = False
                    try:
                        tvio.write_video(epoch_video_dir / f"{base_video}_input.mp4", inp_vid, fps=10)
                        tvio.write_video(epoch_video_dir / f"{base_video}_gen.mp4", gen_vid, fps=10)
                        written = True
                    except Exception as exc:
                        logging.warning("torchvision video write failed for %s: %s", base_video, exc)

                    if (not written) and imageio is not None:
                        try:
                            with imageio.get_writer(epoch_video_dir / f"{base_video}_input.mp4", fps=10, codec="libx264") as w:
                                for frame in inp_vid.numpy():
                                    w.append_data(frame)
                            with imageio.get_writer(epoch_video_dir / f"{base_video}_gen.mp4", fps=10, codec="libx264") as w:
                                for frame in gen_vid.numpy():
                                    w.append_data(frame)
                            written = True
                        except Exception as exc:  # pragma: no cover
                            logging.warning("imageio video write failed for %s: %s", base_video, exc)

                    if not written and imageio is not None:
                        try:
                            imageio.mimsave(epoch_video_dir / f"{base_video}_input.gif", inp_vid.numpy(), duration=0.1)
                            imageio.mimsave(epoch_video_dir / f"{base_video}_gen.gif", gen_vid.numpy(), duration=0.1)
                            written = True
                        except Exception as exc:  # pragma: no cover
                            logging.warning("imageio GIF write failed for %s: %s", base_video, exc)

                    if written:
                        video_written = True
                    else:
                        logging.warning("Failed to save video for sample %s: no available writer", base_video)

                self._saved_this_epoch = sample_id + 1