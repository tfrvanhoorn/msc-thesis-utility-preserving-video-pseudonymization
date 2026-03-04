from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import torch
from torchvision import utils as vutils
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
        face_swapper: object | None = None,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.projector = projector
        self.stylegan = stylegan
        self.face_swapper = face_swapper
        self.device = device or next(projector.parameters()).device
        self._projector_is_lstm = isinstance(projector, ProjectorLSTM)
        self._warned_face_swapper = False
        
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
        self._video_accumulators: Dict[str, Dict[str, list[np.ndarray]]] = {}
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
        sample_context: Optional[str] = None,
        use_face_swapper: bool = False,
    ) -> KfaarResult:
        device = self.device
        frames_t = self._to_sequence_tensor(frames, device=device)
        seq_len = frames_t.shape[0]
        center_idx = max(0, (seq_len - 1) // 2)

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
            projected_seq = self.projector.project(real_emb.unsqueeze(0), k_in)[0]
            projected_z = projected_seq[center_idx : center_idx + 1]
        else:
            real_center = real_emb[center_idx : center_idx + 1]
            projected_z = self.projector.project(real_center, key_t.expand(1, -1))

        # 3. Virtual Processing
        v_embeddings = torch.zeros_like(projected_z)
        gen_mask = [False]
        
        images = None
        swapped_images = None
        
        if self.stylegan is not None:
            w = self.stylegan.map(projected_z, truncation_psi=self.truncation_psi)
            images = self.stylegan.synthesize(w, noise_mode="const")
            img = images[0].clamp(-1, 1).add(1).div(2.0)
            det_input = img
            
            if use_face_swapper:
                if self.face_swapper is None and not self._warned_face_swapper:
                    logger.warning("Face swapping requested but no swapper is configured; proceeding without swap.")
                    self._warned_face_swapper = True
                elif self.face_swapper is not None:
                    
                    # --- PIPELINE ADAPTATION: Align StyleGAN Output First ---
                    stylegan_dets = self.detector.detect(img)
                    if stylegan_dets:
                        stylegan_top = max(stylegan_dets, key=lambda d: d.score)
                        aligned_stylegan = self.aligner.align(img, stylegan_top).to(device)
                    else:
                        aligned_stylegan = img # Fallback to unaligned blob if GAN is untrained
                    
                    target_aligned = aligned_faces[center_idx]
                    
                    # Ensure the target frame actually had a face before swapping
                    if target_aligned.numel() > 0:
                        swapped = self.face_swapper.swap(aligned_stylegan, target_aligned)
                    else:
                        swapped = None

                    if swapped is not None:
                        det_input = swapped
                        swapped_images = swapped.unsqueeze(0)
                        
                        # --- BYPASS DETECTOR: Swap output is already an aligned crop ---
                        v_embeddings[0] = self.embedder.embed([det_input], with_grad=True)[0]
                        gen_mask[0] = True
                    elif not self._warned_face_swapper:
                        logger.warning("Face swapper failed to produce output; proceeding without swap.")
                        self._warned_face_swapper = True

            # Standard detection logic ONLY runs if swapper is disabled or failed
            if not (use_face_swapper and swapped_images is not None):
                dets = self.detector.detect(det_input)
                if dets:
                    top = max(dets, key=lambda d: d.score)
                    aligned = self.aligner.align(det_input, top).to(device)
                    v_embeddings[0] = self.embedder.embed([aligned], with_grad=True)[0]
                    gen_mask[0] = True
                else:
                    v_embeddings[0] = projected_z[0].sum() * 0.0
                    self.stats["gen_no_det"] += 1
            
            self._maybe_save_generated(
                images,
                gen_mask,
                swapped_images=swapped_images,
                sample_label=sample_label,
                key_tag=key_tag,
                batch_index=batch_index,
                input_frames=frames_t[center_idx : center_idx + 1],
                sample_context=sample_context,
            )

        input_mask_t = torch.tensor([input_mask[center_idx]], device=device, dtype=torch.bool)
        gen_mask_t = torch.tensor(gen_mask, device=device, dtype=torch.bool)
        valid_mask = input_mask_t & gen_mask_t

        return KfaarResult(
            detections=[], # Simplified for training speed
            aligned_faces=[aligned_faces[center_idx]],
            real_embeddings=real_emb[center_idx : center_idx + 1].detach(),
            projected_z=projected_z,
            virtual_embeddings=v_embeddings,
            valid_mask=valid_mask,
            gen_mask=gen_mask_t,
        )

    def forward_eval(
        self,
        frames: torch.Tensor,
        key: torch.Tensor,
        *,
        sample_label: Optional[int] = None,
        key_tag: Optional[str] = None,
        batch_index: Optional[int] = None,
        sample_context: Optional[str] = None,
        use_face_swapper: bool = False,
    ) -> KfaarResult:
        with torch.no_grad():
            return self.forward(
                frames,
                key,
                sample_label=sample_label,
                key_tag=key_tag,
                batch_index=batch_index,
                sample_context=sample_context,
                use_face_swapper=use_face_swapper,
            )

    def hpvg_train_step(
        self,
        frames,
        labels,
        seq_lens,
        key_1,
        key_2,
        batch_index: Optional[int] = None,
        use_face_swapper: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        
        # components will return None if criteria not met
        comps = self.hpvg_loss_components(
            frames,
            labels,
            seq_lens,
            key_1,
            key_2,
            batch_index=batch_index,
            use_face_swapper=use_face_swapper,
            **kwargs,
        )
        
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
        use_face_swapper: bool = False,
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
                use_face_swapper=use_face_swapper,
            )
            res2 = self.forward(
                frames[b, :seq_lens_list[b]],
                key_2,
                sample_label=label_int,
                key_tag="k2",
                batch_index=batch_index,
                use_face_swapper=use_face_swapper,
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
        # Evaluation uses single pass; no per-epoch subfolders for saves
        self._saving_active = self._save_enabled
        self._video_accumulators.clear()

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
        swapped_images: Optional[torch.Tensor] = None,
        sample_label: Optional[int],
        key_tag: Optional[str],
        batch_index: Optional[int],
        input_frames: Optional[torch.Tensor] = None,
        sample_context: Optional[str] = None,
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
        if save_images_dir is not None:
            save_images_dir.mkdir(parents=True, exist_ok=True)
        if save_videos_dir is not None:
            save_videos_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
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
                context_part = (sample_context or "context").replace("/", "_").replace("\\", "_")
                video_key = f"{label_part}_{context_part}"

                sample_id = self._saved_this_epoch
                if input_frames is not None:
                    input_img = input_frames[idx].detach().cpu()
                    if input_img.min() < 0.0 or input_img.max() > 1.0:
                        input_img = input_img.add(1).div(2.0)
                    input_img = input_img.clamp(0.0, 1.0)
                    if save_images_dir is not None:
                        vutils.save_image(input_img, save_images_dir / f"{base}_input.png")

                stylegan_img = images[idx].detach().cpu().add(1).div(2.0).clamp(0.0, 1.0)
                if save_images_dir is not None:
                    vutils.save_image(stylegan_img, save_images_dir / f"{base}_stylegan.png")
                    vutils.save_image(stylegan_img, save_images_dir / f"{base}_gen.png")

                swapped_img = None
                if swapped_images is not None and idx < swapped_images.shape[0]:
                    swapped_img = swapped_images[idx].detach().cpu().clamp(0.0, 1.0)
                    if save_images_dir is not None:
                        vutils.save_image(swapped_img, save_images_dir / f"{base}_swapped.png")

                if save_videos_dir is not None and imageio is not None:
                    buffers = self._video_accumulators.setdefault(video_key, {"input": [], "gen": []})
                    vid_frame_source = swapped_img if swapped_img is not None else stylegan_img
                    gen_frame = (vid_frame_source.permute(1, 2, 0) * 255).byte().numpy()
                    buffers["gen"].append(gen_frame)
                    if input_frames is not None:
                        in_frame = (input_img.permute(1, 2, 0) * 255).byte().numpy()
                        buffers["input"].append(in_frame)

                self._saved_this_epoch = sample_id + 1

    def finalize_saving(self) -> None:
        """Flush accumulated video frames into GIF files."""
        if not self._save_enabled or not self._save_videos:
            return
        save_videos_dir = self._save_dir_videos if self._save_dir_videos is not None else None
        if save_videos_dir is None:
            return
        save_videos_dir.mkdir(parents=True, exist_ok=True)
        if imageio is None:
            logging.warning("GIF saving skipped: imageio not available")
            self._video_accumulators.clear()
            return

        for video_key, buffers in list(self._video_accumulators.items()):
            gen_frames = buffers.get("gen", [])
            inp_frames = buffers.get("input", [])
            try:
                if gen_frames:
                    imageio.mimsave(save_videos_dir / f"{video_key}_gen.gif", gen_frames, duration=0.1)
                if inp_frames:
                    imageio.mimsave(save_videos_dir / f"{video_key}_input.gif", inp_frames, duration=0.1)
            except Exception as exc:  # pragma: no cover
                logging.warning("imageio GIF write failed for %s: %s", video_key, exc)
        self._video_accumulators.clear()