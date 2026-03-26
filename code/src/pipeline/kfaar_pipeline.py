from __future__ import annotations
import logging
import shutil
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
from components import EmbeddingModel, FaceDetector, ProjectorMLP
from components.alignment import FaceAligner
from components.detector import Detection
from components import StyleGAN2Generator
from losses import cosine_loss

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
    input_face_frames: Sequence[torch.Tensor]
    generated_face_frames: Sequence[torch.Tensor]
    w_pre_boundary: Optional[torch.Tensor] = None

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
        use_stylegan_mapper: bool = False,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.projector = projector
        self.stylegan = stylegan
        self.face_swapper = face_swapper
        self.use_stylegan_mapper = use_stylegan_mapper
        self.device = device or next(projector.parameters()).device
        self._warned_face_swapper = False
        
        # Optimizer Setup
        self.optimizer = torch.optim.Adam(self.projector.parameters(), lr=1e-4)
        
        self.stats: Dict[str, int] = {"input_no_det": 0, "gen_no_det": 0, "discarded_batches": 0}

        # Transfer and Freeze
        if hasattr(self.embedder, "to"):
            self.embedder.to(self.device)
        if self.stylegan is not None:
            self.stylegan.to(self.device)
            # Freeze StyleGAN weights but allow grad to flow through outputs
            base_g = getattr(self.stylegan, "_G", None)
            if base_g is not None:
                for p in base_g.parameters():
                    p.requires_grad_(False)
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
            self._clear_dir(self._save_dir)
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
        swap_for_visuals_only: bool = True,
        return_frame_pairs: bool = False,
    ) -> KfaarResult:
        device = self.device
        frames_t = self._to_sequence_tensor(frames, device=device)
        seq_len = frames_t.shape[0]
        center_idx = max(0, (seq_len - 1) // 2)

        # 1. Input Processing
        input_mask = []
        aligned_faces = []
        real_emb = torch.zeros(seq_len, 512, device=device)
        
        # --- TRACK THE ORIGINAL DETECTION ---
        # We need to remember the bounding box of the target frame so we can paste it back later.
        center_detection = None 
        
        for i, frame in enumerate(frames_t):
            dets = self.detector.detect(frame)
            if dets:
                top = max(dets, key=lambda d: d.score)
                aligned = self.aligner.align(frame, top).to(device)
                aligned_faces.append(aligned)
                input_mask.append(True)
                
                if i == center_idx:
                    center_detection = top
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
        real_center = real_emb[center_idx : center_idx + 1]
        projected_z = self.projector.project(real_center, key_t.expand(1, -1))

        # 3. Virtual Processing
        v_embeddings = torch.zeros_like(projected_z)
        gen_mask = [False]
        
        images = None
        swapped_images = None
        w_pre_boundary = None
        frame_pair_inputs: list[torch.Tensor] = []
        frame_pair_generated: list[torch.Tensor] = []
        
        if self.stylegan is not None:
            projected_seq = None
            if return_frame_pairs:
                key_seq = key_t.view(1, -1).expand(seq_len, -1)
                projected_seq = self.projector.project(real_emb, key_seq)

                for frame_idx in range(seq_len):
                    aligned_input = aligned_faces[frame_idx]
                    if aligned_input.numel() == 0 or projected_seq is None:
                        frame_pair_inputs.append(torch.empty(0, device=device))
                        frame_pair_generated.append(torch.empty(0, device=device))
                        continue
                    z_i = projected_seq[frame_idx : frame_idx + 1]
                    w_i = self._project_to_stylegan_w(z_i)
                    img_i = self.stylegan.synthesize(w_i, noise_mode="const")[0].clamp(-1, 1).add(1).div(2.0)

                    visual_i = img_i
                    if use_face_swapper and self.face_swapper is not None:
                        stylegan_dets_i = self.detector.detect(img_i)
                        if stylegan_dets_i:
                            stylegan_top_i = max(stylegan_dets_i, key=lambda d: d.score)
                            aligned_stylegan_i = self.aligner.align(img_i, stylegan_top_i).to(device)

                            is_diffusion = type(self.face_swapper).__name__ == "DiffusionFaceSwapper"
                            target_to_swap_i = frames_t[frame_idx] if is_diffusion else aligned_input
                            if target_to_swap_i.numel() > 0:
                                swapped_i = self.face_swapper.swap(aligned_stylegan_i, target_to_swap_i)
                                if swapped_i is not None:
                                    # Diffusion returns full-frame composites; legacy swappers return face crops.
                                    visual_i = swapped_i

                    frame_pair_inputs.append(aligned_input.detach())
                    frame_pair_generated.append(visual_i.detach())

            w_pre_boundary = self._project_to_stylegan_w(projected_z)
            images = self.stylegan.synthesize(w_pre_boundary, noise_mode="const")
            img = images[0].clamp(-1, 1).add(1).div(2.0)
            det_input_embed: Optional[torch.Tensor] = None

            # Detect and align the StyleGAN output once; reuse for embedding
            stylegan_dets = self.detector.detect(img)
            stylegan_detected = bool(stylegan_dets)
            aligned_stylegan = None
            if stylegan_detected:
                stylegan_top = max(stylegan_dets, key=lambda d: d.score)
                aligned_stylegan = self.aligner.align(img, stylegan_top).to(device)

            if use_face_swapper:
                if self.face_swapper is None and not self._warned_face_swapper:
                    logger.warning("Face swapping requested but no swapper is configured; proceeding without swap.")
                    self._warned_face_swapper = True
                elif self.face_swapper is not None:
                    
                    # === SMART ROUTING ===
                    # If it's the diffusion swapper, give it the full frame so it can inverse-warp perfectly.
                    # Otherwise, give it the KFAAR crop for legacy swappers.
                    is_diffusion = type(self.face_swapper).__name__ == "DiffusionFaceSwapper"
                    target_to_swap = frames_t[center_idx] if is_diffusion else aligned_faces[center_idx]
                    
                    swapped = None
                    if target_to_swap.numel() > 0 and aligned_stylegan is not None:
                        swapped = self.face_swapper.swap(aligned_stylegan, target_to_swap)

                    if swapped is not None:
                        det_input_embed = aligned_stylegan if swap_for_visuals_only else swapped

                        if is_diffusion:
                            # DiffusionFaceSwapper natively returns the seamlessly blended full frame
                            swapped_images = swapped.unsqueeze(0)
                        else:
                            # Legacy naive paste-back for other swappers that return tight crops
                            swapped_full = frames_t[center_idx].clone()
                            if center_detection is not None:
                                bbox = center_detection.bbox.to(device)
                                x1, y1, x2, y2 = bbox.round().long()
                                h, w = swapped_full.shape[1], swapped_full.shape[2]

                                x1, x2 = x1.clamp(0, w), x2.clamp(0, w)
                                y1, y2 = y1.clamp(0, h), y2.clamp(0, h)

                                crop_h, crop_w = (y2 - y1).item(), (x2 - x1).item()

                                if crop_h > 0 and crop_w > 0:
                                    swapped_resized = torch.nn.functional.interpolate(
                                        swapped.unsqueeze(0),
                                        size=(crop_h, crop_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).squeeze(0)
                                    swapped_full[:, y1:y2, x1:x2] = swapped_resized

                            swapped_images = swapped_full.unsqueeze(0)
                    elif not self._warned_face_swapper:
                        logger.warning("Face swapper failed to produce output; proceeding without swap.")
                        self._warned_face_swapper = True

            if det_input_embed is None and stylegan_detected:
                det_input_embed = aligned_stylegan

            if det_input_embed is not None:
                v_embeddings[0] = self.embedder.embed([det_input_embed], with_grad=True)[0]
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
            detections=[], 
            aligned_faces=[aligned_faces[center_idx]],
            real_embeddings=real_emb[center_idx : center_idx + 1].detach(),
            projected_z=projected_z,
            virtual_embeddings=v_embeddings,
            valid_mask=valid_mask,
            gen_mask=gen_mask_t,
            input_face_frames=frame_pair_inputs,
            generated_face_frames=frame_pair_generated,
            w_pre_boundary=w_pre_boundary,
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
        swap_for_visuals_only: bool = True,
        return_frame_pairs: bool = False,
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
                swap_for_visuals_only=swap_for_visuals_only,
                return_frame_pairs=return_frame_pairs,
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
        swap_for_visuals_only: bool = True,
        lambda_w_reg: float = 20.0,
        return_components: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
            swap_for_visuals_only=swap_for_visuals_only,
            lambda_w_reg=lambda_w_reg,
            **kwargs,
        )
        
        if comps is None:
            self.stats["discarded_batches"] += 1
            zero = torch.tensor(0.0, device=self.device)
            if return_components:
                return zero, {
                    "ano": zero,
                    "syn": zero,
                    "div": zero,
                    "dif": zero,
                    "w_reg": zero,
                }
            return zero

        ano, syn, div, dif, w_reg, total = comps
        
        # Only backprop if total is linked to a grad_fn
        if total.requires_grad:
            total.backward()
            self.optimizer.step()
            if return_components:
                return total, {
                    "ano": ano.detach(),
                    "syn": syn.detach(),
                    "div": div.detach(),
                    "dif": dif.detach(),
                    "w_reg": w_reg.detach(),
                }
            return total
        
        zero = torch.tensor(0.0, device=self.device)
        if return_components:
            return zero, {
                "ano": ano.detach(),
                "syn": syn.detach(),
                "div": div.detach(),
                "dif": dif.detach(),
                "w_reg": w_reg.detach(),
            }
        return zero

    def hpvg_loss_components(
        self, frames, labels, seq_lens, key_1, key_2,
        margin=0.5, lambda_ano=0.4, lambda_syn=1.0, lambda_div=1.0, lambda_dif=1.0, lambda_temp=0.0,
        lambda_w_reg: float = 20.0,
        batch_index: Optional[int] = None,
        use_face_swapper: bool = False,
        swap_for_visuals_only: bool = True,
    ) -> Optional[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:

        device = self.device
        batch_size = frames.shape[0]
        if batch_size % 3 != 0:
            raise ValueError(
                f"Identity tuple batching expects sample count divisible by 3, got batch_size={batch_size}"
            )

        seq_lens_list = [int(x) for x in (seq_lens.tolist() if torch.is_tensor(seq_lens) else seq_lens)]

        sample_records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        proj_norm_terms = []
        w_reg_terms: list[torch.Tensor] = []
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
                swap_for_visuals_only=swap_for_visuals_only,
            )
            res2 = self.forward(
                frames[b, :seq_lens_list[b]],
                key_2,
                sample_label=label_int,
                key_tag="k2",
                batch_index=batch_index,
                use_face_swapper=use_face_swapper,
                swap_for_visuals_only=swap_for_visuals_only,
            )

            proj_norm_terms.append(res1.projected_z.pow(2).mean())
            proj_norm_terms.append(res2.projected_z.pow(2).mean())

            if lambda_w_reg > 0.0 and self.stylegan is not None and hasattr(self.stylegan, "mapping") and hasattr(self.stylegan.mapping, "w_avg"):
                w_avg = self.stylegan.mapping.w_avg.to(device=device, dtype=res1.w_pre_boundary.dtype) if res1.w_pre_boundary is not None else self.stylegan.mapping.w_avg.to(device=device)

                if res1.w_pre_boundary is not None:
                    w_avg_res1 = w_avg
                    while w_avg_res1.dim() < res1.w_pre_boundary.dim():
                        w_avg_res1 = w_avg_res1.unsqueeze(0)
                    w_reg_terms.append((res1.w_pre_boundary - w_avg_res1).pow(2).mean())

                if res2.w_pre_boundary is not None:
                    w_avg_res2 = w_avg
                    while w_avg_res2.dim() < res2.w_pre_boundary.dim():
                        w_avg_res2 = w_avg_res2.unsqueeze(0)
                    w_reg_terms.append((res2.w_pre_boundary - w_avg_res2).pow(2).mean())

            nondet_faces += int((~res1.gen_mask).sum().item() + (~res2.gen_mask).sum().item())

            mask = res1.valid_mask & res2.valid_mask

            # Track frames where we could not form a valid pair (input or gen missing)
            det_failures += int((~mask).sum().item())

            valid_scalar = bool(mask.any().item())
            if valid_scalar:
                # Collapse any frame-level detections to one representative feature
                # vector per source sample so triplet semantics remain stable.
                real_e = res1.real_embeddings[mask].mean(dim=0, keepdim=True)
                v1_e = res1.virtual_embeddings[mask].mean(dim=0, keepdim=True)
                v2_e = res2.virtual_embeddings[mask].mean(dim=0, keepdim=True)
            else:
                real_e = torch.zeros((1, 512), device=device)
                v1_e = torch.zeros((1, 512), device=device)
                v2_e = torch.zeros((1, 512), device=device)

            sample_records.append(
                (
                    torch.tensor(valid_scalar, device=device, dtype=torch.bool),
                    real_e,
                    v1_e,
                    v2_e,
                    labels[b] if torch.is_tensor(labels) else torch.tensor(label_int, device=device, dtype=torch.long),
                )
            )

        proj_norm = torch.stack(proj_norm_terms).mean() if proj_norm_terms else torch.tensor(0.0, device=device)
        w_reg = torch.stack(w_reg_terms).mean() if w_reg_terms else torch.tensor(0.0, device=device)

        tuple_ano_terms: list[torch.Tensor] = []
        tuple_syn_terms: list[torch.Tensor] = []
        tuple_div_terms: list[torch.Tensor] = []
        tuple_dif_terms: list[torch.Tensor] = []

        tuple_count = batch_size // 3
        for tuple_idx in range(tuple_count):
            i = 3 * tuple_idx
            x1_valid, x1_real, x1_v1, x1_v2, x1_label = sample_records[i]
            x2_valid, _x2_real, x2_v1, _x2_v2, x2_label = sample_records[i + 1]
            y_valid, _y_real, y_v1, _y_v2, y_label = sample_records[i + 2]

            # Guard semantic tuple layout: (x1, x2, y) = (same id, same id, different id)
            if bool((x1_label != x2_label).item()) or bool((x1_label == y_label).item()):
                raise ValueError(
                    "Invalid tuple identity pattern; expected x1/x2 same identity and y different identity"
                )

            tuple_valid = bool((x1_valid & x2_valid & y_valid).item())
            if not tuple_valid:
                continue

            # Losses are computed strictly within each tuple (no cross-tuple pairing).
            tuple_ano_terms.append(cosine_loss(x1_v1, x1_real, label=-1, margin=margin))
            tuple_div_terms.append(cosine_loss(x1_v1, x1_v2, label=-1, margin=margin))
            tuple_syn_terms.append(cosine_loss(x1_v1, x2_v1, label=1, margin=margin))
            tuple_dif_terms.append(cosine_loss(x1_v1, y_v1, label=-1, margin=margin))

        has_valid_tuple = len(tuple_ano_terms) > 0
        if has_valid_tuple:
            ano = torch.stack(tuple_ano_terms).mean()
            syn = torch.stack(tuple_syn_terms).mean()
            div = torch.stack(tuple_div_terms).mean()
            dif = torch.stack(tuple_dif_terms).mean()
            penalty_missing_pairs = proj_norm * (0.1 * float(det_failures))
        else:
            ano = syn = div = dif = torch.tensor(0.0, device=device)
            penalty_missing_pairs = proj_norm * (1.0 + 0.5 * float(det_failures))

        penalty_nondet = proj_norm * (2.0 * float(nondet_faces))

        total = (
            lambda_ano * ano
            + lambda_syn * syn
            + lambda_div * div
            + lambda_dif * dif
            + (lambda_w_reg * w_reg)
            + penalty_nondet
            + penalty_missing_pairs
        )

        return ano, syn, div, dif, w_reg, total

    @staticmethod
    def _to_sequence_tensor(frames, device):
        t = frames if torch.is_tensor(frames) else torch.from_numpy(frames)
        return t.float().to(device)

    def _project_to_stylegan_w(self, projected_z: torch.Tensor) -> torch.Tensor:
        if self.stylegan is None:
            raise RuntimeError("StyleGAN is not initialized")
        if self.use_stylegan_mapper:
            return self.stylegan.map(projected_z, truncation_psi=self.truncation_psi)

        num_ws = getattr(self.stylegan.synthesis, "num_ws", None)
        if num_ws is None:
            raise RuntimeError("StyleGAN synthesis.num_ws is unavailable for mapper-disabled mode")

        # Fetch the mathematical center of W-space
        w_avg = self.stylegan.mapping.w_avg
        
        # Add the center coordinates to the MLP's raw output. 
        # This instantly teleports your vectors out of the zero-origin wilderness 
        # and directly onto the valid face manifold.
        anchored_w = projected_z + w_avg

        # Manually broadcast the anchored W vector into W+ space
        return anchored_w.unsqueeze(1).expand(-1, int(num_ws), -1)

    def _epoch_image_dir(self) -> Optional[Path]:
        if self._save_dir_images is None:
            return None
        if self._current_epoch <= 0:
            return self._save_dir_images
        return self._save_dir_images / f"epoch_{self._current_epoch:04d}"

    def _epoch_video_dir(self) -> Optional[Path]:
        if self._save_dir_videos is None:
            return None
        if self._current_epoch <= 0:
            return self._save_dir_videos
        return self._save_dir_videos / f"epoch_{self._current_epoch:04d}"

    @staticmethod
    def _clear_dir(path: Path) -> None:
        """Remove all contents of a directory without deleting the directory itself."""
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

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
        self._clear_dir(self._save_dir)
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
        self._saving_active = self._save_enabled
        self._video_accumulators.clear()
        images_dir = self._epoch_image_dir()
        videos_dir = self._epoch_video_dir()
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
        if videos_dir is not None and self._save_videos:
            videos_dir.mkdir(parents=True, exist_ok=True)

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
        save_images_dir = self._epoch_image_dir()
        save_videos_dir = self._epoch_video_dir() if self._save_videos else None
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
        save_videos_dir = self._epoch_video_dir()
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