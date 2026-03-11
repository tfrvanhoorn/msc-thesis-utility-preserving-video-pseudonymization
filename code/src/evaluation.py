from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[0]
PROJECT_ROOT = current_file.parents[1]
EXTERNAL_LIB_ROOT = PROJECT_ROOT / "external_libraries"

# Silence PyTorch bilinear align_corners warning
warnings.filterwarnings(
    "ignore",
    message="Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0",
    category=UserWarning,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXTERNAL_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_LIB_ROOT))

from config import (  # noqa: E402
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    EyeglassesBoundaryConfig,
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from metrics import MetricsAccumulator  # noqa: E402
from geometric_metrics import GeometricUtilityEvaluator  # noqa: E402
from pipeline.factory import build_kfaar_pipeline  # noqa: E402
from components import (
    load_stylegan2,
    load_projector_state_dict,
    SimSwapFaceSwapper,
    DiffusionFaceSwapper,
)  # noqa: E402
from data.splits import build_dataloader_for_identities, list_identities  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def _log_pipe(category: str, **fields: Any) -> None:
    parts = [category]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logging.info(" | ".join(parts))


def _to_uint8_rgb_image(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 3:
        raise ValueError(f"Expected CHW tensor image, got shape {tuple(image.shape)}")
    chw = image.detach().to("cpu")
    if chw.dtype != torch.float32:
        chw = chw.float()
    chw = chw.clamp(0.0, 1.0)
    hwc = chw.permute(1, 2, 0).numpy()
    return (hwc * 255.0).round().astype(np.uint8)


def _apply_input_brightness(frames: torch.Tensor, input_brightness_ev: float) -> torch.Tensor:
    if input_brightness_ev == 0.0:
        return frames
    scale = float(2.0 ** float(input_brightness_ev))
    return (frames * scale).clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KFAAR projector")

    parser.add_argument("--checkpoint", type=Path, required=False, default=None, help="Path to a trained projector checkpoint (.pt)")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "celeba", help="Path to the dataset root")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="celeba",
        choices=["celeba", "image_folder", "voxceleb_video", "video_folder"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--stylegan_ckpt",
        type=Path,
        default=SRC_ROOT / "models" / "stylegan2-celebahq-256x256.pkl",
        help="Path to StyleGAN2 .pkl checkpoint",
    )
    parser.add_argument("--truncation_psi", type=float, default=0.5, help="Truncation psi for StyleGAN2 mapping")
    parser.add_argument("--remove_eyeglasses", action="store_true", help="Push StyleGAN away from generating eyeglasses in W-space")
    parser.add_argument("--eyeglasses_boundary_path", type=Path, default=None, help="Path to InterfaceGAN eyeglasses boundary (.npy) in W-space")
    parser.add_argument("--eyeglasses_removal_scale", type=float, default=1.0, help="Scale factor used in W-space eyeglasses removal: w = w - scale * boundary")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SRC_ROOT / "eval_results",
        help="Directory to save evaluation reports",
    )

    # Face swapping selector (visual-only by default)
    parser.add_argument(
        "--face_swapper",
        type=str,
        default="none",
        choices=["none", "simswap", "diffusion"],
        help="Choose face swapper backend for visualization (none=disabled)",
    )
    parser.add_argument("--use_face_swapper", action="store_true", help="Legacy flag to enable face swapping (overridden by face_swapper != none)")
    parser.add_argument(
        "--swap_for_visuals_only",
        action="store_true",
        help="Use swapped faces only for visualization; compute metrics on StyleGAN outputs",
    )
    parser.add_argument(
        "--swap_for_loss",
        dest="swap_for_visuals_only",
        action="store_false",
        help="Use swapped faces for embeddings/metrics (old behavior)",
    )
    parser.set_defaults(swap_for_visuals_only=True)

    # SimSwap options
    parser.add_argument(
        "--simswap_root",
        type=Path,
        default=PROJECT_ROOT / "external_libraries" / "SimSwap",
        help="Path to SimSwap repository root",
    )
    parser.add_argument(
        "--simswap_checkpoints_dir",
        type=Path,
        default=None,
        help="Path to SimSwap checkpoints directory (defaults to simswap_root/checkpoints)",
    )
    parser.add_argument(
        "--simswap_name",
        type=str,
        default="people",
        help="SimSwap experiment name (subfolder in checkpoints_dir)",
    )
    parser.add_argument(
        "--simswap_epoch",
        type=str,
        default="latest",
        help="Generator checkpoint epoch tag (e.g., latest, 0015)",
    )
    parser.add_argument(
        "--simswap_arcface_ckpt",
        type=Path,
        default=None,
        help="Path to ArcFace checkpoint used by SimSwap (defaults to simswap_root/arcface_model/arcface_checkpoint.tar)",
    )
    parser.add_argument(
        "--simswap_parsing_ckpt",
        type=Path,
        default=None,
        help="Path to face parsing checkpoint for SimSwap masking (optional)",
    )
    parser.add_argument(
        "--simswap_crop_size",
        type=int,
        default=224,
        choices=[224, 512],
        help="Input/output resolution for SimSwap",
    )
    parser.add_argument(
        "--simswap_detector_name",
        type=str,
        default="antelopev2",
        help="Face detector name for SimSwap (e.g., antelopev2)",
    )
    parser.add_argument(
        "--simswap_detector_root",
        type=Path,
        default=None,
        help="Path to SimSwap face detector models root (defaults to simswap_root/insightface_func/models)",
    )

    # Diffusion swapper options
    parser.add_argument("--faceadapter_root", type=Path, default=PROJECT_ROOT / "external_libraries" / "Face-Adapter", help="Path to Face-Adapter repository root")
    parser.add_argument("--faceadapter_checkpoint_dir", type=Path, default=None, help="Path to FaceAdapter checkpoints (defaults to faceadapter_root/checkpoints)")
    parser.add_argument("--faceadapter_base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model for FaceAdapter")
    parser.add_argument("--faceadapter_cache_dir", type=Path, default=None, help="Cache directory for HF model files (optional)")
    parser.add_argument("--faceadapter_use_cache", action="store_true", help="Use local-only cached HF model files for FaceAdapter")
    parser.add_argument("--faceadapter_inference_steps", type=int, default=25, help="FaceAdapter diffusion inference steps")
    parser.add_argument("--faceadapter_guidance_scale", type=float, default=5.0, help="FaceAdapter guidance scale")
    parser.add_argument("--faceadapter_crop_ratio", type=float, default=0.81, help="Face crop ratio used by FaceAdapter")
    parser.add_argument("--faceadapter_seed", type=int, default=0, help="Fixed random seed for deterministic FaceAdapter inference")

    # Hyperparameters (Projector)
    parser.add_argument("--key_dim", type=int, default=128, help="Dimension of the pseudonymization key")
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "lstm"], help="Projector architecture")
    parser.add_argument("--lstm_hidden_dim", type=int, default=512, help="Hidden size for LSTM projector")
    parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of layers for LSTM projector")
    parser.add_argument("--lstm_bidirectional", action="store_true", default=True, help="Use bidirectional LSTM")
    parser.add_argument("--no_lstm_bidirectional", dest="lstm_bidirectional", action="store_false", help="Disable bidirectional LSTM")
    parser.add_argument("--lstm_dropout", type=float, default=0.0, help="Dropout for LSTM projector (applied when num_layers>1)")

    # Evaluation thresholds
    parser.add_argument("--ano_threshold", type=float, default=0.7, help="Cosine similarity threshold for anonymization success")
    parser.add_argument("--syn_threshold", type=float, default=0.7, help="Cosine similarity threshold for synchronism success")
    parser.add_argument(
        "--div_threshold",
        dest="div_threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for diversity success (same identity, different keys)",
    )
    parser.add_argument(
        "--diff_threshold",
        dest="diff_threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for differentiation success (different identities, same key)",
    )
    parser.add_argument(
        "--compute_auc_eer",
        action="store_true",
        help="Compute AUC/EER for anonymization, synchronism(all), diversity, and differentiation",
    )

    # Dataset & Split
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Max video files sampled per identity (video datasets)")
    parser.add_argument("--max_videos_per_youtube_id", type=int, default=None, help="Max video files sampled per YouTube ID (voxceleb_video)")
    parser.add_argument("--min_youtube_id_per_identity", type=int, default=None, help="Require at least this many YouTube IDs per identity (voxceleb_video)")
    parser.add_argument("--window_size", type=int, default=16, help="Window size (frames) for video datasets")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride between frames inside a window")
    parser.add_argument("--window_step", type=int, default=None, help="Step between window starts (defaults to window_size*frame_stride)")
    parser.add_argument("--max_windows_per_video", type=int, default=None, help="Max windows sampled per source video (video datasets)")
    parser.add_argument("--max_samples_per_identity", type=int, default=None, help="Cap samples per identity (images) or videos per identity (video datasets)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting and key sampling")

    # Identity batching
    parser.add_argument("--batch_identities", type=int, default=4, help="Number of unique identities per batch")
    parser.add_argument("--batch_videos_per_identity", type=int, default=2, help="Videos per identity per batch (voxceleb: all windows from each video) or samples for image datasets")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--input_brightness_ev",
        type=float,
        default=0.0,
        help="Exposure adjustment in EV stops applied to input frames before processing (scale = 2**EV; negative darkens, positive brightens)",
    )
    parser.add_argument(
        "--skip_pipeline_use_input_as_output",
        action="store_true",
        help="Skip projector/generator/swappers and use detected input faces as outputs for baseline utility metrics",
    )

    # Generated face saving (matches training flags)
    parser.add_argument("--save_generated_faces", action="store_true", help="Store generated faces to disk during evaluation")
    parser.add_argument(
        "--save_generated_mode",
        type=str,
        default="detected",
        choices=["detected", "undetected", "all"],
        help="Which generated frames to store",
    )
    parser.add_argument(
        "--save_generated_dir",
        type=Path,
        default=None,
        help="Directory to store generated face images (defaults to output_dir/generated_faces)",
    )
    parser.add_argument(
        "--save_generated_max_per_epoch",
        type=int,
        default=100,
        help="Maximum number of generated samples to store per evaluation run (set <=0 for no limit)",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Also save each window as an input/gen video alongside frame images",
    )

    return parser.parse_args()


def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str] | None, list[str] | None]:
    frames: Any
    labels: torch.Tensor | None = None
    seq_lens: Any = None
    contexts: list[str] | None = None
    sources: list[str] | None = None

    if isinstance(batch, dict):
        frames = batch.get("frames")
        labels = batch.get("label")
        if labels is None:
            labels = batch.get("labels")
        seq_lens = batch.get("seq_lens")
        ctx_val = batch.get("context")
        if ctx_val is not None:
            if isinstance(ctx_val, (list, tuple)):
                contexts = [str(c) for c in ctx_val]
            else:
                contexts = [str(ctx_val)]
        src_val = batch.get("source")
        if src_val is not None:
            if isinstance(src_val, (list, tuple)):
                sources = [str(s) for s in src_val]
            else:
                sources = [str(src_val)]
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        frames, labels = batch[0], batch[1]
        if len(batch) >= 3:
            seq_lens = batch[2]
    else:
        raise TypeError("Unsupported batch format for evaluation")

    if labels is None:
        raise ValueError("Batch is missing labels for evaluation")
    if frames is None:
        raise ValueError("Batch is missing frames/images for evaluation")

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    frame_tensor = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    if frame_tensor.dim() == 4:
        frame_tensor = frame_tensor.unsqueeze(1)
    if frame_tensor.dim() != 5:
        raise ValueError(f"Expected frames with 5 dimensions (B,Seq,C,H,W), got {tuple(frame_tensor.shape)}")

    seq_len_tensor = torch.as_tensor(seq_lens, dtype=torch.long) if seq_lens is not None else torch.full(
        (frame_tensor.shape[0],), frame_tensor.shape[1], dtype=torch.long
    )

    return frame_tensor, labels_tensor, seq_len_tensor, contexts, sources


def _collect_detected_faces(
    pipeline: Any,
    sample_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    device = sample_frames.device
    aligned_faces: list[torch.Tensor] = []
    input_mask_list: list[bool] = []

    for frame in sample_frames:
        detections = pipeline.detector.detect(frame)
        if detections:
            top = max(detections, key=lambda d: d.score)
            aligned = pipeline.aligner.align(frame, top).to(device)
            aligned_faces.append(aligned)
            input_mask_list.append(True)
        else:
            aligned_faces.append(torch.empty(0, device=device))
            input_mask_list.append(False)

    input_mask = torch.tensor(input_mask_list, dtype=torch.bool, device=device)
    if input_mask.any():
        valid_idx = [i for i, is_valid in enumerate(input_mask_list) if is_valid]
        valid_faces = [aligned_faces[i] for i in valid_idx]
        embeds = pipeline.embedder.embed(valid_faces, with_grad=False)
        embed_dim = int(embeds.shape[1])
        full_embeds = torch.zeros((len(aligned_faces), embed_dim), device=device, dtype=embeds.dtype)
        for emb, idx in zip(embeds, valid_idx):
            full_embeds[idx] = emb
    else:
        probe = sample_frames[0] if sample_frames.shape[0] > 0 else torch.zeros((3, 256, 256), device=device)
        probe_embed = pipeline.embedder.embed([probe], with_grad=False)
        embed_dim = int(probe_embed.shape[1])
        full_embeds = torch.zeros((len(aligned_faces), embed_dim), device=device, dtype=probe_embed.dtype)
    return full_embeds, input_mask, aligned_faces


def main() -> None:
    args = parse_args()
    if not args.skip_pipeline_use_input_as_output and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --skip_pipeline_use_input_as_output is enabled")

    configure_logging()
    device = torch.device(args.device)

    _log_pipe(
        "evaluation_start",
        checkpoint=str(args.checkpoint),
        dataset_type=args.dataset_type,
        passthrough_baseline=bool(args.skip_pipeline_use_input_as_output),
        input_brightness_ev=float(args.input_brightness_ev),
        compute_auc_eer=bool(args.compute_auc_eer),
        anonymization_threshold=float(args.ano_threshold),
        synchronism_threshold=float(args.syn_threshold),
        diversity_threshold=float(args.div_threshold),
        differentiation_threshold=float(args.diff_threshold),
    )

    face_swapper = None
    swapper_choice = (args.face_swapper or "none").lower()
    use_swapper_requested = (args.use_face_swapper or swapper_choice != "none") and not args.skip_pipeline_use_input_as_output
    if use_swapper_requested:
        if swapper_choice == "simswap" or swapper_choice == "none":
            simswap_ckpt_dir = args.simswap_checkpoints_dir or args.simswap_root / "checkpoints"
            arcface_ckpt = args.simswap_arcface_ckpt or args.simswap_root / "arcface_model" / "arcface_checkpoint.tar"
            face_swapper = SimSwapFaceSwapper(
                simswap_root=args.simswap_root,
                checkpoints_dir=simswap_ckpt_dir,
                name=args.simswap_name,
                which_epoch=args.simswap_epoch,
                arcface_ckpt=arcface_ckpt,
                parsing_ckpt=args.simswap_parsing_ckpt,
                detector_name=args.simswap_detector_name,
                detector_root=args.simswap_detector_root,
                crop_size=args.simswap_crop_size,
                device=device,
            )
        elif swapper_choice == "diffusion":
            faceadapter_ckpt_dir = args.faceadapter_checkpoint_dir or args.faceadapter_root / "checkpoints"
            face_swapper = DiffusionFaceSwapper(
                faceadapter_root=args.faceadapter_root,
                checkpoint_dir=faceadapter_ckpt_dir,
                base_model_id=args.faceadapter_base_model,
                cache_dir=args.faceadapter_cache_dir,
                use_cache=args.faceadapter_use_cache,
                inference_steps=args.faceadapter_inference_steps,
                guidance_scale=args.faceadapter_guidance_scale,
                crop_ratio=args.faceadapter_crop_ratio,
                seed=args.faceadapter_seed,
                device=device,
            )

    data_options: dict[str, object] = {
        "max_videos_per_identity": args.max_videos_per_identity,
        "max_videos_per_youtube_id": args.max_videos_per_youtube_id,
        "min_youtube_id_per_identity": args.min_youtube_id_per_identity,
    }
    if args.max_samples_per_identity is not None:
        data_options["max_samples_per_identity"] = args.max_samples_per_identity
        if args.dataset_type in {"voxceleb_video", "video_folder"}:
            data_options["max_videos_per_identity"] = args.max_samples_per_identity
    if args.dataset_type in {"voxceleb_video", "video_folder"}:
        data_options.update(
            {
                "window_size": args.window_size,
                "frame_stride": args.frame_stride,
                "window_step": args.window_step,
                "max_windows_per_video": args.max_windows_per_video,
            }
        )

    data_cfg = DataConfig(
        dataset_path=args.data_path,
        dataset_type=args.dataset_type,
        options=data_options,
    )
    detector_cfg = DetectorConfig(image_size=256, device=str(device))
    embedding_cfg = EmbeddingConfig(method="facenet", pretrained="vggface2", device=str(device))
    projector_cfg = ProjectorConfig(
        type=args.projector_type,
        key_dim=args.key_dim,
        hidden_dims=(1024, 512),
        dropout=args.lstm_dropout if args.projector_type == "lstm" else 0.0,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_bidirectional=args.lstm_bidirectional,
    )

    cfg = PipelineConfig(
        data=data_cfg,
        detector=detector_cfg,
        embedding=embedding_cfg,
        seed=SeedConfig(secret_key="master_thesis_secret"),
        projector=projector_cfg,
        eyeglasses_boundary=EyeglassesBoundaryConfig(
            enabled=args.remove_eyeglasses,
            boundary_path=args.eyeglasses_boundary_path,
            removal_scale=args.eyeglasses_removal_scale,
        ),
    )

    logging.info("Building data loader for evaluation (all identities)...")
    all_identities = list_identities(cfg.data)
    if args.max_identities is not None:
        all_identities = all_identities[: args.max_identities]

    test_loader = build_dataloader_for_identities(
        cfg.data,
        all_identities,
        batch_size=args.batch_identities * args.batch_videos_per_identity,
        identity_batching=True,
        batch_identities=args.batch_identities,
        samples_per_identity=args.batch_videos_per_identity,
        shuffle=False,
        num_workers=args.num_workers,
        group_by_video=cfg.data.dataset_type.lower() in {"voxceleb_video", "video_folder"},
    )

    stylegan = None
    if not args.skip_pipeline_use_input_as_output:
        logging.info("Loading StyleGAN2 from %s...", args.stylegan_ckpt)
        stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(
        cfg,
        stylegan=stylegan,
        device=device,
        truncation_psi=args.truncation_psi,
        face_swapper=face_swapper,
    )
    use_swapper = face_swapper is not None

    if args.save_generated_faces and hasattr(pipeline, "configure_saving"):
        save_dir = args.save_generated_dir if args.save_generated_dir is not None else args.output_dir / "generated_faces"
        save_max = None if args.save_generated_max_per_epoch is not None and args.save_generated_max_per_epoch <= 0 else args.save_generated_max_per_epoch
        pipeline.configure_saving(
            save_dir,
            mode=args.save_generated_mode,
            max_per_epoch=save_max,
            save_videos=args.save_videos,
        )

    if not args.skip_pipeline_use_input_as_output:
        logging.info("Loading checkpoint %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        load_projector_state_dict(pipeline.projector, ckpt["model_state_dict"])
    pipeline.projector.eval()
    if hasattr(pipeline.embedder, "eval"):
        pipeline.embedder.eval()

    metrics = MetricsAccumulator(
        anonymization_threshold=args.ano_threshold,
        synchronism_threshold=args.syn_threshold,
        diversity_threshold=args.div_threshold,
        differentiation_threshold=args.diff_threshold,
        compute_auc_eer=args.compute_auc_eer,
        anonymization_enabled=not args.skip_pipeline_use_input_as_output,
        diversity_enabled=not args.skip_pipeline_use_input_as_output,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    geometric_evaluator = GeometricUtilityEvaluator()

    total_samples = 0
    batch_processing_start_time: float | None = None
    batch_processing_end_time: float | None = None
    try:
        with torch.no_grad():
            for batch in test_loader:
                if batch_processing_start_time is None:
                    batch_processing_start_time = time.perf_counter()

                batch_diff_embeddings: list[torch.Tensor] = []
                batch_diff_labels: list[int] = []
                # Use two random keys shared across the batch.
                # key1 branch is used for differentiation (different identities, same key).
                # key1/key2 pair is used for diversity (same identity, different keys).
                batch_key_1 = torch.randn(args.key_dim, generator=rng, device=device)
                batch_key_2 = torch.randn(args.key_dim, generator=rng, device=device)

                frames, labels, seq_lens, contexts, sources = _extract_batch(batch)
                frames = frames.to(device)
                labels = labels.to(device)
                seq_lens = seq_lens.to(device)

                batch_size = frames.shape[0]
                for idx in range(batch_size):
                    seq_len = int(seq_lens[idx].item())
                    sample_frames = frames[idx, :seq_len]
                    sample_frames = _apply_input_brightness(sample_frames, args.input_brightness_ev)
                    label = int(labels[idx].item())

                    sample_context = None
                    if contexts is not None and idx < len(contexts):
                        sample_context = contexts[idx]

                    source_id = None
                    if sources is not None and idx < len(sources):
                        source_id = sources[idx]

                    if args.skip_pipeline_use_input_as_output:
                        real_full, input_mask, aligned_faces = _collect_detected_faces(pipeline, sample_frames)
                        center_idx = int(sample_frames.shape[0] // 2)
                        center_real = real_full[center_idx : center_idx + 1]
                        center_valid = torch.tensor([bool(input_mask[center_idx].item())], device=device, dtype=torch.bool)
                        input_face_frames = [face for face in aligned_faces if face.numel() > 0]
                        generated_face_frames = [face.clone() for face in input_face_frames]
                        res1_real_embeddings = center_real.detach()
                        res1_virtual_embeddings = center_real.detach()
                        res1_valid_mask = center_valid
                        res1_gen_mask = center_valid
                        res1_input_face_frames = input_face_frames
                        res1_generated_face_frames = generated_face_frames
                        res2_virtual_embeddings = center_real.detach()
                        res2_valid_mask = center_valid
                        res2_generated_face_frames = generated_face_frames
                    else:
                        forward_fn = pipeline.forward_eval if use_swapper else pipeline.forward
                        res1 = forward_fn(
                            sample_frames,
                            batch_key_1,
                            sample_label=label,
                            sample_context=sample_context,
                            use_face_swapper=use_swapper,
                            swap_for_visuals_only=args.swap_for_visuals_only,
                            return_frame_pairs=True,
                        )
                        res2 = forward_fn(
                            sample_frames,
                            batch_key_2,
                            sample_label=label,
                            sample_context=sample_context,
                            use_face_swapper=use_swapper,
                            swap_for_visuals_only=args.swap_for_visuals_only,
                            return_frame_pairs=True,
                        )
                        res1_real_embeddings = res1.real_embeddings
                        res1_virtual_embeddings = res1.virtual_embeddings
                        res1_valid_mask = res1.valid_mask
                        res1_gen_mask = res1.gen_mask
                        res1_input_face_frames = list(res1.input_face_frames)
                        res1_generated_face_frames = list(res1.generated_face_frames)
                        res2_virtual_embeddings = res2.virtual_embeddings
                        res2_valid_mask = res2.valid_mask
                        res2_generated_face_frames = list(res2.generated_face_frames)

                    pair_count = min(len(res1_input_face_frames), len(res1_generated_face_frames), len(res2_generated_face_frames))
                    for frame_idx in range(pair_count):
                        input_face = res1_input_face_frames[frame_idx]
                        gen_face_key1 = res1_generated_face_frames[frame_idx]
                        gen_face_key2 = res2_generated_face_frames[frame_idx]

                        if input_face.numel() == 0 or gen_face_key1.numel() == 0:
                            metrics.update_geometric_utility(None, None)
                        else:
                            errs_key1 = geometric_evaluator.compute_pair_errors(
                                _to_uint8_rgb_image(input_face),
                                _to_uint8_rgb_image(gen_face_key1),
                            )
                            metrics.update_geometric_utility(errs_key1.head_posture_error, errs_key1.facial_expression_error)

                        if input_face.numel() == 0 or gen_face_key2.numel() == 0:
                            metrics.update_geometric_utility(None, None)
                        else:
                            errs_key2 = geometric_evaluator.compute_pair_errors(
                                _to_uint8_rgb_image(input_face),
                                _to_uint8_rgb_image(gen_face_key2),
                            )
                            metrics.update_geometric_utility(errs_key2.head_posture_error, errs_key2.facial_expression_error)

                    # Keep existing metrics based on a single branch for comparability.
                    metrics.update_detection(res1_gen_mask)
                    if not args.skip_pipeline_use_input_as_output:
                        metrics.update_anonymization(res1_real_embeddings, res1_virtual_embeddings, res1_valid_mask)

                        pair_mask = res1_valid_mask & res2_valid_mask
                        if pair_mask.any():
                            pair_v1 = res1_virtual_embeddings[pair_mask]
                            pair_v2 = res2_virtual_embeddings[pair_mask]
                            metrics.update_diversity(pair_v1, pair_v2)

                    valid_virtual = res1_virtual_embeddings[res1_valid_mask]
                    if valid_virtual.numel() > 0:
                        metrics.add_synchronism_embeddings(label, valid_virtual, source_id=source_id)

                        # Collect valid virtual embeddings for differentiation scoring across identities in the batch
                        batch_diff_embeddings.append(valid_virtual.detach())
                        batch_diff_labels.extend([label] * valid_virtual.shape[0])

                    total_samples += 1

                if batch_diff_embeddings:
                    diff_embeds = torch.cat(batch_diff_embeddings, dim=0)
                    diff_labels = torch.as_tensor(batch_diff_labels, device=diff_embeds.device, dtype=torch.long)
                    metrics.update_differentiation(diff_embeds, diff_labels)

                batch_processing_end_time = time.perf_counter()
    finally:
        geometric_evaluator.close()

    if hasattr(pipeline, "finalize_saving"):
        pipeline.finalize_saving()

    batch_processing_seconds = 0.0
    if batch_processing_start_time is not None and batch_processing_end_time is not None:
        batch_processing_seconds = max(0.0, batch_processing_end_time - batch_processing_start_time)

    summary = metrics.finalize()
    _log_pipe(
        "evaluation_summary",
        checkpoint=str(args.checkpoint),
        dataset_type=args.dataset_type,
        passthrough_baseline=bool(args.skip_pipeline_use_input_as_output),
        total_samples=total_samples,
        batch_processing_seconds=batch_processing_seconds,
        detection_rate=summary["detection_rate"],
    )
    _log_pipe(
        "metric_anonymization",
        success_rate=summary["anonymization"]["success_rate"],
        threshold=summary["anonymization"]["threshold"],
        auc=summary["anonymization"]["auc"],
        eer=summary["anonymization"]["eer"],
        eer_threshold=summary["anonymization"]["eer_threshold"],
        success=summary["anonymization"]["counts"]["success"],
        total=summary["anonymization"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_total",
        success_rate=summary["synchronism_total"]["success_rate"],
        threshold=summary["synchronism_total"]["threshold"],
        auc=summary["synchronism_total"]["auc"],
        eer=summary["synchronism_total"]["eer"],
        eer_threshold=summary["synchronism_total"]["eer_threshold"],
        success=summary["synchronism_total"]["counts"]["success"],
        total=summary["synchronism_total"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_within",
        success_rate=summary["synchronism_within"]["success_rate"],
        threshold=summary["synchronism_within"]["threshold"],
        auc=summary["synchronism_within"]["auc"],
        eer=summary["synchronism_within"]["eer"],
        eer_threshold=summary["synchronism_within"]["eer_threshold"],
        success=summary["synchronism_within"]["counts"]["success"],
        total=summary["synchronism_within"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_cross",
        success_rate=summary["synchronism_cross"]["success_rate"],
        threshold=summary["synchronism_cross"]["threshold"],
        auc=summary["synchronism_cross"]["auc"],
        eer=summary["synchronism_cross"]["eer"],
        eer_threshold=summary["synchronism_cross"]["eer_threshold"],
        success=summary["synchronism_cross"]["counts"]["success"],
        total=summary["synchronism_cross"]["counts"]["total"],
    )
    _log_pipe(
        "metric_diversity",
        success_rate=summary["diversity"]["success_rate"],
        threshold=summary["diversity"]["threshold"],
        auc=summary["diversity"]["auc"],
        eer=summary["diversity"]["eer"],
        eer_threshold=summary["diversity"]["eer_threshold"],
        success=summary["diversity"]["counts"]["success"],
        total=summary["diversity"]["counts"]["total"],
    )
    _log_pipe(
        "metric_differentiation",
        success_rate=summary["differentiation"]["success_rate"],
        threshold=summary["differentiation"]["threshold"],
        auc=summary["differentiation"]["auc"],
        eer=summary["differentiation"]["eer"],
        eer_threshold=summary["differentiation"]["eer_threshold"],
        success=summary["differentiation"]["counts"]["success"],
        total=summary["differentiation"]["counts"]["total"],
    )
    _log_pipe(
        "metric_geometric_utility",
        head_posture_error=summary["geometric_utility"]["head_posture_error"],
        facial_expression_error=summary["geometric_utility"]["facial_expression_error"],
        valid_pairs=summary["geometric_utility"]["counts"]["valid_pairs"],
        invalid_pairs=summary["geometric_utility"]["counts"]["invalid_pairs"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = args.checkpoint.stem if args.checkpoint is not None else "passthrough_baseline"
    report_path = args.output_dir / f"{report_stem}_{args.dataset_type}_eval.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "dataset_type": args.dataset_type,
                "seed": args.seed,
                "metrics": summary,
                "total_samples": total_samples,
                "timing": {
                    "batch_processing_seconds": batch_processing_seconds,
                },
                "settings": serialized_args,
                "identities": {
                    "all": all_identities,
                },
            },
            f,
            indent=2,
        )
    _log_pipe(
        "evaluation_report_saved",
        path=str(report_path),
        checkpoint=str(args.checkpoint),
        dataset_type=args.dataset_type,
        batch_processing_seconds=batch_processing_seconds,
    )


if __name__ == "__main__":
    main()
