from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

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
from pipeline.factory import build_kfaar_pipeline  # noqa: E402
from components import (  # noqa: E402
    load_stylegan2,
    load_projector_state_dict,
    SimSwapFaceSwapper,
    DiffusionFaceSwapper,
)
from data.splits import build_dataloader_for_identities, list_identities  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KFAAR inference on image/video folders")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained projector checkpoint (.pt)")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the dataset root")
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["image_folder", "video_folder"],
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
        default=SRC_ROOT / "infer_results",
        help="Directory to save inference outputs/reports",
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
        help="Use swapped faces only for visualization; compute embeddings on StyleGAN outputs",
    )
    parser.add_argument(
        "--swap_for_loss",
        dest="swap_for_visuals_only",
        action="store_false",
        help="Use swapped faces for embeddings (old behavior)",
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

    # Dataset options
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Max video files sampled per identity (video_folder)")
    parser.add_argument("--window_size", type=int, default=1, help="Window size (frames) for video datasets")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride between frames inside a window")
    parser.add_argument("--window_step", type=int, default=None, help="Step between window starts (defaults to window_size*frame_stride)")
    parser.add_argument("--max_windows_per_video", type=int, default=None, help="Max windows sampled per source video (video datasets)")
    parser.add_argument("--max_samples_per_identity", type=int, default=None, help="Cap samples per identity (images) or videos per identity (video datasets)")
    parser.add_argument("--max_files", type=int, default=None, help="Global maximum number of input files to infer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for folder iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for key generation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    # Generated face saving
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
        help="Directory to store generated face outputs (defaults to output_dir/generated_faces)",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Also save each window trajectory as GIF videos alongside frame images",
    )

    return parser.parse_args()


def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str] | None, list[str] | None, list[str] | None]:
    frames: Any
    labels: torch.Tensor | None = None
    seq_lens: Any = None
    identities: list[str] | None = None
    contexts: list[str] | None = None
    sources: list[str] | None = None

    if isinstance(batch, dict):
        frames = batch.get("frames")
        labels = batch.get("label")
        if labels is None:
            labels = batch.get("labels")
        seq_lens = batch.get("seq_lens")
        id_val = batch.get("identity")
        if id_val is not None:
            if isinstance(id_val, (list, tuple)):
                identities = [str(v) for v in id_val]
            else:
                identities = [str(id_val)]
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
        raise TypeError("Unsupported batch format for inference")

    if labels is None:
        raise ValueError("Batch is missing labels for inference")
    if frames is None:
        raise ValueError("Batch is missing frames/images for inference")

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    frame_tensor = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    if frame_tensor.dim() == 4:
        frame_tensor = frame_tensor.unsqueeze(1)
    if frame_tensor.dim() != 5:
        raise ValueError(f"Expected frames with 5 dimensions (B,Seq,C,H,W), got {tuple(frame_tensor.shape)}")

    seq_len_tensor = torch.as_tensor(seq_lens, dtype=torch.long) if seq_lens is not None else torch.full(
        (frame_tensor.shape[0],), frame_tensor.shape[1], dtype=torch.long
    )

    return frame_tensor, labels_tensor, seq_len_tensor, identities, contexts, sources


def _build_face_swapper(args: argparse.Namespace, device: torch.device) -> SimSwapFaceSwapper | DiffusionFaceSwapper | None:
    face_swapper = None
    swapper_choice = (args.face_swapper or "none").lower()
    use_swapper_requested = args.use_face_swapper or swapper_choice != "none"
    if not use_swapper_requested:
        return None

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

    return face_swapper


def main() -> None:
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)

    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max_files must be > 0 when provided")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    face_swapper = _build_face_swapper(args, device)

    data_options: dict[str, object] = {
        "max_videos_per_identity": args.max_videos_per_identity,
    }
    if args.max_samples_per_identity is not None:
        data_options["max_samples_per_identity"] = args.max_samples_per_identity
        if args.dataset_type == "video_folder":
            data_options["max_videos_per_identity"] = args.max_samples_per_identity
    if args.dataset_type == "video_folder":
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

    logging.info("Loading StyleGAN2 from %s...", args.stylegan_ckpt)
    stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(
        cfg,
        stylegan=stylegan,
        device=device,
        truncation_psi=args.truncation_psi,
        face_swapper=face_swapper,
    )

    logging.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_projector_state_dict(pipeline.projector, ckpt["model_state_dict"])
    pipeline.projector.eval()
    if hasattr(pipeline.embedder, "eval"):
        pipeline.embedder.eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = args.save_generated_dir if args.save_generated_dir is not None else output_dir / "generated_faces"
    save_videos = args.save_videos or args.dataset_type == "video_folder"
    pipeline.configure_saving(
        save_dir,
        mode=args.save_generated_mode,
        max_per_epoch=None,
        save_videos=save_videos,
    )

    all_identities = list_identities(cfg.data)
    if args.max_identities is not None:
        all_identities = all_identities[: args.max_identities]

    data_loader = build_dataloader_for_identities(
        cfg.data,
        all_identities,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        identity_batching=False,
        max_samples_per_identity=args.max_samples_per_identity,
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    fixed_key = torch.randn(args.key_dim, generator=rng, device=device)

    use_swapper = face_swapper is not None
    processed_samples = 0
    skipped_samples = 0
    selected_video_sources: set[str] = set()

    with torch.no_grad():
        stop = False
        for batch in data_loader:
            frames, labels, seq_lens, identities, contexts, sources = _extract_batch(batch)
            frames = frames.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            batch_size = frames.shape[0]
            for idx in range(batch_size):
                if args.dataset_type == "image_folder" and args.max_files is not None and processed_samples >= args.max_files:
                    stop = True
                    break

                source_id = None
                if sources is not None and idx < len(sources):
                    src_val = sources[idx].strip()
                    if src_val:
                        source_id = src_val

                if args.dataset_type == "video_folder":
                    if source_id is None:
                        source_id = f"unknown_{idx}"
                    if (
                        args.max_files is not None
                        and source_id not in selected_video_sources
                        and len(selected_video_sources) >= args.max_files
                    ):
                        skipped_samples += 1
                        continue
                    selected_video_sources.add(source_id)

                seq_len = int(seq_lens[idx].item())
                sample_frames = frames[idx, :seq_len]
                label = int(labels[idx].item())

                sample_context = None
                if contexts is not None and idx < len(contexts):
                    sample_context = contexts[idx]

                pipeline.forward_eval(
                    sample_frames,
                    fixed_key,
                    sample_label=label,
                    sample_context=sample_context,
                    use_face_swapper=use_swapper,
                    swap_for_visuals_only=args.swap_for_visuals_only,
                )
                processed_samples += 1

            if stop:
                break

    pipeline.finalize_saving()

    report_path = output_dir / f"{args.checkpoint.stem}_{args.dataset_type}_infer.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "dataset_type": args.dataset_type,
                "seed": args.seed,
                "processed_samples": processed_samples,
                "processed_files": len(selected_video_sources) if args.dataset_type == "video_folder" else processed_samples,
                "skipped_samples": skipped_samples,
                "identities": all_identities,
                "settings": serialized_args,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Inference complete | processed_samples=%d | processed_files=%d | skipped_samples=%d | report=%s",
        processed_samples,
        len(selected_video_sources) if args.dataset_type == "video_folder" else processed_samples,
        skipped_samples,
        report_path,
    )


if __name__ == "__main__":
    main()
