from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

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
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from pipeline.factory import build_kfaar_pipeline  # noqa: E402
from components import (  # noqa: E402
    load_stylegan2,
    load_projector_state_dict,
    SimSwapFaceSwapper,
    FaceAdapterFaceSwap,
    FaceAdapterFaceReenactment,
)
from data.prepared import compile_prepared_regex, parse_prepared_video_path  # noqa: E402
from data.video_io import get_video_fps, load_video_frames, write_mp4  # noqa: E402
from utils.keys import sample_binary_key_bank  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


@dataclass(frozen=True)
class SourceVideo:
    identity: str
    youtube_id: str | None
    source_id: str
    video_path: Path
    sample_index: int
    original_name: str


DEFAULT_VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.avi", "*.mov")


def _sanitize_segment(value: str) -> str:
    cleaned = value.strip().replace("\\", "_").replace("/", "_")
    return cleaned or "unknown"


def _iter_video_paths(root: Path, patterns: tuple[str, ...] = DEFAULT_VIDEO_PATTERNS) -> list[Path]:
    videos: list[Path] = []
    for pattern in patterns:
        videos.extend([p for p in root.rglob(pattern) if p.is_file()])
    videos = sorted(set(videos))
    return videos


def _collect_sources(args: argparse.Namespace) -> list[SourceVideo]:
    dataset_type = args.dataset_type.lower()
    max_files = args.max_files if args.max_files is not None else None
    prepared_regex = compile_prepared_regex(args.filename_regex)

    if dataset_type == "voxceleb_video":
        base = args.data_path / "dev" / "mp4"
        if not base.exists():
            raise FileNotFoundError(f"VoxCeleb path not found: {base}")

        identities = sorted([p.name for p in base.iterdir() if p.is_dir()])
        if args.max_identities is not None:
            identities = identities[: args.max_identities]

        sources: list[SourceVideo] = []
        for identity in identities:
            identity_dir = base / identity
            youtube_dirs = sorted([p for p in identity_dir.iterdir() if p.is_dir()])
            videos_seen_identity = 0
            sample_counter = 1
            for youtube_dir in youtube_dirs:
                candidates = _iter_video_paths(youtube_dir)
                if args.max_videos_per_youtube_id is not None:
                    candidates = candidates[: args.max_videos_per_youtube_id]
                for video_path in candidates:
                    videos_seen_identity += 1
                    if args.max_videos_per_identity is not None and videos_seen_identity > args.max_videos_per_identity:
                        break
                    rel_source = str(video_path.relative_to(base))
                    sources.append(
                        SourceVideo(
                            identity=identity,
                            youtube_id=youtube_dir.name,
                            source_id=rel_source,
                            video_path=video_path,
                            sample_index=sample_counter,
                            original_name=video_path.stem,
                        )
                    )
                    sample_counter += 1
                    if max_files is not None and len(sources) >= max_files:
                        return sources
                if args.max_videos_per_identity is not None and videos_seen_identity >= args.max_videos_per_identity:
                    break
        return sources

    if dataset_type == "video_folder":
        if not args.data_path.exists():
            raise FileNotFoundError(f"Video folder path not found: {args.data_path}")

        all_videos = _iter_video_paths(args.data_path)
        parsed_by_identity: dict[str, list[tuple[int, str, Path]]] = {}
        for video in all_videos:
            parsed = parse_prepared_video_path(video, prepared_regex)
            parsed_by_identity.setdefault(parsed.identity, []).append((parsed.sample_index, parsed.original_name, video))

        identities = sorted(parsed_by_identity.keys())
        if args.max_identities is not None:
            identities = identities[: args.max_identities]

        sources: list[SourceVideo] = []
        for identity in identities:
            candidates = sorted(parsed_by_identity.get(identity, []), key=lambda item: item[0])
            if args.max_videos_per_identity is not None:
                candidates = candidates[: args.max_videos_per_identity]
            for sample_index, original_name, video_path in candidates:
                sources.append(
                    SourceVideo(
                        identity=identity,
                        youtube_id=None,
                        source_id=str(video_path.relative_to(args.data_path).as_posix()),
                        video_path=video_path,
                        sample_index=sample_index,
                        original_name=original_name,
                    )
                )
                if max_files is not None and len(sources) >= max_files:
                    return sources
        return sources

    raise ValueError(f"Unsupported dataset_type for video inference: {args.dataset_type}")


def _build_export_leaf(export_root: Path, source: SourceVideo) -> Path:
    identity_seg = _sanitize_segment(source.identity)
    source_seg = _sanitize_segment(Path(source.source_id).stem)
    if source.youtube_id:
        video_id = _sanitize_segment(f"{source.identity}+{source.youtube_id}+{source_seg}")
    else:
        video_id = _sanitize_segment(f"{source.identity}+{source_seg}")
    return export_root / identity_seg / video_id


def _prepared_output_filename(source: SourceVideo) -> str:
    return f"{source.identity}_sample{source.sample_index}_{source.original_name}.mp4"


def _nested_output_subdir(source: SourceVideo) -> Path:
    rel_source = Path(source.source_id)
    parent = rel_source.parent
    if str(parent) in ("", "."):
        return Path()
    return Path(*[_sanitize_segment(part) for part in parent.parts if part not in ("", ".")])


def _normalize_frame(frame: torch.Tensor, image_size: int | None = 256) -> torch.Tensor:
    fallback_size = 256 if image_size is None else image_size
    if frame.numel() == 0:
        return torch.zeros((3, fallback_size, fallback_size), dtype=torch.float32)
    out = frame.detach().cpu().float()
    if out.dim() != 3:
        return torch.zeros((3, fallback_size, fallback_size), dtype=torch.float32)
    if out.shape[0] != 3 and out.shape[-1] == 3:
        out = out.permute(2, 0, 1)
    if out.shape[0] != 3:
        return torch.zeros((3, fallback_size, fallback_size), dtype=torch.float32)

    if out.min().item() < 0.0 or out.max().item() > 1.0:
        out = out.add(1.0).div(2.0)
    out = out.clamp(0.0, 1.0)

    if image_size is not None and (out.shape[1] != image_size or out.shape[2] != image_size):
        out = torch.nn.functional.interpolate(
            out.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return out


def _apply_input_brightness(frames: torch.Tensor, input_brightness_ev: float) -> torch.Tensor:
    if input_brightness_ev == 0.0:
        return frames
    scale = float(2.0 ** float(input_brightness_ev))
    return (frames * scale).clamp(0.0, 1.0)


def _build_output_path(
    output_root: Path,
    source: SourceVideo,
    prepared_filename: str,
    *,
    preserve_nested_folders: bool,
) -> Path:
    if not preserve_nested_folders:
        return output_root / prepared_filename
    nested_dir = _nested_output_subdir(source)
    return output_root / nested_dir / prepared_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KFAAR inference on prepared videos and export inferred MP4 videos")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained projector checkpoint (.pt)")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the dataset root")
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["video_folder", "voxceleb_video"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--stylegan_ckpt",
        type=Path,
        default=SRC_ROOT / "models" / "stylegan2-celebahq-256x256.pkl",
        help="Path to StyleGAN2 .pkl checkpoint",
    )
    parser.add_argument("--truncation_psi", type=float, default=0.5, help="Truncation psi for StyleGAN2 mapping")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SRC_ROOT / "infer_results",
        help="Directory to save inferred outputs and reports",
    )
    parser.add_argument(
        "--filename_regex",
        type=str,
        default=r"^(?P<identity>[^_]+)_sample(?P<sample>\d+)_(?P<original>.+)$",
        help="Regex used for prepared video names when dataset_type=video_folder",
    )

    # Face postprocessing selector (visual-only by default)
    parser.add_argument(
        "--face_postprocessor",
        type=str,
        default="none",
        choices=["none", "simswap", "faceadapter_swap", "faceadapter_reenactment"],
        help="Choose final face postprocessing backend for visualization (none=disabled)",
    )
    parser.add_argument("--use_face_postprocessor", action="store_true", help="Legacy flag to enable face postprocessing (overridden by face_postprocessor != none)")
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

    # FaceAdapter options
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
    parser.add_argument("--num_keys", type=int, default=2, help="Number of pseudonymization keys to render per source video")
    parser.add_argument("--enable_projector_l2_reg", dest="enable_projector_l2_reg", action="store_true", help="Enable input L2 normalization for both key and z in the projector MLP")
    parser.add_argument("--disable_projector_l2_reg", dest="enable_projector_l2_reg", action="store_false", help="Disable input L2 normalization for key and z in the projector MLP")
    parser.add_argument("--enable_projector_key_upscaler", dest="enable_projector_key_upscaler", action="store_true", help="Enable projector key upscaler to map key_dim to 512 before concatenation")
    parser.add_argument("--disable_projector_key_upscaler", dest="enable_projector_key_upscaler", action="store_false", help="Disable projector key upscaler and concatenate raw key with z")
    parser.add_argument("--use_stylegan_mapper", dest="use_stylegan_mapper", action="store_true", help="Use StyleGAN mapping network (z->W+) before synthesis")
    parser.add_argument("--disable_stylegan_mapper", dest="use_stylegan_mapper", action="store_false", help="Bypass StyleGAN mapping and repeat projected z across W+ layers before synthesis")
    parser.set_defaults(enable_projector_l2_reg=True, enable_projector_key_upscaler=True, use_stylegan_mapper=False)

    # Dataset and sampling options
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Max videos sampled per identity")
    parser.add_argument("--max_videos_per_youtube_id", type=int, default=None, help="Max videos sampled per YouTube ID (voxceleb_video)")
    parser.add_argument("--max_files", type=int, default=None, help="Global maximum number of source videos")
    parser.add_argument("--max_frames_per_video", type=int, default=64, help="Maximum sampled frames per source video")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of sampled frames to process at once during inference")
    parser.add_argument("--target_sample_fps", type=float, default=10.0, help="Target FPS for frame sampling before anonymization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for key generation")
    parser.add_argument("--detector_score_threshold", type=float, default=0.45, help="MTCNN score threshold for detection filtering")
    parser.add_argument("--detector_min_face_size", type=int, default=20, help="Minimum face size in pixels for MTCNN")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--input_brightness_ev",
        type=float,
        default=0.0,
        help="Exposure adjustment in EV stops applied to input frames before processing (scale = 2**EV; negative darkens, positive brightens)",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help="Optional output FPS override. Defaults to source_fps/effective_sample_step.",
    )
    parser.add_argument(
        "--preserve_nested_folders",
        action="store_true",
        help="Preserve nested input folder structure in output (inside key folders when num_keys >= 2)",
    )

    return parser.parse_args()


def _build_face_postprocessor(args: argparse.Namespace, device: torch.device) -> SimSwapFaceSwapper | FaceAdapterFaceSwap | FaceAdapterFaceReenactment | None:
    face_postprocessor = None
    postprocessor_choice = (args.face_postprocessor or "none").lower()
    use_postprocessor_requested = args.use_face_postprocessor or postprocessor_choice != "none"
    if not use_postprocessor_requested:
        return None

    if postprocessor_choice == "simswap" or postprocessor_choice == "none":
        simswap_ckpt_dir = args.simswap_checkpoints_dir or args.simswap_root / "checkpoints"
        arcface_ckpt = args.simswap_arcface_ckpt or args.simswap_root / "arcface_model" / "arcface_checkpoint.tar"
        face_postprocessor = SimSwapFaceSwapper(
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
    elif postprocessor_choice == "faceadapter_swap":
        faceadapter_ckpt_dir = args.faceadapter_checkpoint_dir or args.faceadapter_root / "checkpoints"

        face_postprocessor = FaceAdapterFaceSwap(
            faceadapter_root=args.faceadapter_root,
            checkpoint_dir=faceadapter_ckpt_dir,
            base_model_id=args.faceadapter_base_model,
            cache_dir=args.faceadapter_cache_dir,
            use_cache=args.faceadapter_use_cache,
            inference_steps=args.faceadapter_inference_steps,
            guidance_scale=args.faceadapter_guidance_scale,
            crop_ratio=args.faceadapter_crop_ratio,
            seed=args.faceadapter_seed,
            detector_score_threshold=args.detector_score_threshold,
            detector_min_face_size=args.detector_min_face_size,
            device=device,
        )
    elif postprocessor_choice == "faceadapter_reenactment":
        faceadapter_ckpt_dir = args.faceadapter_checkpoint_dir or args.faceadapter_root / "checkpoints"

        face_postprocessor = FaceAdapterFaceReenactment(
            faceadapter_root=args.faceadapter_root,
            checkpoint_dir=faceadapter_ckpt_dir,
            base_model_id=args.faceadapter_base_model,
            cache_dir=args.faceadapter_cache_dir,
            use_cache=args.faceadapter_use_cache,
            inference_steps=args.faceadapter_inference_steps,
            guidance_scale=args.faceadapter_guidance_scale,
            crop_ratio=args.faceadapter_crop_ratio,
            seed=args.faceadapter_seed,
            detector_score_threshold=args.detector_score_threshold,
            detector_min_face_size=args.detector_min_face_size,
            device=device,
        )
    return face_postprocessor


def main() -> None:
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)

    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max_files must be > 0 when provided")
    if args.max_frames_per_video <= 0:
        raise ValueError("--max_frames_per_video must be > 0")
    if args.target_sample_fps <= 0:
        raise ValueError("--target_sample_fps must be > 0")
    if args.num_keys < 1:
        raise ValueError("--num_keys must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    face_postprocessor = _build_face_postprocessor(args, device)

    data_cfg = DataConfig(
        dataset_path=args.data_path,
        dataset_type=args.dataset_type,
        options={},
    )
    detector_cfg = DetectorConfig(
        image_size=256,
        score_threshold=args.detector_score_threshold,
        min_face_size=args.detector_min_face_size,
        device=str(device),
    )
    embedding_cfg = EmbeddingConfig(method="facenet", pretrained="vggface2", device=str(device))
    projector_cfg = ProjectorConfig(
        key_dim=args.key_dim,
        hidden_dims=(1024, 512),
        dropout=0.0,
        enable_input_l2_norm=args.enable_projector_l2_reg,
        enable_key_upscaler=args.enable_projector_key_upscaler,
    )

    cfg = PipelineConfig(
        data=data_cfg,
        detector=detector_cfg,
        embedding=embedding_cfg,
        seed=SeedConfig(secret_key="master_thesis_secret"),
        projector=projector_cfg,
        use_stylegan_mapper=args.use_stylegan_mapper,
    )

    logging.info("Loading StyleGAN2 from %s...", args.stylegan_ckpt)
    stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(
        cfg,
        stylegan=stylegan,
        device=device,
        truncation_psi=args.truncation_psi,
        face_postprocessor=face_postprocessor,
    )

    logging.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_projector_state_dict(pipeline.projector, ckpt["model_state_dict"])
    pipeline.projector.eval()
    if hasattr(pipeline.embedder, "eval"):
        pipeline.embedder.eval()

    sources = _collect_sources(args)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    projector_dtype = next(pipeline.projector.parameters()).dtype
    key_bank = sample_binary_key_bank(
        args.num_keys,
        args.key_dim,
        device=device,
        generator=rng,
        dtype=projector_dtype,
    )

    processed_videos = 0
    skipped_videos = 0
    manifest_entries: list[dict[str, object]] = []

    with torch.no_grad():
        for src in sources:
            source_fps = get_video_fps(src.video_path)
            effective_sample_step = max(1, int(round(source_fps / float(args.target_sample_fps))))
            sampled = load_video_frames(
                src.video_path,
                max_frames=args.max_frames_per_video,
                frame_step=effective_sample_step,
                convert_rgb=True,
            )
            if sampled is None:
                skipped_videos += 1
                continue

            frames = torch.from_numpy(sampled).permute(0, 3, 1, 2).float() / 255.0
            frames = _apply_input_brightness(frames, args.input_brightness_ev)
            frames = frames.to(device)

            key_outputs: dict[str, list[torch.Tensor]] = {}
            key_stats: dict[str, dict[str, int]] = {}

            for key_idx, key_vec in enumerate(key_bank, start=1):
                key_name = f"key{key_idx}"
                if face_postprocessor is not None and hasattr(face_postprocessor, "reset_sequence"):
                    face_postprocessor.reset_sequence()
                key_detected = 0
                key_processed = 0
                key_composited = 0
                key_skipped = 0
                composited_frames: list[torch.Tensor] = []

                for chunk_start in range(0, int(frames.shape[0]), args.batch_size):
                    chunk_end = min(int(frames.shape[0]), chunk_start + args.batch_size)
                    frame_chunk = frames[chunk_start:chunk_end]
                    chunk_res = pipeline.infer_frames_batched(
                        frame_chunk,
                        key_vec,
                        use_face_postprocessor=face_postprocessor is not None,
                        swap_for_visuals_only=args.swap_for_visuals_only,
                    )

                    key_detected += int(chunk_res.stats.get("detected_faces", 0))
                    key_processed += int(chunk_res.stats.get("processed_faces", 0))
                    key_composited += int(chunk_res.stats.get("composited_faces", 0))
                    key_skipped += int(chunk_res.stats.get("skipped_faces", 0))

                    if len(chunk_res.output_frames) != (chunk_end - chunk_start):
                        skipped_videos += 1
                        composited_frames = []
                        break

                    composited_frames.extend([
                        _normalize_frame(output_frame, image_size=None)
                        for output_frame in chunk_res.output_frames
                    ])

                if len(composited_frames) != int(frames.shape[0]):
                    continue

                key_outputs[key_name] = composited_frames
                key_stats[key_name] = {
                    "detected_faces": key_detected,
                    "processed_faces": key_processed,
                    "composited_faces": key_composited,
                    "skipped_faces": key_skipped,
                }

            if any(len(output_frames) != int(frames.shape[0]) for output_frames in key_outputs.values()):
                skipped_videos += 1
                continue

            output_fps = float(args.output_fps) if args.output_fps is not None else max(1.0, source_fps / float(effective_sample_step))

            prepared_filename = _prepared_output_filename(src)

            outputs: dict[str, str] = {}
            key_codecs: dict[str, str] = {}
            for key_name, output_frames in key_outputs.items():
                if args.num_keys >= 2:
                    key_output_dir = output_dir / key_name
                    output_path = _build_output_path(
                        key_output_dir,
                        src,
                        prepared_filename,
                        preserve_nested_folders=args.preserve_nested_folders,
                    )
                else:
                    output_path = _build_output_path(
                        output_dir,
                        src,
                        prepared_filename,
                        preserve_nested_folders=args.preserve_nested_folders,
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                key_codecs[key_name] = write_mp4(output_path, output_frames, fps=output_fps)
                outputs[key_name] = str(output_path.relative_to(output_dir).as_posix())

            processed_videos += 1
            manifest_entries.append(
                {
                    "identity": src.identity,
                    "youtube_id": src.youtube_id,
                    "sample_index": src.sample_index,
                    "original_name": src.original_name,
                    "prepared_filename": prepared_filename,
                    "source_id": src.source_id,
                    "source_video": str(src.video_path),
                    "source_fps": float(source_fps),
                    "target_sample_fps": float(args.target_sample_fps),
                    "effective_sample_step": int(effective_sample_step),
                    "sampled_frames": int(frames.shape[0]),
                    "outputs": outputs,
                    "multi_face_stats": key_stats,
                    "fps": float(output_fps),
                    "codecs": key_codecs,
                }
            )

            logging.info(
                "Processed video %s | sampled_frames=%d | key_stats=%s",
                src.video_path,
                int(frames.shape[0]),
                key_stats,
            )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": args.dataset_type,
                "num_keys": args.num_keys,
                "nested_key_outputs": bool(args.num_keys >= 2),
                "max_frames_per_video": args.max_frames_per_video,
                "target_sample_fps": float(args.target_sample_fps),
                "entries": manifest_entries,
            },
            handle,
            indent=2,
        )

    report_path = output_dir / f"{args.checkpoint.stem}_{args.dataset_type}_infer.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "dataset_type": args.dataset_type,
                "seed": args.seed,
                "processed_videos": processed_videos,
                "skipped_videos": skipped_videos,
                "output_dir": str(output_dir),
                "nested_key_outputs": bool(args.num_keys >= 2),
                "manifest": str(manifest_path),
                "settings": serialized_args,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Inference export complete | processed_videos=%d | skipped_videos=%d | output_dir=%s | report=%s",
        processed_videos,
        skipped_videos,
        output_dir,
        report_path,
    )


if __name__ == "__main__":
    main()
