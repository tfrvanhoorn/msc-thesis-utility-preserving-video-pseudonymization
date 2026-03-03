from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[2]
PROJECT_ROOT = current_file.parents[3]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar.config import (  # noqa: E402
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from anon_pipeline.kfaar.metrics import MetricsAccumulator  # noqa: E402
from anon_pipeline.kfaar.pipeline.factory import build_kfaar_pipeline  # noqa: E402
from anon_pipeline.kfaar.components import load_stylegan2, load_projector_state_dict, SimSwapFaceSwapper  # noqa: E402
from anon_pipeline.shared.data.splits import build_dataloader_for_identities, list_identities  # noqa: E402
from anon_pipeline.shared.utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KFAAR projector")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained projector checkpoint (.pt)")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "celeba", help="Path to the dataset root")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="celeba",
        choices=["celeba", "image_folder", "voxceleb_video"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--stylegan_ckpt",
        type=Path,
        default=SRC_ROOT / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl",
        help="Path to StyleGAN2 .pkl checkpoint",
    )
    parser.add_argument("--truncation_psi", type=float, default=0.5, help="Truncation psi for StyleGAN2 mapping")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SRC_ROOT / "anon_pipeline" / "kfaar" / "eval_results",
        help="Directory to save evaluation reports",
    )

    # Face swapping via SimSwap
    parser.add_argument("--use_face_swapper", action="store_true", help="Enable SimSwap face swapping")
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
        "--simswap_crop_size",
        type=int,
        default=224,
        choices=[224, 512],
        help="Input/output resolution for SimSwap",
    )

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

    # Dataset & Split
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Max video files sampled per identity (voxceleb_video)")
    parser.add_argument("--max_videos_per_youtube_id", type=int, default=None, help="Max video files sampled per YouTube ID (voxceleb_video)")
    parser.add_argument("--min_youtube_id_per_identity", type=int, default=None, help="Require at least this many YouTube IDs per identity (voxceleb_video)")
    parser.add_argument("--window_size", type=int, default=16, help="Window size (frames) for voxceleb_video sequences")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride between frames inside a window")
    parser.add_argument("--window_step", type=int, default=None, help="Step between window starts (defaults to window_size*frame_stride)")
    parser.add_argument("--max_windows_per_video", type=int, default=None, help="Max windows sampled per source video (voxceleb_video)")
    parser.add_argument("--max_samples_per_identity", type=int, default=None, help="Cap samples per identity (images) or videos per identity (voxceleb)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting and key sampling")

    # Identity batching
    parser.add_argument("--batch_identities", type=int, default=4, help="Number of unique identities per batch")
    parser.add_argument("--batch_videos_per_identity", type=int, default=2, help="Videos per identity per batch (voxceleb: all windows from each video) or samples for image datasets")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

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


def main() -> None:
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)

    face_swapper = None
    if args.use_face_swapper:
        simswap_ckpt_dir = args.simswap_checkpoints_dir or args.simswap_root / "checkpoints"
        arcface_ckpt = args.simswap_arcface_ckpt or args.simswap_root / "arcface_model" / "arcface_checkpoint.tar"
        face_swapper = SimSwapFaceSwapper(
            simswap_root=args.simswap_root,
            checkpoints_dir=simswap_ckpt_dir,
            name=args.simswap_name,
            which_epoch=args.simswap_epoch,
            arcface_ckpt=arcface_ckpt,
            crop_size=args.simswap_crop_size,
            device=device,
        )

    data_options: dict[str, object] = {
        "max_videos_per_identity": args.max_videos_per_identity,
        "max_videos_per_youtube_id": args.max_videos_per_youtube_id,
        "min_youtube_id_per_identity": args.min_youtube_id_per_identity,
    }
    if args.max_samples_per_identity is not None:
        data_options["max_samples_per_identity"] = args.max_samples_per_identity
        if args.dataset_type == "voxceleb_video":
            data_options["max_videos_per_identity"] = args.max_samples_per_identity
    if args.dataset_type == "voxceleb_video":
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
        group_by_video=cfg.data.dataset_type.lower() == "voxceleb_video",
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
    use_swapper = args.use_face_swapper and face_swapper is not None

    if args.save_generated_faces and hasattr(pipeline, "configure_saving"):
        save_dir = args.save_generated_dir if args.save_generated_dir is not None else args.output_dir / "generated_faces"
        save_max = None if args.save_generated_max_per_epoch is not None and args.save_generated_max_per_epoch <= 0 else args.save_generated_max_per_epoch
        pipeline.configure_saving(
            save_dir,
            mode=args.save_generated_mode,
            max_per_epoch=save_max,
            save_videos=args.save_videos,
        )
        if hasattr(pipeline, "begin_epoch"):
            pipeline.begin_epoch(1)

    logging.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_projector_state_dict(pipeline.projector, ckpt["model_state_dict"])
    pipeline.projector.eval()
    if hasattr(pipeline.embedder, "eval"):
        pipeline.embedder.eval()

    metrics = MetricsAccumulator(anonymization_threshold=args.ano_threshold, synchronism_threshold=args.syn_threshold)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    identity_keys: dict[int, torch.Tensor] = {}

    def _key_for(label: int) -> torch.Tensor:
        if label not in identity_keys:
            identity_keys[label] = torch.randn(args.key_dim, generator=rng, device=device)
        return identity_keys[label]

    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_div_embeddings: list[torch.Tensor] = []
            batch_div_labels: list[int] = []

            frames, labels, seq_lens, contexts, sources = _extract_batch(batch)
            frames = frames.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            batch_size = frames.shape[0]
            for idx in range(batch_size):
                seq_len = int(seq_lens[idx].item())
                sample_frames = frames[idx, :seq_len]
                label = int(labels[idx].item())
                key = _key_for(label)

                sample_context = None
                if contexts is not None and idx < len(contexts):
                    sample_context = contexts[idx]

                source_id = None
                if sources is not None and idx < len(sources):
                    source_id = sources[idx]

                forward_fn = pipeline.forward_eval if use_swapper else pipeline.forward
                res = forward_fn(
                    sample_frames,
                    key,
                    sample_label=label,
                    sample_context=sample_context,
                    use_face_swapper=use_swapper,
                )

                metrics.update_detection(res.gen_mask)
                metrics.update_anonymization(res.real_embeddings, res.virtual_embeddings, res.valid_mask)

                valid_virtual = res.virtual_embeddings[res.valid_mask]
                if valid_virtual.numel() > 0:
                    metrics.add_synchronism_embeddings(label, valid_virtual, source_id=source_id)

                    # Collect valid virtual embeddings for diversity scoring across identities in the batch
                    batch_div_embeddings.append(valid_virtual.detach())
                    batch_div_labels.extend([label] * valid_virtual.shape[0])

                total_samples += 1

            if batch_div_embeddings:
                div_embeds = torch.cat(batch_div_embeddings, dim=0)
                div_labels = torch.as_tensor(batch_div_labels, device=div_embeds.device, dtype=torch.long)
                metrics.update_diversity(div_embeds, div_labels)

    if hasattr(pipeline, "finalize_saving"):
        pipeline.finalize_saving()

    summary = metrics.finalize()
    logging.info(
        "Evaluation complete | detection_rate=%.4f | anonymization_success=%.4f | synchronism_success=%.4f | syn_within=%.4f | syn_cross=%.4f | diversity_success=%.4f | samples=%d",
        summary["detection_rate"],
        summary["anonymization_success_rate"],
        summary["synchronism_success_rate"],
        summary["synchronism_within_success_rate"],
        summary["synchronism_cross_success_rate"],
        summary.get("diversity_success_rate", 0.0),
        total_samples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.checkpoint.stem}_{args.dataset_type}_eval.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "dataset_type": args.dataset_type,
                "seed": args.seed,
                "metrics": summary,
                "total_samples": total_samples,
                "settings": serialized_args,
                "identities": {
                    "all": all_identities,
                },
            },
            f,
            indent=2,
        )
    logging.info("Saved evaluation report to %s", report_path)


if __name__ == "__main__":
    main()
