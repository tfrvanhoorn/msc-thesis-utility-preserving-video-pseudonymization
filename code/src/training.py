import argparse
from collections import Counter
import logging
import sys
import warnings
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

from trainer import KfaarTrainer
from config import (
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from pipeline.factory import build_kfaar_pipeline
from components import (
    load_stylegan2,
    load_projector_state_dict,
    SimSwapFaceSwapper,
    DiffusionFaceSwapper,
)
from data.splits import build_train_test_loaders
from data.prepared import DEFAULT_PREPARED_REGEX, collect_prepared_images, compile_prepared_regex, PreparedNameError
from utils.logging import configure_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train the KFAAR Projector for Face Pseudonymization")

    # Path Arguments
    parser.add_argument("--input_dir", type=Path, default=PROJECT_ROOT / "data" / "prepared_celeba", help="Path to prepared input images root")
    parser.add_argument("--prepared_filename_regex", type=str, default=DEFAULT_PREPARED_REGEX, help="Regex used to parse prepared image filenames")
    parser.add_argument("--stylegan_ckpt", type=Path, default=SRC_ROOT / "models" / "stylegan2-celebahq-256x256.pkl", help="Path to StyleGAN2 .pkl checkpoint")
    parser.add_argument("--truncation_psi", type=float, default=0.5, help="Truncation psi for StyleGAN2 mapping")
    parser.add_argument("--output_dir", type=Path, default=SRC_ROOT / "train_results", help="Directory to save checkpoints")

    # Hyperparameters (Projector & Trainer)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--key_dim", type=int, default=128, help="Dimension of the pseudonymization key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the projector")
    
    # Loss Weights (The KFAAR Lambda parameters)
    parser.add_argument("--lambda_ano", type=float, default=0.4, help="Weight for Anonymity loss")
    parser.add_argument("--lambda_syn", type=float, default=1.0, help="Weight for Synchronism loss")
    parser.add_argument("--lambda_div", type=float, default=1.0, help="Weight for Diversity loss")
    parser.add_argument("--lambda_dif", type=float, default=1.0, help="Weight for Differentiation loss")
    parser.add_argument("--lambda_temp", type=float, default=0.0, help="Weight for temporal smoothness loss")
    parser.add_argument("--lambda_w_reg", type=float, default=20.0, help="Weight for StyleGAN W-space regularization loss")
    parser.add_argument("--enable_projector_l2_reg", dest="enable_projector_l2_reg", action="store_true", help="Enable input L2 normalization for both key and z in the projector MLP")
    parser.add_argument("--disable_projector_l2_reg", dest="enable_projector_l2_reg", action="store_false", help="Disable input L2 normalization for key and z in the projector MLP")
    parser.add_argument("--enable_projector_key_upscaler", dest="enable_projector_key_upscaler", action="store_true", help="Enable projector key upscaler to map key_dim to 512 before concatenation")
    parser.add_argument("--disable_projector_key_upscaler", dest="enable_projector_key_upscaler", action="store_false", help="Disable projector key upscaler and concatenate raw key with z")
    parser.add_argument("--use_stylegan_mapper", dest="use_stylegan_mapper", action="store_true", help="Use StyleGAN mapping network (z->W+) before synthesis")
    parser.add_argument("--disable_stylegan_mapper", dest="use_stylegan_mapper", action="store_false", help="Bypass StyleGAN mapping and repeat projected z across W+ layers before synthesis")
    parser.set_defaults(enable_projector_l2_reg=True, enable_projector_key_upscaler=True, use_stylegan_mapper=False)
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for triplet/cosine losses")

    # Dataset & Split
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of identities used for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Logical tuple batch size; each tuple is (x1, x2, y), so physical samples per step are 3*batch_size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes")
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--max_samples_per_identity", type=int, default=None, help="Cap samples per identity (images) or videos per identity (video datasets)")
    parser.add_argument("--shuffle_batches", dest="shuffle_batches", action="store_true", help="Shuffle training batches each epoch")
    parser.add_argument("--no_shuffle_batches", dest="shuffle_batches", action="store_false", help="Disable training batch shuffling; keep identical batch composition/order across epochs")
    parser.set_defaults(shuffle_batches=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # Resuming
    parser.add_argument("--resume_ckpt", type=Path, default=None, help="Path to a checkpoint (.pt) to resume from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Epoch index to start from (overrides checkpoint epoch)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    # Generated face saving
    parser.add_argument("--save_generated_faces", action="store_true", help="Store generated faces to disk during training")
    parser.add_argument("--save_generated_mode", type=str, default="detected", choices=["detected", "undetected", "all"], help="Which generated frames to store")
    parser.add_argument("--save_generated_dir", type=Path, default=None, help="Directory to store generated face images (defaults to output_dir/generated_faces)")
    parser.add_argument("--save_generated_max_per_epoch", type=int, default=100, help="Maximum number of generated samples to store per epoch (set <=0 for no limit)")

    # Face swapping selector (visual-only by default)
    parser.add_argument(
        "--face_swapper",
        type=str,
        default="none",
        choices=["none", "simswap", "diffusion"],
        help="Choose face swapper backend (none=disabled, simswap, diffusion)",
    )
    parser.add_argument("--use_face_swapper", action="store_true", help="Legacy flag to enable face swapping (overridden by face_swapper != none)")
    parser.add_argument(
        "--swap_for_visuals_only",
        action="store_true",
        help="Use swapped faces only for visualization; compute losses on StyleGAN outputs",
    )
    parser.add_argument(
        "--swap_for_loss",
        dest="swap_for_visuals_only",
        action="store_false",
        help="Use swapped faces as loss inputs (old behavior)",
    )
    parser.set_defaults(swap_for_visuals_only=True)

    # SimSwap options
    parser.add_argument("--simswap_root", type=Path, default=PROJECT_ROOT / "external_libraries" / "SimSwap", help="Path to SimSwap repository root")
    parser.add_argument("--simswap_checkpoints_dir", type=Path, default=None, help="Path to SimSwap checkpoints directory (defaults to simswap_root/checkpoints)")
    parser.add_argument("--simswap_name", type=str, default="people", help="SimSwap experiment name (subfolder in checkpoints_dir)")
    parser.add_argument("--simswap_epoch", type=str, default="latest", help="Generator checkpoint epoch tag (e.g., latest, 0015)")
    parser.add_argument("--simswap_arcface_ckpt", type=Path, default=None, help="Path to ArcFace checkpoint used by SimSwap (defaults to simswap_root/arcface_model/arcface_checkpoint.tar)")
    parser.add_argument("--simswap_parsing_ckpt", type=Path, default=None, help="Path to face parsing checkpoint for SimSwap masking (optional)")
    parser.add_argument("--simswap_crop_size", type=int, default=224, choices=[224, 512], help="Input/output resolution for SimSwap")
    parser.add_argument("--simswap_detector_name", type=str, default="antelopev2", help="Face detector name for SimSwap (e.g., antelopev2)")
    parser.add_argument("--simswap_detector_root", type=Path, default=None, help="Path to SimSwap face detector models root (defaults to simswap_root/insightface_func/models)")

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

    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging()
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {args.num_workers}")
    device = torch.device(args.device)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Prepared input directory not found: {args.input_dir}")
    try:
        logging.info(
            "Scanning prepared inputs | input_dir=%s | max_identities=%s | filename_regex=%s",
            args.input_dir,
            args.max_identities,
            args.prepared_filename_regex,
        )
        prepared_regex = compile_prepared_regex(args.prepared_filename_regex)
        prepared_refs = collect_prepared_images(
            args.input_dir,
            prepared_regex,
            max_identities=args.max_identities,
            stop_after_max_identities=bool(args.max_identities is not None),
        )
    except PreparedNameError as exc:
        raise ValueError(f"Invalid prepared input naming: {exc}") from exc
    if not prepared_refs:
        raise FileNotFoundError(f"No prepared images found in input_dir: {args.input_dir}")

    discovered_identities = sorted({ref.identity for ref in prepared_refs})
    per_identity_counts = Counter(ref.identity for ref in prepared_refs)
    min_samples = min(per_identity_counts.values())
    max_samples = max(per_identity_counts.values())
    logging.info(
        "Prepared scan complete | files=%d | identities=%d | min_samples_per_identity=%d | max_samples_per_identity=%d",
        len(prepared_refs),
        len(discovered_identities),
        min_samples,
        max_samples,
    )

    face_swapper = None
    swapper_choice = (args.face_swapper or "none").lower()
    use_swapper_requested = args.use_face_swapper or swapper_choice != "none"
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
    
    # 1. Setup Configurations
    data_options: dict[str, object] = {}
    if args.max_samples_per_identity is not None:
        data_options["max_samples_per_identity"] = args.max_samples_per_identity
    data_options["prepared_filename_regex"] = args.prepared_filename_regex
    data_options["prepared_prefetched_refs"] = prepared_refs

    data_cfg = DataConfig(
        dataset_path=args.input_dir,
        dataset_type="prepared_images",
        options=data_options,
    )
    detector_cfg = DetectorConfig(image_size=256, device=str(device))
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

    # 2. Build Data Loaders
    logging.info("Building data loaders...")
    batch_seed = None if args.shuffle_batches else args.seed
    split, train_loader, val_loader = build_train_test_loaders(
        cfg.data,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_identities=args.max_identities,
        max_samples_per_identity=args.max_samples_per_identity,
        batch_size=args.batch_size,
        identity_batching=True,
        shuffle_train=args.shuffle_batches,
        batch_seed=batch_seed,
        shuffle_test=False,
        num_workers=args.num_workers,
    )

    # 3. Initialize Pipeline Components
    logging.info(f"Loading StyleGAN2 from {args.stylegan_ckpt}...")
    stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(
        cfg,
        stylegan=stylegan,
        device=device,
        truncation_psi=args.truncation_psi,
        face_swapper=face_swapper,
    )

    start_epoch = args.start_epoch if args.start_epoch is not None else 0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        load_projector_state_dict(pipeline.projector, ckpt["model_state_dict"])
        pipeline.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if args.start_epoch is None:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
        logging.info("Resumed from %s at epoch %s", args.resume_ckpt, start_epoch)

    save_dir = args.save_generated_dir if args.save_generated_dir is not None else args.output_dir / "generated_faces"
    save_max = None if args.save_generated_max_per_epoch is not None and args.save_generated_max_per_epoch <= 0 else args.save_generated_max_per_epoch

    # 4. Initialize and Run Trainer
    use_swapper_flag = face_swapper is not None

    trainer = KfaarTrainer(
        pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        key_dim=args.key_dim,
        margin=args.margin,
        lambda_ano=args.lambda_ano,
        lambda_syn=args.lambda_syn,
        lambda_div=args.lambda_div,
        lambda_dif=args.lambda_dif,
        lambda_temp=args.lambda_temp,
        lambda_w_reg=args.lambda_w_reg,
        checkpoint_dir=args.output_dir,
        device=device,
        train_identities=split.train,
        val_identities=split.test,
        start_epoch=start_epoch,
        save_generated_faces=args.save_generated_faces,
        save_generated_dir=save_dir,
        save_generated_mode=args.save_generated_mode,
        save_generated_max_per_epoch=save_max,
        use_face_swapper=use_swapper_flag,
        swap_for_visuals_only=args.swap_for_visuals_only,
    )

    logging.info("Starting training loop...")
    trainer.train()
    logging.info(f"Training complete. Checkpoints saved to {args.output_dir}")

if __name__ == "__main__":
    main()