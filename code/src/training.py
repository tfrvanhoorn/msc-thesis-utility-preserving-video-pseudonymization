import argparse
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
    EyeglassesBoundaryConfig,
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
from utils.logging import configure_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train the KFAAR Projector for Face Pseudonymization")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "celeba", help="Path to the dataset root")
    parser.add_argument("--dataset_type", type=str, default="celeba", choices=["celeba", "image_folder", "voxceleb_video", "video_folder"], help="Dataset type to use")
    parser.add_argument("--stylegan_ckpt", type=Path, default=SRC_ROOT / "models" / "stylegan2-celebahq-256x256.pkl", help="Path to StyleGAN2 .pkl checkpoint")
    parser.add_argument("--truncation_psi", type=float, default=0.5, help="Truncation psi for StyleGAN2 mapping")
    parser.add_argument("--remove_eyeglasses", action="store_true", help="Push StyleGAN away from generating eyeglasses in W-space")
    parser.add_argument("--eyeglasses_boundary_path", type=Path, default=None, help="Path to InterfaceGAN eyeglasses boundary (.npy) in W-space")
    parser.add_argument("--eyeglasses_removal_scale", type=float, default=1.0, help="Scale factor used in W-space eyeglasses removal: w = w - scale * boundary")
    parser.add_argument("--output_dir", type=Path, default=SRC_ROOT / "train_results", help="Directory to save checkpoints")

    # Hyperparameters (Projector & Trainer)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_identities", type=int, default=4, help="Number of unique identities per batch")
    parser.add_argument("--batch_samples_per_identity", type=int, default=2, help="Images per identity in a batch")
    parser.add_argument("--key_dim", type=int, default=128, help="Dimension of the pseudonymization key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the projector")

    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "lstm"], help="Projector architecture")
    parser.add_argument("--lstm_hidden_dim", type=int, default=512, help="Hidden size for LSTM projector")
    parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of layers for LSTM projector")
    parser.add_argument("--lstm_bidirectional", action="store_true", default=True, help="Use bidirectional LSTM")
    parser.add_argument("--no_lstm_bidirectional", dest="lstm_bidirectional", action="store_false", help="Disable bidirectional LSTM")
    parser.add_argument("--lstm_dropout", type=float, default=0.0, help="Dropout for LSTM projector (applied when num_layers>1)")
    
    # Loss Weights (The KFAAR Lambda parameters)
    parser.add_argument("--lambda_ano", type=float, default=0.4, help="Weight for Anonymity loss")
    parser.add_argument("--lambda_syn", type=float, default=1.0, help="Weight for Synchronism loss")
    parser.add_argument("--lambda_div", type=float, default=1.0, help="Weight for Diversity loss")
    parser.add_argument("--lambda_dif", type=float, default=1.0, help="Weight for Differentiation loss")
    parser.add_argument("--lambda_temp", type=float, default=0.0, help="Weight for temporal smoothness loss (LSTM + seq>1 only)")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for triplet/cosine losses")

    # Dataset & Split
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of identities used for training")
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--window_size", type=int, default=16, help="Window size for video datasets")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride inside a window for video datasets")
    parser.add_argument("--window_step", type=int, default=None, help="Step between window starts for video datasets (defaults to window_size*frame_stride)")
    parser.add_argument("--max_windows_per_video", type=int, default=None, help="Max windows to sample per source video for video datasets")
    parser.add_argument("--max_samples_per_identity", type=int, default=None, help="Cap samples per identity (images) or videos per identity (video datasets)")
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

    # Deprecated diffusion aliases kept for backward compatibility
    parser.add_argument("--diffusion_base_model", type=str, default=None, help="[Deprecated] Alias for --faceadapter_base_model")
    parser.add_argument("--diffusion_ip_adapter_id", type=str, default=None, help="[Deprecated] Unused for FaceAdapter backend")
    parser.add_argument("--diffusion_ip_adapter_weight", type=str, default=None, help="[Deprecated] Unused for FaceAdapter backend")
    parser.add_argument("--diffusion_inference_steps", type=int, default=None, help="[Deprecated] Alias for --faceadapter_inference_steps")
    parser.add_argument("--diffusion_ip_adapter_scale", type=float, default=None, help="[Deprecated] Alias for --faceadapter_guidance_scale")

    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)
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
            faceadapter_base_model = args.diffusion_base_model or args.faceadapter_base_model
            faceadapter_steps = args.diffusion_inference_steps if args.diffusion_inference_steps is not None else args.faceadapter_inference_steps
            faceadapter_guidance = args.diffusion_ip_adapter_scale if args.diffusion_ip_adapter_scale is not None else args.faceadapter_guidance_scale

            face_swapper = DiffusionFaceSwapper(
                faceadapter_root=args.faceadapter_root,
                checkpoint_dir=faceadapter_ckpt_dir,
                base_model_id=faceadapter_base_model,
                cache_dir=args.faceadapter_cache_dir,
                use_cache=args.faceadapter_use_cache,
                inference_steps=faceadapter_steps,
                guidance_scale=faceadapter_guidance,
                crop_ratio=args.faceadapter_crop_ratio,
                seed=args.faceadapter_seed,
                device=device,
            )
    
    # 1. Setup Configurations
    data_options: dict[str, object] = {}
    if args.max_samples_per_identity is not None:
        data_options["max_samples_per_identity"] = args.max_samples_per_identity
    if args.dataset_type in {"voxceleb_video", "video_folder"}:
        data_options.update(
            {
                "max_videos_per_identity": args.max_samples_per_identity,
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

    # 2. Build Data Loaders
    logging.info("Building data loaders...")
    split, train_loader, val_loader = build_train_test_loaders(
        cfg.data,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_identities=args.max_identities,
        max_samples_per_identity=args.max_samples_per_identity,
        batch_size=args.batch_identities * args.batch_samples_per_identity,
        identity_batching=True,
        batch_identities=args.batch_identities,
        samples_per_identity=args.batch_samples_per_identity,
        shuffle_train=True,
        shuffle_test=False,
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
        batch_identities=args.batch_identities,
        samples_per_identity=args.batch_samples_per_identity,
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