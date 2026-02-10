import argparse
import logging
import sys
from pathlib import Path

import torch

current_file = Path(__file__).resolve()

SRC_ROOT = current_file.parents[2] 
PROJECT_ROOT = current_file.parents[3]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar import KfaarTrainer
from anon_pipeline.kfaar.config import (
    DataConfig,
    DetectorConfig,
    EmbeddingConfig,
    PipelineConfig,
    ProjectorConfig,
    SeedConfig,
)
from anon_pipeline.kfaar.pipeline.factory import build_kfaar_pipeline
from anon_pipeline.kfaar.components import load_stylegan2
from anon_pipeline.shared.data.splits import build_train_test_loaders
from anon_pipeline.shared.utils.logging import configure_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train the KFAAR Projector for Face Pseudonymization")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "celeba", help="Path to the dataset root")
    parser.add_argument("--dataset_type", type=str, default="celeba", choices=["celeba", "image_folder", "voxceleb_video"], help="Dataset type to use")
    parser.add_argument("--stylegan_ckpt", type=Path, default=SRC_ROOT / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl", help="Path to StyleGAN2 .pkl checkpoint")
    parser.add_argument("--output_dir", type=Path, default=SRC_ROOT / "anon_pipeline" / "kfaar" / "train_results", help="Directory to save checkpoints")

    # Hyperparameters (Projector & Trainer)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_identities", type=int, default=4, help="Number of unique identities per batch")
    parser.add_argument("--batch_samples_per_identity", type=int, default=2, help="Images per identity in a batch")
    parser.add_argument("--min_samples_per_identity", type=int, default=2, help="Minimum samples required to include an identity in a batch")
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
    parser.add_argument("--max_per_identity", type=int, default=None, help="Max samples per identity to use from dataset")
    parser.add_argument("--window_size", type=int, default=16, help="Window size for voxceleb_video sequences")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride inside a window for voxceleb_video")
    parser.add_argument("--window_step", type=int, default=None, help="Step between window starts for voxceleb_video (defaults to window_size*frame_stride)")
    parser.add_argument("--max_windows_per_video", type=int, default=None, help="Max windows to sample per video for voxceleb_video")
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

    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)
    
    # 1. Setup Configurations
    data_options: dict[str, object] = {"max_per_identity": args.max_per_identity}
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
        projector=projector_cfg
    )

    # 2. Build Data Loaders
    logging.info("Building data loaders...")
    split, train_loader, val_loader = build_train_test_loaders(
        cfg.data,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_identities=args.max_identities,
        batch_size=args.batch_identities * args.batch_samples_per_identity,
        identity_batching=True,
        batch_identities=args.batch_identities,
        samples_per_identity=args.batch_samples_per_identity,
        min_samples_per_identity=args.min_samples_per_identity,
        shuffle_train=True,
        shuffle_test=False,
    )

    # 3. Initialize Pipeline Components
    logging.info(f"Loading StyleGAN2 from {args.stylegan_ckpt}...")
    stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(cfg, stylegan=stylegan, device=device)

    start_epoch = args.start_epoch if args.start_epoch is not None else 0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        pipeline.projector.load_state_dict(ckpt["model_state_dict"])
        pipeline.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if args.start_epoch is None:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
        logging.info("Resumed from %s at epoch %s", args.resume_ckpt, start_epoch)

    save_dir = args.save_generated_dir if args.save_generated_dir is not None else args.output_dir / "generated_faces"
    save_max = None if args.save_generated_max_per_epoch is not None and args.save_generated_max_per_epoch <= 0 else args.save_generated_max_per_epoch

    # 4. Initialize and Run Trainer
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
    )

    logging.info("Starting training loop...")
    trainer.train()
    logging.info(f"Training complete. Checkpoints saved to {args.output_dir}")

if __name__ == "__main__":
    main()