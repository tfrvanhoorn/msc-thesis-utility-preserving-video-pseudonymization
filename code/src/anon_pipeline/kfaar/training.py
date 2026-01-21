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

def parse_args():
    parser = argparse.ArgumentParser(description="Train the KFAAR Projector for Face Pseudonymization")

    # Path Arguments
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "celeba", help="Path to the dataset")
    parser.add_argument("--stylegan_ckpt", type=Path, default=SRC_ROOT / "anon_pipeline" / "kfaar" / "models" / "stylegan2-celebahq-256x256.pkl", help="Path to StyleGAN2 .pkl checkpoint")
    parser.add_argument("--output_dir", type=Path, default=SRC_ROOT / "anon_pipeline" / "kfaar" / "train_results", help="Directory to save checkpoints")

    # Hyperparameters (Projector & Trainer)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_identities", type=int, default=4, help="Number of unique identities per batch")
    parser.add_argument("--batch_samples_per_identity", type=int, default=2, help="Images per identity in a batch")
    parser.add_argument("--key_dim", type=int, default=128, help="Dimension of the pseudonymization key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the projector")
    
    # Loss Weights (The KFAAR Lambda parameters)
    parser.add_argument("--lambda_ano", type=float, default=0.4, help="Weight for Anonymity loss")
    parser.add_argument("--lambda_syn", type=float, default=1.0, help="Weight for Synchronism loss")
    parser.add_argument("--lambda_div", type=float, default=1.0, help="Weight for Diversity loss")
    parser.add_argument("--lambda_dif", type=float, default=1.0, help="Weight for Differentiation loss")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for triplet/cosine losses")

    # Dataset & Split
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of identities used for training")
    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities (useful for debugging)")
    parser.add_argument("--max_per_identity", type=int, default=None, help="Max samples per identity to use from dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)
    
    # 1. Setup Configurations
    data_cfg = DataConfig(
        dataset_path=args.data_path,
        dataset_type="celeba",
        options={"max_per_identity": args.max_per_identity}, # Adjust based on dataset availability
    )
    detector_cfg = DetectorConfig(image_size=256, device=str(device))
    embedding_cfg = EmbeddingConfig(method="facenet", pretrained="vggface2", device=str(device))
    projector_cfg = ProjectorConfig(key_dim=args.key_dim)
    
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
        shuffle_train=True,
        shuffle_test=False,
    )

    # 3. Initialize Pipeline Components
    logging.info(f"Loading StyleGAN2 from {args.stylegan_ckpt}...")
    stylegan = load_stylegan2(ckpt_path=args.stylegan_ckpt, device=device)
    pipeline = build_kfaar_pipeline(cfg, stylegan=stylegan, device=device)

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
        batch_identities=args.batch_identities,
        samples_per_identity=args.batch_samples_per_identity,
        checkpoint_dir=args.output_dir,
        device=device,
        train_identities=split.train,
        val_identities=split.test,
    )

    logging.info("Starting training loop...")
    trainer.train()
    logging.info(f"Training complete. Checkpoints saved to {args.output_dir}")

if __name__ == "__main__":
    main()