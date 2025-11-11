#!/usr/bin/env python3
"""
Standalone training script
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from src import (
    train_malware_detector,
    setup_logging,
    set_seed,
    get_device
)


def main():
    parser = argparse.ArgumentParser(description='Train malware detection model')
    parser.add_argument('--model-type', type=str, default='dnn',
                       choices=['dnn', 'cnn', 'lstm', 'ensemble'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    set_seed(args.seed)
    device = get_device(use_cuda=not args.no_cuda)
    
    logger.info(f"Training {args.model_type} model")
    logger.info(f"Adversarial training: {args.adversarial}")
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train
    model, history = train_malware_detector(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        adversarial=args.adversarial,
        device=device
    )
    
    print("\n✓ Training complete!")
    print(f"Model saved to: models/best_model.pth")


if __name__ == "__main__":
    main()
