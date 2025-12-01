#!/usr/bin/env python3
"""
Training Script for Malware Detection Model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from src.training import train_malware_detector, evaluate_model
from src.evaluation import create_evaluation_report
from src.utils import setup_logging, set_seed, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train malware detection model')
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type',
        type=str,
        default='dnn',
        choices=['dnn', 'dnn_residual', 'cnn', 'ensemble'],
        help='Type of model architecture'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--adversarial',
        action='store_true',
        help='Use adversarial training'
    )
    
    # System arguments
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA (use CPU)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Output arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate existing model (no training)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    set_seed(args.seed)
    device = get_device(use_cuda=not args.no_cuda)
    
    print("\n" + "="*70)
    print("VIRUSHUNTER - Malware Detection Training")
    print("="*70)
    print(f"Model Type:      {args.model_type}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.learning_rate}")
    print(f"Adversarial:     {args.adversarial}")
    print(f"Device:          {device}")
    print(f"Seed:            {args.seed}")
    print(f"Save Directory:  {args.save_dir}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading preprocessed data...")
    try:
        data_dir = Path(args.data_dir)
        
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_val = np.load(data_dir / 'X_val.npy')
        y_val = np.load(data_dir / 'y_val.npy')
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        print(f"✓ Data loaded successfully")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
    except FileNotFoundError:
        print("❌ Error: Processed data not found!")
        print("Please run preprocessing first:")
        print("  python scripts/preprocess.py")
        sys.exit(1)
    
    if args.eval_only:
        # Evaluation only mode
        print("\n" + "="*70)
        print("EVALUATION MODE")
        print("="*70 + "\n")
        
        if not Path(args.save_dir, 'best_model.pth').exists():
            print("❌ Error: No trained model found!")
            print(f"Expected path: {args.save_dir}/best_model.pth")
            sys.exit(1)
        
        from src.model import create_model
        model = create_model(args.model_type, input_dim=X_train.shape[1])
        checkpoint = load_checkpoint(
            Path(args.save_dir) / 'best_model.pth',
            model,
            device=device
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(model, X_test, y_test, device, args.batch_size)
        
        # Generate full evaluation report
        model.eval()
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                probs = torch.sigmoid(output)
                pred = (probs > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        create_evaluation_report(
            y_test,
            np.array(all_preds).flatten(),
            np.array(all_probs).flatten(),
            history=None,
            save_dir='results/evaluations'
        )
        
    else:
        # Training mode
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        # Train model
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
            device=device,
            save_dir=args.save_dir
        )
        
        # Evaluate on test set
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70 + "\n")
        
        test_metrics = evaluate_model(model, X_test, y_test, device, args.batch_size)
        
        # Generate full evaluation report
        model.eval()
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                probs = torch.sigmoid(output)
                pred = (probs > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        create_evaluation_report(
            y_test,
            np.array(all_preds).flatten(),
            np.array(all_probs).flatten(),
            history=history,
            save_dir='results/evaluations'
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"✓ Best model saved to: {args.save_dir}/best_model.pth")
        print(f"✓ Evaluation results saved to: results/evaluations/")
        print("\nFinal Test Metrics:")
        print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC:       {test_metrics['auc']:.4f}")
        print("="*70 + "\n")
        
        print("Next step: Run the Streamlit application:")
        print("  streamlit run app/streamlit_complete.py")


if __name__ == "__main__":
    main()