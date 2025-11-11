#!/usr/bin/env python3
"""
VirusHunter - Complete Pipeline
Main orchestration script for training and evaluation
"""
import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    prepare_ember_dataset,
    train_malware_detector,
    comprehensive_evaluation,
    generate_explanation_report,
    setup_logging,
    set_seed,
    get_device,
    create_dir_structure,
    download_ember_dataset,
    load_model
)
import numpy as np


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='VirusHunter - Advanced Malware Detection Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['download', 'preprocess', 'train', 'evaluate', 'explain', 'all'],
        help='Pipeline mode'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ember',
        help='EMBER data directory'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='dnn',
        choices=['dnn', 'cnn', 'lstm', 'ensemble'],
        help='Model architecture'
    )
    
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
        help='Batch size'
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
        help='Enable adversarial training'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def download_phase(args):
    """Download EMBER dataset"""
    print("\n" + "="*70)
    print("PHASE 1: DOWNLOADING EMBER DATASET")
    print("="*70)
    
    download_ember_dataset(args.data_dir)
    
    print("\n✓ Download phase complete!")


def preprocess_phase(args):
    """Preprocess data"""
    print("\n" + "="*70)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*70)
    
    data = prepare_ember_dataset(
        data_dir=args.data_dir,
        output_dir="data/processed",
        test_size=0.2,
        augment=args.adversarial
    )
    
    print("\n✓ Preprocessing phase complete!")
    return data


def train_phase(args):
    """Train model"""
    print("\n" + "="*70)
    print("PHASE 3: MODEL TRAINING")
    print("="*70)
    
    # Load processed data
    print("\nLoading preprocessed data...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Get device
    device = get_device(use_cuda=not args.no_cuda)
    
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
        device=device
    )
    
    print("\n✓ Training phase complete!")
    return model


def evaluate_phase(args):
    """Evaluate model"""
    print("\n" + "="*70)
    print("PHASE 4: MODEL EVALUATION")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    print(f"Test: {X_test.shape}")
    
    # Get device
    device = get_device(use_cuda=not args.no_cuda)
    
    # Load model
    print("\nLoading trained model...")
    from src.model import create_model
    model = create_model(args.model_type, input_size=X_test.shape[1])
    checkpoint = load_model(
        "models/best_model.pth",
        model_type=args.model_type,
        device=device,
        input_size=X_test.shape[1]
    )
    
    # Evaluate
    results = comprehensive_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=device,
        save_dir="results"
    )
    
    print("\n✓ Evaluation phase complete!")
    return results


def explain_phase(args):
    """Generate explanations"""
    print("\n" + "="*70)
    print("PHASE 5: EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load("data/processed/X_test.npy")
    
    # Get device
    device = get_device(use_cuda=not args.no_cuda)
    
    # Load model
    print("\nLoading trained model...")
    model = load_model(
        "models/best_model.pth",
        model_type=args.model_type,
        device=device,
        input_size=X_test.shape[1]
    )
    
    # Generate explanation for a sample
    print("\nGenerating explanation for sample...")
    sample_idx = 0
    sample = X_test[sample_idx]
    
    # Load background data for SHAP
    background_indices = np.random.choice(len(X_test), size=min(100, len(X_test)), replace=False)
    background_data = X_test[background_indices]
    
    explanations = generate_explanation_report(
        model=model,
        features=sample,
        background_data=background_data,
        device=device,
        save_dir="results/explanations"
    )
    
    print("\n✓ Explainability phase complete!")
    return explanations


def main():
    """Main pipeline execution"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    logger = setup_logging()
    set_seed(args.seed)
    create_dir_structure()
    
    print("\n" + "="*70)
    print("VIRUSHUNTER - ADVERSARIAL MALWARE DETECTION")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_type}")
    print(f"Adversarial Training: {args.adversarial}")
    print(f"Device: {'CUDA' if not args.no_cuda else 'CPU'}")
    
    try:
        # Execute pipeline based on mode
        if args.mode in ['download', 'all']:
            download_phase(args)
        
        if args.mode in ['preprocess', 'all']:
            preprocess_phase(args)
        
        if args.mode in ['train', 'all']:
            train_phase(args)
        
        if args.mode in ['evaluate', 'all']:
            evaluate_phase(args)
        
        if args.mode in ['explain', 'all']:
            explain_phase(args)
        
        # Success
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Total Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        print("\nResults saved to:")
        print("  - Models: models/")
        print("  - Metrics: results/")
        print("  - Logs: logs/")
        print("\nTo run the web interface:")
        print("  streamlit run app/streamlit_app.py")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
