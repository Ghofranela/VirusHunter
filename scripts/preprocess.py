#!/usr/bin/env python3
"""
Preprocessing Script for EMBER Dataset
Loads, cleans, normalizes and saves processed data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from src.preprocessing import prepare_malware_dataset
from src.utils import setup_logging, set_seed


def main():
    parser = argparse.ArgumentParser(description='Preprocess EMBER malware dataset')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/ember2018',
        help='Directory containing EMBER raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply adversarial augmentation'
    )
    
    parser.add_argument(
        '--scaler-type',
        type=str,
        default='standard',
        choices=['standard', 'robust'],
        help='Type of scaler to use'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    set_seed(args.seed)
    
    print("\n" + "="*60)
    print("EMBER Dataset Preprocessing")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Augmentation: {args.augment}")
    print(f"Scaler: {args.scaler_type}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")
    
    try:
        # Run preprocessing pipeline
        data = prepare_malware_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            augment=args.augment,
            scaler_type=args.scaler_type
        )
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        print(f"Training samples:   {len(data['X_train']):,}")
        print(f"Validation samples: {len(data['X_val']):,}")
        print(f"Test samples:       {len(data['X_test']):,}")
        print(f"Feature dimension:  {data['X_train'].shape[1]}")
        print("="*60)
        
        # Display class distribution
        print("\nClass Distribution:")
        print("-" * 60)
        for split_name, y in [('Train', data['y_train']), 
                               ('Val', data['y_val']), 
                               ('Test', data['y_test'])]:
            unique, counts = np.unique(y, return_counts=True)
            print(f"{split_name:10s} - Benign: {counts[0]:,} ({counts[0]/len(y)*100:.1f}%), "
                  f"Malware: {counts[1]:,} ({counts[1]/len(y)*100:.1f}%)")
        print("="*60 + "\n")
        
        print("✓ All data saved successfully!")
        print(f"✓ Files saved to: {args.output_dir}/")
        print("\nNext step: Train the model with:")
        print(f"  python run_training.py")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        logger.exception("Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()