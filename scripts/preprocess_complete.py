#!/usr/bin/env python3
"""
Complete preprocessing script with all features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src import (
    prepare_ember_dataset,
    setup_logging,
    set_seed
)


def main():
    parser = argparse.ArgumentParser(description='Preprocess EMBER dataset')
    parser.add_argument('--data-dir', type=str, default='data/ember')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--augment', action='store_true', help='Apply adversarial augmentation')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    set_seed(args.seed)
    
    logger.info("Starting data preprocessing")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Preprocess
    data = prepare_ember_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        augment=args.augment
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
