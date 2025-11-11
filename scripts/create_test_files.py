#!/usr/bin/env python3
"""
Create test sample files for testing detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib


def create_test_files(output_dir='test_samples'):
    """
    Create synthetic test files for malware detection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating test samples in: {output_dir}")
    
    # Load preprocessor if available
    try:
        scaler = joblib.load('models/preprocessor.pkl')
        print("Using trained preprocessor")
    except:
        print("Preprocessor not found, using raw features")
        scaler = None
    
    n_features = 2381
    
    # Create malware samples (high positive values)
    print("\nCreating malware samples...")
    for i in range(1, 6):
        # Malware-like features
        features = np.random.randn(n_features) + 2.5
        features += np.random.uniform(-0.5, 0.5, n_features)
        
        if scaler:
            features = scaler.transform(features.reshape(1, -1))[0]
        
        filename = output_dir / f"malware_sample_{i}.npy"
        np.save(filename, features)
        print(f"  ✓ Created {filename.name}")
    
    # Create benign samples (low/negative values)
    print("\nCreating benign samples...")
    for i in range(1, 6):
        # Benign-like features
        features = np.random.randn(n_features) - 1.5
        features += np.random.uniform(-0.5, 0.5, n_features)
        
        if scaler:
            features = scaler.transform(features.reshape(1, -1))[0]
        
        filename = output_dir / f"benign_sample_{i}.npy"
        np.save(filename, features)
        print(f"  ✓ Created {filename.name}")
    
    # Create ambiguous samples (near decision boundary)
    print("\nCreating ambiguous samples...")
    for i in range(1, 4):
        # Ambiguous features
        features = np.random.randn(n_features) + 0.5
        features += np.random.uniform(-1, 1, n_features)
        
        if scaler:
            features = scaler.transform(features.reshape(1, -1))[0]
        
        filename = output_dir / f"ambiguous_sample_{i}.npy"
        np.save(filename, features)
        print(f"  ✓ Created {filename.name}")
    
    print(f"\n✓ Created {13} test samples in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test sample files')
    parser.add_argument('--output-dir', type=str, default='test_samples')
    
    args = parser.parse_args()
    
    create_test_files(args.output_dir)
