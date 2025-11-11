#!/usr/bin/env python3
"""
Light preprocessing - for quick testing with subset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_light(n_samples=10000):
    """
    Quick preprocessing for testing
    Uses subset of data for fast iteration
    """
    print(f"Light preprocessing with {n_samples} samples")
    
    # Load subset
    try:
        import ember
        X_train, y_train = ember.read_vectorized_features('data/ember', subset='train')
        
        # Take subset
        mask = y_train != -1
        X_train = X_train[mask][:n_samples]
        y_train = y_train[mask][:n_samples]
        
        print(f"Loaded {len(X_train)} samples")
        
        # Quick preprocessing
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Save
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/X_val.npy', X_val)
        np.save('data/processed/y_val.npy', y_val)
        joblib.dump(scaler, 'models/preprocessor.pkl')
        
        print("✓ Light preprocessing complete!")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
    except ImportError:
        print("Error: ember package not installed")
        print("Install with: pip install ember")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=10000)
    args = parser.parse_args()
    
    preprocess_light(args.n_samples)
