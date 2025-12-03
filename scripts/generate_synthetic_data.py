#!/usr/bin/env env python
"""
Generate synthetic training data for development/testing
Simulates EMBER dataset structure without needing 11GB download
"""
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_ember_data(output_dir: Path, n_samples: int = 10000):
    """
    Generate synthetic EMBER-like data for testing

    Args:
        output_dir: Directory to save synthetic data
        n_samples: Number of samples to generate (default 10k for testing)
    """
    print(f"🎲 Generating {n_samples} synthetic samples...")

    # Create directory structure
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # EMBER has 2381 features
    n_features = 2381

    # Split: 80% train, 10% val, 10% test
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)
    n_test = n_samples - n_train - n_val

    print(f"📊 Split: {n_train} train, {n_val} val, {n_test} test")

    # Generate synthetic features (realistic distributions)
    # Malware features tend to have different patterns than benign

    # Training data
    X_train_benign = np.random.randn(n_train // 2, n_features) - 1.0  # Benign: negative bias
    X_train_malware = np.random.randn(n_train // 2, n_features) + 1.5  # Malware: positive bias
    X_train = np.vstack([X_train_benign, X_train_malware])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    # Shuffle training data
    shuffle_idx = np.random.permutation(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # Validation data
    X_val_benign = np.random.randn(n_val // 2, n_features) - 1.0
    X_val_malware = np.random.randn(n_val // 2, n_features) + 1.5
    X_val = np.vstack([X_val_benign, X_val_malware])
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2))

    # Test data
    X_test_benign = np.random.randn(n_test // 2, n_features) - 1.0
    X_test_malware = np.random.randn(n_test // 2, n_features) + 1.5
    X_test = np.vstack([X_test_benign, X_test_malware])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))

    # Save as .npy files
    print("💾 Saving to disk...")
    np.save(processed_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(processed_dir / "y_train.npy", y_train.astype(np.int32))
    np.save(processed_dir / "X_val.npy", X_val.astype(np.float32))
    np.save(processed_dir / "y_val.npy", y_val.astype(np.int32))
    np.save(processed_dir / "X_test.npy", X_test.astype(np.float32))
    np.save(processed_dir / "y_test.npy", y_test.astype(np.int32))

    # Create a simple preprocessor (use sklearn for proper pickling)
    from sklearn.preprocessing import StandardScaler
    preprocessor = StandardScaler()
    preprocessor.fit(X_train)  # Fit on training data

    with open(processed_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("✅ Synthetic data generated successfully!")
    print(f"📁 Location: {processed_dir}")
    print(f"📊 Shapes:")
    print(f"   - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"   - X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"   - Features: {n_features}")

    # Calculate expected accuracy (should be ~85-90% due to separation)
    print(f"\n💡 Expected accuracy: ~85-90% (synthetic data has clear separation)")

    return processed_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic EMBER data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)"
    )

    args = parser.parse_args()

    generate_synthetic_ember_data(args.output, args.samples)
