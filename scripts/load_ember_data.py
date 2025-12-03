#!/usr/bin/env python3
"""
Load EMBER dataset from various sources
Supports: processed .npy, raw EMBER .dat, or synthetic generation
"""
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_processed_data(data_dir: Path):
    """
    Load preprocessed .npy files

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
    """
    processed_dir = data_dir / "processed"

    if not processed_dir.exists():
        return None

    required_files = [
        "X_train.npy", "y_train.npy",
        "X_val.npy", "y_val.npy",
        "X_test.npy", "y_test.npy",
        "preprocessor.pkl"
    ]

    if not all((processed_dir / f).exists() for f in required_files):
        return None

    print(f"📂 Loading processed data from {processed_dir}")

    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    with open(processed_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    print(f"✅ Loaded preprocessed data:")
    print(f"   - Train: {X_train.shape}")
    print(f"   - Val: {X_val.shape}")
    print(f"   - Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def load_ember_raw_data(data_dir: Path):
    """
    Load raw EMBER .dat files (X_train.dat, y_train.dat, etc.)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, preprocessor)
    """
    ember_dir = data_dir / "raw" / "ember2018"

    if not ember_dir.exists():
        return None

    required_files = ["X_train.dat", "y_train.dat", "X_test.dat", "y_test.dat"]

    if not all((ember_dir / f).exists() for f in required_files):
        return None

    print(f"📂 Loading raw EMBER data from {ember_dir}")

    # Load EMBER .dat files (they're memory-mapped numpy arrays)
    X_train = np.memmap(ember_dir / "X_train.dat", dtype=np.float32, mode='r')
    y_train = np.memmap(ember_dir / "y_train.dat", dtype=np.float32, mode='r')
    X_test = np.memmap(ember_dir / "X_test.dat", dtype=np.float32, mode='r')
    y_test = np.memmap(ember_dir / "y_test.dat", dtype=np.float32, mode='r')

    # EMBER format: flattened array, need to reshape to (n_samples, 2381)
    n_features = 2381
    n_train = len(X_train) // n_features
    n_test = len(X_test) // n_features

    X_train = X_train.reshape(n_train, n_features)
    X_test = X_test.reshape(n_test, n_features)

    # Filter out unlabeled samples (-1)
    train_idx = y_train != -1
    test_idx = y_test != -1

    X_train = X_train[train_idx]
    y_train = y_train[train_idx].astype(np.int32)
    X_test = X_test[test_idx]
    y_test = y_test[test_idx].astype(np.int32)

    # Create validation split (10% of training data)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Create and fit preprocessor
    from sklearn.preprocessing import StandardScaler
    preprocessor = StandardScaler()
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    print(f"✅ Loaded raw EMBER data:")
    print(f"   - Train: {X_train.shape}")
    print(f"   - Val: {X_val.shape}")
    print(f"   - Test: {X_test.shape}")

    # Save processed data for next time
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving processed data to {processed_dir} for future use...")
    np.save(processed_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val.astype(np.float32))
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test.astype(np.float32))
    np.save(processed_dir / "y_test.npy", y_test)

    with open(processed_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("✅ Processed data saved!")

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def load_data(data_dir: Path = Path("data"), auto_generate: bool = True):
    """
    Smart data loader: tries multiple sources in order

    1. Check for processed/ .npy files (fastest)
    2. Check for raw/ember2018/ .dat files (slower, will process)
    3. Generate synthetic data if auto_generate=True

    Args:
        data_dir: Root data directory
        auto_generate: Whether to generate synthetic data if nothing found

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
    """
    print(f"🔍 Looking for training data in {data_dir}")

    # Try 1: Load processed data
    result = load_processed_data(data_dir)
    if result is not None:
        print("✅ Using preprocessed data")
        return result

    # Try 2: Load raw EMBER data
    result = load_ember_raw_data(data_dir)
    if result is not None:
        print("✅ Using raw EMBER data (processed and saved)")
        return result

    # Try 3: Generate synthetic data
    if auto_generate:
        print("⚠️ No EMBER data found, generating synthetic data...")
        from scripts.generate_synthetic_data import generate_synthetic_ember_data
        generate_synthetic_ember_data(data_dir, n_samples=10000)
        return load_processed_data(data_dir)

    raise FileNotFoundError(
        f"No training data found in {data_dir}. "
        "Please either:\n"
        "1. Download EMBER dataset to data/raw/ember2018/\n"
        "2. Run: python scripts/generate_synthetic_data.py\n"
        "3. Use auto_generate=True"
    )


if __name__ == "__main__":
    # Test the loader
    data = load_data(Path("data"))
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = data

    print("\n📊 Dataset Summary:")
    print(f"   Training samples: {len(y_train):,} ({np.sum(y_train==1):,} malware, {np.sum(y_train==0):,} benign)")
    print(f"   Validation samples: {len(y_val):,}")
    print(f"   Test samples: {len(y_test):,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Malware ratio: {np.mean(y_train):.1%}")
