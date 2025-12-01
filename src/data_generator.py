"""
Synthetic Data Generator for Malware Detection
Creates realistic EMBER-like dataset when original is unavailable
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class MalwareDataGenerator:
    """
    Generate synthetic malware detection dataset
    Mimics EMBER dataset structure and characteristics
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.feature_dim = 2381  # EMBER feature dimension

    def generate_pe_features(self, n_samples: int, malware_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic PE file features

        Args:
            n_samples: Number of samples to generate
            malware_ratio: Ratio of malware samples

        Returns:
            Features and labels
        """
        n_malware = int(n_samples * malware_ratio)
        n_benign = n_samples - n_malware

        # Initialize feature arrays
        features = np.zeros((n_samples, self.feature_dim))
        labels = np.zeros(n_samples, dtype=int)

        # Generate benign samples
        for i in range(n_benign):
            features[i] = self._generate_benign_sample()
            labels[i] = 0

        # Generate malware samples
        for i in range(n_benign, n_samples):
            features[i] = self._generate_malware_sample()
            labels[i] = 1

        return features, labels

    def _generate_benign_sample(self) -> np.ndarray:
        """Generate a benign PE file feature vector"""
        features = np.zeros(self.feature_dim)

        # Basic file features (0-49)
        file_size = np.random.randint(10000, 1000000)  # 10KB to 1MB
        features[0] = file_size
        features[1] = np.random.randint(100, 1000)  # Unique bytes
        features[2] = np.random.uniform(0.1, 0.3)  # Null byte ratio
        features[3] = np.random.uniform(100, 150)  # Average byte value
        features[4] = np.random.uniform(6.5, 7.5)  # Entropy (benign files less random)

        # Byte frequency features (50-305)
        # Benign files have more structured byte patterns
        base_freq = np.random.uniform(0.001, 0.01, 256)
        # Boost printable ASCII characters
        base_freq[32:127] *= np.random.uniform(2, 5)
        base_freq /= base_freq.sum()  # Normalize
        features[50:306] = base_freq

        # N-gram features (306-1000) - benign files have common patterns
        features[306:400] = np.random.uniform(0.001, 0.01, 94)

        # PE header features (1000-1100)
        features[1000] = 1  # PE file indicator
        features[1001] = 1  # Valid PE signature
        features[1002] = np.random.randint(3, 8)  # Number of sections

        # Statistical features (1881-1920)
        features[1881] = np.random.uniform(100, 140)  # Mean byte value
        features[1882] = np.random.uniform(20, 40)  # Std byte value
        features[1883] = np.random.uniform(95, 135)  # Median byte value
        features[1884] = np.random.uniform(0, 50)  # Min byte value
        features[1885] = np.random.uniform(200, 255)  # Max byte value
        features[1886] = np.random.randint(1, 10)  # Max run length

        return features

    def _generate_malware_sample(self) -> np.ndarray:
        """Generate a malware PE file feature vector"""
        features = np.zeros(self.feature_dim)

        # Basic file features (0-49) - malware often smaller but more packed
        file_size = np.random.randint(5000, 500000)  # 5KB to 500KB
        features[0] = file_size
        features[1] = np.random.randint(50, 500)  # Unique bytes (often fewer)
        features[2] = np.random.uniform(0.3, 0.7)  # Null byte ratio (higher in packed files)
        features[3] = np.random.uniform(80, 120)  # Average byte value
        features[4] = np.random.uniform(7.0, 7.8)  # Entropy (more random)

        # Byte frequency features (50-305)
        # Malware often has unusual byte distributions
        base_freq = np.random.uniform(0.001, 0.01, 256)
        # Less emphasis on printable ASCII, more on control chars
        base_freq[0:32] *= np.random.uniform(1.5, 3)
        base_freq[32:127] *= np.random.uniform(0.5, 1.5)
        base_freq /= base_freq.sum()
        features[50:306] = base_freq

        # N-gram features (306-1000) - malware has suspicious patterns
        features[306:400] = np.random.uniform(0.01, 0.05, 94)  # Higher frequency

        # PE header features (1000-1100)
        features[1000] = 1  # PE file indicator
        features[1001] = np.random.choice([0, 1], p=[0.3, 0.7])  # Sometimes invalid signatures
        features[1002] = np.random.randint(1, 15)  # Variable sections, often more

        # Add some suspicious features
        features[1300] = np.random.uniform(0.1, 0.8)  # Suspicious keywords
        features[1350] = np.random.uniform(0.1, 0.5)  # Hex encoding
        features[1351] = np.random.uniform(0.1, 0.4)  # Non-ASCII chars

        # Statistical features (1881-1920) - more varied
        features[1881] = np.random.uniform(80, 130)  # Mean byte value
        features[1882] = np.random.uniform(30, 60)  # Std byte value (higher variance)
        features[1883] = np.random.uniform(70, 140)  # Median byte value
        features[1884] = np.random.uniform(0, 30)  # Min byte value
        features[1885] = np.random.uniform(180, 255)  # Max byte value
        features[1886] = np.random.randint(5, 50)  # Max run length (longer runs in packed files)

        return features

    def add_noise_and_artifacts(self, features: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add realistic noise and artifacts to features"""
        noise = np.random.normal(0, noise_level, features.shape)
        features_noisy = features + noise

        # Ensure non-negative values for some features
        features_noisy[:, 0] = np.maximum(0, features_noisy[:, 0])  # File size
        features_noisy[:, 1] = np.maximum(0, features_noisy[:, 1])  # Unique bytes

        # Clip byte frequencies to [0, 1]
        features_noisy[:, 50:306] = np.clip(features_noisy[:, 50:306], 0, 1)
        # Renormalize byte frequencies
        row_sums = features_noisy[:, 50:306].sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        features_noisy[:, 50:306] /= row_sums

        return features_noisy

    def generate_dataset(
        self,
        n_samples: int = 10000,
        malware_ratio: float = 0.5,
        test_size: float = 0.2,
        val_size: float = 0.2,
        noise_level: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete train/val/test dataset

        Args:
            n_samples: Total number of samples
            malware_ratio: Ratio of malware samples
            test_size: Test set ratio
            val_size: Validation set ratio
            noise_level: Amount of noise to add

        Returns:
            Dictionary with train/val/test splits
        """
        print(f"Generating synthetic dataset with {n_samples} samples...")

        # Generate features and labels
        X, y = self.generate_pe_features(n_samples, malware_ratio)

        # Add noise and artifacts
        X = self.add_noise_and_artifacts(X, noise_level)

        # Split the data
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.seed
        )

        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=self.seed
        )

        print(f"Generated dataset:")
        print(f"  Train: {X_train.shape}, Malware: {y_train.sum()}/{len(y_train)} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Val: {X_val.shape}, Malware: {y_val.sum()}/{len(y_val)} ({y_val.sum()/len(y_val)*100:.1f}%)")
        print(f"  Test: {X_test.shape}, Malware: {y_test.sum()}/{len(y_test)} ({y_test.sum()/len(y_test)*100:.1f}%)")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def save_dataset(self, data: Dict[str, np.ndarray], output_dir: str = "data/processed"):
        """Save generated dataset to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving dataset to {output_dir}...")

        # Save numpy arrays
        np.save(output_dir / "X_train.npy", data['X_train'])
        np.save(output_dir / "y_train.npy", data['y_train'])
        np.save(output_dir / "X_val.npy", data['X_val'])
        np.save(output_dir / "y_val.npy", data['y_val'])
        np.save(output_dir / "X_test.npy", data['X_test'])
        np.save(output_dir / "y_test.npy", data['y_test'])

        # Fit and save scaler
        scaler = StandardScaler()
        scaler.fit(data['X_train'])
        joblib.dump(scaler, output_dir / "preprocessor.pkl")

        print("Dataset saved successfully!")

    def load_or_generate_dataset(
        self,
        output_dir: str = "data/processed",
        n_samples: int = 10000,
        force_regenerate: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load existing dataset or generate new one

        Args:
            output_dir: Directory to load/save data
            n_samples: Number of samples to generate if needed
            force_regenerate: Force regeneration even if data exists

        Returns:
            Dataset dictionary
        """
        output_dir = Path(output_dir)

        # Check if data already exists
        data_files = [
            output_dir / "X_train.npy",
            output_dir / "y_train.npy",
            output_dir / "X_val.npy",
            output_dir / "y_val.npy",
            output_dir / "X_test.npy",
            output_dir / "y_test.npy"
        ]

        if not force_regenerate and all(f.exists() for f in data_files):
            print("Loading existing dataset...")
            try:
                data = {
                    'X_train': np.load(output_dir / "X_train.npy"),
                    'y_train': np.load(output_dir / "y_train.npy"),
                    'X_val': np.load(output_dir / "X_val.npy"),
                    'y_val': np.load(output_dir / "y_val.npy"),
                    'X_test': np.load(output_dir / "X_test.npy"),
                    'y_test': np.load(output_dir / "y_test.npy")
                }
                print(f"Loaded dataset: Train {data['X_train'].shape}, Val {data['X_val'].shape}, Test {data['X_test'].shape}")
                return data
            except Exception as e:
                print(f"Error loading existing data: {e}")
                print("Regenerating dataset...")

        # Generate new dataset
        data = self.generate_dataset(n_samples)
        self.save_dataset(data, output_dir)

        return data


def create_test_samples(output_dir: str = "test_samples", n_samples: int = 10):
    """Create test samples for quick testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MalwareDataGenerator(seed=42)

    # Generate samples
    X, y = generator.generate_pe_features(n_samples, malware_ratio=0.5)

    # Save individual samples
    for i in range(n_samples):
        sample_type = "malware" if y[i] == 1 else "benign"
        filename = f"{sample_type}_sample_{i+1}.npy"
        np.save(output_dir / filename, X[i:i+1])  # Save as (1, 2381)

    # Create some ambiguous samples
    for i in range(3):
        # Mix benign and malware features
        benign = generator._generate_benign_sample()
        malware = generator._generate_malware_sample()
        ambiguous = 0.5 * benign + 0.5 * malware + np.random.normal(0, 0.05, generator.feature_dim)
        np.save(output_dir / f"ambiguous_sample_{i+1}.npy", ambiguous.reshape(1, -1))

    print(f"Created {n_samples + 3} test samples in {output_dir}")


if __name__ == "__main__":
    # Test the generator
    print("Testing MalwareDataGenerator...")

    generator = MalwareDataGenerator()

    # Generate small dataset for testing
    data = generator.generate_dataset(n_samples=1000, malware_ratio=0.4)

    print(f"Generated {len(data['X_train'])} training samples")
    print(f"Feature dimension: {data['X_train'].shape[1]}")
    print(f"Malware ratio in train: {data['y_train'].mean():.3f}")

    # Create test samples
    create_test_samples()

    print("✓ Data generator tested successfully!")
