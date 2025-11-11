"""
EMBER Dataset Preprocessing and Feature Engineering
Handles static PE features, dynamic behavior, and raw bytes
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
import hashlib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import ember


class EMBERPreprocessor:
    """
    Complete preprocessing pipeline for EMBER dataset
    """
    def __init__(self, data_dir: str = "data/ember", version: int = 2):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Directory containing EMBER data
            version: EMBER dataset version (1 or 2)
        """
        self.data_dir = Path(data_dir)
        self.version = version
        self.scaler = None
        self.feature_names = None
        
    def download_ember(self):
        """Download EMBER dataset"""
        print(f"Downloading EMBER v{self.version}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if self.version == 2:
            ember.create_vectorized_features(str(self.data_dir))
            ember.create_metadata(str(self.data_dir))
        
        print("Download complete!")
    
    def load_data(
        self,
        subset: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EMBER dataset
        
        Args:
            subset: 'train' or 'test'
        
        Returns:
            X: Features array
            y: Labels array
        """
        print(f"Loading {subset} data...")
        
        X, y = ember.read_vectorized_features(
            str(self.data_dir),
            subset=subset
        )
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def preprocess_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        remove_unlabeled: bool = True,
        handle_missing: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features
        
        Args:
            X: Raw features
            y: Labels (-1: unlabeled, 0: benign, 1: malware)
            remove_unlabeled: Remove unlabeled samples
            handle_missing: Strategy for missing values
        
        Returns:
            X_processed, y_processed
        """
        print("Preprocessing features...")
        
        # Remove unlabeled samples
        if remove_unlabeled:
            mask = y != -1
            X = X[mask]
            y = y[mask]
            print(f"After removing unlabeled: {len(X)} samples")
        
        # Handle missing values
        if handle_missing == 'mean':
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
        elif handle_missing == 'zero':
            X = np.nan_to_num(X, nan=0.0)
        
        # Handle inf values
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        
        print(f"Processed: {X.shape}, Labels: {np.bincount(y.astype(int))}")
        return X, y
    
    def engineer_features(
        self,
        X: np.ndarray,
        method: str = 'statistics'
    ) -> np.ndarray:
        """
        Engineer additional features
        
        Args:
            X: Input features
            method: Feature engineering method
        
        Returns:
            Enhanced feature matrix
        """
        if method == 'statistics':
            # Add statistical features
            features_list = [X]
            
            # Row-wise statistics
            features_list.append(np.mean(X, axis=1, keepdims=True))
            features_list.append(np.std(X, axis=1, keepdims=True))
            features_list.append(np.min(X, axis=1, keepdims=True))
            features_list.append(np.max(X, axis=1, keepdims=True))
            
            X_enhanced = np.concatenate(features_list, axis=1)
            return X_enhanced
        
        return X
    
    def fit_scaler(
        self,
        X: np.ndarray,
        scaler_type: str = 'standard'
    ):
        """
        Fit scaler on training data
        
        Args:
            X: Training features
            scaler_type: 'standard' or 'robust'
        """
        print(f"Fitting {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler.fit(X)
        print("Scaler fitted!")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler transformation"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(
        self,
        X: np.ndarray,
        scaler_type: str = 'standard'
    ) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit_scaler(X, scaler_type)
        return self.transform(X)
    
    def save_preprocessor(self, save_path: str):
        """Save scaler and metadata"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, save_path)
        print(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """Load scaler and metadata"""
        self.scaler = joblib.load(load_path)
        print(f"Preprocessor loaded from {load_path}")


class AdversarialAugmenter:
    """
    Generate adversarial samples for training robustness
    """
    def __init__(self, noise_level: float = 0.01):
        """
        Initialize augmenter
        
        Args:
            noise_level: Magnitude of perturbations
        """
        self.noise_level = noise_level
    
    def add_gaussian_noise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise to features
        
        Args:
            X: Features
            y: Labels
            ratio: Proportion of samples to augment
        
        Returns:
            Augmented X, y
        """
        n_samples = int(len(X) * ratio)
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        X_noise = X[indices].copy()
        y_noise = y[indices].copy()
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, X_noise.shape)
        X_noise = X_noise + noise
        
        # Concatenate
        X_aug = np.vstack([X, X_noise])
        y_aug = np.concatenate([y, y_noise])
        
        return X_aug, y_aug
    
    def feature_masking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_prob: float = 0.1,
        ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random feature masking
        
        Args:
            X: Features
            y: Labels
            mask_prob: Probability of masking each feature
            ratio: Proportion of samples to augment
        """
        n_samples = int(len(X) * ratio)
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        X_masked = X[indices].copy()
        y_masked = y[indices].copy()
        
        # Random masking
        mask = np.random.random(X_masked.shape) < mask_prob
        X_masked[mask] = 0
        
        X_aug = np.vstack([X, X_masked])
        y_aug = np.concatenate([y, y_masked])
        
        return X_aug, y_aug
    
    def mimicry_attack_simulation(
        self,
        X_malware: np.ndarray,
        X_benign: np.ndarray,
        alpha: float = 0.1
    ) -> np.ndarray:
        """
        Simulate mimicry attacks: blend malware with benign features
        
        Args:
            X_malware: Malware samples
            X_benign: Benign samples
            alpha: Blending factor
        
        Returns:
            Adversarial malware samples
        """
        n_samples = min(len(X_malware), len(X_benign))
        
        X_mal_subset = X_malware[:n_samples]
        X_ben_subset = X_benign[:n_samples]
        
        # Blend features
        X_adv = (1 - alpha) * X_mal_subset + alpha * X_ben_subset
        
        return X_adv


class FeatureAnalyzer:
    """
    Analyze and select important features
    """
    def __init__(self):
        self.feature_importance = None
        self.selected_features = None
    
    def calculate_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'variance'
    ) -> np.ndarray:
        """
        Calculate feature importance
        
        Args:
            X: Features
            y: Labels
            method: 'variance', 'correlation', or 'mutual_info'
        
        Returns:
            Feature importance scores
        """
        if method == 'variance':
            # High variance features
            importance = np.var(X, axis=0)
        
        elif method == 'correlation':
            # Correlation with target
            importance = np.abs([
                np.corrcoef(X[:, i], y)[0, 1]
                for i in range(X.shape[1])
            ])
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            importance = mutual_info_classif(X, y)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.feature_importance = importance
        return importance
    
    def select_features(
        self,
        X: np.ndarray,
        top_k: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-k features
        
        Args:
            X: Features
            top_k: Number of features to select
        
        Returns:
            Selected features, indices
        """
        if self.feature_importance is None:
            raise ValueError("Calculate importance first")
        
        indices = np.argsort(self.feature_importance)[::-1][:top_k]
        self.selected_features = indices
        
        return X[:, indices], indices


def prepare_ember_dataset(
    data_dir: str = "data/ember",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    augment: bool = True
) -> Dict[str, np.ndarray]:
    """
    Complete pipeline to prepare EMBER dataset
    
    Args:
        data_dir: EMBER data directory
        output_dir: Output directory for processed data
        test_size: Validation split size
        augment: Apply adversarial augmentation
    
    Returns:
        Dictionary with train/val/test splits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = EMBERPreprocessor(data_dir)
    
    # Load training data
    print("\n=== Loading Training Data ===")
    X_train_raw, y_train = preprocessor.load_data("train")
    
    # Preprocess
    X_train, y_train = preprocessor.preprocess_features(X_train_raw, y_train)
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        stratify=y_train,
        random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Adversarial augmentation
    if augment:
        print("\n=== Adversarial Augmentation ===")
        augmenter = AdversarialAugmenter(noise_level=0.01)
        
        X_train, y_train = augmenter.add_gaussian_noise(
            X_train, y_train, ratio=0.1
        )
        print(f"After augmentation: {len(X_train)} samples")
    
    # Fit scaler on training data
    print("\n=== Fitting Scaler ===")
    X_train = preprocessor.fit_transform(X_train, scaler_type='standard')
    
    # Transform validation
    X_val = preprocessor.transform(X_val)
    
    # Save preprocessor
    preprocessor.save_preprocessor(output_dir / "preprocessor.pkl")
    
    # Load and preprocess test data
    print("\n=== Loading Test Data ===")
    X_test_raw, y_test = preprocessor.load_data("test")
    X_test, y_test = preprocessor.preprocess_features(X_test_raw, y_test)
    X_test = preprocessor.transform(X_test)
    
    # Save processed data
    print("\n=== Saving Processed Data ===")
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"\nData saved to {output_dir}")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == "__main__":
    # Prepare dataset
    data = prepare_ember_dataset(
        data_dir="data/ember",
        output_dir="data/processed",
        test_size=0.2,
        augment=True
    )
    
    print("\nDataset preparation complete!")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")
