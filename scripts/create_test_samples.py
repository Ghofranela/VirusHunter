#!/usr/bin/env python3
"""
Create Test Samples for VirusHunter
Generates synthetic malware and benign samples for testing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler


def create_malware_sample(feature_dim=2381, normalize=True):
    """
    Create synthetic malware-like sample
    
    Characteristics:
    - Higher entropy values
    - Suspicious PE features
    - Higher standard deviation
    """
    sample = np.zeros(feature_dim)
    
    # Basic features (0-50): High values indicating complexity
    sample[0] = np.random.randint(50000, 500000)  # File size
    sample[1] = np.random.randint(200, 256)  # Unique bytes
    sample[2] = np.random.uniform(0.01, 0.1)  # Null byte ratio
    sample[3] = np.random.uniform(100, 200)  # Avg byte value
    sample[4] = np.random.uniform(6.0, 8.0)  # High entropy
    
    # Byte frequency features (50-306): Diverse distribution
    byte_freq = np.random.dirichlet(np.ones(256)) * np.random.uniform(0.8, 1.2, 256)
    sample[50:306] = byte_freq / byte_freq.sum()
    
    # N-gram features (306-1000): Higher values
    sample[306:1000] = np.random.exponential(0.01, 694)
    
    # PE features (1000-1300): Suspicious indicators
    sample[1000] = 1  # Is PE
    sample[1001] = 1  # Valid PE signature
    sample[1002] = np.random.randint(5, 20)  # Many sections
    sample[1010:1030] = np.random.uniform(0.5, 1.0, 20)  # Suspicious imports
    
    # Script features (1300-1400): Suspicious keywords
    sample[1300:1350] = np.random.poisson(2, 50)  # Obfuscation indicators
    
    # Statistical features (1881-1900)
    sample[1881] = np.random.uniform(120, 150)  # Mean
    sample[1882] = np.random.uniform(60, 90)  # Std
    sample[1883] = np.random.uniform(100, 140)  # Median
    sample[1884] = 0  # Min
    sample[1885] = 255  # Max
    sample[1886] = np.random.randint(50, 200)  # Max run
    
    # Add random noise to rest
    mask = sample == 0
    sample[mask] = np.random.randn(mask.sum()) * 0.1
    
    if normalize:
        # Apply standard scaling similar to training
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample.reshape(1, -1))
        sample = sample.flatten()
    
    return sample


def create_benign_sample(feature_dim=2381, normalize=True):
    """
    Create synthetic benign sample
    
    Characteristics:
    - Lower entropy
    - Normal PE features
    - Lower complexity
    """
    sample = np.zeros(feature_dim)
    
    # Basic features (0-50): Normal values
    sample[0] = np.random.randint(10000, 100000)  # File size
    sample[1] = np.random.randint(100, 180)  # Unique bytes
    sample[2] = np.random.uniform(0.1, 0.3)  # Null byte ratio
    sample[3] = np.random.uniform(60, 100)  # Avg byte value
    sample[4] = np.random.uniform(3.0, 5.5)  # Normal entropy
    
    # Byte frequency features (50-306): Normal distribution
    byte_freq = np.random.dirichlet(np.ones(256)) * np.random.uniform(0.5, 0.8, 256)
    sample[50:306] = byte_freq / byte_freq.sum()
    
    # N-gram features (306-1000): Lower values
    sample[306:1000] = np.random.exponential(0.005, 694)
    
    # PE features (1000-1300): Normal indicators
    sample[1000] = 1  # Is PE
    sample[1001] = 1  # Valid PE signature
    sample[1002] = np.random.randint(3, 8)  # Normal sections
    sample[1010:1030] = np.random.uniform(0.1, 0.4, 20)  # Normal imports
    
    # Script features (1300-1400): Low suspicious activity
    sample[1300:1350] = np.random.poisson(0.5, 50)
    
    # Statistical features (1881-1900)
    sample[1881] = np.random.uniform(80, 110)  # Mean
    sample[1882] = np.random.uniform(30, 50)  # Std
    sample[1883] = np.random.uniform(70, 100)  # Median
    sample[1884] = 0  # Min
    sample[1885] = 255  # Max
    sample[1886] = np.random.randint(10, 50)  # Max run
    
    # Add random noise to rest
    mask = sample == 0
    sample[mask] = np.random.randn(mask.sum()) * 0.05
    
    if normalize:
        # Apply standard scaling
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample.reshape(1, -1))
        sample = sample.flatten()
    
    return sample


def create_ambiguous_sample(feature_dim=2381, normalize=True):
    """
    Create ambiguous sample (between malware and benign)
    Useful for testing edge cases
    """
    sample = np.zeros(feature_dim)
    
    # Mix of malware and benign characteristics
    sample[0] = np.random.randint(20000, 200000)
    sample[1] = np.random.randint(150, 220)
    sample[2] = np.random.uniform(0.05, 0.2)
    sample[3] = np.random.uniform(80, 130)
    sample[4] = np.random.uniform(4.5, 6.5)  # Medium entropy
    
    # Byte frequency
    byte_freq = np.random.dirichlet(np.ones(256))
    sample[50:306] = byte_freq
    
    # Mixed features
    sample[306:1000] = np.random.exponential(0.007, 694)
    sample[1000:1300] = np.random.uniform(0.2, 0.7, 300)
    sample[1300:1400] = np.random.poisson(1, 100)
    
    # Statistical features
    sample[1881:1890] = np.random.uniform(50, 150, 9)
    
    # Noise
    mask = sample == 0
    sample[mask] = np.random.randn(mask.sum()) * 0.08
    
    if normalize:
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample.reshape(1, -1))
        sample = sample.flatten()
    
    return sample


def main():
    parser = argparse.ArgumentParser(description='Create test samples')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_samples',
        help='Output directory for test samples'
    )
    parser.add_argument(
        '--num-malware',
        type=int,
        default=5,
        help='Number of malware samples to create'
    )
    parser.add_argument(
        '--num-benign',
        type=int,
        default=5,
        help='Number of benign samples to create'
    )
    parser.add_argument(
        '--num-ambiguous',
        type=int,
        default=3,
        help='Number of ambiguous samples to create'
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=2381,
        help='Feature dimension'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Test Samples")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Malware samples:  {args.num_malware}")
    print(f"Benign samples:   {args.num_benign}")
    print(f"Ambiguous samples: {args.num_ambiguous}")
    print(f"Feature dimension: {args.feature_dim}")
    print("="*60 + "\n")
    
    # Create malware samples
    print(f"Creating {args.num_malware} malware samples...")
    for i in range(1, args.num_malware + 1):
        sample = create_malware_sample(args.feature_dim)
        filepath = output_dir / f"malware_sample_{i}.npy"
        np.save(filepath, sample)
        print(f"  ✓ Saved: {filepath}")
    
    # Create benign samples
    print(f"\nCreating {args.num_benign} benign samples...")
    for i in range(1, args.num_benign + 1):
        sample = create_benign_sample(args.feature_dim)
        filepath = output_dir / f"benign_sample_{i}.npy"
        np.save(filepath, sample)
        print(f"  ✓ Saved: {filepath}")
    
    # Create ambiguous samples
    print(f"\nCreating {args.num_ambiguous} ambiguous samples...")
    for i in range(1, args.num_ambiguous + 1):
        sample = create_ambiguous_sample(args.feature_dim)
        filepath = output_dir / f"ambiguous_sample_{i}.npy"
        np.save(filepath, sample)
        print(f"  ✓ Saved: {filepath}")
    
    print("\n" + "="*60)
    print("Test Samples Created Successfully!")
    print("="*60)
    print(f"Total samples: {args.num_malware + args.num_benign + args.num_ambiguous}")
    print(f"Location: {output_dir.absolute()}")
    print("\nYou can now test the application by uploading these .npy files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()