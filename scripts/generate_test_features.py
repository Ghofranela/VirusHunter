#!/usr/bin/env python3
"""
Generate Test Feature Files for VirusHunter
Creates .npy files with correct 2381 features
"""
import numpy as np
from pathlib import Path
import sys

def generate_malware_sample(output_file="test_malware.npy"):
    """Generate a malware-like feature vector"""
    # Create features with higher values (malware-like)
    features = np.random.randn(2381) + 2.5
    
    # Add some specific patterns
    features[0:50] = np.abs(np.random.randn(50) * 3)  # High metadata values
    features[1000:1100] = np.random.uniform(0.8, 1.0, 100)  # PE indicators
    features[1300:1350] = np.random.randint(5, 20, 50)  # Suspicious keywords
    
    # Reshape to (1, 2381) for single sample
    features = features.reshape(1, -1)
    
    # Save
    np.save(output_file, features)
    print(f"✅ Generated malware sample: {output_file}")
    print(f"   Shape: {features.shape}")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")
    
    return features

def generate_benign_sample(output_file="test_benign.npy"):
    """Generate a benign-like feature vector"""
    # Create features with lower values (benign-like)
    features = np.random.randn(2381) - 1.5
    
    # Add some specific patterns
    features[0:50] = np.abs(np.random.randn(50) * 0.5)  # Normal metadata
    features[1000:1100] = np.random.uniform(0.0, 0.3, 100)  # Low PE indicators
    features[1300:1350] = np.zeros(50)  # No suspicious keywords
    
    # Reshape to (1, 2381)
    features = features.reshape(1, -1)
    
    # Save
    np.save(output_file, features)
    print(f"✅ Generated benign sample: {output_file}")
    print(f"   Shape: {features.shape}")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")
    
    return features

def generate_batch_samples(output_file="test_batch.npy", n_samples=10):
    """Generate multiple samples"""
    features_list = []
    
    for i in range(n_samples):
        if i % 2 == 0:
            # Malware
            features = np.random.randn(2381) + 2.0
        else:
            # Benign
            features = np.random.randn(2381) - 1.0
        
        features_list.append(features)
    
    # Stack into batch
    batch = np.vstack(features_list)
    
    # Save
    np.save(output_file, batch)
    print(f"✅ Generated batch: {output_file}")
    print(f"   Shape: {batch.shape}")
    print(f"   Samples: {n_samples}")
    
    return batch

def validate_npy_file(file_path):
    """Validate a .npy file"""
    try:
        data = np.load(file_path)
        print(f"\n📊 Validation: {file_path}")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        print(f"   Min: {data.min():.4f}")
        print(f"   Max: {data.max():.4f}")
        print(f"   Mean: {data.mean():.4f}")
        
        if len(data.shape) == 2 and data.shape[1] == 2381:
            print(f"   ✅ Valid format!")
        elif len(data.shape) == 1 and data.shape[0] == 2381:
            print(f"   ⚠️  1D array - will be reshaped to (1, 2381)")
        else:
            print(f"   ❌ Invalid shape! Expected (n, 2381)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("VirusHunter - Test Feature Generator")
    print("="*60)
    
    # Create output directory
    output_dir = Path("test_features")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}\n")
    
    # Generate samples
    print("🔬 Generating test samples...\n")
    
    malware_file = output_dir / "malware_sample.npy"
    benign_file = output_dir / "benign_sample.npy"
    batch_file = output_dir / "batch_samples.npy"
    
    generate_malware_sample(malware_file)
    print()
    
    generate_benign_sample(benign_file)
    print()
    
    generate_batch_samples(batch_file, n_samples=10)
    print()
    
    # Validate all files
    print("\n" + "="*60)
    print("Validating generated files...")
    print("="*60)
    
    validate_npy_file(malware_file)
    validate_npy_file(benign_file)
    validate_npy_file(batch_file)
    
    print("\n" + "="*60)
    print("✅ Generation complete!")
    print("="*60)
    print("\nYou can now upload these files to VirusHunter:")
    print(f"  • {malware_file}")
    print(f"  • {benign_file}")
    print(f"  • {batch_file}")
    print("\nOr use them in scripts:")
    print(f"  features = np.load('{malware_file}')")

if __name__ == "__main__":
    main()