#!/usr/bin/env python3
"""
Simple data generator for testing - No complex dependencies required
"""
import sys
import os
from pathlib import Path

def create_minimal_dataset():
    """Create a minimal dataset for testing"""
    print("Creating minimal dataset...")
    
    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results/evaluations").mkdir(parents=True, exist_ok=True)
    
    # Try to import numpy, if not available, create simple data
    try:
        import numpy as np
        print("✅ NumPy available")
        
        # Create synthetic data
        n_samples = 1000
        n_features = 2381
        
        # Generate benign samples (lower values)
        benign_features = np.random.normal(-1, 1, (n_samples//2, n_features))
        benign_labels = np.zeros(n_samples//2)
        
        # Generate malware samples (higher values)  
        malware_features = np.random.normal(1, 1, (n_samples//2, n_features))
        malware_labels = np.ones(n_samples//2)
        
        # Combine
        X = np.vstack([benign_features, malware_features])
        y = np.concatenate([benign_labels, malware_labels])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split data (80% train, 10% val, 10% test)
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Save data
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/X_val.npy", X_val)
        np.save("data/processed/y_val.npy", y_val)
        np.save("data/processed/X_test.npy", X_test)
        np.save("data/processed/y_test.npy", y_test)
        
        print(f"✅ Generated dataset with {len(X_train)} training samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
    except ImportError:
        print("❌ NumPy not available - creating simple text files instead")
        
        # Create simple text files with dataset info
        with open("data/processed/dataset_info.txt", "w") as f:
            f.write("Minimal Dataset Information\n")
            f.write("===========================\n")
            f.write("Training samples: 800\n")
            f.write("Validation samples: 100\n")
            f.write("Test samples: 100\n")
            f.write("Features: 2381\n")
            f.write("Classes: Benign (0), Malware (1)\n")
        
        print("✅ Created dataset info file")
        return None

def show_simple_split_analysis():
    """Show simple split analysis without complex dependencies"""
    print("\n" + "="*60)
    print("SIMPLE DATASET SPLIT ANALYSIS")
    print("="*60)
    
    try:
        import numpy as np
        
        # Load data
        X_train = np.load("data/processed/X_train.npy")
        y_train = np.load("data/processed/y_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        y_val = np.load("data/processed/y_val.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_test = np.load("data/processed/y_test.npy")
        
        splits = {
            'TRAIN': (X_train, y_train),
            'VALIDATION': (X_val, y_val),
            'TEST': (X_test, y_test)
        }
        
        print("\n📊 DATASET SPLIT STATISTICS")
        print("="*40)
        
        total_samples = 0
        for split_name, (X, y) in splits.items():
            n_samples = len(X)
            n_malware = int(y.sum())
            n_benign = len(y) - n_malware
            malware_ratio = n_malware / n_samples
            
            total_samples += n_samples
            
            print(f"\n{split_name}:")
            print(f"  📁 Samples:     {n_samples:,}")
            print(f"  🔧 Features:    {X.shape[1]:,}")
            print(f"  🦠 Malware:     {n_malware:,} ({malware_ratio:.1%})")
            print(f"  ✅ Benign:      {n_benign:,} ({(1-malware_ratio):.1%})")
        
        print(f"\n📋 SUMMARY")
        print("="*40)
        print(f"Total Samples: {total_samples:,}")
        print(f"Train/Val/Test Ratio: {len(X_train)/total_samples:.1%}/"
              f"{len(X_val)/total_samples:.1%}/{len(X_test)/total_samples:.1%}")
        
        # Create simple visualization using text
        print(f"\n📈 SPLIT VISUALIZATION")
        print("="*40)
        
        max_bar_length = 50
        for split_name, (X, y) in splits.items():
            n_samples = len(X)
            n_malware = int(y.sum())
            n_benign = len(y) - n_malware
            
            # Create ASCII bar chart
            total_bar = "█" * max_bar_length
            malware_bar = "█" * int((n_malware / n_samples) * max_bar_length)
            benign_bar = "█" * int((n_benign / n_samples) * max_bar_length)
            
            print(f"\n{split_name}:")
            print(f"  Total:    {total_bar} {n_samples:,}")
            print(f"  Malware:  {malware_bar} {n_malware:,}")
            print(f"  Benign:   {benign_bar} {n_benign:,}")
            
    except ImportError:
        print("NumPy not available - cannot load binary data")
        print("Please install NumPy: pip install numpy")
    except FileNotFoundError:
        print("Processed data not found - please run data generation first")

if __name__ == "__main__":
    # Create minimal dataset
    data = create_minimal_dataset()
    
    # Show split analysis
    show_simple_split_analysis()