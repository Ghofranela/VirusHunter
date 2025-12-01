#!/usr/bin/env python3
"""
Test preprocessing directly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import prepare_malware_dataset

if __name__ == "__main__":
    print("Testing preprocessing...")
    
    try:
        data = prepare_malware_dataset(
            data_dir="data/ember",
            output_dir="data/processed",
            test_size=0.2,
            augment=True,
            scaler_type='standard'
        )
        
        print("Preprocessing successful!")
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        print(f"Test samples: {len(data['X_test'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()