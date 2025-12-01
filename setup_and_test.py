#!/usr/bin/env python3
"""
Setup and test the entire VirusHunter system
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/ember",
        "models",
        "results/evaluations",
        "results/explanations",
        "logs",
        "test_samples"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print("✓ Directory structure created")

def test_imports():
    """Test that all modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from src import (
            MalwareDetector, create_model, prepare_malware_dataset,
            MalwareTrainer, calculate_metrics, setup_logging
        )
        print("✓ All main modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_generator():
    """Test the data generator"""
    print("\nTesting data generator...")
    
    try:
        from src.data_generator import MalwareDataGenerator
        
        generator = MalwareDataGenerator(seed=42)
        X, y = generator.generate_pe_features(100, malware_ratio=0.5)
        
        print(f"✓ Generated {len(X)} samples with {X.shape[1]} features")
        print(f"  Malware: {y.sum()}, Benign: {len(y) - y.sum()}")
        return True
    except Exception as e:
        print(f"✗ Data generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("VIRUSHUNTER - Setup and Test")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed!")
        return
    
    # Test data generator
    if not test_data_generator():
        print("❌ Data generator test failed!")
        return
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate synthetic data:")
    print("   python run_preprocessing.py")
    print("2. Train the model:")
    print("   python run_training.py")
    print("3. Run the application:")
    print("   streamlit run app/streamlit_complete.py")
    print("="*60)

if __name__ == "__main__":
    main()