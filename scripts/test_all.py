#!/usr/bin/env python3
"""
Comprehensive test suite for VirusHunter
Tests all modules and functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        from src import (
            MalwareDetector, CNNMalwareDetector, LSTMMalwareDetector,
            EnsembleDetector, EMBERPreprocessor, Trainer,
            IntegratedGradients, ModelEvaluator, setup_logging
        )
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*60)
    print("TEST 2: Model Creation")
    print("="*60)
    
    try:
        from src import create_model
        
        models = ['dnn', 'cnn', 'lstm', 'ensemble']
        
        for model_type in models:
            model = create_model(model_type, input_size=2381)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {model_type.upper()}: {params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through models"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    try:
        from src import create_model
        
        batch_size = 32
        input_size = 2381
        
        models = ['dnn', 'cnn', 'lstm', 'ensemble']
        
        for model_type in models:
            model = create_model(model_type, input_size=input_size)
            model.eval()
            
            if model_type == 'cnn':
                x = torch.randn(batch_size, 1, input_size)
            else:
                x = torch.randn(batch_size, input_size)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[0] == batch_size
            print(f"✓ {model_type.upper()}: Input {tuple(x.shape)} → Output {tuple(output.shape)}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing functionality"""
    print("\n" + "="*60)
    print("TEST 4: Preprocessing")
    print("="*60)
    
    try:
        from src import EMBERPreprocessor
        from sklearn.preprocessing import StandardScaler
        
        # Create dummy data
        X = np.random.randn(1000, 2381)
        y = np.random.randint(0, 2, 1000)
        
        # Test scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✓ Scaling: Mean={X_scaled.mean():.4f}, Std={X_scaled.std():.4f}")
        
        # Test preprocessing
        preprocessor = EMBERPreprocessor('data/ember')
        X_clean, y_clean = preprocessor.preprocess_features(X, y, remove_unlabeled=False)
        
        print(f"✓ Preprocessing: {X_clean.shape[0]} samples")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False


def test_training_components():
    """Test training components"""
    print("\n" + "="*60)
    print("TEST 5: Training Components")
    print("="*60)
    
    try:
        from src import MalwareDataset
        from torch.utils.data import DataLoader
        
        # Create dummy data
        X = np.random.randn(100, 2381)
        y = np.random.randint(0, 2, 100)
        
        # Test dataset
        dataset = MalwareDataset(X, y)
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        X_batch, y_batch = next(iter(loader))
        
        print(f"✓ DataLoader: Batch shape {tuple(X_batch.shape)}")
        
        return True
    except Exception as e:
        print(f"✗ Training components failed: {e}")
        return False


def test_explainability():
    """Test explainability methods"""
    print("\n" + "="*60)
    print("TEST 6: Explainability")
    print("="*60)
    
    try:
        from src import IntegratedGradients, create_model
        
        # Create model
        model = create_model('dnn', input_size=2381)
        model.eval()
        
        # Test Integrated Gradients
        ig = IntegratedGradients(model, device='cpu')
        features = np.random.randn(2381)
        
        explanation = ig.explain_prediction(features, top_k=5)
        
        print(f"✓ Integrated Gradients: {len(explanation['top_features'])} features")
        print(f"  Prediction: {explanation['prediction']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Explainability failed: {e}")
        return False


def test_evaluation():
    """Test evaluation functionality"""
    print("\n" + "="*60)
    print("TEST 7: Evaluation")
    print("="*60)
    
    try:
        from src import ModelEvaluator, create_model
        
        # Create model
        model = create_model('dnn', input_size=2381)
        model.eval()
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device='cpu')
        
        # Test predictions
        X = np.random.randn(100, 2381)
        probs, preds = evaluator.predict(X, batch_size=32)
        
        print(f"✓ Predictions: {len(preds)} samples")
        print(f"  Malware detected: {preds.sum()}/{len(preds)}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("TEST 8: CUDA Support")
    print("="*60)
    
    try:
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available (CPU only)")
        
        return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def test_utilities():
    """Test utility functions"""
    print("\n" + "="*60)
    print("TEST 9: Utilities")
    print("="*60)
    
    try:
        from src import set_seed, get_device, count_parameters, create_model
        
        # Test seed
        set_seed(42)
        print("✓ Random seed set")
        
        # Test device
        device = get_device()
        print(f"✓ Device selected: {device}")
        
        # Test parameter counting
        model = create_model('dnn', input_size=2381)
        params = count_parameters(model)
        print(f"✓ Parameter counting: {params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def test_file_structure():
    """Test project file structure"""
    print("\n" + "="*60)
    print("TEST 10: File Structure")
    print("="*60)
    
    required_files = [
        'src/__init__.py',
        'src/model.py',
        'src/preprocessing.py',
        'src/training.py',
        'src/explainability.py',
        'src/evaluation.py',
        'src/utils.py',
        'main.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_present = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_present = False
    
    return all_present


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("VIRUSHUNTER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Preprocessing", test_preprocessing),
        ("Training Components", test_training_components),
        ("Explainability", test_explainability),
        ("Evaluation", test_evaluation),
        ("CUDA Support", test_cuda),
        ("Utilities", test_utilities),
        ("File Structure", test_file_structure)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<10} {name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✓ ALL TESTS PASSED! System is ready.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check errors above.")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
