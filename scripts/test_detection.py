#!/usr/bin/env python3
"""
Test malware detection on samples
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import joblib
from src import load_model


def test_detection(sample_path, model_path='models/best_model.pth', 
                   preprocessor_path='models/preprocessor.pkl'):
    """
    Test malware detection on a sample
    """
    print(f"Testing detection on: {sample_path}")
    
    # Load model and preprocessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(model_path, model_type='dnn', device=device)
    scaler = joblib.load(preprocessor_path)
    
    # Load sample
    features = np.load(sample_path)
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Preprocess
    features_scaled = scaler.transform(features)
    
    # Predict
    with torch.no_grad():
        X = torch.FloatTensor(features_scaled).to(device)
        output = model(X)
        prob = torch.sigmoid(output).item()
    
    # Results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"Malware Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Classification: {'MALWARE' if prob > 0.5 else 'BENIGN'}")
    print(f"Confidence: {abs(prob - 0.5) * 200:.2f}%")
    print("="*60)
    
    return prob


def test_directory(test_dir='test_samples'):
    """
    Test all samples in directory
    """
    test_dir = Path(test_dir)
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    results = []
    
    print(f"\nTesting all samples in: {test_dir}\n")
    
    for sample_file in sorted(test_dir.glob('*.npy')):
        prob = test_detection(str(sample_file))
        results.append({
            'file': sample_file.name,
            'probability': prob,
            'classification': 'MALWARE' if prob > 0.5 else 'BENIGN'
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for result in results:
        status = "🔴" if result['classification'] == 'MALWARE' else "🟢"
        print(f"{status} {result['file']:<30} {result['probability']:.4f} {result['classification']}")
    
    malware_count = sum(1 for r in results if r['classification'] == 'MALWARE')
    print(f"\nTotal: {len(results)} samples")
    print(f"Malware: {malware_count}")
    print(f"Benign: {len(results) - malware_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test malware detection')
    parser.add_argument('--sample', type=str, help='Single sample to test')
    parser.add_argument('--directory', type=str, default='test_samples', 
                       help='Directory of samples to test')
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.pkl')
    
    args = parser.parse_args()
    
    if args.sample:
        test_detection(args.sample, args.model, args.preprocessor)
    else:
        test_directory(args.directory)
