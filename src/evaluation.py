"""
Evaluation Module for Malware Detection
Comprehensive metrics, robustness testing, and comparative analysis
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions
        
        Args:
            X: Input features
            batch_size: Batch size for inference
        
        Returns:
            Probabilities, predictions
        """
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
        
        all_probs = np.concatenate(all_probs)
        predictions = (all_probs > 0.5).astype(int)
        
        return all_probs, predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Complete evaluation with all metrics
        
        Args:
            X: Features
            y: True labels
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        print("Evaluating model...")
        
        # Get predictions
        probs, preds = self.predict(X)
        
        # Adjust predictions based on threshold
        preds = (probs > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds, zero_division=0),
            'recall': recall_score(y, preds, zero_division=0),
            'f1_score': f1_score(y, preds, zero_division=0),
            'roc_auc': roc_auc_score(y, probs),
            'average_precision': average_precision_score(y, probs),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted way"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1-Score:           {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Average Precision:  {metrics['average_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:     {metrics['true_negatives']}")
        print(f"  False Positives:    {metrics['false_positives']}")
        print(f"  False Negatives:    {metrics['false_negatives']}")
        print(f"  True Positives:     {metrics['true_positives']}")
        
        print(f"\nAdditional Metrics:")
        print(f"  Specificity:        {metrics['specificity']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'],
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: str = None
    ):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.4f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class RobustnessEvaluator:
    """
    Evaluate model robustness against adversarial attacks
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize robustness evaluator
        
        Args:
            model: PyTorch model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def fgsm_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Fast Gradient Sign Method attack
        
        Args:
            X: Input features
            y: True labels
            epsilon: Perturbation magnitude
        
        Returns:
            Adversarial examples, success rate
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        X_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = nn.BCEWithLogitsLoss()(outputs, y_tensor)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        perturbation = epsilon * X_tensor.grad.sign()
        X_adv = X_tensor + perturbation
        X_adv = X_adv.detach().cpu().numpy()
        
        # Evaluate attack success
        with torch.no_grad():
            X_adv_tensor = torch.FloatTensor(X_adv).to(self.device)
            outputs_adv = self.model(X_adv_tensor)
            preds_adv = (torch.sigmoid(outputs_adv) > 0.5).float()
            
            # Success: prediction changed
            success_rate = (preds_adv != y_tensor).float().mean().item()
        
        return X_adv, success_rate
    
    def evaluate_robustness(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilons: List[float] = None
    ) -> Dict:
        """
        Evaluate robustness across different epsilon values
        
        Args:
            X_test: Test features
            y_test: Test labels
            epsilons: List of perturbation magnitudes
        
        Returns:
            Robustness metrics
        """
        if epsilons is None:
            epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        print("Evaluating robustness against FGSM attacks...")
        
        results = {
            'epsilons': epsilons,
            'clean_accuracy': 0.0,
            'adversarial_accuracy': [],
            'attack_success_rates': []
        }
        
        # Clean accuracy
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_tensor)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            results['clean_accuracy'] = accuracy_score(y_test, preds)
        
        # Test each epsilon
        for epsilon in tqdm(epsilons, desc="Testing epsilon values"):
            X_adv, success_rate = self.fgsm_attack(X_test, y_test, epsilon)
            
            # Adversarial accuracy
            with torch.no_grad():
                X_adv_tensor = torch.FloatTensor(X_adv).to(self.device)
                outputs_adv = self.model(X_adv_tensor)
                preds_adv = (torch.sigmoid(outputs_adv) > 0.5).float().cpu().numpy()
                adv_acc = accuracy_score(y_test, preds_adv)
            
            results['adversarial_accuracy'].append(adv_acc)
            results['attack_success_rates'].append(success_rate)
        
        return results
    
    def plot_robustness(self, results: Dict, save_path: str = None):
        """Plot robustness results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epsilons = results['epsilons']
        
        # Accuracy vs Epsilon
        ax1.plot(epsilons, [results['clean_accuracy']] * len(epsilons), 
                 'k--', label='Clean Accuracy', linewidth=2)
        ax1.plot(epsilons, results['adversarial_accuracy'], 
                 'r-o', label='Adversarial Accuracy', linewidth=2)
        ax1.set_xlabel('Epsilon (Perturbation Magnitude)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Robustness vs Perturbation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # Attack Success Rate
        ax2.plot(epsilons, results['attack_success_rates'], 
                 'b-o', linewidth=2)
        ax2.set_xlabel('Epsilon (Perturbation Magnitude)', fontsize=12)
        ax2.set_ylabel('Attack Success Rate', fontsize=12)
        ax2.set_title('FGSM Attack Success Rate', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def comprehensive_evaluation(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cuda',
    save_dir: str = "results"
) -> Dict:
    """
    Complete evaluation pipeline
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to use
        save_dir: Directory to save results
    
    Returns:
        Complete evaluation metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard evaluation
    print("\n=== Standard Evaluation ===")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(X_test, y_test)
    evaluator.print_metrics(metrics)
    
    # Get predictions for plotting
    probs, preds = evaluator.predict(X_test)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        y_test, preds,
        save_path=save_dir / "confusion_matrix.png"
    )
    
    # Plot ROC curve
    evaluator.plot_roc_curve(
        y_test, probs,
        save_path=save_dir / "roc_curve.png"
    )
    
    # Plot PR curve
    evaluator.plot_precision_recall_curve(
        y_test, probs,
        save_path=save_dir / "precision_recall_curve.png"
    )
    
    # Robustness evaluation
    print("\n=== Robustness Evaluation ===")
    robustness_evaluator = RobustnessEvaluator(model, device)
    
    # Sample subset for robustness testing
    n_samples = min(1000, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    robustness_results = robustness_evaluator.evaluate_robustness(
        X_test[indices],
        y_test[indices]
    )
    
    # Plot robustness
    robustness_evaluator.plot_robustness(
        robustness_results,
        save_path=save_dir / "robustness_analysis.png"
    )
    
    # Combine results
    complete_results = {
        'standard_metrics': metrics,
        'robustness': robustness_results
    }
    
    # Save results
    with open(save_dir / "evaluation_results.json", 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\nAll results saved to {save_dir}")
    
    return complete_results


if __name__ == "__main__":
    # Test evaluation
    from model import MalwareDetector
    
    print("Testing Evaluation Module...")
    
    # Create dummy model and data
    model = MalwareDetector(input_size=2381)
    model.eval()
    
    X_test = np.random.randn(1000, 2381)
    y_test = np.random.randint(0, 2, 1000)
    
    # Evaluate
    results = comprehensive_evaluation(
        model, X_test, y_test,
        device='cpu',
        save_dir='test_results'
    )
    
    print("\nEvaluation module tested successfully!")
