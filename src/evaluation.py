"""
Model Evaluation Module
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from pathlib import Path


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Malware'],
        yticklabels=['Benign', 'Malware'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def plot_roc_curve(y_true, y_proba, save_path=None, figsize=(8, 6)):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    return fpr, tpr, auc


def plot_precision_recall_curve(y_true, y_proba, save_path=None, figsize=(8, 6)):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
        figsize: Figure size
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()
    return precision, recall


def plot_training_history(history, save_path=None, figsize=(15, 5)):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # F1 & AUC plot
    axes[2].plot(epochs, history['val_f1'], 'g-', label='F1 Score', linewidth=2)
    axes[2].plot(epochs, history['val_auc'], 'm-', label='AUC', linewidth=2)
    axes[2].set_title('F1 Score & AUC', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def generate_classification_report(y_true, y_pred, save_path=None):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save report
    
    Returns:
        report: Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Benign', 'Malware'],
        digits=4
    )
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")
    
    return report


def evaluate_threshold_impact(y_true, y_proba, thresholds=None):
    """
    Evaluate impact of different classification thresholds
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate
    
    Returns:
        results: DataFrame with threshold analysis
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'fpr': metrics['fpr']
        })
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Threshold Analysis")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    return df


def plot_feature_importance(feature_importance, top_k=20, save_path=None, figsize=(10, 8)):
    """
    Plot feature importance
    
    Args:
        feature_importance: Array of feature importance scores
        top_k: Number of top features to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get top k features
    indices = np.argsort(feature_importance)[-top_k:]
    values = feature_importance[indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(top_k), values, color='steelblue')
    plt.yticks(range(top_k), [f'Feature {i}' for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def create_evaluation_report(
    y_true,
    y_pred,
    y_proba,
    history=None,
    save_dir='results/evaluations'
):
    """
    Create comprehensive evaluation report with all plots
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        history: Training history
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Creating Evaluation Report")
    print(f"{'='*60}\n")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, save_path=save_dir / 'confusion_matrix.png')
    plot_roc_curve(y_true, y_proba, save_path=save_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_true, y_proba, save_path=save_dir / 'pr_curve.png')
    
    if history:
        plot_training_history(history, save_path=save_dir / 'training_history.png')
    
    # Generate classification report
    generate_classification_report(y_true, y_pred, save_path=save_dir / 'classification_report.txt')
    
    # Threshold analysis
    threshold_df = evaluate_threshold_impact(y_true, y_proba)
    threshold_df.to_csv(save_dir / 'threshold_analysis.csv', index=False)
    
    # Save metrics
    import json
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Report Complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # Create dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_proba = np.random.random(1000)
    y_pred = (y_proba > 0.5).astype(int)
    
    # Test metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print("Metrics:", metrics)
    
    # Test plots
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    
    print("\n✓ Evaluation module tested successfully!")