"""
Utility Functions for VirusHunter Project
Helper functions, logging, and common utilities
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import random

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"virushunter_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('VirusHunter')


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(use_cuda: bool = True) -> str:
    """
    Get computing device
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        Device string
    """
    if use_cuda and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load model on
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint


def save_json(data: Dict, save_path: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        save_path: Path to save JSON
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON saved to {save_path}")


def load_json(file_path: str) -> Dict:
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Architecture: {model.__class__.__name__}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model Size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    print("="*60 + "\n")


def create_dir_structure(base_dir: str = "."):
    """
    Create project directory structure
    
    Args:
        base_dir: Base directory for project
    """
    base_dir = Path(base_dir)
    
    dirs = [
        "data/raw",
        "data/processed",
        "data/ember",
        "models",
        "results/explanations",
        "results/evaluations",
        "logs",
        "scripts",
        "src",
        "app",
        "notebooks",
        "docs"
    ]
    
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created in {base_dir}")


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        labels: Array of labels
    
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return torch.FloatTensor(weights)


def calculate_metrics_summary(metrics_list: List[Dict]) -> Dict:
    """
    Calculate summary statistics from multiple metric dictionaries
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Summary statistics
    """
    summary = {}
    
    if not metrics_list:
        return summary
    
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values and isinstance(values[0], (int, float)):
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return summary


class EarlyStopping:
    """
    Early stopping handler
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float):
        """
        Check if should stop
        
        Args:
            score: Current score (higher is better)
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MetricsTracker:
    """
    Track and log metrics during training
    """
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_dict: Dict, step: int):
        """
        Update metrics
        
        Args:
            metric_dict: Dictionary of metrics
            step: Current step/epoch
        """
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for metric"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1][1]
        return None
    
    def get_history(self, key: str) -> List[Tuple[int, float]]:
        """Get full history for metric"""
        return self.metrics.get(key, [])
    
    def save(self, save_path: str):
        """Save metrics to file"""
        save_json(self.metrics, save_path)


def download_ember_dataset(data_dir: str = "data/ember", version: int = 2):
    """
    Download EMBER dataset
    
    Args:
        data_dir: Directory to save data
        version: EMBER version (1 or 2)
    """
    try:
        import ember
        
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading EMBER v{version} dataset...")
        print(f"This may take a while (dataset is ~13GB)")
        
        if version == 2:
            ember.create_vectorized_features(str(data_dir))
            ember.create_metadata(str(data_dir))
        
        print(f"EMBER dataset downloaded to {data_dir}")
    
    except ImportError:
        print("ember package not installed.")
        print("Install with: pip install ember")
    
    except Exception as e:
        print(f"Error downloading EMBER dataset: {e}")
        print("\nAlternative: Manually download from:")
        print("https://github.com/elastic/ember")


if __name__ == "__main__":
    # Test utilities
    print("Testing Utilities...")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Logging initialized")
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create directory structure
    create_dir_structure("test_project")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({'loss': 0.5, 'acc': 0.8}, step=1)
    tracker.update({'loss': 0.3, 'acc': 0.85}, step=2)
    print(f"Latest accuracy: {tracker.get_latest('acc')}")
    
    print("\nUtilities tested successfully!")
