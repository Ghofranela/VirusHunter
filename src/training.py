"""
Training Module with Adversarial Robustness
Implements standard training and adversarial training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt


class MalwareDataset(Dataset):
    """Custom dataset for malware detection"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    """
    Complete training pipeline with adversarial training
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function with class weights
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        adversarial: bool = False,
        epsilon: float = 0.01
    ) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            adversarial: Use adversarial training
            epsilon: Adversarial perturbation magnitude
        
        Returns:
            Average loss, accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Standard training
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Adversarial training
            if adversarial:
                # Generate adversarial examples
                X_batch.requires_grad = True
                outputs_adv = self.model(X_batch)
                loss_adv = self.criterion(outputs_adv, y_batch)
                loss_adv.backward(retain_graph=True)
                
                # FGSM attack
                perturbation = epsilon * X_batch.grad.sign()
                X_adv = X_batch + perturbation
                X_adv = X_adv.detach()
                
                # Train on adversarial examples
                outputs_adv = self.model(X_adv)
                loss_adv = self.criterion(outputs_adv, y_batch)
                
                # Combined loss
                loss = 0.5 * loss + 0.5 * loss_adv
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, Dict]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average loss, accuracy, detailed metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        # Detailed metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions),
            'auc': roc_auc_score(all_labels, all_probs)
        }
        
        return avg_loss, accuracy, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        adversarial: bool = True,
        epsilon: float = 0.01,
        early_stopping_patience: int = 10,
        save_dir: str = "models"
    ):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            adversarial: Use adversarial training
            epsilon: Adversarial perturbation
            early_stopping_patience: Early stopping patience
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        print(f"\nTraining for {epochs} epochs")
        print(f"Adversarial training: {adversarial}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader,
                adversarial=adversarial,
                epsilon=epsilon
            )
            
            # Validate
            val_loss, val_acc, metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'metrics': metrics
                }
                
                torch.save(checkpoint, save_dir / "best_model.pth")
                print(f"✓ Best model saved (val_acc: {val_acc:.4f})")
            
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model loaded (val_acc: {self.best_val_acc:.4f})")
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            save_dir / "final_model.pth"
        )
        
        # Save training history
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete! Models saved to {save_dir}")
    
    def plot_training_curves(self, save_path: str = "results/training_curves.png"):
        """Plot training curves"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")


def train_malware_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'dnn',
    batch_size: int = 256,
    epochs: int = 50,
    learning_rate: float = 0.001,
    adversarial: bool = True,
    device: str = None
) -> Tuple[nn.Module, Dict]:
    """
    Complete training pipeline
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: 'dnn', 'cnn', 'lstm', 'ensemble'
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        adversarial: Use adversarial training
        device: Device to train on
    
    Returns:
        Trained model, training history
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MalwareDataset(X_train, y_train)
    val_dataset = MalwareDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    from model import create_model
    
    input_size = X_train.shape[1]
    model = create_model(model_type, input_size=input_size)
    
    print(f"\nModel: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        adversarial=adversarial,
        save_dir="models"
    )
    
    # Plot curves
    trainer.plot_training_curves()
    
    return model, trainer.history


if __name__ == "__main__":
    # Load processed data
    print("Loading data...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train model
    model, history = train_malware_detector(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type='dnn',
        batch_size=256,
        epochs=50,
        learning_rate=0.001,
        adversarial=True
    )
    
    print("\nTraining complete!")