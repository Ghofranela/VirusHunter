"""
Training Module for Malware Detection Models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from .model import create_model, count_parameters
from .utils import EarlyStopping, save_checkpoint, get_device
from .evaluation import calculate_metrics
from .adversarial import AdversarialTrainer, evaluate_adversarial_robustness, AdversarialAttack


class MalwareTrainer:
    """
    Trainer class for malware detection models
    """
    def __init__(
        self,
        model,
        device='cuda',
        learning_rate=0.001,
        weight_decay=1e-5,
        class_weights=None,
        adversarial=False
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        if class_weights is not None:
            pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Adversarial training
        self.adversarial = adversarial
        if adversarial:
            self.adversarial_trainer = AdversarialTrainer(model, device=device)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': [],
            'robustness_score': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch with optional adversarial training"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            target = target.float().unsqueeze(1)
            
            if self.adversarial:
                # Adversarial training step
                loss = self.adversarial_trainer.adversarial_training_step(
                    data, target.squeeze(), self.optimizer, alpha=0.3
                )
                
                # Get predictions for metrics
                with torch.no_grad():
                    output = self.model(data)
                    pred = (torch.sigmoid(output) > 0.5).float()
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            else:
                # Standard training
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
            
            total_loss += loss if isinstance(loss, float) else loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss if isinstance(loss, float) else loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target_float = target.float().unsqueeze(1)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target_float)
                
                total_loss += loss.item()
                
                # Predictions
                probs = torch.sigmoid(output)
                pred = (probs > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return avg_loss, metrics
    
    def evaluate_robustness(self, val_loader):
        """Evaluate model robustness against adversarial attacks"""
        if not self.adversarial:
            return 0.0
            
        self.model.eval()
        
        # Extract data from loader
        all_data = []
        all_targets = []
        for data, target in val_loader:
            all_data.append(data)
            all_targets.append(target)
        
        X_val = torch.cat(all_data, dim=0)
        y_val = torch.cat(all_targets, dim=0)
        
        # Evaluate robustness
        attack = AdversarialAttack(self.model)
        robustness_results = evaluate_adversarial_robustness(
            self.model, X_val.cpu().numpy(), y_val.cpu().numpy(), attack, self.device
        )
        
        return robustness_results['robustness_score']
    
    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        early_stopping_patience=10,
        save_dir='models'
    ):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_f1 = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Adversarial Training: {self.adversarial}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Evaluate robustness
            robustness_score = self.evaluate_robustness(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['f1'])
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'] * 100)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['robustness_score'].append(robustness_score)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.2f}s)")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            if self.adversarial:
                print(f"Robustness: {robustness_score:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {**val_metrics, 'robustness': robustness_score},
                    save_dir / 'best_model.pth'
                )
                print(f"✓ Best model saved (F1: {best_f1:.4f})")
            
            # Early stopping
            early_stopping(val_metrics['f1'])
            if early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
            
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best F1 Score: {best_f1:.4f}")
        if self.adversarial:
            print(f"Final Robustness Score: {self.history['robustness_score'][-1]:.4f}")
        print(f"{'='*60}\n")
        
        return self.history


def train_malware_detector(
    X_train,
    y_train,
    X_val,
    y_val,
    model_type='dnn',
    batch_size=256,
    epochs=50,
    learning_rate=0.001,
    adversarial=False,
    device=None,
    save_dir='models'
):
    """
    High-level function to train malware detector
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        adversarial: Use adversarial training
        device: Computing device
        save_dir: Directory to save models
    
    Returns:
        model: Trained model
        history: Training history
    """
    if device is None:
        device = get_device()
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = dict(zip(unique, counts))
    
    # Create model
    model = create_model(model_type, input_dim=X_train.shape[1])
    
    # Create trainer
    trainer = MalwareTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights,
        adversarial=adversarial
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    return trainer.model, history


def evaluate_model(model, X_test, y_test, device='cuda', batch_size=256):
    """
    Evaluate trained model on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Computing device
        batch_size: Batch size
    
    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    
    # Create data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print("="*60 + "\n")
    
    return metrics


if __name__ == "__main__":
    print("Testing training module...")
    
    # Create dummy data
    X_train = np.random.randn(1000, 2381)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 2381)
    y_val = np.random.randint(0, 2, 200)
    
    # Train model
    model, history = train_malware_detector(
        X_train, y_train, X_val, y_val,
        epochs=2,
        batch_size=64
    )
    
    print("✓ Training module tested successfully!")