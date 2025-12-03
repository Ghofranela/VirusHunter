#!/usr/bin/env python3
"""
Train malware detection model with real-time progress tracking
"""
import sys
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model import MalwareDetector
from scripts.load_ember_data import load_data


def train_model(
    model_type="DNN",
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
    data_dir=Path("data"),
    model_save_path=Path("models/best_model.pth"),
    progress_file=Path("training_progress.json")
):
    """
    Train malware detection model with progress tracking

    Args:
        model_type: Model architecture (DNN, CNN, LSTM, Ensemble)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        data_dir: Directory containing training data
        model_save_path: Path to save best model
        progress_file: JSON file to write training progress
    """

    # Initialize progress tracking
    progress = {
        "status": "loading_data",
        "epoch": 0,
        "train_loss": 0.0,
        "train_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "best_val_acc": 0.0,
        "message": "Loading dataset..."
    }
    _save_progress(progress, progress_file)

    # Load data
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_data(data_dir)
        print(f"✅ Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    except Exception as e:
        progress["status"] = "error"
        progress["message"] = f"Failed to load data: {str(e)}"
        _save_progress(progress, progress_file)
        raise

    # Convert to PyTorch tensors
    progress["status"] = "preparing"
    progress["message"] = "Preparing datasets..."
    _save_progress(progress, progress_file)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    if model_type == "DNN":
        model = MalwareDetector(input_size=X_train.shape[1]).to(device)
    else:
        # For now, only DNN is implemented
        # CNN, LSTM, Ensemble can be added later
        model = MalwareDetector(input_size=X_train.shape[1]).to(device)
        print(f"⚠️  {model_type} not yet implemented, using DNN")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Track history for visualization
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"\n🎓 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Collect predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': model_type,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs
                }
            }, model_save_path)
            print(f"💾 Saved best model (val_acc: {val_acc:.4f})")

        # Update progress
        progress = {
            "status": "training",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "best_val_acc": float(best_val_acc),
            "message": f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        }
        _save_progress(progress, progress_file)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final evaluation on test set
    progress["status"] = "evaluating"
    progress["message"] = "Evaluating on test set..."
    _save_progress(progress, progress_file)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)

    print(f"\n📊 Final Test Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")

    # Training complete
    progress = {
        "status": "completed",
        "epoch": epochs,
        "total_epochs": epochs,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "message": f"Training completed! Test Accuracy: {test_acc:.4f}",
        "history": history
    }
    _save_progress(progress, progress_file)

    print(f"\n✅ Training completed! Best model saved to {model_save_path}")

    return progress


def _save_progress(progress, progress_file):
    """Save progress to JSON file"""
    progress["timestamp"] = time.time()
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train malware detection model")
    parser.add_argument("--model-type", type=str, default="DNN", choices=["DNN", "CNN", "LSTM", "Ensemble"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pth"))

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        model_save_path=args.output
    )
