"""
Advanced Malware Detection Models with Adversarial Robustness
Supports DNN, CNN, LSTM, and Ensemble architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MalwareDetector(nn.Module):
    """
    Deep Neural Network for malware detection
    Robust architecture with dropout and batch normalization
    """
    def __init__(self, input_size: int = 2381, hidden_sizes: list = None):
        super(MalwareDetector, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256, 128]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CNNMalwareDetector(nn.Module):
    """
    Convolutional Neural Network for sequence-based malware detection
    Processes raw byte sequences
    """
    def __init__(self, input_channels: int = 1, sequence_length: int = 2381):
        super(CNNMalwareDetector, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # Calculate flattened size
        self.flat_size = 256 * (sequence_length // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, sequence_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze(-1)


class LSTMMalwareDetector(nn.Module):
    """
    LSTM-based malware detector for sequential features
    Captures temporal patterns in execution behavior
    """
    def __init__(self, input_size: int = 2381, hidden_size: int = 256, num_layers: int = 2):
        super(LSTMMalwareDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Reshape input to sequence
        self.seq_len = 100
        self.feature_dim = input_size // self.seq_len
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to sequence: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Fully connected
        out = self.fc(pooled)
        return out.squeeze(-1)


class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple models for robust detection
    Combines DNN, CNN, and LSTM predictions
    """
    def __init__(self, input_size: int = 2381):
        super(EnsembleDetector, self).__init__()
        
        self.dnn = MalwareDetector(input_size)
        self.cnn = CNNMalwareDetector(sequence_length=input_size)
        self.lstm = LSTMMalwareDetector(input_size)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DNN prediction
        dnn_out = self.dnn(x)
        
        # CNN prediction (reshape input)
        cnn_input = x.unsqueeze(1)  # Add channel dimension
        cnn_out = self.cnn(cnn_input)
        
        # LSTM prediction
        lstm_out = self.lstm(x)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = (
            weights[0] * dnn_out +
            weights[1] * cnn_out +
            weights[2] * lstm_out
        )
        
        return ensemble_out


class AdversarialDefense(nn.Module):
    """
    Wrapper for adversarial training and defense mechanisms
    """
    def __init__(self, base_model: nn.Module):
        super(AdversarialDefense, self).__init__()
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    def generate_adversarial_examples(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.01,
        method: str = 'fgsm'
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM or PGD
        
        Args:
            x: Input features
            y: True labels
            epsilon: Perturbation magnitude
            method: 'fgsm' or 'pgd'
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        if method == 'fgsm':
            # Fast Gradient Sign Method
            output = self.forward(x_adv)
            loss = F.binary_cross_entropy_with_logits(
                output, y.float()
            )
            loss.backward()
            
            # Create perturbation
            perturbation = epsilon * x_adv.grad.sign()
            x_adv = x + perturbation
            
        elif method == 'pgd':
            # Projected Gradient Descent
            alpha = epsilon / 5
            num_iter = 10
            
            for _ in range(num_iter):
                x_adv.requires_grad_(True)
                output = self.forward(x_adv)
                loss = F.binary_cross_entropy_with_logits(
                    output, y.float()
                )
                loss.backward()
                
                # Update with projection
                perturbation = alpha * x_adv.grad.sign()
                x_adv = x_adv + perturbation
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = x_adv.detach()
        
        return x_adv.detach()


class FeatureExtractor(nn.Module):
    """
    Extract learned features for explainability
    """
    def __init__(self, model: nn.Module, layer_name: str = None):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.features = None
        self.layer_name = layer_name
        
        # Register hook
        if hasattr(model, 'network'):
            # For MalwareDetector
            model.network[-2].register_forward_hook(self.save_features)
        
    def save_features(self, module, input, output):
        self.features = output
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        return output, self.features


def create_model(model_type: str = 'dnn', **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'dnn', 'cnn', 'lstm', 'ensemble'
        **kwargs: Additional model parameters
    """
    models = {
        'dnn': MalwareDetector,
        'cnn': CNNMalwareDetector,
        'lstm': LSTMMalwareDetector,
        'ensemble': EnsembleDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


def load_model(
    checkpoint_path: str,
    model_type: str = 'dnn',
    device: str = 'cpu',
    **kwargs
) -> nn.Module:
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model to create
        device: Device to load model on
        **kwargs: Additional model parameters
    """
    model = create_model(model_type, **kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test models
    batch_size = 32
    input_size = 2381
    
    print("Testing MalwareDetector...")
    model = MalwareDetector(input_size)
    x = torch.randn(batch_size, input_size)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting CNNMalwareDetector...")
    cnn_model = CNNMalwareDetector(sequence_length=input_size)
    x_cnn = torch.randn(batch_size, 1, input_size)
    output_cnn = cnn_model(x_cnn)
    print(f"Output shape: {output_cnn.shape}")
    
    print("\nTesting LSTMMalwareDetector...")
    lstm_model = LSTMMalwareDetector(input_size)
    output_lstm = lstm_model(x)
    print(f"Output shape: {output_lstm.shape}")
    
    print("\nTesting EnsembleDetector...")
    ensemble_model = EnsembleDetector(input_size)
    output_ensemble = ensemble_model(x)
    print(f"Output shape: {output_ensemble.shape}")
    
    print("\nAll models initialized successfully!")
