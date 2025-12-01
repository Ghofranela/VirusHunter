#!/usr/bin/env python3
"""
Simple MalwareDetector model definition for VirusHunter
"""
import torch
import torch.nn as nn

class MalwareDetector(nn.Module):
    """
    Deep Neural Network for malware detection
    Input: 2381 features (EMBER dataset format)
    Output: Binary classification (malware vs benign)
    """
    def __init__(self, input_size=2381, hidden_sizes=[512, 256, 128], dropout=0.3):
        super(MalwareDetector, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)