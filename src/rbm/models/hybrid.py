"""
Hybrid CNN-RBM model implementations.

This module contains hybrid architectures that combine CNNs with RBMs for
enhanced feature learning capabilities.
"""

import torch
import torch.nn as nn
from .rbm import RBM


class MNISTFeatureExtractor(nn.Module):
    """
    A lightweight CNN feature extractor designed specifically for MNIST.
    Uses a simple architecture with two convolutional layers.
    """
    
    def __init__(self, feature_dim: int = 64):
        super(MNISTFeatureExtractor, self).__init__()
        
        # Simple CNN for MNIST (28x28 grayscale images)
        self.features = nn.Sequential(
            # First conv block: 1 -> 16 channels, 28x28 -> 14x14
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block: 16 -> 32 channels, 14x14 -> 7x7
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Final conv block: 32 -> feature_dim, 7x7 -> 3x3
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate the flattened feature dimension (feature_dim * 3 * 3)
        self.flat_dim = feature_dim * 3 * 3
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) for MNIST.
            
        Returns:
            Flattened features of shape (batch_size, flat_dim).
        """
        # x shape: (batch_size, 1, 28, 28) for MNIST
        x = self.features(x)  # Output: (batch_size, feature_dim, 3, 3)
        
        # Flatten: (batch_size, feature_dim * 3 * 3)
        x = x.view(x.size(0), -1)
        
        return x


class CNN_RBM(nn.Module):
    """
    Hybrid model combining a CNN feature extractor with an RBM.
    The CNN extracts features which become the visible layer for the RBM.
    
    Args:
        cnn_feature_dim (int): Feature dimension for the CNN output.
        rbm_hidden_dim (int): Number of hidden units in the RBM.
    """
    
    def __init__(self, cnn_feature_dim: int = 64, rbm_hidden_dim: int = 128):
        super(CNN_RBM, self).__init__()
        
        # CNN feature extractor (frozen)
        self.cnn = MNISTFeatureExtractor(feature_dim=cnn_feature_dim)
        
        # Calculate the flattened feature dimension
        cnn_output_dim = self.cnn.flat_dim
        
        # RBM that operates on CNN features
        self.rbm = RBM(n_visible=cnn_output_dim, n_hidden=rbm_hidden_dim)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from input images.
        
        Args:
            x: Input images tensor.
            
        Returns:
            Extracted CNN features.
        """
        with torch.no_grad():  # CNN is frozen
            return self.cnn(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN to get features.
        
        Args:
            x: Input images tensor.
            
        Returns:
            CNN features.
        """
        return self.extract_features(x)