"""
Base abstract class for EBM training methods.

This module defines the abstract interface that all EBM training methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EBMTrainingMethod(ABC):
    """
    Abstract base class for Energy-Based Model training methods.
    
    This class defines the interface that all EBM training methods must implement.
    It provides shared functionality for the positive phase and gradient computation,
    while requiring subclasses to implement the specific negative phase sampling strategy.
    
    Args:
        model: The EBM model to train.
        optimizer: PyTorch optimizer for parameter updates.
        config: Training configuration dictionary.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, config: Dict[str, Any]):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        # Configuration with defaults
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 10)
        self.learning_rate = training_config.get('learning_rate', 0.01)
        self.batch_limit = training_config.get('batch_limit', None)
        
        self.model.train()
    
    @abstractmethod
    def negative_phase(self, v_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative samples for gradient computation.
        
        This is the core method that differentiates training methods.
        Each method implements its own strategy for sampling from the model distribution.
        
        Args:
            v_positive: Positive phase visible data batch.
            
        Returns:
            Tuple of (v_negative, h_negative) negative phase samples.
        """
        pass
    
    def positive_phase(self, v_data: torch.Tensor) -> torch.Tensor:
        """
        Compute positive phase hidden probabilities.
        
        This is shared across all training methods as it's simply the forward pass.
        
        Args:
            v_data: Visible data batch.
            
        Returns:
            Hidden unit probabilities.
        """
        return torch.sigmoid(torch.nn.functional.linear(v_data, self.model.W, self.model.c))
    
    def compute_gradients(self, v_pos: torch.Tensor, h_pos: torch.Tensor, 
                         v_neg: torch.Tensor, h_neg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute parameter gradients using contrastive divergence.
        
        Args:
            v_pos: Positive phase visible samples.
            h_pos: Positive phase hidden samples/probabilities.
            v_neg: Negative phase visible samples.
            h_neg: Negative phase hidden samples/probabilities.
            
        Returns:
            Dictionary of gradients for each parameter.
        """
        batch_size = v_pos.size(0)
        
        # Compute positive and negative phase associations
        pos_assoc = torch.bmm(h_pos.unsqueeze(2), v_pos.unsqueeze(1))
        neg_assoc = torch.bmm(h_neg.unsqueeze(2), v_neg.unsqueeze(1))
        
        # Compute gradients
        grad_W = torch.mean(pos_assoc - neg_assoc, dim=0)
        grad_b = torch.mean(v_pos - v_neg, dim=0)
        grad_c = torch.mean(h_pos - h_neg, dim=0)
        
        return {
            'W': grad_W,
            'b': grad_b,
            'c': grad_c
        }
    
    def train_step(self, batch_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch_data: Batch of training data.
            
        Returns:
            Dictionary of training metrics.
        """
        # Positive phase
        v_pos = batch_data.view(batch_data.size(0), -1)
        v_pos = (v_pos > torch.rand_like(v_pos)).float()
        h_pos = self.positive_phase(v_pos)
        
        # Negative phase (method-specific)
        v_neg, h_neg = self.negative_phase(v_pos)
        
        # Compute gradients
        gradients = self.compute_gradients(v_pos, h_pos, v_neg, h_neg)
        
        # Apply gradients
        self.model.W.grad = -gradients['W']
        self.model.b.grad = -gradients['b']
        self.model.c.grad = -gradients['c']
        
        # Update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Compute metrics
        with torch.no_grad():
            v_reconstructed = self.model.reconstruct(h_pos)
            reconstruction_error = torch.mean((v_pos - v_reconstructed)**2)
            
            # Compute pseudo-likelihood or other metrics
            metrics = {
                'reconstruction_error': reconstruction_error.item(),
                'positive_energy': self._compute_energy(v_pos, h_pos).mean().item(),
                'negative_energy': self._compute_energy(v_neg, h_neg).mean().item(),
            }
        
        return metrics
    
    def _compute_energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy of (v, h) configurations.
        
        Args:
            v: Visible units.
            h: Hidden units.
            
        Returns:
            Energy values.
        """
        return -(torch.sum(v * self.model.b, dim=1) + 
                torch.sum(h * self.model.c, dim=1) + 
                torch.sum((v @ self.model.W.t()) * h, dim=1))
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this training method."""
        pass
    
    @property
    @abstractmethod
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this training method."""
        pass