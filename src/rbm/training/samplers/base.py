"""
Base abstract class for sampling strategies.

This module defines the abstract interface that all sampling strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


class AbstractSampler(ABC):
    """
    Abstract base class for sampling strategies used in EBM training.
    
    This class defines the interface that all sampling strategies must implement.
    Different sampling strategies (QUBO, Gibbs, Langevin) can be plugged into
    training methods through this interface.
    
    Args:
        model: The EBM model to sample from.
        config: Sampler-specific configuration dictionary.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Common configuration
        self.batch_size = config.get('batch_size', 64)
        self.seed = config.get('seed', 42)
        
        # Set random seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
    
    @abstractmethod
    def sample(self, v_init: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples from the model distribution.
        
        Args:
            v_init: Initial visible state for sampling (may be ignored by some samplers).
            num_samples: Number of samples to generate.
            
        Returns:
            Tuple of (v_samples, h_samples) where:
            - v_samples: Visible unit samples
            - h_samples: Hidden unit samples/probabilities
        """
        pass
    
    def _validate_input(self, v_init: torch.Tensor, num_samples: int) -> None:
        """
        Validate input parameters.
        
        Args:
            v_init: Initial visible state.
            num_samples: Number of samples to generate.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        if not isinstance(v_init, torch.Tensor):
            raise ValueError("v_init must be a torch.Tensor")
        
        if v_init.dim() != 2:
            raise ValueError("v_init must be a 2D tensor (batch_size, n_visible)")
        
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if v_init.size(1) != self.model.n_visible:
            raise ValueError(f"v_init has wrong number of visible units: "
                           f"expected {self.model.n_visible}, got {v_init.size(1)}")
    
    def _sample_binary(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Sample binary values from probabilities.
        
        Args:
            probs: Probability tensor.
            
        Returns:
            Binary samples.
        """
        return torch.bernoulli(probs)
    
    def _compute_visible_probs(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute visible unit probabilities given hidden units.
        
        Args:
            h: Hidden unit values.
            
        Returns:
            Visible unit probabilities.
        """
        return torch.sigmoid(torch.nn.functional.linear(h, self.model.W.t(), self.model.b))
    
    def _compute_hidden_probs(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute hidden unit probabilities given visible units.
        
        Args:
            v: Visible unit values.
            
        Returns:
            Hidden unit probabilities.
        """
        return torch.sigmoid(torch.nn.functional.linear(v, self.model.W, self.model.c))
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this sampling strategy."""
        pass
    
    @property
    @abstractmethod
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this sampling strategy."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get sampling statistics (e.g., acceptance rates, convergence metrics).
        
        Returns:
            Dictionary of sampling statistics.
        """
        return {
            'sampler': self.name,
            'hyperparameters': self.hyperparameters
        }