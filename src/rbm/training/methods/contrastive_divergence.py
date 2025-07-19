"""
Contrastive Divergence training method for RBMs.

This module implements the Contrastive Divergence (CD) training algorithm,
including CD-k and Persistent Contrastive Divergence (PCD) variants.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import EBMTrainingMethod
from ..samplers.gibbs_sampler import GibbsSampler


class ContrastiveDivergenceTraining(EBMTrainingMethod):
    """
    Contrastive Divergence training method for RBMs.
    
    This class implements the standard Contrastive Divergence algorithm and its variants.
    It uses Gibbs sampling for the negative phase, supporting both CD-k and PCD.
    
    Args:
        model: The RBM model to train.
        optimizer: PyTorch optimizer for parameter updates.
        config: Training configuration dictionary.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, config: Dict[str, Any]):
        super().__init__(model, optimizer, config)
        
        # Get CD-specific configuration
        cd_config = config.get('cd_params', {})
        
        # Initialize Gibbs sampler
        sampler_config = {
            'k_steps': cd_config.get('k_steps', 1),
            'persistent': cd_config.get('persistent', False),
            'sampling_mode': cd_config.get('sampling_mode', 'stochastic'),
            'burn_in_steps': cd_config.get('burn_in_steps', 0),
            'batch_size': config.get('training', {}).get('batch_size', 64),
            'seed': config.get('seed', 42)
        }
        
        self.sampler = GibbsSampler(model, sampler_config)
        
        # CD-specific parameters
        self.k_steps = sampler_config['k_steps']
        self.persistent = sampler_config['persistent']
        self.sampling_mode = sampler_config['sampling_mode']
        
        # Training statistics
        self.cd_statistics = {
            'total_samples': 0,
            'reconstruction_errors': [],
            'energy_gaps': [],
            'acceptance_rates': []
        }
        
        print(f"Initialized {self.name} with k_steps={self.k_steps}, persistent={self.persistent}")
    
    def negative_phase(self, v_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative samples using Gibbs sampling.
        
        For CD-k: Start Gibbs chains from the positive phase data.
        For PCD: Use persistent fantasy particles maintained across batches.
        
        Args:
            v_positive: Positive phase visible data batch.
            
        Returns:
            Tuple of (v_negative, h_negative) negative phase samples.
        """
        batch_size = v_positive.size(0)
        
        # Generate negative samples using Gibbs sampling
        v_negative, h_negative = self.sampler.sample(v_positive, batch_size)
        
        # Update statistics
        self.cd_statistics['total_samples'] += batch_size
        
        # Compute acceptance rate for monitoring
        if hasattr(self, '_last_v_negative'):
            acceptance_rate = self.sampler.compute_acceptance_rate(
                self._last_v_negative, v_negative
            )
            self.cd_statistics['acceptance_rates'].append(acceptance_rate)
        
        self._last_v_negative = v_negative.clone()
        
        return v_negative, h_negative
    
    def train_step(self, batch_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single CD training step.
        
        Args:
            batch_data: Batch of training data.
            
        Returns:
            Dictionary of training metrics.
        """
        # Call parent train_step which handles the standard CD workflow
        metrics = super().train_step(batch_data)
        
        # Add CD-specific metrics
        sampler_stats = self.sampler.get_statistics()
        metrics.update({
            'cd_k_steps': self.k_steps,
            'cd_persistent': self.persistent,
            'cd_sample_count': sampler_stats.get('sample_count', 0),
            'cd_acceptance_rate': np.mean(self.cd_statistics['acceptance_rates']) 
                                if self.cd_statistics['acceptance_rates'] else 0.0
        })
        
        # Store metrics for analysis
        self.cd_statistics['reconstruction_errors'].append(metrics['reconstruction_error'])
        self.cd_statistics['energy_gaps'].append(
            metrics['positive_energy'] - metrics['negative_energy']
        )
        
        return metrics
    
    def reset_persistent_chain(self):
        """Reset the persistent fantasy particles (for PCD)."""
        if self.persistent:
            self.sampler.reset_fantasy_particles()
            print("Reset persistent fantasy particles")
    
    def get_fantasy_particles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current fantasy particles (for PCD).
        
        Returns:
            Tuple of (v_fantasy, h_fantasy) if available, None otherwise.
        """
        return self.sampler.get_fantasy_particles()
    
    def set_fantasy_particles(self, v_fantasy: torch.Tensor, h_fantasy: torch.Tensor):
        """
        Set fantasy particles (for PCD).
        
        Args:
            v_fantasy: Fantasy visible particles.
            h_fantasy: Fantasy hidden particles.
        """
        self.sampler.set_fantasy_particles(v_fantasy, h_fantasy)
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Get convergence diagnostics for CD training.
        
        Returns:
            Dictionary with convergence metrics.
        """
        diagnostics = {
            'method': self.name,
            'total_samples': self.cd_statistics['total_samples'],
            'mean_reconstruction_error': np.mean(self.cd_statistics['reconstruction_errors']) 
                                       if self.cd_statistics['reconstruction_errors'] else 0.0,
            'mean_energy_gap': np.mean(self.cd_statistics['energy_gaps']) 
                             if self.cd_statistics['energy_gaps'] else 0.0,
            'recent_acceptance_rate': np.mean(self.cd_statistics['acceptance_rates'][-10:]) 
                                    if len(self.cd_statistics['acceptance_rates']) >= 10 else 0.0
        }
        
        # Add sampler-specific diagnostics
        sampler_diagnostics = self.sampler.get_convergence_diagnostics()
        diagnostics.update({f'sampler_{k}': v for k, v in sampler_diagnostics.items()})
        
        return diagnostics
    
    def adapt_learning_rate(self, metrics: Dict[str, float]) -> bool:
        """
        Adaptive learning rate adjustment based on CD metrics.
        
        Args:
            metrics: Current training metrics.
            
        Returns:
            True if learning rate was adjusted, False otherwise.
        """
        # Simple adaptive learning rate based on energy gap
        if len(self.cd_statistics['energy_gaps']) > 10:
            recent_gaps = self.cd_statistics['energy_gaps'][-10:]
            gap_trend = np.mean(recent_gaps[-5:]) - np.mean(recent_gaps[:5])
            
            # If energy gap is not decreasing, reduce learning rate
            if gap_trend > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                return True
        
        return False
    
    @property
    def name(self) -> str:
        """Return the name of this training method."""
        if self.persistent:
            return f"PersistentContrastiveDivergence(k={self.k_steps})"
        else:
            return f"ContrastiveDivergence(k={self.k_steps})"
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this training method."""
        return {
            'k_steps': self.k_steps,
            'persistent': self.persistent,
            'sampling_mode': self.sampling_mode,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'sampler_hyperparameters': self.sampler.hyperparameters
        }
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary of training statistics.
        """
        stats = {
            'method': self.name,
            'hyperparameters': self.hyperparameters,
            'cd_statistics': self.cd_statistics.copy(),
            'convergence_diagnostics': self.get_convergence_diagnostics(),
            'sampler_statistics': self.sampler.get_statistics()
        }
        
        return stats