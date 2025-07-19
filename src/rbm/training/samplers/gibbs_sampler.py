"""
Gibbs sampler for Contrastive Divergence training.

This module implements the Gibbs sampling strategy used in Contrastive Divergence
and related training methods for RBMs.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from .base import AbstractSampler


class GibbsSampler(AbstractSampler):
    """
    Gibbs sampler for alternating sampling between visible and hidden units.
    
    This sampler implements k-step Gibbs sampling starting from the data distribution.
    It supports both standard Contrastive Divergence (CD-k) and Persistent Contrastive
    Divergence (PCD) variants.
    
    Args:
        model: The RBM model to sample from.
        config: Configuration dictionary with sampler parameters.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Gibbs sampler specific configuration
        self.k_steps = config.get('k_steps', 1)
        self.persistent = config.get('persistent', False)
        self.sampling_mode = config.get('sampling_mode', 'stochastic')  # 'stochastic' or 'deterministic'
        self.burn_in_steps = config.get('burn_in_steps', 0)
        
        # Persistent CD state
        self.fantasy_particles = None
        self.fantasy_hidden = None
        self.initialized = False
        
        # Statistics tracking
        self.sample_count = 0
        self.acceptance_rates = []
        
        # Validation
        if self.k_steps < 1:
            raise ValueError("k_steps must be at least 1")
        if self.sampling_mode not in ['stochastic', 'deterministic']:
            raise ValueError("sampling_mode must be 'stochastic' or 'deterministic'")
    
    def sample(self, v_init: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples using k-step Gibbs sampling.
        
        Args:
            v_init: Initial visible state (used for CD, ignored for PCD after initialization).
            num_samples: Number of samples to generate.
            
        Returns:
            Tuple of (v_samples, h_samples) where:
            - v_samples: Final visible unit samples
            - h_samples: Final hidden unit probabilities
        """
        self._validate_input(v_init, num_samples)
        
        # Initialize sampling chains
        if self.persistent and self.initialized:
            # Use persistent fantasy particles
            v_samples = self.fantasy_particles
            batch_size = v_samples.size(0)
            
            # Adjust batch size if needed
            if batch_size != num_samples:
                if batch_size < num_samples:
                    # Duplicate fantasy particles to match batch size
                    repeats = (num_samples + batch_size - 1) // batch_size
                    v_samples = v_samples.repeat(repeats, 1)[:num_samples]
                else:
                    # Truncate fantasy particles
                    v_samples = v_samples[:num_samples]
        else:
            # Initialize from data (CD) or random (PCD initial)
            if self.persistent and not self.initialized:
                # For PCD, start with random initialization
                v_samples = torch.randint(0, 2, (num_samples, self.model.n_visible)).float()
                self.initialized = True
            else:
                # For CD, start from data
                v_samples = v_init[:num_samples].clone()
        
        # Burn-in phase (for PCD initialization)
        if self.burn_in_steps > 0 and (not self.persistent or not self.initialized):
            v_samples = self._gibbs_steps(v_samples, self.burn_in_steps)
        
        # Main sampling phase
        v_samples = self._gibbs_steps(v_samples, self.k_steps)
        
        # Compute final hidden probabilities
        h_samples = self._compute_hidden_probs(v_samples)
        
        # Update persistent state
        if self.persistent:
            self.fantasy_particles = v_samples.detach()
            self.fantasy_hidden = h_samples.detach()
        
        # Update statistics
        self.sample_count += num_samples
        
        return v_samples, h_samples
    
    def _gibbs_steps(self, v_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Perform num_steps of alternating Gibbs sampling.
        
        Args:
            v_init: Initial visible state.
            num_steps: Number of Gibbs steps to perform.
            
        Returns:
            Final visible state after num_steps.
        """
        v_sample = v_init.clone()
        
        for step in range(num_steps):
            # Sample hidden given visible: h ~ p(h|v)
            h_probs = self._compute_hidden_probs(v_sample)
            
            if self.sampling_mode == 'stochastic':
                h_sample = self._sample_binary(h_probs)
            else:  # deterministic
                h_sample = (h_probs > 0.5).float()
            
            # Sample visible given hidden: v ~ p(v|h)
            v_probs = self._compute_visible_probs(h_sample)
            
            if self.sampling_mode == 'stochastic':
                v_sample = self._sample_binary(v_probs)
            else:  # deterministic
                v_sample = (v_probs > 0.5).float()
        
        return v_sample
    
    def reset_fantasy_particles(self):
        """Reset persistent fantasy particles (for PCD)."""
        self.fantasy_particles = None
        self.fantasy_hidden = None
        self.initialized = False
    
    def get_fantasy_particles(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get current fantasy particles (for PCD).
        
        Returns:
            Tuple of (v_fantasy, h_fantasy) if available, None otherwise.
        """
        if self.persistent and self.fantasy_particles is not None:
            return self.fantasy_particles, self.fantasy_hidden
        return None
    
    def set_fantasy_particles(self, v_fantasy: torch.Tensor, h_fantasy: torch.Tensor):
        """
        Set fantasy particles (for PCD).
        
        Args:
            v_fantasy: Fantasy visible particles.
            h_fantasy: Fantasy hidden particles.
        """
        self.fantasy_particles = v_fantasy.clone()
        self.fantasy_hidden = h_fantasy.clone()
        self.initialized = True
    
    def compute_acceptance_rate(self, v_old: torch.Tensor, v_new: torch.Tensor) -> float:
        """
        Compute acceptance rate for monitoring convergence.
        
        Args:
            v_old: Previous visible state.
            v_new: New visible state.
            
        Returns:
            Acceptance rate (fraction of units that changed).
        """
        if v_old.shape != v_new.shape:
            return 0.0
        
        changes = torch.sum(v_old != v_new, dim=1).float()
        acceptance_rate = torch.mean(changes / v_old.size(1)).item()
        self.acceptance_rates.append(acceptance_rate)
        
        return acceptance_rate
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Get convergence diagnostics for the Gibbs sampler.
        
        Returns:
            Dictionary with convergence metrics.
        """
        diagnostics = {
            'sample_count': self.sample_count,
            'mean_acceptance_rate': np.mean(self.acceptance_rates) if self.acceptance_rates else 0.0,
            'acceptance_rate_std': np.std(self.acceptance_rates) if self.acceptance_rates else 0.0,
            'is_persistent': self.persistent,
            'fantasy_particles_available': self.fantasy_particles is not None
        }
        
        return diagnostics
    
    @property
    def name(self) -> str:
        """Return the name of this sampling strategy."""
        if self.persistent:
            return f"PersistentGibbsSampler(k={self.k_steps})"
        else:
            return f"GibbsSampler(k={self.k_steps})"
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this sampling strategy."""
        return {
            'k_steps': self.k_steps,
            'persistent': self.persistent,
            'sampling_mode': self.sampling_mode,
            'burn_in_steps': self.burn_in_steps,
            'seed': self.seed
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sampling statistics.
        
        Returns:
            Dictionary of sampling statistics.
        """
        stats = super().get_statistics()
        stats.update(self.get_convergence_diagnostics())
        return stats