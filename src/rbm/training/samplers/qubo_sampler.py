"""
QUBO sampler for Perturb-and-MAP training.

This module implements the QUBO sampling strategy used in Perturb-and-MAP
training methods for RBMs.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from .base import AbstractSampler
from ...solvers.base import QUBOSolver


class QUBOSampler(AbstractSampler):
    """
    QUBO sampler for Perturb-and-MAP methodology.
    
    This sampler uses QUBO optimization to generate samples from the
    Gumbel-perturbed energy distribution, implementing the "MAP" part
    of the Perturb-and-MAP algorithm.
    
    Args:
        model: The RBM model to sample from.
        solver: The QUBO solver to use for optimization.
        config: Configuration dictionary with sampler parameters.
    """
    
    def __init__(self, model: nn.Module, solver: QUBOSolver, config: Dict[str, Any]):
        super().__init__(model, config)
        
        self.solver = solver
        
        # QUBO sampler specific configuration
        self.gumbel_scale = config.get('gumbel_scale', 1.0)
        self.solver_timeout = config.get('solver_timeout', 60.0)
        self.max_retries = config.get('max_retries', 3)
        
        # Statistics tracking
        self.solve_times = []
        self.solve_successes = []
        self.total_samples = 0
        
        print(f"Initialized QUBOSampler with {solver.name} solver")
    
    def sample(self, v_init: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples using QUBO optimization.
        
        Args:
            v_init: Initial visible state (not used in QUBO sampling).
            num_samples: Number of samples to generate.
            
        Returns:
            Tuple of (v_samples, h_samples) where:
            - v_samples: Visible unit samples from QUBO optimization
            - h_samples: Hidden unit samples from QUBO optimization
        """
        self._validate_input(v_init, num_samples)
        
        v_samples = []
        h_samples = []
        
        for i in range(num_samples):
            # Generate single sample using QUBO
            v_sample, h_sample = self._generate_single_sample()
            v_samples.append(v_sample)
            h_samples.append(h_sample)
        
        v_samples = torch.stack(v_samples)
        h_samples = torch.stack(h_samples)
        
        self.total_samples += num_samples
        
        return v_samples, h_samples
    
    def _generate_single_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single sample using QUBO optimization.
        
        Returns:
            Tuple of (v_sample, h_sample).
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Create joint QUBO matrix with Gumbel perturbation
                Q_joint = self.model.create_joint_qubo()
                
                # Solve QUBO problem
                vh_sample_np = self.solver.solve(Q_joint)
                
                solve_time = time.time() - start_time
                self.solve_times.append(solve_time)
                
                # Extract visible and hidden parts
                v_sample = torch.from_numpy(vh_sample_np[:self.model.n_visible]).float()
                h_sample = torch.from_numpy(vh_sample_np[self.model.n_visible:]).float()
                
                self.solve_successes.append(True)
                return v_sample, h_sample
                
            except Exception as e:
                self.solve_successes.append(False)
                
                if attempt == self.max_retries - 1:
                    # Last attempt failed, use random sample as fallback
                    print(f"Warning: QUBO solver failed after {self.max_retries} attempts, using random sample")
                    v_sample = torch.randint(0, 2, (self.model.n_visible,)).float()
                    h_sample = torch.randint(0, 2, (self.model.n_hidden,)).float()
                    return v_sample, h_sample
                else:
                    print(f"QUBO solver attempt {attempt + 1} failed: {e}")
                    continue
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """
        Get QUBO solver performance statistics.
        
        Returns:
            Dictionary with solver statistics.
        """
        if not self.solve_times:
            return {
                'avg_solve_time': 0.0,
                'success_rate': 0.0,
                'total_solves': 0
            }
        
        return {
            'avg_solve_time': np.mean(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'success_rate': np.mean(self.solve_successes),
            'total_solves': len(self.solve_times),
            'solver_name': self.solver.name
        }
    
    def reset_statistics(self):
        """Reset solver statistics."""
        self.solve_times = []
        self.solve_successes = []
        self.total_samples = 0
    
    @property
    def name(self) -> str:
        """Return the name of this sampling strategy."""
        return f"QUBOSampler({self.solver.name})"
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this sampling strategy."""
        return {
            'solver': self.solver.name,
            'gumbel_scale': self.gumbel_scale,
            'solver_timeout': self.solver_timeout,
            'max_retries': self.max_retries,
            'seed': self.seed
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sampling statistics.
        
        Returns:
            Dictionary of sampling statistics.
        """
        stats = super().get_statistics()
        stats.update(self.get_solver_statistics())
        stats['total_samples'] = self.total_samples
        return stats