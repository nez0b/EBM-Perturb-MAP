"""
Perturb-and-MAP training method for RBMs.

This module implements the Perturb-and-MAP (P&M) training algorithm,
which uses QUBO optimization for the negative phase sampling.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import EBMTrainingMethod
from ..samplers.qubo_sampler import QUBOSampler
from ...solvers.base import QUBOSolver


class PerturbMapTraining(EBMTrainingMethod):
    """
    Perturb-and-MAP training method for RBMs.
    
    This class implements the Perturb-and-MAP algorithm, which uses the Gumbel trick
    to convert sampling from the model distribution into a optimization problem
    solved using QUBO solvers.
    
    Args:
        model: The RBM model to train.
        optimizer: PyTorch optimizer for parameter updates.
        solver: QUBO solver for MAP optimization.
        config: Training configuration dictionary.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 solver: QUBOSolver, config: Dict[str, Any]):
        super().__init__(model, optimizer, config)
        
        self.solver = solver
        
        # Get P&M-specific configuration
        pm_config = config.get('pm_params', {})
        
        # Initialize QUBO sampler
        sampler_config = {
            'gumbel_scale': pm_config.get('gumbel_scale', 1.0),
            'solver_timeout': pm_config.get('solver_timeout', 60.0),
            'max_retries': pm_config.get('max_retries', 3),
            'batch_size': config.get('training', {}).get('batch_size', 64),
            'seed': config.get('seed', 42)
        }
        
        self.sampler = QUBOSampler(model, solver, sampler_config)
        
        # P&M-specific parameters
        self.gumbel_scale = sampler_config['gumbel_scale']
        self.solver_timeout = sampler_config['solver_timeout']
        
        # Training statistics
        self.pm_statistics = {
            'total_samples': 0,
            'reconstruction_errors': [],
            'energy_gaps': [],
            'solver_failures': 0
        }
        
        print(f"Initialized {self.name} with {solver.name} solver")
    
    def negative_phase(self, v_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative samples using QUBO optimization.
        
        This method implements the "MAP" part of Perturb-and-MAP by solving
        the QUBO optimization problem on the Gumbel-perturbed energy function.
        
        Args:
            v_positive: Positive phase visible data batch.
            
        Returns:
            Tuple of (v_negative, h_negative) negative phase samples.
        """
        batch_size = v_positive.size(0)
        
        # Generate negative samples using QUBO sampling
        v_negative, h_negative = self.sampler.sample(v_positive, batch_size)
        
        # Update statistics
        self.pm_statistics['total_samples'] += batch_size
        
        # Check for solver failures
        solver_stats = self.sampler.get_solver_statistics()
        if solver_stats['success_rate'] < 1.0:
            recent_failures = len(self.sampler.solve_successes) - sum(self.sampler.solve_successes)
            self.pm_statistics['solver_failures'] += recent_failures
        
        return v_negative, h_negative
    
    def train_step(self, batch_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single P&M training step.
        
        Args:
            batch_data: Batch of training data.
            
        Returns:
            Dictionary of training metrics.
        """
        # Call parent train_step which handles the standard P&M workflow
        metrics = super().train_step(batch_data)
        
        # Add P&M-specific metrics
        solver_stats = self.sampler.get_solver_statistics()
        metrics.update({
            'pm_gumbel_scale': self.gumbel_scale,
            'pm_solver_timeout': self.solver_timeout,
            'pm_avg_solve_time': solver_stats.get('avg_solve_time', 0.0),
            'pm_success_rate': solver_stats.get('success_rate', 0.0),
            'pm_total_solves': solver_stats.get('total_solves', 0)
        })
        
        # Store metrics for analysis
        self.pm_statistics['reconstruction_errors'].append(metrics['reconstruction_error'])
        self.pm_statistics['energy_gaps'].append(
            metrics['positive_energy'] - metrics['negative_energy']
        )
        
        return metrics
    
    def reset_solver_statistics(self):
        """Reset QUBO solver statistics."""
        self.sampler.reset_statistics()
        self.pm_statistics['solver_failures'] = 0
    
    def get_solver_diagnostics(self) -> Dict[str, Any]:
        """
        Get QUBO solver diagnostics.
        
        Returns:
            Dictionary with solver diagnostics.
        """
        solver_stats = self.sampler.get_solver_statistics()
        
        diagnostics = {
            'solver_name': self.solver.name,
            'avg_solve_time': solver_stats.get('avg_solve_time', 0.0),
            'min_solve_time': solver_stats.get('min_solve_time', 0.0),
            'max_solve_time': solver_stats.get('max_solve_time', 0.0),
            'success_rate': solver_stats.get('success_rate', 0.0),
            'total_solves': solver_stats.get('total_solves', 0),
            'total_failures': self.pm_statistics['solver_failures']
        }
        
        return diagnostics
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Get convergence diagnostics for P&M training.
        
        Returns:
            Dictionary with convergence metrics.
        """
        diagnostics = {
            'method': self.name,
            'total_samples': self.pm_statistics['total_samples'],
            'mean_reconstruction_error': np.mean(self.pm_statistics['reconstruction_errors']) 
                                       if self.pm_statistics['reconstruction_errors'] else 0.0,
            'mean_energy_gap': np.mean(self.pm_statistics['energy_gaps']) 
                             if self.pm_statistics['energy_gaps'] else 0.0,
            'solver_failure_rate': self.pm_statistics['solver_failures'] / max(1, self.pm_statistics['total_samples'])
        }
        
        # Add solver-specific diagnostics
        solver_diagnostics = self.get_solver_diagnostics()
        diagnostics.update({f'solver_{k}': v for k, v in solver_diagnostics.items()})
        
        return diagnostics
    
    def adapt_solver_parameters(self, metrics: Dict[str, float]) -> bool:
        """
        Adaptive solver parameter adjustment based on performance.
        
        Args:
            metrics: Current training metrics.
            
        Returns:
            True if parameters were adjusted, False otherwise.
        """
        # Simple adaptive timeout based on solve times
        solver_stats = self.sampler.get_solver_statistics()
        avg_solve_time = solver_stats.get('avg_solve_time', 0.0)
        success_rate = solver_stats.get('success_rate', 1.0)
        
        adjusted = False
        
        # If success rate is low, increase timeout
        if success_rate < 0.8 and avg_solve_time > 0:
            self.solver_timeout *= 1.2
            adjusted = True
            print(f"Increased solver timeout to {self.solver_timeout:.1f}s (success rate: {success_rate:.3f})")
        
        # If solve times are consistently very fast, we might reduce timeout
        elif success_rate > 0.95 and avg_solve_time < self.solver_timeout * 0.5:
            self.solver_timeout *= 0.9
            adjusted = True
            print(f"Reduced solver timeout to {self.solver_timeout:.1f}s (avg solve time: {avg_solve_time:.3f}s)")
        
        return adjusted
    
    @property
    def name(self) -> str:
        """Return the name of this training method."""
        return f"PerturbAndMAP({self.solver.name})"
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return the hyperparameters specific to this training method."""
        return {
            'solver': self.solver.name,
            'gumbel_scale': self.gumbel_scale,
            'solver_timeout': self.solver_timeout,
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
            'pm_statistics': self.pm_statistics.copy(),
            'convergence_diagnostics': self.get_convergence_diagnostics(),
            'solver_diagnostics': self.get_solver_diagnostics(),
            'sampler_statistics': self.sampler.get_statistics()
        }
        
        return stats