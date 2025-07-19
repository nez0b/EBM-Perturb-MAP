"""
Training manager for coordinating different EBM training methods.

This module provides a high-level interface for training EBMs with different
methods, handling configuration, model creation, and training orchestration.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from tqdm import tqdm

from .methods.base import EBMTrainingMethod
from .methods.contrastive_divergence import ContrastiveDivergenceTraining
from ..models.rbm import RBM
from ..utils.config import validate_config


class TrainingManager:
    """
    High-level training manager for EBM training methods.
    
    This class coordinates the training process, handles configuration,
    creates appropriate training methods, and manages the training loop.
    
    Args:
        config: Training configuration dictionary.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = validate_config(config)
        
        # Create model and optimizer
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        
        # Create training method
        self.training_method = self._create_training_method()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.best_loss = float('inf')
        
        # Configuration shortcuts
        training_config = self.config.get('training', {})
        self.epochs = training_config.get('epochs', 10)
        self.batch_limit = training_config.get('batch_limit', None)
        self.checkpoint_every = training_config.get('checkpoint_every', 5)
        self.checkpoint_path = training_config.get('checkpoint_path', 'rbm_checkpoint.pth')
        
        print(f"Initialized TrainingManager with {self.training_method.name}")
    
    def _create_model(self) -> nn.Module:
        """Create the RBM model based on configuration."""
        model_config = self.config['model']
        model_type = model_config['model_type']
        
        if model_type == 'rbm':
            return RBM(
                n_visible=model_config['n_visible'],
                n_hidden=model_config['n_hidden']
            )
        elif model_type == 'cnn_rbm':
            # Future: implement CNN-RBM
            raise NotImplementedError("CNN-RBM not implemented yet")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer based on configuration."""
        training_config = self.config['training']
        optimizer_name = training_config.get('optimizer', 'sgd')
        learning_rate = training_config['learning_rate']
        
        if optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_solver(self):
        """Create QUBO solver for P&M training."""
        solver_config = self.config['solver']
        solver_name = solver_config['name']
        
        if solver_name == 'gurobi':
            from ..solvers.gurobi import GurobiSolver
            return GurobiSolver(
                suppress_output=solver_config.get('suppress_output', True),
                time_limit=solver_config.get('time_limit', 60.0)
            )
        elif solver_name == 'scip':
            from ..solvers.scip import ScipSolver
            return ScipSolver(
                time_limit=solver_config.get('time_limit', 60.0)
            )
        elif solver_name == 'hexaly':
            from ..solvers.hexaly import HexalySolver
            return HexalySolver(
                time_limit=solver_config.get('time_limit', 120.0),
                nb_threads=solver_config.get('nb_threads', 4),
                seed=solver_config.get('seed', 42)
            )
        elif solver_name == 'dirac':
            from ..solvers.dirac import DiracSolver
            return DiracSolver(
                num_samples=solver_config.get('num_samples', 10),
                relaxation_schedule=solver_config.get('relaxation_schedule', 1)
            )
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
    
    def _create_training_method(self) -> EBMTrainingMethod:
        """Create the training method based on configuration."""
        training_config = self.config['training']
        method_name = training_config.get('method', 'perturb_map')
        
        if method_name == 'contrastive_divergence':
            return ContrastiveDivergenceTraining(self.model, self.optimizer, self.config)
        elif method_name == 'perturb_map':
            # Import here to avoid circular dependencies
            from .methods.perturb_map import PerturbMapTraining
            
            # Create QUBO solver for P&M
            solver = self._create_solver()
            return PerturbMapTraining(self.model, self.optimizer, solver, self.config)
        else:
            raise ValueError(f"Unknown training method: {method_name}")
    
    def train(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the model using the configured training method.
        
        Args:
            data_loader: DataLoader providing training data.
            
        Returns:
            Training results dictionary.
        """
        print(f"Starting training with {self.training_method.name}")
        print(f"Epochs: {self.epochs}, Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Model: {self.model.__class__.__name__} "
              f"({self.model.n_visible} visible, {self.model.n_hidden} hidden)")
        if self.batch_limit is not None:
            print(f"Batch limit: {self.batch_limit} batches per epoch")
        print("-" * 60)
        
        total_start_time = time.time()
        
        # Create overall progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            epoch_metrics = self._train_epoch(data_loader, epoch)
            epoch_time = time.time() - epoch_start_time
            
            # Add timing information
            epoch_metrics['epoch_time'] = epoch_time
            epoch_metrics['epoch'] = epoch + 1
            
            # Update training history
            self.training_history.append(epoch_metrics)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'loss': f"{epoch_metrics.get('reconstruction_error', 0):.4f}",
                'batches': epoch_metrics.get('num_batches', 0)
            })
            
            # Print progress
            self._print_epoch_progress(epoch + 1, epoch_metrics)
            
            # Save checkpoint if needed
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, epoch_metrics)
            
            # Check for early stopping or learning rate adaptation
            self._check_training_progress(epoch_metrics)
            
            self.current_epoch = epoch + 1
        
        epoch_pbar.close()
        
        total_time = time.time() - total_start_time
        
        # Save final checkpoint if training completed (and not already saved)
        if self.training_history and self.current_epoch % self.checkpoint_every != 0:
            final_metrics = self.training_history[-1]
            self._save_checkpoint(self.current_epoch, final_metrics)
        
        print("-" * 60)
        print(f"Training completed in {total_time:.2f}s")
        
        return {
            'history': self.training_history,
            'final_metrics': self.training_history[-1] if self.training_history else {},
            'total_epochs': self.current_epoch,
            'total_time': total_time,
            'method': self.training_method.name,
            'hyperparameters': self.training_method.hyperparameters
        }
    
    def _train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for this epoch.
            epoch: Current epoch number.
            
        Returns:
            Epoch metrics.
        """
        epoch_metrics = []
        batch_count = 0
        
        # Calculate total batches for progress bar
        total_batches = len(data_loader)
        if self.batch_limit is not None:
            total_batches = min(total_batches, self.batch_limit)
        
        # Create progress bar for this epoch
        desc = f"Epoch {epoch + 1}/{self.epochs}"
        pbar = tqdm(enumerate(data_loader), desc=desc, total=total_batches, leave=False)
        
        for i, (batch_data, _) in pbar:
            # Apply batch limit if specified
            if self.batch_limit is not None and i >= self.batch_limit:
                break
            
            batch_count += 1
            
            # Train step
            batch_metrics = self.training_method.train_step(batch_data)
            epoch_metrics.append(batch_metrics)
            
            # Update progress bar with current metrics
            if batch_metrics:
                postfix = {
                    'loss': f"{batch_metrics.get('reconstruction_error', 0):.4f}",
                    'pos_energy': f"{batch_metrics.get('positive_energy', 0):.2f}",
                    'neg_energy': f"{batch_metrics.get('negative_energy', 0):.2f}"
                }
                pbar.set_postfix(postfix)
        
        pbar.close()
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(epoch_metrics)
        aggregated_metrics['num_batches'] = batch_count
        
        return aggregated_metrics
    
    def _aggregate_metrics(self, batch_metrics: list) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple batches.
        
        Args:
            batch_metrics: List of batch metric dictionaries.
            
        Returns:
            Aggregated metrics.
        """
        if not batch_metrics:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in batch_metrics:
            all_keys.update(metrics.keys())
        
        # Aggregate metrics
        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in batch_metrics if key in m]
            if values:
                if isinstance(values[0], (int, float)):
                    aggregated[key] = sum(values) / len(values)
                elif isinstance(values[0], bool):
                    aggregated[key] = sum(values) / len(values)
                else:
                    aggregated[key] = values[-1]  # Take last value for non-numeric
        
        return aggregated
    
    def _print_epoch_progress(self, epoch: int, metrics: Dict[str, Any]):
        """Print epoch progress."""
        print(f"Epoch {epoch}/{self.epochs} | "
              f"Loss: {metrics.get('reconstruction_error', 0):.4f} | "
              f"Pos Energy: {metrics.get('positive_energy', 0):.4f} | "
              f"Neg Energy: {metrics.get('negative_energy', 0):.4f} | "
              f"Time: {metrics.get('epoch_time', 0):.2f}s")
    
    def _check_training_progress(self, metrics: Dict[str, Any]):
        """Check training progress for early stopping or adaptation."""
        current_loss = metrics.get('reconstruction_error', float('inf'))
        
        # Update best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        
        # Adaptive learning rate (if supported by training method)
        if hasattr(self.training_method, 'adapt_learning_rate'):
            lr_changed = self.training_method.adapt_learning_rate(metrics)
            if lr_changed:
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning rate adapted to {new_lr:.6f}")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'method': self.training_method.name,
            'hyperparameters': self.training_method.hyperparameters,
            'best_loss': self.best_loss
        }
        
        # Add method-specific state
        if hasattr(self.training_method, 'get_training_statistics'):
            checkpoint['method_statistics'] = self.training_method.get_training_statistics()
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"  Checkpoint saved: {self.checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint data.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Load method-specific state
        if 'method_statistics' in checkpoint and hasattr(self.training_method, 'load_training_statistics'):
            self.training_method.load_training_statistics(checkpoint['method_statistics'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        
        Args:
            data_loader: DataLoader for evaluation data.
            
        Returns:
            Evaluation metrics.
        """
        self.model.eval()
        eval_metrics = []
        
        with torch.no_grad():
            for batch_data, _ in data_loader:
                # Simple evaluation using reconstruction error
                v_data = batch_data.view(batch_data.size(0), -1)
                v_data = (v_data > 0.5).float()
                
                h_probs = self.training_method.positive_phase(v_data)
                v_recon = self.model.reconstruct(h_probs)
                
                recon_error = torch.mean((v_data - v_recon)**2)
                eval_metrics.append({'reconstruction_error': recon_error.item()})
        
        self.model.train()
        
        return self._aggregate_metrics(eval_metrics)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Training summary dictionary.
        """
        summary = {
            'method': self.training_method.name,
            'hyperparameters': self.training_method.hyperparameters,
            'config': self.config,
            'current_epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        # Add method-specific statistics
        if hasattr(self.training_method, 'get_training_statistics'):
            summary['method_statistics'] = self.training_method.get_training_statistics()
        
        return summary