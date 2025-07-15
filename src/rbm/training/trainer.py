"""
Training orchestration for RBM models.

This module contains the Trainer class that handles the training process
for RBM models using Perturb-and-MAP methodology.
"""

import torch
import torch.optim as optim
import time
from typing import Dict, Any, Optional, Callable
import numpy as np
from torch.utils.data import DataLoader

from ..models.rbm import RBM
from ..solvers.base import QUBOSolver


class Trainer:
    """
    Trainer class for RBM models using Perturb-and-MAP methodology.
    
    This class orchestrates the training process, handles checkpointing,
    logging, and provides a clean interface for training RBM models.
    
    Args:
        model: The RBM model to train.
        solver: The QUBO solver to use for MAP optimization.
        optimizer: PyTorch optimizer for parameter updates.
        config: Training configuration dictionary.
    """
    
    def __init__(
        self,
        model: RBM,
        solver: QUBOSolver,
        optimizer: optim.Optimizer,
        config: Dict[str, Any]
    ):
        self.model = model
        self.solver = solver
        self.optimizer = optimizer
        self.config = config
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        # Configuration with defaults
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 10)
        self.learning_rate = training_config.get('learning_rate', 0.01)
        self.batch_limit = training_config.get('batch_limit', None)  # For limiting batches during training
        self.checkpoint_every = training_config.get('checkpoint_every', 5)
        self.checkpoint_path = training_config.get('checkpoint_path', 'rbm_checkpoint.pth')
        
        self.model.train()
    
    def train(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the RBM model using Perturb-and-MAP methodology.
        
        Args:
            data_loader: DataLoader providing training data.
            
        Returns:
            Training history dictionary containing losses and metrics.
        """
        print(f"Starting RBM training with {self.solver.name} solver")
        print(f"Epochs: {self.epochs}, Learning Rate: {self.learning_rate}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            epoch_loss = self._train_epoch(data_loader, epoch)
            epoch_time = time.time() - epoch_start_time
            
            # Log progress
            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Avg Recon Error: {epoch_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'time': epoch_time
            })
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, epoch_loss)
            
            self.current_epoch = epoch + 1
        
        print("-" * 50)
        print("Training completed!")
        
        return {
            'history': self.training_history,
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None,
            'total_epochs': self.current_epoch
        }
    
    def _train_epoch(self, data_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch using the corrected P&M batch training approach.
        
        Args:
            data_loader: DataLoader for this epoch.
            epoch: Current epoch number.
            
        Returns:
            Average reconstruction error for this epoch.
        """
        total_error = 0.0
        batch_count = 0
        
        for i, (batch_data, _) in enumerate(data_loader):
            # Apply batch limit if specified
            if self.batch_limit is not None and i >= self.batch_limit:
                break
            
            batch_count += 1
            batch_size = batch_data.size(0)
            
            # Prepare positive phase data
            v_positive = batch_data.view(batch_size, -1)
            v_positive = (v_positive > torch.rand_like(v_positive)).float()
            
            # --- POSITIVE PHASE (Analytic & Vectorized) ---
            h_positive_probs = self.model(v_positive)
            
            # --- NEGATIVE PHASE (P&M Sampling & Vectorized) ---
            v_negative_samples, h_negative_samples = [], []
            
            for _ in range(batch_size):
                # Generate joint sample using P&M
                Q_joint = self.model.create_joint_qubo()
                vh_sample_np = self.solver.solve(Q_joint)
                
                v_sample = torch.from_numpy(vh_sample_np[:self.model.n_visible]).float()
                h_sample = torch.from_numpy(vh_sample_np[self.model.n_visible:]).float()
                
                v_negative_samples.append(v_sample)
                h_negative_samples.append(h_sample)
            
            v_negative = torch.stack(v_negative_samples)
            h_negative = torch.stack(h_negative_samples)
            
            # --- GRADIENT CALCULATION (Vectorized) ---
            pos_assoc = torch.bmm(h_positive_probs.unsqueeze(2), v_positive.unsqueeze(1))
            neg_assoc = torch.bmm(h_negative.unsqueeze(2), v_negative.unsqueeze(1))
            
            grad_W = torch.mean(pos_assoc - neg_assoc, dim=0)
            grad_b = torch.mean(v_positive - v_negative, dim=0)
            grad_c = torch.mean(h_positive_probs - h_negative, dim=0)
            
            # --- PARAMETER UPDATE ---
            self.model.W.grad = -grad_W
            self.model.b.grad = -grad_b
            self.model.c.grad = -grad_c
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Track reconstruction error
            with torch.no_grad():
                v_reconstructed = self.model.reconstruct(h_positive_probs)
                error = torch.mean((v_positive - v_reconstructed)**2)
                total_error += error.item()
        
        return total_error / batch_count if batch_count > 0 else 0.0
    
    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number.
            loss: Current loss value.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            
        Returns:
            Checkpoint data dictionary.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint