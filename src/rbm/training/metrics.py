"""
Training metrics and logging utilities.

This module provides functions for calculating training metrics and
logging training progress.
"""

import torch
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import os


def calculate_reconstruction_error(
    original: torch.Tensor, 
    reconstructed: torch.Tensor
) -> float:
    """
    Calculate mean squared reconstruction error.
    
    Args:
        original: Original input tensor.
        reconstructed: Reconstructed tensor.
        
    Returns:
        Mean squared error between original and reconstructed.
    """
    return torch.mean((original - reconstructed)**2).item()


def calculate_free_energy(model, v: torch.Tensor) -> float:
    """
    Calculate the free energy of visible units.
    
    Args:
        model: RBM model.
        v: Visible units tensor.
        
    Returns:
        Free energy value.
    """
    with torch.no_grad():
        # F(v) = -b^T v - sum_j log(1 + exp(c_j + W_j^T v))
        bias_term = torch.sum(model.b * v, dim=1)
        hidden_term = torch.sum(
            torch.log(1 + torch.exp(torch.nn.functional.linear(v, model.W, model.c))),
            dim=1
        )
        free_energy = -bias_term - hidden_term
        return torch.mean(free_energy).item()


def plot_training_history(
    history: List[Dict[str, Any]], 
    save_path: str = "training_history.png"
) -> None:
    """
    Plot training history showing loss over epochs.
    
    Args:
        history: List of training history dictionaries.
        save_path: Path to save the plot.
    """
    if not history:
        print("No training history to plot.")
        return
    
    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    times = [h['time'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot timing
    ax2.plot(epochs, times, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Epoch Duration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_path}")


class TrainingLogger:
    """
    Logger for tracking training progress and metrics.
    """
    
    def __init__(self, log_file: str = "training.log"):
        self.log_file = log_file
        self.metrics_history = []
    
    def log_epoch(
        self, 
        epoch: int, 
        loss: float, 
        time: float, 
        additional_metrics: Dict[str, float] = None
    ) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number.
            loss: Training loss.
            time: Epoch duration.
            additional_metrics: Additional metrics to log.
        """
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'time': time
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.metrics_history.append(metrics)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: loss={loss:.4f}, time={time:.2f}s")
            if additional_metrics:
                for key, value in additional_metrics.items():
                    f.write(f", {key}={value:.4f}")
            f.write("\n")
    
    def get_best_epoch(self, metric: str = 'loss', minimize: bool = True) -> Dict[str, Any]:
        """
        Get the epoch with the best metric value.
        
        Args:
            metric: Metric name to optimize.
            minimize: Whether to minimize (True) or maximize (False) the metric.
            
        Returns:
            Dictionary with the best epoch's metrics.
        """
        if not self.metrics_history:
            return {}
        
        if minimize:
            best_epoch = min(self.metrics_history, key=lambda x: x.get(metric, float('inf')))
        else:
            best_epoch = max(self.metrics_history, key=lambda x: x.get(metric, float('-inf')))
        
        return best_epoch