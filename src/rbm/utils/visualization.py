"""
Visualization utilities for RBM experiments.

This module provides functions for plotting and visualizing RBM training results,
reconstructions, and generated samples.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import os
from pathlib import Path


def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    noisy: Optional[torch.Tensor] = None,
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None,
    title: str = "RBM Reconstruction"
) -> None:
    """
    Plot original vs reconstructed images.
    
    Args:
        original: Original images tensor.
        reconstructed: Reconstructed images tensor.
        noisy: Optional noisy input images.
        image_shape: Shape to reshape images to (height, width).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    num_images = min(5, original.size(0))
    
    if noisy is not None:
        fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
        row_labels = ["Original", "Noisy", "Reconstructed"]
        images_list = [original, noisy, reconstructed]
    else:
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        row_labels = ["Original", "Reconstructed"]
        images_list = [original, reconstructed]
    
    fig.suptitle(title, fontsize=16)
    
    for row, (images, label) in enumerate(zip(images_list, row_labels)):
        for col in range(num_images):
            if len(axes.shape) == 1:  # Single row
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Reshape and plot image
            img = images[col].detach().view(image_shape).numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col == 0:
                ax.set_ylabel(label)
    
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction plot saved to {save_path}")
    
    plt.show()


def plot_generation(
    generated_samples: torch.Tensor,
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None,
    title: str = "RBM Generated Samples"
) -> None:
    """
    Plot generated samples in a grid.
    
    Args:
        generated_samples: Generated images tensor.
        image_shape: Shape to reshape images to (height, width).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    num_samples = generated_samples.size(0)
    
    # Determine grid size
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=16)
    
    # Handle case where axes might be 1D
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        img = generated_samples[i].detach().view(image_shape).numpy()
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Sample {i+1}")
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generation plot saved to {save_path}")
    
    plt.show()


def plot_interpolation(
    interpolated_images: torch.Tensor,
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None,
    title: str = "RBM Interpolation"
) -> None:
    """
    Plot interpolated images in a row.
    
    Args:
        interpolated_images: Interpolated images tensor.
        image_shape: Shape to reshape images to (height, width).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    num_steps = interpolated_images.size(0)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    fig.suptitle(title, fontsize=16)
    
    if num_steps == 1:
        axes = [axes]
    
    for i in range(num_steps):
        img = interpolated_images[i].detach().view(image_shape).numpy()
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Step {i+1}")
    
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Interpolation plot saved to {save_path}")
    
    plt.show()


def plot_training_progress(
    losses: List[float],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Progress"
) -> None:
    """
    Plot training loss over epochs.
    
    Args:
        losses: List of loss values.
        epochs: List of epoch numbers (if None, uses 1, 2, ...).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    if epochs is None:
        epochs = list(range(1, len(losses) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    
    plt.show()


def plot_weight_matrices(
    weights: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "RBM Weight Matrices"
) -> None:
    """
    Visualize RBM weight matrices as images.
    
    Args:
        weights: Weight tensor of shape (n_hidden, n_visible).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    n_hidden, n_visible = weights.shape
    
    # Try to infer image shape from n_visible
    img_size = int(np.sqrt(n_visible))
    if img_size * img_size != n_visible:
        print(f"Warning: Cannot visualize weights - n_visible ({n_visible}) is not a perfect square")
        return
    
    # Determine grid size for displaying weight vectors
    cols = min(8, n_hidden)
    rows = (n_hidden + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=16)
    
    # Handle case where axes might be 1D
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(min(n_hidden, len(axes))):
        weight_img = weights[i].detach().view(img_size, img_size).numpy()
        im = axes[i].imshow(weight_img, cmap='RdBu_r', vmin=-weight_img.std(), vmax=weight_img.std())
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Hidden {i+1}")
    
    # Hide extra subplots
    for i in range(n_hidden, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Weight matrices plot saved to {save_path}")
    
    plt.show()


def plot_comparison_grid(
    images_dict: dict,
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None,
    title: str = "Comparison Grid"
) -> None:
    """
    Plot a comparison grid of different image sets.
    
    Args:
        images_dict: Dictionary with keys as labels and values as image tensors.
        image_shape: Shape to reshape images to (height, width).
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    labels = list(images_dict.keys())
    num_conditions = len(labels)
    num_samples = min(5, min(images.size(0) for images in images_dict.values()))
    
    fig, axes = plt.subplots(num_conditions, num_samples, 
                            figsize=(num_samples * 2, num_conditions * 2))
    fig.suptitle(title, fontsize=16)
    
    # Handle different axis configurations
    if num_conditions == 1 and num_samples == 1:
        axes = [[axes]]
    elif num_conditions == 1:
        axes = [axes]
    elif num_samples == 1:
        axes = [[ax] for ax in axes]
    
    for row, label in enumerate(labels):
        images = images_dict[label]
        for col in range(num_samples):
            img = images[col].detach().view(image_shape).numpy()
            axes[row][col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            
            if col == 0:
                axes[row][col].set_ylabel(label)
            if row == 0:
                axes[row][col].set_title(f"Sample {col+1}")
    
    plt.tight_layout()
    
    if save_path:
        _ensure_dir_exists(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison grid saved to {save_path}")
    
    plt.show()


def _ensure_dir_exists(file_path: str) -> None:
    """Ensure the directory for the file path exists."""
    directory = os.path.dirname(file_path)
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)