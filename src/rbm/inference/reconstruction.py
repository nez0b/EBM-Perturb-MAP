"""
Image reconstruction utilities for RBM models.

This module provides functions for reconstructing images using trained RBM models
with Perturb-and-MAP sampling.
"""

import torch
import numpy as np
from typing import Union

from ..models.rbm import RBM
from ..solvers.base import QUBOSolver


def reconstruct_image(
    model: RBM, 
    image_tensor: torch.Tensor, 
    solver: QUBOSolver
) -> torch.Tensor:
    """
    Perform a single pass of reconstruction on a given image.
    
    This is equivalent to one full step of Gibbs sampling (v -> h -> v').
    Uses Perturb-and-MAP sampling for the hidden layer.
    
    Args:
        model: The trained RBM model.
        image_tensor: A single image tensor (can be 2D or 3D).
        solver: QUBO solver for MAP optimization.
        
    Returns:
        The reconstructed image tensor.
    """
    model.eval()
    
    # Flatten the input tensor to be a 1D vector
    v_input = (image_tensor.view(-1) > 0.5).float()

    # --- POSITIVE PHASE: v -> h ---
    # Use the P&M and QUBO solver to sample the hidden layer
    Q_h, _ = model.create_qubo_for_sampling(v_input)
    h_sample_np = solver.solve(Q_h)
    h_sample = torch.from_numpy(h_sample_np).float()
    
    # --- NEGATIVE PHASE: h -> v' ---
    # Reconstruct the visible layer from the hidden sample
    v_reconstructed_probs = model.reconstruct(h_sample)
    
    return v_reconstructed_probs


def reconstruct_batch(
    model: RBM,
    batch_tensor: torch.Tensor,
    solver: QUBOSolver
) -> torch.Tensor:
    """
    Reconstruct a batch of images.
    
    Args:
        model: The trained RBM model.
        batch_tensor: Batch of images tensor.
        solver: QUBO solver for MAP optimization.
        
    Returns:
        Batch of reconstructed images.
    """
    model.eval()
    batch_size = batch_tensor.size(0)
    reconstructions = []
    
    for i in range(batch_size):
        reconstructed = reconstruct_image(model, batch_tensor[i], solver)
        reconstructions.append(reconstructed)
    
    return torch.stack(reconstructions)


def denoise_image(
    model: RBM,
    noisy_image: torch.Tensor,
    solver: QUBOSolver,
    num_iterations: int = 1
) -> torch.Tensor:
    """
    Denoise an image using multiple reconstruction iterations.
    
    Args:
        model: The trained RBM model.
        noisy_image: Noisy input image tensor.
        solver: QUBO solver for MAP optimization.
        num_iterations: Number of denoising iterations.
        
    Returns:
        Denoised image tensor.
    """
    current_image = noisy_image.clone()
    
    for _ in range(num_iterations):
        current_image = reconstruct_image(model, current_image, solver)
        # Binarize for next iteration
        current_image = (current_image > 0.5).float()
    
    return current_image


def compute_reconstruction_error(
    model: RBM,
    test_images: torch.Tensor,
    solver: QUBOSolver,
    num_samples: int = None
) -> float:
    """
    Compute average reconstruction error on test images.
    
    Args:
        model: The trained RBM model.
        test_images: Test images tensor.
        solver: QUBO solver for MAP optimization.
        num_samples: Number of samples to test (None for all).
        
    Returns:
        Average reconstruction error.
    """
    model.eval()
    
    if num_samples is not None:
        test_images = test_images[:num_samples]
    
    total_error = 0.0
    num_images = test_images.size(0)
    
    with torch.no_grad():
        for i in range(num_images):
            original = test_images[i]
            reconstructed = reconstruct_image(model, original, solver)
            
            # Flatten for error calculation
            original_flat = original.view(-1)
            reconstructed_flat = reconstructed.view(-1)
            
            error = torch.mean((original_flat - reconstructed_flat)**2)
            total_error += error.item()
    
    return total_error / num_images


def interpolate_reconstructions(
    model: RBM,
    image1: torch.Tensor,
    image2: torch.Tensor,
    solver: QUBOSolver,
    num_steps: int = 5
) -> torch.Tensor:
    """
    Create interpolations between two images in the latent space.
    
    Args:
        model: The trained RBM model.
        image1: First image tensor.
        image2: Second image tensor.
        solver: QUBO solver for MAP optimization.
        num_steps: Number of interpolation steps.
        
    Returns:
        Tensor containing interpolated reconstructions.
    """
    model.eval()
    
    # Get hidden representations
    v1 = (image1.view(-1) > 0.5).float()
    v2 = (image2.view(-1) > 0.5).float()
    
    Q_h1, _ = model.create_qubo_for_sampling(v1)
    Q_h2, _ = model.create_qubo_for_sampling(v2)
    
    h1 = torch.from_numpy(solver.solve(Q_h1)).float()
    h2 = torch.from_numpy(solver.solve(Q_h2)).float()
    
    # Create interpolations
    interpolations = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        h_interp = (1 - alpha) * h1 + alpha * h2
        
        # Binarize interpolated hidden state
        h_interp = (h_interp > 0.5).float()
        
        v_interp = model.reconstruct(h_interp)
        interpolations.append(v_interp)
    
    return torch.stack(interpolations)