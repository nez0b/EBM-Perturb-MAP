#!/usr/bin/env python3
"""
Inference script for RBM models trained with Contrastive Divergence.

This script provides a command-line interface for running inference tasks
with CD-trained RBM models, including reconstruction, generation, and denoising.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbm.models.rbm import RBM
from rbm.models.hybrid import CNN_RBM
from rbm.data.mnist import load_mnist_data, create_noisy_mnist
from rbm.utils.config import ConfigManager, validate_config, load_config
from rbm.utils.visualization import plot_reconstruction, plot_generation
import matplotlib.pyplot as plt


def load_cd_checkpoint(checkpoint_path: str, config: dict) -> tuple:
    """
    Load CD model from TrainingManager checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (model, checkpoint_data).
    """
    # Create model based on config
    model_config = config['model']
    model_type = model_config['model_type']
    
    if model_type == 'rbm':
        model = RBM(
            n_visible=model_config['n_visible'],
            n_hidden=model_config['n_hidden']
        )
    elif model_type == 'cnn_rbm':
        model = CNN_RBM(
            cnn_feature_dim=64,
            rbm_hidden_dim=model_config['n_hidden']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded CD model from {checkpoint_path}")
    print(f"Training method: {checkpoint.get('method', 'Unknown')}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    # Extract final metrics from training history
    if 'training_history' in checkpoint and checkpoint['training_history']:
        final_metrics = checkpoint['training_history'][-1]
        print(f"Final reconstruction error: {final_metrics.get('reconstruction_error', 'N/A'):.6f}")
        print(f"Final positive energy: {final_metrics.get('positive_energy', 'N/A'):.6f}")
        print(f"Final negative energy: {final_metrics.get('negative_energy', 'N/A'):.6f}")
    
    return model, checkpoint


def cd_gibbs_step(model: RBM, v: torch.Tensor, deterministic: bool = False) -> tuple:
    """
    Perform one step of Gibbs sampling: v -> h -> v'.
    
    Args:
        model: RBM model.
        v: Current visible state.
        deterministic: If True, use deterministic sampling.
        
    Returns:
        Tuple of (v_new, h_sample, v_prob, h_prob).
    """
    # v -> h (positive phase)
    h_prob = model.forward(v)
    
    if deterministic:
        h_sample = (h_prob > 0.5).float()
    else:
        h_sample = torch.bernoulli(h_prob)
    
    # h -> v' (negative phase)
    v_prob = model.reconstruct(h_sample)
    
    if deterministic:
        v_new = (v_prob > 0.5).float()
    else:
        v_new = torch.bernoulli(v_prob)
    
    return v_new, h_sample, v_prob, h_prob


def cd_reconstruct_image(model: RBM, image: torch.Tensor, k_steps: int = 1) -> torch.Tensor:
    """
    Reconstruct image using CD-style Gibbs sampling.
    
    Args:
        model: RBM model.
        image: Input image tensor.
        k_steps: Number of Gibbs sampling steps.
        
    Returns:
        Reconstructed image tensor.
    """
    model.eval()
    with torch.no_grad():
        # Flatten and binarize input
        v = image.view(-1)
        v = (v > 0.5).float()
        
        # Run k steps of Gibbs sampling
        for _ in range(k_steps):
            v, _, v_prob, _ = cd_gibbs_step(model, v.unsqueeze(0))
            v = v.squeeze(0)
        
        return v_prob.squeeze(0)


def cd_reconstruct_batch(model: RBM, images: torch.Tensor, k_steps: int = 1) -> torch.Tensor:
    """
    Reconstruct batch of images using CD-style Gibbs sampling.
    
    Args:
        model: RBM model.
        images: Batch of input images.
        k_steps: Number of Gibbs sampling steps.
        
    Returns:
        Batch of reconstructed images.
    """
    model.eval()
    with torch.no_grad():
        batch_size = images.size(0)
        
        # Flatten and binarize input
        v = images.view(batch_size, -1)
        v = (v > 0.5).float()
        
        # Run k steps of Gibbs sampling
        for _ in range(k_steps):
            v, _, v_prob, _ = cd_gibbs_step(model, v)
        
        return v_prob


def cd_generate_samples(model: RBM, num_samples: int, gibbs_steps: int = 1000, 
                       initial_state: torch.Tensor = None, verbose: bool = True) -> torch.Tensor:
    """
    Generate samples using CD-style Gibbs sampling.
    
    Args:
        model: RBM model.
        num_samples: Number of samples to generate.
        gibbs_steps: Number of Gibbs sampling steps.
        initial_state: Optional initial state.
        verbose: Whether to print progress.
        
    Returns:
        Generated samples tensor.
    """
    model.eval()
    with torch.no_grad():
        # Initialize visible state
        if initial_state is not None:
            v = initial_state.clone()
            if v.dim() == 1:
                v = v.unsqueeze(0).repeat(num_samples, 1)
        else:
            v = torch.bernoulli(torch.ones(num_samples, model.n_visible) * 0.5)
        
        if verbose:
            print(f"Generating {num_samples} samples with {gibbs_steps} Gibbs steps...")
        
        # Run Gibbs sampling chain
        for step in range(gibbs_steps):
            v, _, _, _ = cd_gibbs_step(model, v)
            
            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{gibbs_steps}")
        
        return v


def cd_denoise_image(model: RBM, noisy_image: torch.Tensor, k_steps: int = 5) -> torch.Tensor:
    """
    Denoise image using CD-style Gibbs sampling.
    
    Args:
        model: RBM model.
        noisy_image: Noisy input image.
        k_steps: Number of Gibbs sampling steps.
        
    Returns:
        Denoised image tensor.
    """
    return cd_reconstruct_image(model, noisy_image, k_steps)


def run_reconstruction_task(model: RBM, config: dict, output_dir: Path, k_steps: int = 1):
    """Run reconstruction and denoising tasks."""
    print(f"\n=== Running CD Reconstruction Task (k={k_steps}) ===")
    
    # Load test data
    test_loader, _ = load_mnist_data(config, train=False)
    test_batch = next(iter(test_loader))[0]
    
    # Select a few images
    num_images = min(8, test_batch.size(0))
    test_images = test_batch[:num_images]
    
    # Create noisy versions
    noisy_images = create_noisy_mnist(test_images, noise_prob=0.1)
    
    # Reconstruct clean images
    print("Reconstructing clean images...")
    clean_reconstructed = cd_reconstruct_batch(model, test_images, k_steps)
    
    # Reconstruct noisy images (denoising)
    print("Denoising noisy images...")
    denoised_images = cd_reconstruct_batch(model, noisy_images, k_steps)
    
    # Get image shape from config
    image_size = config['data']['image_size']
    image_shape = tuple(image_size)
    
    # Plot results
    plot_reconstruction(
        original=test_images,
        reconstructed=clean_reconstructed,
        image_shape=image_shape,
        save_path=str(output_dir / "cd_reconstruction_clean.png"),
        title=f"CD Clean Reconstruction (k={k_steps})"
    )
    
    plot_reconstruction(
        original=test_images,
        reconstructed=denoised_images,
        noisy=noisy_images,
        image_shape=image_shape,
        save_path=str(output_dir / "cd_reconstruction_denoising.png"),
        title=f"CD Image Denoising (k={k_steps})"
    )
    
    # Calculate reconstruction error
    clean_error = torch.mean((test_images.view(num_images, -1) - clean_reconstructed)**2)
    denoise_error = torch.mean((test_images.view(num_images, -1) - denoised_images)**2)
    
    print(f"Clean reconstruction error: {clean_error:.6f}")
    print(f"Denoising reconstruction error: {denoise_error:.6f}")
    print("CD reconstruction task completed!")


def run_generation_task(model: RBM, config: dict, output_dir: Path, gibbs_steps: int = 1000):
    """Run sample generation tasks."""
    print(f"\n=== Running CD Generation Task ({gibbs_steps} steps) ===")
    
    inference_config = config.get('inference', {})
    num_samples = inference_config.get('num_generated_samples', 10)
    
    # Generate samples using CD Gibbs sampling
    print("Generating samples using CD Gibbs sampling...")
    start_time = time.time()
    generated_samples = cd_generate_samples(
        model=model,
        num_samples=num_samples,
        gibbs_steps=gibbs_steps,
        verbose=True
    )
    generation_time = time.time() - start_time
    
    # Get image shape from config
    image_size = config['data']['image_size']
    image_shape = tuple(image_size)
    
    # Plot results
    plot_generation(
        generated_samples=generated_samples,
        image_shape=image_shape,
        save_path=str(output_dir / "cd_generation_samples.png"),
        title=f"CD Generated Samples ({gibbs_steps} steps)"
    )
    
    print(f"Generation completed in {generation_time:.2f}s")
    print(f"Average time per sample: {generation_time/num_samples:.2f}s")
    print("CD generation task completed!")


def run_interpolation_task(model: RBM, config: dict, output_dir: Path, num_steps: int = 8):
    """Run interpolation task between two samples."""
    print(f"\n=== Running CD Interpolation Task ({num_steps} steps) ===")
    
    # Load test data
    test_loader, _ = load_mnist_data(config, train=False)
    test_batch = next(iter(test_loader))[0]
    
    # Select two different images
    img1 = test_batch[0].view(-1)
    img2 = test_batch[1].view(-1)
    
    # Create interpolation steps
    interpolations = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        interpolated = (1 - alpha) * img1 + alpha * img2
        interpolated = (interpolated > 0.5).float()  # Binarize
        
        # Reconstruct interpolated image
        reconstructed = cd_reconstruct_image(model, interpolated, k_steps=1)
        interpolations.append(reconstructed)
    
    interpolations = torch.stack(interpolations)
    
    # Get image shape from config
    image_size = config['data']['image_size']
    image_shape = tuple(image_size)
    
    # Plot interpolation
    fig, axes = plt.subplots(2, num_steps, figsize=(2*num_steps, 4))
    
    for i in range(num_steps):
        # Original interpolation
        axes[0, i].imshow(((1 - i/(num_steps-1)) * img1 + (i/(num_steps-1)) * img2).view(image_shape), 
                         cmap='gray')
        axes[0, i].set_title(f'Step {i+1}')
        axes[0, i].axis('off')
        
        # CD reconstruction
        axes[1, i].imshow(interpolations[i].view(image_shape), cmap='gray')
        axes[1, i].set_title(f'CD Recon')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', rotation=90, size='large')
    axes[1, 0].set_ylabel('CD Reconstruction', rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "cd_interpolation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("CD interpolation task completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with CD-trained RBM models"
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to CD model checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='contrastive_divergence',
        help='Configuration name to use (default: contrastive_divergence)'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file (overrides --config)'
    )
    parser.add_argument(
        '--task',
        choices=['reconstruction', 'generation', 'interpolation', 'all'],
        default='all',
        help='Inference task to run (default: all)'
    )
    parser.add_argument(
        '--k-steps',
        type=int,
        default=1,
        help='Number of Gibbs steps for reconstruction (default: 1)'
    )
    parser.add_argument(
        '--gibbs-steps',
        type=int,
        default=1000,
        help='Number of Gibbs steps for generation (default: 1000)'
    )
    parser.add_argument(
        '--digit-filter',
        type=int,
        help='Filter test data to specific digit (0-9)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cd_inference_results',
        help='Directory for outputs (default: ./cd_inference_results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.config_file:
        config = load_config(args.config_file)
    else:
        try:
            config = config_manager.load(args.config)
        except FileNotFoundError:
            print(f"Configuration '{args.config}' not found. Using default configuration.")
            config = config_manager.create_default()
    
    # Apply digit filter if specified
    if args.digit_filter is not None:
        config.setdefault('data', {})['digit_filter'] = args.digit_filter
    
    # Validate configuration
    config = validate_config(config)
    
    print("=== CD-RBM Inference ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configuration: {args.config}")
    print(f"Task: {args.task}")
    print(f"K-steps (reconstruction): {args.k_steps}")
    print(f"Gibbs steps (generation): {args.gibbs_steps}")
    if args.digit_filter is not None:
        print(f"Digit filter: {args.digit_filter}")
    print(f"Output directory: {output_dir}")
    print("-" * 40)
    
    try:
        # Load model
        model, checkpoint = load_cd_checkpoint(args.checkpoint, config)
        print(f"Model loaded: {model.n_visible} visible, {model.n_hidden} hidden units")
        
        # Run inference tasks
        if args.task in ['reconstruction', 'all']:
            run_reconstruction_task(model, config, output_dir, args.k_steps)
        
        if args.task in ['generation', 'all']:
            run_generation_task(model, config, output_dir, args.gibbs_steps)
        
        if args.task in ['interpolation', 'all']:
            run_interpolation_task(model, config, output_dir)
        
        print(f"\nCD inference completed! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()