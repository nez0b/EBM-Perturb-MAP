#!/usr/bin/env python3
"""
Inference script for trained RBM models.

This script provides a command-line interface for running inference tasks
with trained RBM models, including reconstruction and generation.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbm.models.rbm import RBM
from rbm.models.hybrid import CNN_RBM
from rbm.data.mnist import load_mnist_data, create_noisy_mnist
from rbm.inference.reconstruction import reconstruct_image, reconstruct_batch, denoise_image
from rbm.inference.generation import generate_samples, generate_from_joint_sampling
from rbm.utils.config import ConfigManager, validate_config, load_config
from rbm.utils.visualization import plot_reconstruction, plot_generation, plot_interpolation
from rbm.solvers.gurobi import GurobiSolver
from rbm.solvers.scip import ScipSolver
from rbm.solvers.dirac import DiracSolver


def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> torch.nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        
    Returns:
        Loaded model.
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.6f}")
    
    return model


def create_solver(config: dict):
    """Create QUBO solver based on configuration."""
    solver_config = config['solver']
    solver_name = solver_config['name']
    
    if solver_name == 'gurobi':
        if not GurobiSolver.is_available:
            raise RuntimeError("Gurobi solver is not available")
        return GurobiSolver(
            suppress_output=solver_config.get('suppress_output', True),
            time_limit=solver_config.get('time_limit', 60.0)
        )
    elif solver_name == 'scip':
        if not ScipSolver.is_available:
            raise RuntimeError("SCIP solver is not available")
        return ScipSolver(
            time_limit=solver_config.get('time_limit', 60.0)
        )
    elif solver_name == 'dirac':
        if not DiracSolver.is_available:
            raise RuntimeError("Dirac solver is not available")
        return DiracSolver(
            num_samples=solver_config.get('num_samples', 10),
            relaxation_schedule=solver_config.get('relaxation_schedule', 1)
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def run_reconstruction_task(model, solver, config, output_dir):
    """Run reconstruction and denoising tasks."""
    print("\n=== Running Reconstruction Task ===")
    
    # Load test data
    test_loader, _ = load_mnist_data(config, train=False)
    test_batch = next(iter(test_loader))[0]
    
    # Select a few images
    num_images = min(5, test_batch.size(0))
    test_images = test_batch[:num_images]
    
    # Create noisy versions
    noisy_images = create_noisy_mnist(test_images, noise_prob=0.1)
    
    # Reconstruct clean images
    clean_reconstructed = reconstruct_batch(model, test_images, solver)
    
    # Reconstruct noisy images (denoising)
    denoised_images = []
    for i in range(num_images):
        denoised = denoise_image(model, noisy_images[i], solver, num_iterations=1)
        denoised_images.append(denoised)
    denoised_images = torch.stack(denoised_images)
    
    # Get image shape from config
    image_size = config['data']['image_size']
    image_shape = tuple(image_size)
    
    # Plot results
    plot_reconstruction(
        original=test_images,
        reconstructed=clean_reconstructed,
        image_shape=image_shape,
        save_path=str(output_dir / "reconstruction_clean.png"),
        title="Clean Image Reconstruction"
    )
    
    plot_reconstruction(
        original=test_images,
        reconstructed=denoised_images,
        noisy=noisy_images,
        image_shape=image_shape,
        save_path=str(output_dir / "reconstruction_denoising.png"),
        title="Image Denoising"
    )
    
    print("Reconstruction task completed!")


def run_generation_task(model, solver, config, output_dir):
    """Run sample generation tasks."""
    print("\n=== Running Generation Task ===")
    
    inference_config = config['inference']
    
    # Generate samples using Gibbs sampling
    print("Generating samples using Gibbs sampling...")
    gibbs_samples = generate_samples(
        model=model,
        solver=solver,
        num_samples=inference_config['num_generated_samples'],
        gibbs_steps=inference_config['gibbs_steps'],
        verbose=True
    )
    
    # Generate samples using joint sampling
    print("Generating samples using joint P&M sampling...")
    joint_samples = generate_from_joint_sampling(
        model=model,
        solver=solver,
        num_samples=inference_config['num_generated_samples']
    )
    
    # Get image shape from config
    image_size = config['data']['image_size']
    image_shape = tuple(image_size)
    
    # Plot results
    plot_generation(
        generated_samples=gibbs_samples,
        image_shape=image_shape,
        save_path=str(output_dir / "generation_gibbs.png"),
        title="Generated Samples (Gibbs Sampling)"
    )
    
    plot_generation(
        generated_samples=joint_samples,
        image_shape=image_shape,
        save_path=str(output_dir / "generation_joint.png"),
        title="Generated Samples (Joint P&M)"
    )
    
    print("Generation task completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained RBM models"
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='default',
        help='Configuration name to use (default: default)'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file (overrides --config)'
    )
    parser.add_argument(
        '--task',
        choices=['reconstruction', 'generation', 'both'],
        default='both',
        help='Inference task to run (default: both)'
    )
    parser.add_argument(
        '--solver',
        choices=['gurobi', 'scip', 'dirac'],
        help='Override solver choice'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_results',
        help='Directory for outputs (default: ./inference_results)'
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
    
    # Apply command-line overrides
    if args.solver:
        config['solver']['name'] = args.solver
    
    # Validate configuration
    config = validate_config(config)
    
    print("=== RBM Inference ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configuration: {args.config}")
    print(f"Task: {args.task}")
    print(f"Solver: {config['solver']['name']}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)
    
    try:
        # Load model
        model = load_model_from_checkpoint(args.checkpoint, config)
        
        # Create solver
        solver = create_solver(config)
        print(f"Using {solver.name} solver")
        
        # Run inference tasks
        if args.task in ['reconstruction', 'both']:
            run_reconstruction_task(model, solver, config, output_dir)
        
        if args.task in ['generation', 'both']:
            run_generation_task(model, solver, config, output_dir)
        
        print(f"\nInference completed! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()