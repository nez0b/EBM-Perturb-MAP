#!/usr/bin/env python3
"""
QUBO Sampler RBM Training and Inference Script

This script demonstrates both training and inference using the QUBO sampler
(Perturb-and-MAP method) for Restricted Boltzmann Machines. It provides a
comprehensive example of how to use the EBM framework with various QUBO solvers.

=== USAGE OVERVIEW ===

The script performs two main tasks:
1. Training: Uses Perturb-and-MAP with QUBO optimization for the negative phase
2. Inference: Uses the trained model for reconstruction, denoising, and generation

=== BASIC USAGE ===

# Train and run inference with default settings (Gurobi solver)
python train_and_infer_qubo.py

# Use specific QUBO solver
python train_and_infer_qubo.py --solver hexaly

# Train on specific digit (figure 6 experiment)
python train_and_infer_qubo.py --figure6

# Custom configuration
python train_and_infer_qubo.py --config-file custom_config.yaml

# Train only (skip inference)
python train_and_infer_qubo.py --task training

# Inference only (requires existing checkpoint)
python train_and_infer_qubo.py --task inference --checkpoint path/to/checkpoint.pth

=== QUBO SOLVER OPTIONS ===

--solver gurobi    : Commercial optimizer (requires license)
--solver scip      : Open-source solver (free)
--solver hexaly    : Commercial local search optimizer
--solver dirac     : Research solver with relaxation

=== ADVANCED USAGE ===

# Performance tuning
python train_and_infer_qubo.py --solver-timeout 120 --max-retries 5

# Custom Gumbel perturbation scale
python train_and_infer_qubo.py --gumbel-scale 0.5

# Extended training with checkpointing
python train_and_infer_qubo.py --epochs 50 --checkpoint-every 10

# Quick testing with batch limit
python train_and_infer_qubo.py --epochs 3 --batch-limit 10

# Resume training from checkpoint
python train_and_infer_qubo.py --resume ./outputs/rbm_pm_checkpoint.pth --epochs 50

# Generate more samples with longer optimization
python train_and_infer_qubo.py --task inference --num-samples 20

=== SOLVER COMPARISON ===

Gurobi:     Best performance, requires license
SCIP:       Good performance, free and open-source
Hexaly:     Fast local search, good for large problems
Dirac:      Research solver, good for experimentation

=== PERFORMANCE TIPS ===

1. Start with SCIP for free experimentation
2. Use Gurobi for best performance and quality
3. Adjust solver timeout based on problem size
4. Monitor solver statistics for optimization
5. Use figure 6 experiments for faster convergence

=== OUTPUT FILES ===

Training:
- rbm_pm_checkpoint.pth: Model checkpoint
- pm_training_summary.txt: Training statistics
- figures/: Training plots and visualizations

Inference:
- qubo_reconstruction_clean.png: Clean image reconstruction
- qubo_reconstruction_denoising.png: Denoising results
- qubo_generation_samples.png: Generated samples
- qubo_solver_stats.txt: Solver performance metrics

=== TROUBLESHOOTING ===

1. Solver not available: Install required solver or use --solver scip
2. Slow convergence: Increase --solver-timeout or use --gumbel-scale 0.8
3. Poor quality: Try different solver or adjust --max-retries
4. Memory issues: Reduce --batch-size or use smaller model

=== EXAMPLES ===

# Quick test with SCIP solver
python train_and_infer_qubo.py --solver scip --epochs 5

# High-quality training with Gurobi
python train_and_infer_qubo.py --solver gurobi --epochs 30 --solver-timeout 180

# Figure 6 experiment comparison
python train_and_infer_qubo.py --figure6 --epochs 25 --output-dir ./digit6_qubo

# Solver performance comparison
python train_and_infer_qubo.py --solver scip --output-dir ./scip_results
python train_and_infer_qubo.py --solver gurobi --output-dir ./gurobi_results

# Extended training with resume capability
python train_and_infer_qubo.py --epochs 20 --output-dir ./training_run1
python train_and_infer_qubo.py --resume ./training_run1/rbm_pm_checkpoint.pth --epochs 40

"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbm.training.training_manager import TrainingManager
from rbm.data.mnist import load_mnist_data, create_noisy_mnist
from rbm.utils.config import load_config, validate_config
from rbm.utils.visualization import plot_reconstruction, plot_generation


def print_header():
    """Print script header and information."""
    print("=" * 60)
    print("QUBO Sampler RBM Training and Inference")
    print("Using Perturb-and-MAP methodology with QUBO optimization")
    print("=" * 60)


def safe_format_number(value, format_spec=".6f", fallback="N/A"):
    """
    Safely format a numeric value with fallback for non-numeric values.
    
    Args:
        value: Value to format (can be numeric or string)
        format_spec: Format specification (default: ".6f")
        fallback: Fallback value for non-numeric inputs
        
    Returns:
        Formatted string
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:{format_spec}}"
    else:
        return str(fallback)


def check_solver_availability(solver_name: str) -> bool:
    """Check if a QUBO solver is available."""
    try:
        if solver_name == 'gurobi':
            from rbm.solvers.gurobi import GurobiSolver
            return GurobiSolver.is_available
        elif solver_name == 'scip':
            from rbm.solvers.scip import ScipSolver
            return ScipSolver.is_available
        elif solver_name == 'hexaly':
            from rbm.solvers.hexaly import HexalySolver
            return HexalySolver.is_available
        elif solver_name == 'dirac':
            from rbm.solvers.dirac import DiracSolver
            return DiracSolver.is_available
        else:
            return False
    except ImportError:
        return False


def get_available_solvers() -> list:
    """Get list of available QUBO solvers."""
    solvers = ['gurobi', 'scip', 'hexaly', 'dirac']
    return [s for s in solvers if check_solver_availability(s)]


def load_default_config() -> Dict[str, Any]:
    """Load default Perturb-and-MAP configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "perturb_map.yaml"
    if config_path.exists():
        return load_config(str(config_path))
    else:
        # Fallback default configuration
        return {
            'model': {
                'n_visible': 784,
                'n_hidden': 64,
                'model_type': 'rbm'
            },
            'training': {
                'epochs': 20,
                'learning_rate': 0.01,
                'batch_size': 64,
                'optimizer': 'sgd',
                'method': 'perturb_map',
                'checkpoint_every': 5,
                'checkpoint_path': 'rbm_pm_checkpoint.pth',
                'batch_limit': None
            },
            'pm_params': {
                'gumbel_scale': 1.0,
                'solver_timeout': 60.0,
                'max_retries': 3,
                'seed': 42
            },
            'solver': {
                'name': 'gurobi',
                'time_limit': 60.0,
                'suppress_output': True
            },
            'data': {
                'dataset': 'mnist',
                'digit_filter': None,
                'image_size': [28, 28],
                'data_root': './data',
                'download': True
            },
            'inference': {
                'num_generated_samples': 10,
                'reconstruction_samples': 5
            },
            'logging': {
                'log_file': 'pm_training.log',
                'figures_dir': './figures'
            }
        }


def train_rbm_with_qubo(config: Dict[str, Any], output_dir: Path, resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Train RBM using Perturb-and-MAP with QUBO sampling.
    
    Args:
        config: Training configuration dictionary.
        output_dir: Directory for saving outputs.
        resume_checkpoint: Optional path to checkpoint file for resuming training.
        
    Returns:
        Training results dictionary.
    """
    print("\n=== Training Phase ===")
    print(f"Method: Perturb-and-MAP")
    print(f"Solver: {config['solver']['name']}")
    print(f"Model: {config['model']['n_visible']} visible, {config['model']['n_hidden']} hidden")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Update checkpoint path
    config['training']['checkpoint_path'] = str(output_dir / 'rbm_pm_checkpoint.pth')
    config['logging']['figures_dir'] = str(output_dir / 'figures')
    
    # Load training data
    print("\nLoading training data...")
    train_loader, dataset_size = load_mnist_data(config, train=True)
    
    digit_filter = config['data'].get('digit_filter')
    if digit_filter is not None:
        print(f"Training on digit {digit_filter}: {dataset_size} samples")
    else:
        print(f"Training on all digits: {dataset_size} samples")
    
    # Create training manager
    print("Initializing training manager...")
    manager = TrainingManager(config)
    
    # Handle resume training
    if resume_checkpoint:
        print(f"\nLoading checkpoint: {resume_checkpoint}")
        try:
            checkpoint_data = manager.load_checkpoint(resume_checkpoint)
            
            # Check if training is already complete
            current_epoch = manager.current_epoch
            total_epochs = config['training']['epochs']
            
            if current_epoch >= total_epochs:
                print(f"Training already completed! Current epoch: {current_epoch}, Target epochs: {total_epochs}")
                print("To continue training, increase --epochs parameter")
                return {
                    "status": "already_complete", 
                    "checkpoint_epoch": current_epoch,
                    "total_epochs": total_epochs,
                    "message": "Training was already completed"
                }
            
            remaining_epochs = total_epochs - current_epoch
            print(f"Resuming from epoch {current_epoch}")
            print(f"Remaining epochs: {remaining_epochs}")
            print(f"Best loss so far: {getattr(manager, 'best_loss', 'N/A')}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting fresh training instead...")
            resume_checkpoint = None
    
    # Start training
    if resume_checkpoint:
        print(f"\nResuming training...")
    else:
        print(f"\nStarting training...")
    start_time = time.time()
    
    try:
        results = manager.train(train_loader)
        training_time = time.time() - start_time
        
        print(f"\n=== Training Results ===")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Total epochs: {results.get('total_epochs', 'N/A')}")
        
        # Extract final reconstruction error from final_metrics
        final_metrics = results.get('final_metrics', {})
        final_error = final_metrics.get('reconstruction_error', 'N/A')
        print(f"Final reconstruction error: {safe_format_number(final_error)}")
        print(f"Checkpoint saved to: {config['training']['checkpoint_path']}")
        
        # Save training summary
        summary_path = output_dir / 'pm_training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== RBM Perturb-and-MAP Training Summary ===\n")
            f.write(f"Configuration: {config['solver']['name']}\n")
            f.write(f"Method: {manager.training_method.name}\n")
            f.write(f"Total epochs: {results.get('total_epochs', 'N/A')}\n")
            f.write(f"Total time: {training_time:.2f}s\n")
            f.write(f"Dataset size: {dataset_size}\n")
            f.write(f"Final reconstruction error: {safe_format_number(final_error)}\n")
            f.write(f"Hyperparameters: {manager.training_method.hyperparameters}\n")
            
            # Add solver statistics if available
            if hasattr(manager.training_method, 'get_solver_diagnostics'):
                solver_stats = manager.training_method.get_solver_diagnostics()
                f.write(f"Solver statistics: {solver_stats}\n")
        
        print(f"Training summary saved to: {summary_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


def run_inference_with_qubo(config: Dict[str, Any], checkpoint_path: str, output_dir: Path):
    """
    Run inference tasks using trained QUBO model.
    
    Args:
        config: Inference configuration dictionary.
        checkpoint_path: Path to trained model checkpoint.
        output_dir: Directory for saving outputs.
    """
    print("\n=== Inference Phase ===")
    print(f"Loading model from: {checkpoint_path}")
    
    # Load the trained model
    manager = TrainingManager(config)
    checkpoint = manager.load_checkpoint(checkpoint_path)
    
    print(f"Model loaded successfully")
    print(f"Training method: {manager.training_method.name}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Load test data
    print("Loading test data...")
    test_loader, _ = load_mnist_data(config, train=False)
    test_batch = next(iter(test_loader))[0]
    
    # Select images for inference
    inference_config = config['inference']
    num_samples = min(inference_config['reconstruction_samples'], test_batch.size(0))
    test_images = test_batch[:num_samples]
    
    print(f"Running inference on {num_samples} test images")
    
    # Task 1: Clean Image Reconstruction
    print("\n--- Clean Image Reconstruction ---")
    try:
        reconstructed_images = []
        reconstruction_errors = []
        
        # Progress bar for reconstruction
        pbar = tqdm(range(num_samples), desc="Reconstructing", unit="image")
        
        for i in pbar:
            image = test_images[i:i+1]
            
            # Use the training method's negative phase for reconstruction
            # This demonstrates the QUBO sampling process
            v_pos = image.view(image.size(0), -1)
            v_neg, h_neg = manager.training_method.negative_phase(v_pos)
            
            # Reconstruct from the negative phase
            reconstructed = manager.model.reconstruct(h_neg)
            
            # Ensure values are in [0,1] range for visualization
            reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
            
            reconstructed_images.append(reconstructed)
            
            # Calculate reconstruction error
            error = torch.mean((v_pos - reconstructed) ** 2).item()
            reconstruction_errors.append(error)
            
            # Update progress bar
            pbar.set_postfix({'error': f"{error:.6f}"})
        
        reconstructed_images = torch.cat(reconstructed_images, dim=0)
        
        # Ensure both tensors are on CPU for plotting
        test_images_cpu = test_images.cpu()
        reconstructed_images_cpu = reconstructed_images.cpu()
        
        # Ensure both are in [0,1] range
        test_images_cpu = torch.clamp(test_images_cpu, 0.0, 1.0)
        reconstructed_images_cpu = torch.clamp(reconstructed_images_cpu, 0.0, 1.0)
        
        # Fix shape mismatch: flatten original images to match reconstructed
        if len(test_images_cpu.shape) == 4:  # [batch, channels, height, width]
            test_images_cpu = test_images_cpu.view(test_images_cpu.size(0), -1)
        
        # Plot reconstruction results
        image_size = config['data']['image_size']
        plot_reconstruction(
            original=test_images_cpu,
            reconstructed=reconstructed_images_cpu,
            image_shape=tuple(image_size),
            save_path=str(output_dir / "qubo_reconstruction_clean.png"),
            title="QUBO Reconstruction (Clean Images)"
        )
        
        avg_error = np.mean(reconstruction_errors)
        print(f"Average reconstruction error: {avg_error:.6f}")
        
    except Exception as e:
        print(f"Error during clean reconstruction: {str(e)}")
    
    # Task 2: Image Denoising
    print("\n--- Image Denoising ---")
    try:
        # Create noisy versions of test images
        noisy_images = create_noisy_mnist(test_images, noise_prob=0.1)
        
        denoised_images = []
        denoising_errors = []
        
        # Progress bar for denoising
        pbar = tqdm(range(num_samples), desc="Denoising", unit="image")
        
        for i in pbar:
            noisy_image = noisy_images[i:i+1]
            
            # Use QUBO sampling for denoising
            v_pos = noisy_image.view(noisy_image.size(0), -1)
            v_neg, h_neg = manager.training_method.negative_phase(v_pos)
            
            # Reconstruct the denoised image
            denoised = manager.model.reconstruct(h_neg)
            denoised_images.append(denoised)
            
            # Calculate denoising error (compared to original)
            original = test_images[i:i+1].view(1, -1)
            error = torch.mean((original - denoised) ** 2).item()
            denoising_errors.append(error)
            
            # Update progress bar
            pbar.set_postfix({'error': f"{error:.6f}"})
        
        denoised_images = torch.cat(denoised_images, dim=0)
        
        # Ensure all tensors are on CPU and in [0,1] range for plotting
        test_images_cpu = test_images.cpu()
        noisy_images_cpu = noisy_images.cpu()
        denoised_images_cpu = denoised_images.cpu()
        
        test_images_cpu = torch.clamp(test_images_cpu, 0.0, 1.0)
        noisy_images_cpu = torch.clamp(noisy_images_cpu, 0.0, 1.0)
        denoised_images_cpu = torch.clamp(denoised_images_cpu, 0.0, 1.0)
        
        # Fix shape mismatch: flatten image tensors to match denoised
        if len(test_images_cpu.shape) == 4:  # [batch, channels, height, width]
            test_images_cpu = test_images_cpu.view(test_images_cpu.size(0), -1)
        if len(noisy_images_cpu.shape) == 4:  # [batch, channels, height, width]
            noisy_images_cpu = noisy_images_cpu.view(noisy_images_cpu.size(0), -1)
        
        # Plot denoising results
        plot_reconstruction(
            original=test_images_cpu,
            reconstructed=denoised_images_cpu,
            noisy=noisy_images_cpu,
            image_shape=tuple(image_size),
            save_path=str(output_dir / "qubo_reconstruction_denoising.png"),
            title="QUBO Denoising Results"
        )
        
        avg_denoising_error = np.mean(denoising_errors)
        print(f"Average denoising error: {avg_denoising_error:.6f}")
        
    except Exception as e:
        print(f"Error during denoising: {str(e)}")
    
    # Task 3: Sample Generation
    print("\n--- Sample Generation ---")
    try:
        num_generate = inference_config['num_generated_samples']
        print(f"Generating {num_generate} samples using QUBO joint sampling...")
        
        generated_samples = []
        generation_times = []
        
        # Progress bar for generation
        pbar = tqdm(range(num_generate), desc="Generating", unit="sample")
        
        for i in pbar:
            start_time = time.time()
            
            # Use QUBO joint sampling for generation
            # Start with random visible state
            random_v = torch.rand(1, config['model']['n_visible'])
            
            # Sample from joint distribution using QUBO
            v_sample, h_sample = manager.training_method.negative_phase(random_v)
            generated_samples.append(v_sample)
            
            gen_time = time.time() - start_time
            generation_times.append(gen_time)
            
            # Update progress bar
            pbar.set_postfix({'time': f"{gen_time:.4f}s"})
        
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Plot generation results
        plot_generation(
            generated_samples=generated_samples,
            image_shape=tuple(image_size),
            save_path=str(output_dir / "qubo_generation_samples.png"),
            title="QUBO Generated Samples"
        )
        
        avg_gen_time = np.mean(generation_times)
        print(f"Average generation time: {avg_gen_time:.4f}s")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
    
    # Display solver statistics
    print("\n--- Solver Performance Statistics ---")
    if hasattr(manager.training_method, 'get_solver_diagnostics'):
        solver_stats = manager.training_method.get_solver_diagnostics()
        print(f"Solver: {config['solver']['name']}")
        print(f"Statistics: {solver_stats}")
        
        # Save solver statistics
        stats_path = output_dir / 'qubo_solver_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("=== QUBO Solver Performance Statistics ===\n")
            f.write(f"Solver: {config['solver']['name']}\n")
            f.write(f"Configuration: {config['solver']}\n")
            f.write(f"Statistics: {solver_stats}\n")
            f.write(f"Average reconstruction error: {avg_error:.6f}\n")
            f.write(f"Average denoising error: {avg_denoising_error:.6f}\n")
            f.write(f"Average generation time: {avg_gen_time:.4f}s\n")
        
        print(f"Solver statistics saved to: {stats_path}")
    
    print("\nInference completed successfully!")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train and run inference with QUBO sampler RBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Train and infer with default settings
  %(prog)s --solver scip             # Use SCIP solver
  %(prog)s --figure6                 # Train on digit 6
  %(prog)s --task inference --checkpoint model.pth  # Inference only
        """
    )
    
    # Main options
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file (default: uses perturb_map.yaml)'
    )
    parser.add_argument(
        '--task',
        choices=['training', 'inference', 'both'],
        default='both',
        help='Task to perform (default: both)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (required for inference-only)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory for outputs (default: ./outputs)'
    )
    
    # Solver options
    parser.add_argument(
        '--solver',
        choices=['gurobi', 'scip', 'hexaly', 'dirac'],
        help='QUBO solver to use (default: auto-detect)'
    )
    parser.add_argument(
        '--solver-timeout',
        type=float,
        help='Solver timeout in seconds (default: 60.0)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        help='Maximum solver retries (default: 3)'
    )
    parser.add_argument(
        '--gumbel-scale',
        type=float,
        help='Gumbel noise scale for perturbation (default: 1.0)'
    )
    
    # Training options
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        help='Checkpoint frequency (default: 5)'
    )
    parser.add_argument(
        '--batch-limit',
        type=int,
        help='Limit number of batches per epoch (useful for testing)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint file'
    )
    
    # Data options
    parser.add_argument(
        '--figure6',
        action='store_true',
        help='Train on digit 6 only (figure 6 experiment)'
    )
    parser.add_argument(
        '--digit-filter',
        type=int,
        choices=range(10),
        help='Train on specific digit (0-9)'
    )
    
    # Inference options
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to generate (default: 10)'
    )
    
    # Utility options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--list-solvers',
        action='store_true',
        help='List available QUBO solvers and exit'
    )
    
    args = parser.parse_args()
    
    # List available solvers and exit if requested
    if args.list_solvers:
        available = get_available_solvers()
        print("Available QUBO solvers:")
        for solver in available:
            print(f"  - {solver}")
        if not available:
            print("  No QUBO solvers available!")
        sys.exit(0)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print header
    print_header()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.config_file:
        config = load_config(args.config_file)
    else:
        config = load_default_config()
    
    # Apply command-line overrides
    if args.solver:
        config['solver']['name'] = args.solver
    if args.solver_timeout:
        config['solver']['time_limit'] = args.solver_timeout
        config['pm_params']['solver_timeout'] = args.solver_timeout
    if args.max_retries:
        config['pm_params']['max_retries'] = args.max_retries
    if args.gumbel_scale:
        config['pm_params']['gumbel_scale'] = args.gumbel_scale
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.checkpoint_every:
        config['training']['checkpoint_every'] = args.checkpoint_every
    if args.batch_limit:
        config['training']['batch_limit'] = args.batch_limit
    
    if args.figure6:
        config['data']['digit_filter'] = 6
    elif args.digit_filter is not None:
        config['data']['digit_filter'] = args.digit_filter
    
    if args.num_samples:
        config['inference']['num_generated_samples'] = args.num_samples
    
    # Validate configuration
    config = validate_config(config)
    
    # Check solver availability
    solver_name = config['solver']['name']
    if not check_solver_availability(solver_name):
        available = get_available_solvers()
        if available:
            print(f"Warning: {solver_name} solver not available.")
            print(f"Available solvers: {available}")
            config['solver']['name'] = available[0]
            print(f"Using {config['solver']['name']} instead.")
        else:
            print("Error: No QUBO solvers available!")
            print("Please install at least one QUBO solver (e.g., python-scip)")
            sys.exit(1)
    
    # Validate resume checkpoint
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Resume checkpoint not found: {args.resume}")
            sys.exit(1)
    
    # Validate inference-only mode
    if args.task == 'inference':
        if not args.checkpoint:
            print("Error: --checkpoint is required for inference-only mode")
            sys.exit(1)
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  Solver: {config['solver']['name']}")
    print(f"  Output directory: {output_dir}")
    if config['data']['digit_filter'] is not None:
        print(f"  Digit filter: {config['data']['digit_filter']}")
    print(f"  Random seed: {args.seed}")
    
    try:
        # Execute tasks
        if args.task in ['training', 'both']:
            results = train_rbm_with_qubo(config, output_dir, resume_checkpoint=args.resume)
            checkpoint_path = config['training']['checkpoint_path']
        
        if args.task in ['inference', 'both']:
            if args.task == 'inference':
                checkpoint_path = args.checkpoint
            run_inference_with_qubo(config, checkpoint_path, output_dir)
        
        print(f"\n{'='*60}")
        print("QUBO Sampler RBM experiment completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()