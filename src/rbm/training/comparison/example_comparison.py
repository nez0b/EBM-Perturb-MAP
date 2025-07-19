"""
Example script showing how to use the comparison framework.

This script demonstrates how to compare Contrastive Divergence and 
Perturb-and-MAP training methods using the comparison framework.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.rbm.training.comparison.framework import MethodComparison


def create_data_loader(batch_size: int = 64, digit_filter: int = None) -> DataLoader:
    """Create MNIST data loader for comparison."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    # Load MNIST dataset
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Filter for specific digit if requested
    if digit_filter is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label == digit_filter]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    return data_loader


def main():
    """Main comparison script."""
    
    # Base configuration shared by all methods
    base_config = {
        'model': {
            'n_visible': 784,
            'n_hidden': 32,  # Smaller for faster comparison
            'model_type': 'rbm'
        },
        'training': {
            'epochs': 5,  # Fewer epochs for quick comparison
            'learning_rate': 0.01,
            'batch_size': 64,
            'batch_limit': 10,  # Limit batches for quick comparison
            'optimizer': 'sgd',
            'checkpoint_every': 2,
        },
        'data': {
            'dataset': 'mnist',
            'digit_filter': 1,  # Focus on digit 1 for faster training
            'image_size': [28, 28],
            'data_root': './data',
            'train_split': True,
            'download': True
        },
        'solver': {
            'name': 'gurobi',
            'time_limit': 30.0,  # Shorter timeout for comparison
            'suppress_output': True
        },
        'logging': {
            'save_plots': True,
            'plot_every': 2,
            'figures_dir': './comparison_figures'
        }
    }
    
    # Initialize comparison framework
    comparison = MethodComparison(base_config, output_dir="./comparison_results")
    
    # Add Contrastive Divergence method configurations
    cd_config = {
        'training': {
            'method': 'contrastive_divergence',
            'checkpoint_path': 'cd_checkpoint.pth'
        },
        'cd_params': {
            'k_steps': 1,
            'persistent': False,
            'use_momentum': False,
            'temperature': 1.0
        }
    }
    comparison.add_method_config('CD-1', cd_config)
    
    # Add Persistent CD configuration
    pcd_config = {
        'training': {
            'method': 'contrastive_divergence',
            'checkpoint_path': 'pcd_checkpoint.pth'
        },
        'cd_params': {
            'k_steps': 3,
            'persistent': True,
            'use_momentum': True,
            'momentum': 0.7,
            'temperature': 1.0
        }
    }
    comparison.add_method_config('PCD-3', pcd_config)
    
    # Add Perturb-and-MAP method configuration
    pm_config = {
        'training': {
            'method': 'perturb_map',
            'checkpoint_path': 'pm_checkpoint.pth'
        },
        'pm_params': {
            'gumbel_scale': 1.0,
            'solver_timeout': 30.0,
            'max_retries': 2
        }
    }
    comparison.add_method_config('P&M-Gurobi', pm_config)
    
    # Add P&M with different solver
    pm_scip_config = {
        'training': {
            'method': 'perturb_map',
            'checkpoint_path': 'pm_scip_checkpoint.pth'
        },
        'pm_params': {
            'gumbel_scale': 1.0,
            'solver_timeout': 30.0,
            'max_retries': 2
        },
        'solver': {
            'name': 'scip',
            'time_limit': 30.0
        }
    }
    comparison.add_method_config('P&M-SCIP', pm_scip_config)
    
    # Create data loader
    print("Creating data loader...")
    data_loader = create_data_loader(batch_size=64, digit_filter=1)
    print(f"Data loader created with {len(data_loader)} batches")
    
    # Run comparison
    print("\nStarting method comparison...")
    results = comparison.run_comparison(
        data_loader,
        methods=['CD-1', 'PCD-3', 'P&M-Gurobi'],  # Run subset for demo
        runs_per_method=1,  # Single run for demo
        save_checkpoints=True
    )
    
    # Print summary
    comparison.print_summary()
    
    # Print detailed analysis
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS")
    print("=" * 50)
    
    stats = results['statistical_analysis']
    
    print(f"\nTraining Time Analysis:")
    print(f"  Fastest: {stats['training_time']['fastest']}")
    print(f"  Slowest: {stats['training_time']['slowest']}")
    print(f"  Speedup: {stats['training_time']['speedup']:.2f}x")
    
    print(f"\nFinal Loss Analysis:")
    print(f"  Best: {stats['final_loss']['best']}")
    print(f"  Worst: {stats['final_loss']['worst']}")
    print(f"  Improvement: {stats['final_loss']['improvement']:.2f}x")
    
    print(f"\nConvergence Analysis:")
    print(f"  Converged methods: {stats['convergence']['converged_methods']}")
    print(f"  Convergence rate: {stats['convergence']['convergence_rate']:.1%}")
    
    print(f"\nFiles saved in: {comparison.output_dir}")
    print("  - comparison_results.json (detailed results)")
    print("  - summary_report.txt (human-readable summary)")
    print("  - plots/ (visualization plots)")
    
    return results


if __name__ == "__main__":
    # Example of quick comparison
    print("EBM Training Methods Comparison Example")
    print("=" * 50)
    
    try:
        results = main()
        print("\n✓ Comparison completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()