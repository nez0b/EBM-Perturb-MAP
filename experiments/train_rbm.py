#!/usr/bin/env python3
"""
Training script for RBM models using Perturb-and-MAP.

This script provides a command-line interface for training RBM models
with various QUBO solvers and configuration options.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbm.models.rbm import RBM
from rbm.models.hybrid import CNN_RBM
from rbm.training.trainer import Trainer
from rbm.data.mnist import load_mnist_data
from rbm.utils.config import ConfigManager, validate_config, load_config
from rbm.solvers.gurobi import GurobiSolver
from rbm.solvers.scip import ScipSolver
from rbm.solvers.dirac import DiracSolver


def create_model(config: dict) -> torch.nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Model instance.
    """
    model_config = config['model']
    model_type = model_config['model_type']
    
    if model_type == 'rbm':
        return RBM(
            n_visible=model_config['n_visible'],
            n_hidden=model_config['n_hidden']
        )
    elif model_type == 'cnn_rbm':
        return CNN_RBM(
            cnn_feature_dim=64,  # Fixed for now
            rbm_hidden_dim=model_config['n_hidden']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_solver(config: dict):
    """
    Create QUBO solver based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Solver instance.
    """
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


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize.
        config: Configuration dictionary.
        
    Returns:
        Optimizer instance.
    """
    training_config = config['training']
    optimizer_name = training_config.get('optimizer', 'sgd')
    learning_rate = training_config['learning_rate']
    
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RBM models using Perturb-and-MAP"
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
        '--solver',
        choices=['gurobi', 'scip', 'dirac'],
        help='Override solver choice'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory for outputs (default: ./outputs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.config_file:
        config = load_config(args.config_file)
    else:
        try:
            config = config_manager.load(args.config)
        except FileNotFoundError:
            print(f"Configuration '{args.config}' not found. Creating default configuration.")
            config = config_manager.create_default()
    
    # Apply command-line overrides
    if args.solver:
        config['solver']['name'] = args.solver
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Update output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config['training']['checkpoint_path'] = str(output_dir / 'checkpoint.pth')
    config['logging']['figures_dir'] = str(output_dir / 'figures')
    
    # Validate configuration
    config = validate_config(config)
    
    print("=== RBM Training with Perturb-and-MAP ===")
    print(f"Configuration: {args.config}")
    print(f"Model: {config['model']['model_type']}")
    print(f"Visible units: {config['model']['n_visible']}")
    print(f"Hidden units: {config['model']['n_hidden']}")
    print(f"Solver: {config['solver']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print("-" * 50)
    
    try:
        # Create model
        model = create_model(config)
        print(f"Created {config['model']['model_type']} model")
        
        # Create solver
        solver = create_solver(config)
        print(f"Using {solver.name} solver")
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Load data
        train_loader, dataset_size = load_mnist_data(config, train=True)
        print(f"Loaded training data: {dataset_size} samples")
        
        # Create trainer
        trainer = Trainer(model, solver, optimizer, config)
        
        # Train the model
        results = trainer.train(train_loader)
        
        print("=== Training Results ===")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Total epochs: {results['total_epochs']}")
        print(f"Checkpoint saved to: {config['training']['checkpoint_path']}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()