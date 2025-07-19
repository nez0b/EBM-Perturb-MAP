#!/usr/bin/env python3
"""
Training script for RBM models using Contrastive Divergence.

This script provides a command-line interface for training RBM models
with Contrastive Divergence using the modular training framework.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
import logging
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbm.training.training_manager import TrainingManager
from rbm.data.mnist import load_mnist_data
from rbm.utils.config import ConfigManager, validate_config, load_config


def setup_epoch_logging(output_dir: Path, log_filename: str) -> logging.Logger:
    """
    Setup file logging for epoch metrics.
    
    Args:
        output_dir: Output directory for log file.
        log_filename: Name of log file.
        
    Returns:
        Configured logger instance.
    """
    log_path = output_dir / log_filename
    
    # Create logger
    logger = logging.getLogger('cd_epoch_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Log header
    logger.info("epoch,reconstruction_error,positive_energy,negative_energy,epoch_time,cd_k_steps,cd_persistent,cd_acceptance_rate")
    
    print(f"Epoch logging enabled: {log_path}")
    return logger


def log_epoch_metrics(logger: logging.Logger, epoch: int, metrics: dict):
    """
    Log epoch metrics to file.
    
    Args:
        logger: Logger instance.
        epoch: Epoch number.
        metrics: Epoch metrics dictionary.
    """
    # Extract key metrics
    recon_error = metrics.get('reconstruction_error', 0.0)
    pos_energy = metrics.get('positive_energy', 0.0)
    neg_energy = metrics.get('negative_energy', 0.0)
    epoch_time = metrics.get('epoch_time', 0.0)
    cd_k_steps = metrics.get('cd_k_steps', 1.0)
    cd_persistent = metrics.get('cd_persistent', 0.0)
    cd_acceptance_rate = metrics.get('cd_acceptance_rate', 0.0)
    
    # Log as CSV format
    log_entry = f"{epoch},{recon_error:.6f},{pos_energy:.6f},{neg_energy:.6f},{epoch_time:.4f},{cd_k_steps},{cd_persistent},{cd_acceptance_rate:.6f}"
    logger.info(log_entry)


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
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RBM models using Contrastive Divergence"
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
        '--k-steps',
        type=int,
        help='Number of Gibbs sampling steps (CD-k)'
    )
    parser.add_argument(
        '--persistent',
        action='store_true',
        help='Enable Persistent Contrastive Divergence (PCD)'
    )
    parser.add_argument(
        '--figure6',
        action='store_true',
        help='Train on digit 6 only (figure 6 experiment)'
    )
    parser.add_argument(
        '--digit-filter',
        type=int,
        help='Filter to specific digit (0-9)'
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
        '--batch-limit',
        type=int,
        help='Limit number of batches per epoch (default: unlimited)'
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
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint file'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Check resume checkpoint if specified
    resume_training = False
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Resume checkpoint not found: {args.resume}")
            sys.exit(1)
        resume_training = True
        print(f"Resume mode: Will load checkpoint from {args.resume}")
    
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
    
    # Ensure training method is set to contrastive divergence
    config['training']['method'] = 'contrastive_divergence'
    
    # Apply command-line overrides
    if args.k_steps:
        config.setdefault('cd_params', {})['k_steps'] = args.k_steps
    if args.persistent:
        config.setdefault('cd_params', {})['persistent'] = True
    if args.figure6:
        config.setdefault('data', {})['digit_filter'] = 6
    if args.digit_filter is not None:
        config.setdefault('data', {})['digit_filter'] = args.digit_filter
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.batch_limit:
        config['training']['batch_limit'] = args.batch_limit
    
    # Update output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config['training']['checkpoint_path'] = str(output_dir / 'rbm_cd_checkpoint.pth')
    config['logging']['figures_dir'] = str(output_dir / 'figures')
    
    # Create figures directory
    figures_dir = Path(config['logging']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate configuration
    config = validate_config(config)
    
    # Print configuration summary
    print("=== RBM Training with Contrastive Divergence ===")
    print(f"Configuration: {args.config}")
    print(f"Model: {config['model']['model_type']}")
    print(f"Visible units: {config['model']['n_visible']}")
    print(f"Hidden units: {config['model']['n_hidden']}")
    print(f"Training method: {config['training']['method']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # CD-specific parameters
    cd_params = config.get('cd_params', {})
    print(f"CD-k steps: {cd_params.get('k_steps', 1)}")
    print(f"Persistent CD: {cd_params.get('persistent', False)}")
    
    # Data filtering
    data_config = config.get('data', {})
    if 'digit_filter' in data_config:
        print(f"Digit filter: {data_config['digit_filter']}")
    
    print("-" * 50)
    
    try:
        # Load data
        train_loader, dataset_size = load_mnist_data(config, train=True)
        print(f"Loaded training data: {dataset_size} samples")
        
        # Setup epoch logging
        logging_config = config.get('logging', {})
        log_filename = logging_config.get('log_file', 'cd_training.log')
        epoch_logger = setup_epoch_logging(output_dir, log_filename)
        
        # Create training manager
        training_manager = TrainingManager(config)
        
        # Handle resume training
        if resume_training:
            print(f"\nLoading checkpoint: {args.resume}")
            checkpoint_data = training_manager.load_checkpoint(args.resume)
            
            # Check if training is already complete
            current_epoch = training_manager.current_epoch
            total_epochs = config['training']['epochs']
            
            if current_epoch >= total_epochs:
                print(f"Training already completed! Current epoch: {current_epoch}, Target epochs: {total_epochs}")
                print("To continue training, increase --epochs parameter")
                sys.exit(0)
            
            remaining_epochs = total_epochs - current_epoch
            print(f"Resuming from epoch {current_epoch}")
            print(f"Remaining epochs: {remaining_epochs}")
            print(f"Best loss so far: {training_manager.best_loss:.6f}")
            
            # Log previous training history to file
            print("Logging previous training history...")
            for i, hist_metrics in enumerate(training_manager.training_history):
                log_epoch_metrics(epoch_logger, i + 1, hist_metrics)
        
        # Train the model
        print("\nStarting training...")
        
        # Custom training loop with epoch logging
        print(f"Starting training with {training_manager.training_method.name}")
        print(f"Epochs: {training_manager.epochs}, Learning Rate: {training_manager.optimizer.param_groups[0]['lr']}")
        print(f"Model: {training_manager.model.__class__.__name__} "
              f"({training_manager.model.n_visible} visible, {training_manager.model.n_hidden} hidden)")
        print("-" * 60)
        
        import time
        total_start_time = time.time()
        
        start_epoch = training_manager.current_epoch if resume_training else 0
        
        for epoch in range(start_epoch, training_manager.epochs):
            epoch_start_time = time.time()
            epoch_metrics = training_manager._train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start_time
            
            # Add timing information
            epoch_metrics['epoch_time'] = epoch_time
            epoch_metrics['epoch'] = epoch + 1
            
            # Update training history
            training_manager.training_history.append(epoch_metrics)
            
            # Print progress
            training_manager._print_epoch_progress(epoch + 1, epoch_metrics)
            
            # Log epoch metrics to file
            log_epoch_metrics(epoch_logger, epoch + 1, epoch_metrics)
            
            # Save checkpoint if needed
            if (epoch + 1) % training_manager.checkpoint_every == 0:
                training_manager._save_checkpoint(epoch + 1, epoch_metrics)
            
            # Check for early stopping or learning rate adaptation
            training_manager._check_training_progress(epoch_metrics)
            
            training_manager.current_epoch = epoch + 1
        
        total_time = time.time() - total_start_time
        
        print("-" * 60)
        print(f"Training completed in {total_time:.2f}s")
        
        # Create results summary
        results = {
            'history': training_manager.training_history,
            'final_metrics': training_manager.training_history[-1] if training_manager.training_history else {},
            'total_epochs': training_manager.current_epoch,
            'total_time': total_time,
            'method': training_manager.training_method.name,
            'hyperparameters': training_manager.training_method.hyperparameters
        }
        
        print("\n=== Training Results ===")
        print(f"Training method: {results['method']}")
        print(f"Total epochs: {results['total_epochs']}")
        print(f"Total time: {results['total_time']:.2f}s")
        
        # Print final metrics
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"Final reconstruction error: {final_metrics.get('reconstruction_error', 'N/A'):.6f}")
            print(f"Final positive energy: {final_metrics.get('positive_energy', 'N/A'):.6f}")
            print(f"Final negative energy: {final_metrics.get('negative_energy', 'N/A'):.6f}")
        
        # Print hyperparameters
        print(f"\nHyperparameters:")
        for key, value in results.get('hyperparameters', {}).items():
            print(f"  {key}: {value}")
        
        print(f"\nCheckpoint saved to: {config['training']['checkpoint_path']}")
        print(f"Figures saved to: {config['logging']['figures_dir']}")
        
        # Save training summary
        summary_path = output_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== RBM Contrastive Divergence Training Summary ===\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Method: {results['method']}\n")
            f.write(f"Total epochs: {results['total_epochs']}\n")
            f.write(f"Total time: {results['total_time']:.2f}s\n")
            f.write(f"Dataset size: {dataset_size}\n")
            f.write(f"Final metrics: {final_metrics}\n")
            f.write(f"Hyperparameters: {results.get('hyperparameters', {})}\n")
        
        print(f"Training summary saved to: {summary_path}")
        
        # Close epoch logging
        for handler in epoch_logger.handlers:
            handler.close()
        print(f"Epoch log saved to: {output_dir / log_filename}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()