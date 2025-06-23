"""
Configuration management utilities.

This module provides functions for loading and managing configuration files
for RBM experiments and training.
"""

import yaml
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
import os


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file format is unsupported.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for RBM training.
    
    Returns:
        Default configuration dictionary.
    """
    return {
        'model': {
            'n_visible': 784,
            'n_hidden': 64,
            'model_type': 'rbm'  # 'rbm' or 'cnn_rbm'
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.01,
            'batch_size': 64,
            'batch_limit': None,
            'optimizer': 'sgd',
            'checkpoint_every': 5,
            'checkpoint_path': 'rbm_checkpoint.pth'
        },
        'data': {
            'dataset': 'mnist',
            'digit_filter': None,  # None for all digits, int for specific digit
            'image_size': [28, 28],
            'data_root': './data',
            'train_split': True,
            'download': True
        },
        'solver': {
            'name': 'gurobi',  # 'gurobi', 'scip', 'dirac'
            'time_limit': 60.0,
            'suppress_output': True,
            'num_samples': 10,  # For Dirac solver
            'relaxation_schedule': 1  # For Dirac solver
        },
        'inference': {
            'gibbs_steps': 1000,
            'num_generated_samples': 10,
            'reconstruction_samples': 5
        },
        'logging': {
            'log_file': 'training.log',
            'save_plots': True,
            'plot_every': 5,
            'figures_dir': './figures'
        }
    }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and complete configuration with defaults.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        Validated and completed configuration.
        
    Raises:
        ValueError: If configuration contains invalid values.
    """
    default_config = get_default_config()
    config = merge_configs(default_config, config)
    
    # Validate model parameters
    if config['model']['n_visible'] <= 0:
        raise ValueError("n_visible must be positive")
    if config['model']['n_hidden'] <= 0:
        raise ValueError("n_hidden must be positive")
    if config['model']['model_type'] not in ['rbm', 'cnn_rbm']:
        raise ValueError("model_type must be 'rbm' or 'cnn_rbm'")
    
    # Validate training parameters
    if config['training']['epochs'] <= 0:
        raise ValueError("epochs must be positive")
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    # Validate solver parameters
    valid_solvers = ['gurobi', 'scip', 'dirac']
    if config['solver']['name'] not in valid_solvers:
        raise ValueError(f"solver name must be one of {valid_solvers}")
    
    return config


class ConfigManager:
    """
    Manager class for handling experiment configurations.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config = None
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a named configuration.
        
        Args:
            config_name: Name of the configuration (without extension).
            
        Returns:
            Configuration dictionary.
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        self.current_config = load_config(config_path)
        return self.current_config
    
    def save(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save a configuration with a given name.
        
        Args:
            config: Configuration dictionary.
            config_name: Name for the configuration.
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        save_config(config, config_path)
        self.current_config = config
    
    def create_default(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Create and save a default configuration.
        
        Args:
            config_name: Name for the default configuration.
            
        Returns:
            Default configuration dictionary.
        """
        default_config = get_default_config()
        self.save(default_config, config_name)
        return default_config
    
    def list_configs(self) -> list:
        """
        List all available configuration files.
        
        Returns:
            List of configuration names.
        """
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        return [f.stem for f in config_files]
    
    def update_current(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply.
            
        Returns:
            Updated configuration.
        """
        if self.current_config is None:
            self.current_config = get_default_config()
        
        self.current_config = merge_configs(self.current_config, updates)
        return self.current_config