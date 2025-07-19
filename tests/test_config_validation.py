"""
Unit tests for configuration validation and management.

This module tests the updated configuration system that supports
multiple training methods.
"""

import pytest
import tempfile
import os
import yaml
import json
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rbm.utils.config import (
    load_config, save_config, validate_config, get_default_config,
    merge_configs, ConfigManager
)


class TestConfigurationLoading:
    """Test configuration loading and saving functionality."""
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration files."""
        config_data = {
            'model': {'n_visible': 784, 'n_hidden': 64},
            'training': {'method': 'contrastive_divergence', 'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config(self):
        """Test loading JSON configuration files."""
        config_data = {
            'model': {'n_visible': 784, 'n_hidden': 64},
            'training': {'method': 'perturb_map', 'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('invalid config format')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_yaml_config(self):
        """Test saving YAML configuration files."""
        config_data = {
            'model': {'n_visible': 784, 'n_hidden': 64},
            'training': {'method': 'contrastive_divergence', 'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config_data, temp_path)
            
            # Verify saved config
            with open(temp_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_save_json_config(self):
        """Test saving JSON configuration files."""
        config_data = {
            'model': {'n_visible': 784, 'n_hidden': 64},
            'training': {'method': 'perturb_map', 'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config_data, temp_path)
            
            # Verify saved config
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config == config_data
        finally:
            os.unlink(temp_path)


class TestConfigurationValidation:
    """Test configuration validation for different training methods."""
    
    def test_default_config_validation(self):
        """Test that default configuration is valid."""
        default_config = get_default_config()
        validated_config = validate_config(default_config)
        
        assert validated_config is not None
        assert validated_config['training']['method'] == 'perturb_map'
        assert 'cd_params' in validated_config
        assert 'pm_params' in validated_config
    
    def test_cd_method_validation(self):
        """Test validation of contrastive divergence configuration."""
        cd_config = {
            'training': {'method': 'contrastive_divergence', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'},
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
        
        validated_config = validate_config(cd_config)
        assert validated_config['training']['method'] == 'contrastive_divergence'
        assert validated_config['cd_params']['k_steps'] == 1
        assert validated_config['cd_params']['persistent'] is False
    
    def test_pm_method_validation(self):
        """Test validation of perturb-and-map configuration."""
        pm_config = {
            'training': {'method': 'perturb_map', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'},
            'pm_params': {
                'gumbel_scale': 1.0,
                'solver_timeout': 60.0,
                'max_retries': 3,
                'seed': 42
            }
        }
        
        validated_config = validate_config(pm_config)
        assert validated_config['training']['method'] == 'perturb_map'
        assert validated_config['pm_params']['gumbel_scale'] == 1.0
        assert validated_config['pm_params']['solver_timeout'] == 60.0
    
    def test_invalid_training_method(self):
        """Test validation rejects invalid training methods."""
        invalid_config = {
            'training': {'method': 'invalid_method', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'}
        }
        
        with pytest.raises(ValueError, match="training method must be one of"):
            validate_config(invalid_config)
    
    def test_invalid_cd_parameters(self):
        """Test validation rejects invalid CD parameters."""
        # Test negative k_steps
        invalid_config = {
            'training': {'method': 'contrastive_divergence', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'},
            'cd_params': {
                'k_steps': -1,  # Invalid
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0
            }
        }
        
        with pytest.raises(ValueError, match="CD k_steps must be positive"):
            validate_config(invalid_config)
        
        # Test invalid momentum
        invalid_config['cd_params']['k_steps'] = 1
        invalid_config['cd_params']['momentum'] = 1.5  # Invalid (> 1)
        
        with pytest.raises(ValueError, match="CD momentum must be between 0 and 1"):
            validate_config(invalid_config)
        
        # Test invalid temperature
        invalid_config['cd_params']['momentum'] = 0.5
        invalid_config['cd_params']['temperature'] = -1.0  # Invalid
        
        with pytest.raises(ValueError, match="CD temperature must be positive"):
            validate_config(invalid_config)
    
    def test_invalid_pm_parameters(self):
        """Test validation rejects invalid P&M parameters."""
        # Test negative gumbel_scale
        invalid_config = {
            'training': {'method': 'perturb_map', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'},
            'pm_params': {
                'gumbel_scale': -1.0,  # Invalid
                'solver_timeout': 60.0,
                'max_retries': 3
            }
        }
        
        with pytest.raises(ValueError, match="P&M gumbel_scale must be positive"):
            validate_config(invalid_config)
        
        # Test negative solver_timeout
        invalid_config['pm_params']['gumbel_scale'] = 1.0
        invalid_config['pm_params']['solver_timeout'] = -10.0  # Invalid
        
        with pytest.raises(ValueError, match="P&M solver_timeout must be positive"):
            validate_config(invalid_config)
        
        # Test zero max_retries
        invalid_config['pm_params']['solver_timeout'] = 60.0
        invalid_config['pm_params']['max_retries'] = 0  # Invalid
        
        with pytest.raises(ValueError, match="P&M max_retries must be positive"):
            validate_config(invalid_config)
    
    def test_invalid_model_parameters(self):
        """Test validation rejects invalid model parameters."""
        # Test negative n_visible
        invalid_config = {
            'model': {'n_visible': -1, 'n_hidden': 64, 'model_type': 'rbm'},
            'training': {'method': 'contrastive_divergence', 'epochs': 10, 'learning_rate': 0.01, 'batch_size': 64}
        }
        
        with pytest.raises(ValueError, match="n_visible must be positive"):
            validate_config(invalid_config)
        
        # Test negative n_hidden
        invalid_config['model']['n_visible'] = 784
        invalid_config['model']['n_hidden'] = -1
        
        with pytest.raises(ValueError, match="n_hidden must be positive"):
            validate_config(invalid_config)
        
        # Test invalid model_type
        invalid_config['model']['n_hidden'] = 64
        invalid_config['model']['model_type'] = 'invalid_type'
        
        with pytest.raises(ValueError, match="model_type must be 'rbm' or 'cnn_rbm'"):
            validate_config(invalid_config)
    
    def test_invalid_training_parameters(self):
        """Test validation rejects invalid training parameters."""
        # Test negative epochs
        invalid_config = {
            'training': {'method': 'contrastive_divergence', 'epochs': -1, 'learning_rate': 0.01, 'batch_size': 64},
            'model': {'n_visible': 784, 'n_hidden': 64, 'model_type': 'rbm'}
        }
        
        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_config(invalid_config)
        
        # Test negative learning_rate
        invalid_config['training']['epochs'] = 10
        invalid_config['training']['learning_rate'] = -0.01
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(invalid_config)
        
        # Test negative batch_size
        invalid_config['training']['learning_rate'] = 0.01
        invalid_config['training']['batch_size'] = -1
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_config(invalid_config)


class TestConfigurationMerging:
    """Test configuration merging functionality."""
    
    def test_merge_configs_flat(self):
        """Test merging flat configuration dictionaries."""
        base_config = {'a': 1, 'b': 2}
        override_config = {'b': 3, 'c': 4}
        
        merged = merge_configs(base_config, override_config)
        
        assert merged == {'a': 1, 'b': 3, 'c': 4}
    
    def test_merge_configs_nested(self):
        """Test merging nested configuration dictionaries."""
        base_config = {
            'model': {'n_visible': 784, 'n_hidden': 64},
            'training': {'epochs': 10, 'learning_rate': 0.01}
        }
        
        override_config = {
            'model': {'n_hidden': 128},
            'training': {'epochs': 20, 'batch_size': 32}
        }
        
        merged = merge_configs(base_config, override_config)
        
        expected = {
            'model': {'n_visible': 784, 'n_hidden': 128},
            'training': {'epochs': 20, 'learning_rate': 0.01, 'batch_size': 32}
        }
        
        assert merged == expected
    
    def test_merge_configs_preserve_original(self):
        """Test that merging preserves original configurations."""
        base_config = {'a': 1, 'b': 2}
        override_config = {'b': 3, 'c': 4}
        
        original_base = base_config.copy()
        original_override = override_config.copy()
        
        merged = merge_configs(base_config, override_config)
        
        # Original configs should be unchanged
        assert base_config == original_base
        assert override_config == original_override
        
        # Merged config should be correct
        assert merged == {'a': 1, 'b': 3, 'c': 4}


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)
            assert manager.config_dir == Path(temp_dir)
            assert manager.config_dir.exists()
    
    def test_config_manager_save_load(self):
        """Test saving and loading configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)
            
            config = {
                'model': {'n_visible': 784, 'n_hidden': 64},
                'training': {'method': 'contrastive_divergence', 'epochs': 10}
            }
            
            # Save config
            manager.save(config, 'test_config')
            
            # Load config
            loaded_config = manager.load('test_config')
            
            assert loaded_config == config
            assert manager.current_config == config
    
    def test_config_manager_create_default(self):
        """Test creating default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)
            
            default_config = manager.create_default('default')
            
            assert default_config is not None
            assert manager.current_config == default_config
            
            # Verify file was created
            config_file = Path(temp_dir) / 'default.yaml'
            assert config_file.exists()
    
    def test_config_manager_list_configs(self):
        """Test listing available configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)
            
            # Create some configs
            config1 = {'model': {'n_visible': 784}}
            config2 = {'model': {'n_visible': 1024}}
            
            manager.save(config1, 'config1')
            manager.save(config2, 'config2')
            
            # List configs
            configs = manager.list_configs()
            
            assert 'config1' in configs
            assert 'config2' in configs
            assert len(configs) == 2
    
    def test_config_manager_update_current(self):
        """Test updating current configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)
            
            # Create initial config
            initial_config = {
                'model': {'n_visible': 784, 'n_hidden': 64},
                'training': {'epochs': 10}
            }
            manager.save(initial_config, 'initial')
            
            # Update config
            updates = {
                'model': {'n_hidden': 128},
                'training': {'epochs': 20, 'batch_size': 32}
            }
            
            updated_config = manager.update_current(updates)
            
            expected = {
                'model': {'n_visible': 784, 'n_hidden': 128},
                'training': {'epochs': 20, 'batch_size': 32}
            }
            
            assert updated_config == expected
            assert manager.current_config == expected