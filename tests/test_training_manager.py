"""
Unit tests for TrainingManager.

This module tests the TrainingManager class that orchestrates
the training process with different methods.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rbm.training.training_manager import TrainingManager


class TestTrainingManagerInitialization:
    """Test TrainingManager initialization."""
    
    @pytest.fixture
    def cd_config(self):
        """Configuration for CD training."""
        return {
            'model': {
                'n_visible': 784,
                'n_hidden': 64,
                'model_type': 'rbm'
            },
            'training': {
                'method': 'contrastive_divergence',
                'epochs': 5,
                'learning_rate': 0.01,
                'batch_size': 32,
                'optimizer': 'sgd',
                'checkpoint_every': 2,
                'checkpoint_path': 'test_checkpoint.pth'
            },
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
    
    @pytest.fixture
    def pm_config(self):
        """Configuration for P&M training."""
        return {
            'model': {
                'n_visible': 784,
                'n_hidden': 64,
                'model_type': 'rbm'
            },
            'training': {
                'method': 'perturb_map',
                'epochs': 5,
                'learning_rate': 0.01,
                'batch_size': 32,
                'optimizer': 'sgd',
                'checkpoint_every': 2,
                'checkpoint_path': 'test_checkpoint.pth'
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
            }
        }
    
    def test_cd_training_manager_initialization(self, cd_config):
        """Test TrainingManager initialization with CD method."""
        manager = TrainingManager(cd_config)
        
        assert manager.config['training']['method'] == 'contrastive_divergence'
        assert manager.epochs == 5
        assert manager.checkpoint_every == 2
        assert manager.training_method.name.startswith('ContrastiveDivergence')
    
    def test_pm_training_manager_initialization(self, pm_config):
        """Test TrainingManager initialization with P&M method."""
        # Mock the solver creation to avoid dependencies
        with patch.object(TrainingManager, '_create_solver') as mock_solver:
            mock_solver.return_value = Mock()
            mock_solver.return_value.name = 'MockSolver'
            
            manager = TrainingManager(pm_config)
            
            assert manager.config['training']['method'] == 'perturb_map'
            assert manager.epochs == 5
            assert manager.checkpoint_every == 2
            assert manager.training_method.name.startswith('PerturbAndMAP')
    
    def test_invalid_training_method(self, cd_config):
        """Test that invalid training method raises error."""
        cd_config['training']['method'] = 'invalid_method'
        
        with pytest.raises(ValueError, match="training method must be one of"):
            TrainingManager(cd_config)
    
    def test_invalid_model_type(self, cd_config):
        """Test that invalid model type raises error."""
        cd_config['model']['model_type'] = 'invalid_model'
        
        with pytest.raises(ValueError, match="model_type must be 'rbm' or 'cnn_rbm'"):
            TrainingManager(cd_config)
    
    def test_invalid_optimizer(self, cd_config):
        """Test that invalid optimizer raises error."""
        cd_config['training']['optimizer'] = 'invalid_optimizer'
        
        with pytest.raises(ValueError, match="Unknown optimizer"):
            TrainingManager(cd_config)


class TestTrainingManagerTraining:
    """Test TrainingManager training functionality."""
    
    @pytest.fixture
    def sample_data_loader(self):
        """Create a small sample data loader for testing."""
        # Create simple binary data
        data = torch.randint(0, 2, (64, 784)).float()
        dataset = TensorDataset(data, torch.zeros(64))  # dummy labels
        
        return DataLoader(dataset, batch_size=16, shuffle=False)
    
    @pytest.fixture
    def cd_config(self):
        """Configuration for CD training."""
        return {
            'model': {
                'n_visible': 784,
                'n_hidden': 32,  # Smaller for faster testing
                'model_type': 'rbm'
            },
            'training': {
                'method': 'contrastive_divergence',
                'epochs': 2,  # Few epochs for testing
                'learning_rate': 0.01,
                'batch_size': 16,
                'batch_limit': 2,  # Limit batches for testing
                'optimizer': 'sgd',
                'checkpoint_every': 1,
                'checkpoint_path': 'test_checkpoint.pth'
            },
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
    
    def test_cd_training_execution(self, cd_config, sample_data_loader):
        """Test CD training execution."""
        manager = TrainingManager(cd_config)
        
        # Run training
        results = manager.train(sample_data_loader)
        
        # Check results structure
        assert 'history' in results
        assert 'final_metrics' in results
        assert 'total_epochs' in results
        assert 'total_time' in results
        assert 'method' in results
        assert 'hyperparameters' in results
        
        # Check that training actually ran
        assert results['total_epochs'] == 2
        assert len(results['history']) == 2
        assert results['method'].startswith('ContrastiveDivergence')
        assert results['total_time'] > 0
    
    def test_pm_training_execution(self, sample_data_loader):
        """Test P&M training execution with mocked solver."""
        pm_config = {
            'model': {
                'n_visible': 784,
                'n_hidden': 32,
                'model_type': 'rbm'
            },
            'training': {
                'method': 'perturb_map',
                'epochs': 2,
                'learning_rate': 0.01,
                'batch_size': 16,
                'batch_limit': 2,
                'optimizer': 'sgd',
                'checkpoint_every': 1,
                'checkpoint_path': 'test_checkpoint.pth'
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
            }
        }
        
        # Mock the solver creation
        with patch.object(TrainingManager, '_create_solver') as mock_solver_factory:
            mock_solver = Mock()
            mock_solver.name = 'MockSolver'
            mock_solver.solve.return_value = torch.randint(0, 2, (784 + 32,)).numpy()
            mock_solver_factory.return_value = mock_solver
            
            manager = TrainingManager(pm_config)
            
            # Run training
            results = manager.train(sample_data_loader)
            
            # Check results structure
            assert 'history' in results
            assert 'final_metrics' in results
            assert 'total_epochs' in results
            assert 'total_time' in results
            assert 'method' in results
            assert 'hyperparameters' in results
            
            # Check that training actually ran
            assert results['total_epochs'] == 2
            assert len(results['history']) == 2
            assert results['method'].startswith('PerturbAndMAP')
            assert results['total_time'] > 0
    
    def test_training_metrics_structure(self, cd_config, sample_data_loader):
        """Test that training metrics have correct structure."""
        manager = TrainingManager(cd_config)
        
        # Run training
        results = manager.train(sample_data_loader)
        
        # Check epoch metrics structure
        for epoch_metrics in results['history']:
            assert 'epoch' in epoch_metrics
            assert 'epoch_time' in epoch_metrics
            assert 'num_batches' in epoch_metrics
            assert 'reconstruction_error' in epoch_metrics
            assert 'positive_energy' in epoch_metrics
            assert 'negative_energy' in epoch_metrics
    
    def test_checkpointing(self, cd_config, sample_data_loader):
        """Test model checkpointing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
            cd_config['training']['checkpoint_path'] = checkpoint_path
            cd_config['training']['checkpoint_every'] = 1  # Save every epoch
            
            manager = TrainingManager(cd_config)
            
            # Run training
            results = manager.train(sample_data_loader)
            
            # Check that checkpoint was created
            assert os.path.exists(checkpoint_path)
            
            # Test loading checkpoint
            checkpoint_data = manager.load_checkpoint(checkpoint_path)
            
            assert 'epoch' in checkpoint_data
            assert 'model_state_dict' in checkpoint_data
            assert 'optimizer_state_dict' in checkpoint_data
            assert 'config' in checkpoint_data
            assert 'method' in checkpoint_data
    
    def test_evaluation(self, cd_config, sample_data_loader):
        """Test model evaluation functionality."""
        manager = TrainingManager(cd_config)
        
        # Run short training
        manager.train(sample_data_loader)
        
        # Test evaluation
        eval_results = manager.evaluate(sample_data_loader)
        
        assert 'reconstruction_error' in eval_results
        assert eval_results['reconstruction_error'] > 0
    
    def test_training_summary(self, cd_config, sample_data_loader):
        """Test training summary functionality."""
        manager = TrainingManager(cd_config)
        
        # Run training
        results = manager.train(sample_data_loader)
        
        # Get training summary
        summary = manager.get_training_summary()
        
        assert 'method' in summary
        assert 'hyperparameters' in summary
        assert 'config' in summary
        assert 'current_epoch' in summary
        assert 'total_epochs' in summary
        assert 'best_loss' in summary
        assert 'training_history' in summary
        
        assert summary['method'] == results['method']
        assert summary['current_epoch'] == results['total_epochs']


class TestTrainingManagerSolverCreation:
    """Test solver creation in TrainingManager."""
    
    def test_gurobi_solver_creation(self):
        """Test Gurobi solver creation."""
        config = {
            'solver': {
                'name': 'gurobi',
                'time_limit': 60.0,
                'suppress_output': True
            }
        }
        
        manager = TrainingManager.__new__(TrainingManager)  # Create without __init__
        manager.config = config
        
        with patch('rbm.solvers.gurobi.GurobiSolver') as mock_gurobi:
            mock_solver = Mock()
            mock_solver.name = 'Gurobi'
            mock_gurobi.return_value = mock_solver
            
            solver = manager._create_solver()
            
            assert solver is mock_solver
            mock_gurobi.assert_called_once_with(
                suppress_output=True,
                time_limit=60.0
            )
            # This is more of a structure test
    
    def test_invalid_solver_name(self):
        """Test that invalid solver name raises error."""
        config = {
            'solver': {
                'name': 'invalid_solver',
                'time_limit': 60.0
            }
        }
        
        manager = TrainingManager.__new__(TrainingManager)
        manager.config = config
        
        with pytest.raises(ValueError, match="Unknown solver"):
            manager._create_solver()


class TestTrainingManagerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_loader(self):
        """Test training with empty data loader."""
        config = {
            'model': {'n_visible': 784, 'n_hidden': 32, 'model_type': 'rbm'},
            'training': {
                'method': 'contrastive_divergence',
                'epochs': 1,
                'learning_rate': 0.01,
                'batch_size': 32,
                'optimizer': 'sgd'
            },
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
        
        # Create empty data loader
        empty_dataset = TensorDataset(torch.empty(0, 784), torch.empty(0))
        empty_loader = DataLoader(empty_dataset, batch_size=32)
        
        manager = TrainingManager(config)
        
        # Training should handle empty loader gracefully
        results = manager.train(empty_loader)
        
        # Should still return valid results structure
        assert 'history' in results
        assert 'total_epochs' in results
        assert results['total_epochs'] == 1
    
    def test_batch_limit_functionality(self):
        """Test batch limit functionality."""
        config = {
            'model': {'n_visible': 784, 'n_hidden': 32, 'model_type': 'rbm'},
            'training': {
                'method': 'contrastive_divergence',
                'epochs': 1,
                'learning_rate': 0.01,
                'batch_size': 16,
                'batch_limit': 2,  # Limit to 2 batches
                'optimizer': 'sgd'
            },
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
        
        # Create data loader with more than 2 batches
        data = torch.randint(0, 2, (64, 784)).float()
        dataset = TensorDataset(data, torch.zeros(64))
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)  # 4 batches
        
        manager = TrainingManager(config)
        results = manager.train(data_loader)
        
        # Should only process 2 batches
        assert results['history'][0]['num_batches'] == 2