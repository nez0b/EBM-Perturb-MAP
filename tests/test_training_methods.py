"""
Unit tests for EBM training methods.

This module tests the modular training method implementations
including ContrastiveDivergenceTraining and PerturbMapTraining.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rbm.training.methods.base import EBMTrainingMethod
from rbm.training.methods.contrastive_divergence import ContrastiveDivergenceTraining
from rbm.training.methods.perturb_map import PerturbMapTraining
from rbm.training.samplers.gibbs_sampler import GibbsSampler
from rbm.training.samplers.qubo_sampler import QUBOSampler
from rbm.models.rbm import RBM


class TestEBMTrainingMethod:
    """Test the abstract base class for EBM training methods."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            EBMTrainingMethod(None, None, {})
    
    def test_concrete_implementation_requires_negative_phase(self):
        """Test that concrete implementations must implement negative_phase."""
        
        class IncompleteMethod(EBMTrainingMethod):
            pass
        
        with pytest.raises(TypeError):
            IncompleteMethod(None, None, {})


class TestContrastiveDivergenceTraining:
    """Test the Contrastive Divergence training method."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock RBM model for testing."""
        model = Mock(spec=RBM)
        model.n_visible = 784
        model.n_hidden = 64
        model.forward.return_value = torch.randn(32, 64)
        model.reconstruct.return_value = torch.randn(32, 784)
        return model
    
    @pytest.fixture
    def mock_optimizer(self):
        """Mock optimizer for testing."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 0.01}]
        return optimizer
    
    @pytest.fixture
    def cd_config(self):
        """Configuration for CD training."""
        return {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.01
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
    
    def test_cd_initialization(self, mock_model, mock_optimizer, cd_config):
        """Test CD method initialization."""
        cd_method = ContrastiveDivergenceTraining(mock_model, mock_optimizer, cd_config)
        
        assert cd_method.model is mock_model
        assert cd_method.optimizer is mock_optimizer
        assert cd_method.k_steps == 1
        assert cd_method.persistent is False
        assert cd_method.sampling_mode == 'stochastic'
        assert cd_method.name == "ContrastiveDivergence(k=1)"
    
    def test_cd_hyperparameters(self, mock_model, mock_optimizer, cd_config):
        """Test CD hyperparameters property."""
        cd_method = ContrastiveDivergenceTraining(mock_model, mock_optimizer, cd_config)
        
        hyperparams = cd_method.hyperparameters
        assert hyperparams['k_steps'] == 1
        assert hyperparams['persistent'] is False
        assert hyperparams['sampling_mode'] == 'stochastic'
        assert hyperparams['learning_rate'] == 0.01
        assert 'sampler_hyperparameters' in hyperparams
    
    def test_cd_negative_phase_shape(self, mock_model, mock_optimizer, cd_config):
        """Test that CD negative phase returns correct shapes."""
        cd_method = ContrastiveDivergenceTraining(mock_model, mock_optimizer, cd_config)
        
        # Mock the sampler to return specific shapes
        v_positive = torch.randn(32, 784)
        
        with patch.object(cd_method.sampler, 'sample') as mock_sample:
            mock_sample.return_value = (
                torch.randn(32, 784),  # v_negative
                torch.randn(32, 64)    # h_negative
            )
            
            v_neg, h_neg = cd_method.negative_phase(v_positive)
            
            assert v_neg.shape == (32, 784)
            assert h_neg.shape == (32, 64)
            mock_sample.assert_called_once_with(v_positive, 32)
    
    def test_cd_persistent_mode(self, mock_model, mock_optimizer, cd_config):
        """Test CD persistent mode configuration."""
        cd_config['cd_params']['persistent'] = True
        cd_method = ContrastiveDivergenceTraining(mock_model, mock_optimizer, cd_config)
        
        assert cd_method.persistent is True
        assert cd_method.name == "PersistentContrastiveDivergence(k=1)"


class TestPerturbMapTraining:
    """Test the Perturb-and-MAP training method."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock RBM model for testing."""
        model = Mock(spec=RBM)
        model.n_visible = 784
        model.n_hidden = 64
        model.forward.return_value = torch.randn(32, 64)
        model.reconstruct.return_value = torch.randn(32, 784)
        return model
    
    @pytest.fixture
    def mock_optimizer(self):
        """Mock optimizer for testing."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 0.01}]
        return optimizer
    
    @pytest.fixture
    def mock_solver(self):
        """Mock QUBO solver for testing."""
        solver = Mock()
        solver.name = "MockSolver"
        solver.solve.return_value = np.random.randint(0, 2, 784 + 64)
        return solver
    
    @pytest.fixture
    def pm_config(self):
        """Configuration for P&M training."""
        return {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.01
            },
            'pm_params': {
                'gumbel_scale': 1.0,
                'solver_timeout': 60.0,
                'max_retries': 3,
                'seed': 42
            }
        }
    
    def test_pm_initialization(self, mock_model, mock_optimizer, mock_solver, pm_config):
        """Test P&M method initialization."""
        pm_method = PerturbMapTraining(mock_model, mock_optimizer, mock_solver, pm_config)
        
        assert pm_method.model is mock_model
        assert pm_method.optimizer is mock_optimizer
        assert pm_method.solver is mock_solver
        assert pm_method.gumbel_scale == 1.0
        assert pm_method.solver_timeout == 60.0
        assert pm_method.name == "PerturbAndMAP(MockSolver)"
    
    def test_pm_hyperparameters(self, mock_model, mock_optimizer, mock_solver, pm_config):
        """Test P&M hyperparameters property."""
        pm_method = PerturbMapTraining(mock_model, mock_optimizer, mock_solver, pm_config)
        
        hyperparams = pm_method.hyperparameters
        assert hyperparams['solver'] == "MockSolver"
        assert hyperparams['gumbel_scale'] == 1.0
        assert hyperparams['solver_timeout'] == 60.0
        assert hyperparams['learning_rate'] == 0.01
        assert 'sampler_hyperparameters' in hyperparams
    
    def test_pm_negative_phase_shape(self, mock_model, mock_optimizer, mock_solver, pm_config):
        """Test that P&M negative phase returns correct shapes."""
        pm_method = PerturbMapTraining(mock_model, mock_optimizer, mock_solver, pm_config)
        
        # Mock the sampler to return specific shapes
        v_positive = torch.randn(32, 784)
        
        with patch.object(pm_method.sampler, 'sample') as mock_sample:
            mock_sample.return_value = (
                torch.randn(32, 784),  # v_negative
                torch.randn(32, 64)    # h_negative
            )
            
            v_neg, h_neg = pm_method.negative_phase(v_positive)
            
            assert v_neg.shape == (32, 784)
            assert h_neg.shape == (32, 64)
            mock_sample.assert_called_once_with(v_positive, 32)
    
    def test_pm_solver_statistics(self, mock_model, mock_optimizer, mock_solver, pm_config):
        """Test P&M solver statistics tracking."""
        pm_method = PerturbMapTraining(mock_model, mock_optimizer, mock_solver, pm_config)
        
        # Mock solver statistics
        with patch.object(pm_method.sampler, 'get_solver_statistics') as mock_stats:
            mock_stats.return_value = {
                'avg_solve_time': 0.5,
                'success_rate': 0.95,
                'total_solves': 100
            }
            
            diagnostics = pm_method.get_solver_diagnostics()
            
            assert diagnostics['solver_name'] == "MockSolver"
            assert diagnostics['avg_solve_time'] == 0.5
            assert diagnostics['success_rate'] == 0.95
            assert diagnostics['total_solves'] == 100
    
    def test_pm_adaptive_solver_parameters(self, mock_model, mock_optimizer, mock_solver, pm_config):
        """Test adaptive solver parameter adjustment."""
        pm_method = PerturbMapTraining(mock_model, mock_optimizer, mock_solver, pm_config)
        
        # Mock solver statistics with low success rate
        with patch.object(pm_method.sampler, 'get_solver_statistics') as mock_stats:
            mock_stats.return_value = {
                'avg_solve_time': 30.0,
                'success_rate': 0.7,  # Low success rate
                'total_solves': 100
            }
            
            original_timeout = pm_method.solver_timeout
            adjusted = pm_method.adapt_solver_parameters({})
            
            assert adjusted is True
            assert pm_method.solver_timeout > original_timeout


class TestTrainingMethodIntegration:
    """Integration tests for training methods with real components."""
    
    @pytest.fixture
    def real_model(self):
        """Create a real RBM model for integration testing."""
        return RBM(n_visible=28, n_hidden=16)
    
    @pytest.fixture
    def real_optimizer(self, real_model):
        """Create a real optimizer for integration testing."""
        return torch.optim.SGD(real_model.parameters(), lr=0.01)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return torch.randint(0, 2, (8, 28)).float()
    
    def test_cd_train_step_integration(self, real_model, real_optimizer, sample_data):
        """Test CD training step with real components."""
        config = {
            'training': {'batch_size': 8, 'learning_rate': 0.01},
            'cd_params': {
                'k_steps': 1,
                'persistent': False,
                'use_momentum': False,
                'momentum': 0.5,
                'temperature': 1.0,
                'seed': 42
            }
        }
        
        cd_method = ContrastiveDivergenceTraining(real_model, real_optimizer, config)
        
        # Test that train_step runs without errors
        metrics = cd_method.train_step(sample_data)
        
        assert isinstance(metrics, dict)
        assert 'reconstruction_error' in metrics
        assert 'positive_energy' in metrics
        assert 'negative_energy' in metrics
        assert metrics['reconstruction_error'] > 0
    
    def test_pm_train_step_integration(self, real_model, real_optimizer, sample_data):
        """Test P&M training step with real components (mocked solver)."""
        config = {
            'training': {'batch_size': 8, 'learning_rate': 0.01},
            'pm_params': {
                'gumbel_scale': 1.0,
                'solver_timeout': 60.0,
                'max_retries': 3,
                'seed': 42
            }
        }
        
        # Mock solver for integration test
        mock_solver = Mock()
        mock_solver.name = "TestSolver"
        mock_solver.solve.return_value = np.random.randint(0, 2, 28 + 16)
        
        pm_method = PerturbMapTraining(real_model, real_optimizer, mock_solver, config)
        
        # Test that train_step runs without errors
        metrics = pm_method.train_step(sample_data)
        
        assert isinstance(metrics, dict)
        assert 'reconstruction_error' in metrics
        assert 'positive_energy' in metrics
        assert 'negative_energy' in metrics
        assert metrics['reconstruction_error'] > 0
        assert 'pm_gumbel_scale' in metrics
        assert 'pm_solver_timeout' in metrics