"""
Unit tests for EBM samplers.

This module tests the different sampling strategies including 
GibbsSampler and QUBOSampler.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rbm.training.samplers.base import AbstractSampler
from rbm.training.samplers.gibbs_sampler import GibbsSampler
from rbm.training.samplers.qubo_sampler import QUBOSampler
from rbm.models.rbm import RBM


class TestAbstractSampler:
    """Test the abstract base class for samplers."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            AbstractSampler(None, {})
    
    def test_concrete_implementation_requires_sample_method(self):
        """Test that concrete implementations must implement sample method."""
        
        class IncompleteSampler(AbstractSampler):
            pass
        
        with pytest.raises(TypeError):
            IncompleteSampler(None, {})


class TestGibbsSampler:
    """Test the Gibbs sampler for Contrastive Divergence."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock RBM model for testing."""
        model = Mock(spec=RBM)
        model.n_visible = 784
        model.n_hidden = 64
        
        # Mock the weight matrices for direct access
        model.W = torch.randn(64, 784)
        model.b = torch.randn(784)
        model.c = torch.randn(64)
        
        return model
    
    @pytest.fixture
    def gibbs_config(self):
        """Configuration for Gibbs sampler."""
        return {
            'k_steps': 1,
            'persistent': False,
            'sampling_mode': 'stochastic',
            'burn_in_steps': 0,
            'seed': 42
        }
    
    def test_gibbs_initialization(self, mock_model, gibbs_config):
        """Test Gibbs sampler initialization."""
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        assert sampler.model is mock_model
        assert sampler.k_steps == 1
        assert sampler.persistent is False
        assert sampler.sampling_mode == 'stochastic'
        assert sampler.name == "GibbsSampler(k=1)"
    
    def test_gibbs_hyperparameters(self, mock_model, gibbs_config):
        """Test Gibbs sampler hyperparameters property."""
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        hyperparams = sampler.hyperparameters
        assert hyperparams['k_steps'] == 1
        assert hyperparams['persistent'] is False
        assert hyperparams['sampling_mode'] == 'stochastic'
        assert hyperparams['burn_in_steps'] == 0
        assert hyperparams['seed'] == 42
    
    def test_gibbs_sample_shape(self, mock_model, gibbs_config):
        """Test that Gibbs sampler returns correct shapes."""
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        v_init = torch.randn(32, 784)
        v_samples, h_samples = sampler.sample(v_init, 32)
        
        assert v_samples.shape == (32, 784)
        assert h_samples.shape == (32, 64)
    
    def test_gibbs_k_steps(self, mock_model, gibbs_config):
        """Test different k_steps values."""
        gibbs_config['k_steps'] = 5
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        assert sampler.k_steps == 5
        assert sampler.name == "GibbsSampler(k=5)"
    
    def test_gibbs_persistent_mode(self, mock_model, gibbs_config):
        """Test persistent mode initialization."""
        gibbs_config['persistent'] = True
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        assert sampler.persistent is True
        assert sampler.name == "PersistentGibbsSampler(k=1)"
        
        # Test that persistent chains are initialized
        v_init = torch.randn(32, 784)
        sampler.sample(v_init, 32)
        
        assert sampler.fantasy_particles is not None
        assert sampler.fantasy_particles.shape == (32, 784)
    
    def test_gibbs_momentum(self, mock_model, gibbs_config):
        """Test momentum configuration - momentum not implemented yet."""
        # Note: momentum is not implemented in current GibbsSampler
        # This test is a placeholder for future implementation
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        # Test that sampler still works without momentum
        assert sampler.sampling_mode == 'stochastic'
    
    def test_gibbs_temperature(self, mock_model, gibbs_config):
        """Test temperature scaling - temperature not implemented yet."""
        # Note: temperature scaling is not implemented in current GibbsSampler
        # This test is a placeholder for future implementation
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        # Test that sampler still works without temperature scaling
        assert sampler.sampling_mode == 'stochastic'
    
    def test_gibbs_sample_input_validation(self, mock_model, gibbs_config):
        """Test input validation for sample method."""
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            v_init = torch.randn(32, 100)  # Wrong visible dimension
            sampler.sample(v_init, 32)
        
        # Test with negative num_samples
        with pytest.raises(ValueError):
            v_init = torch.randn(32, 784)
            sampler.sample(v_init, -1)
    
    def test_gibbs_statistics_tracking(self, mock_model, gibbs_config):
        """Test statistics tracking."""
        sampler = GibbsSampler(mock_model, gibbs_config)
        
        v_init = torch.randn(32, 784)
        sampler.sample(v_init, 32)
        
        stats = sampler.get_statistics()
        assert 'sample_count' in stats
        assert stats['sample_count'] == 32
        assert 'sampler' in stats
        assert stats['sampler'] == 'GibbsSampler(k=1)'


class TestQUBOSampler:
    """Test the QUBO sampler for Perturb-and-MAP."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock RBM model for testing."""
        model = Mock(spec=RBM)
        model.n_visible = 784
        model.n_hidden = 64
        model.create_joint_qubo.return_value = np.random.randn(784 + 64, 784 + 64)
        return model
    
    @pytest.fixture
    def mock_solver(self):
        """Mock QUBO solver for testing."""
        solver = Mock()
        solver.name = "MockSolver"
        solver.solve.return_value = np.random.randint(0, 2, 784 + 64)
        return solver
    
    @pytest.fixture
    def qubo_config(self):
        """Configuration for QUBO sampler."""
        return {
            'gumbel_scale': 1.0,
            'solver_timeout': 60.0,
            'max_retries': 3,
            'seed': 42
        }
    
    def test_qubo_initialization(self, mock_model, mock_solver, qubo_config):
        """Test QUBO sampler initialization."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        assert sampler.model is mock_model
        assert sampler.solver is mock_solver
        assert sampler.gumbel_scale == 1.0
        assert sampler.solver_timeout == 60.0
        assert sampler.max_retries == 3
        assert sampler.name == "QUBOSampler(MockSolver)"
    
    def test_qubo_hyperparameters(self, mock_model, mock_solver, qubo_config):
        """Test QUBO sampler hyperparameters property."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        hyperparams = sampler.hyperparameters
        assert hyperparams['solver'] == "MockSolver"
        assert hyperparams['gumbel_scale'] == 1.0
        assert hyperparams['solver_timeout'] == 60.0
        assert hyperparams['max_retries'] == 3
        assert hyperparams['seed'] == 42
    
    def test_qubo_sample_shape(self, mock_model, mock_solver, qubo_config):
        """Test that QUBO sampler returns correct shapes."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        v_init = torch.randn(32, 784)
        v_samples, h_samples = sampler.sample(v_init, 32)
        
        assert v_samples.shape == (32, 784)
        assert h_samples.shape == (32, 64)
        
        # Check that solver was called for each sample
        assert mock_solver.solve.call_count == 32
    
    def test_qubo_solver_failure_handling(self, mock_model, mock_solver, qubo_config):
        """Test handling of solver failures."""
        # Make solver fail
        mock_solver.solve.side_effect = Exception("Solver failed")
        
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        v_init = torch.randn(1, 784)
        v_samples, h_samples = sampler.sample(v_init, 1)
        
        # Should fall back to random samples
        assert v_samples.shape == (1, 784)
        assert h_samples.shape == (1, 64)
        
        # Check that solver was attempted max_retries times
        assert mock_solver.solve.call_count == 3
    
    def test_qubo_solver_statistics(self, mock_model, mock_solver, qubo_config):
        """Test solver statistics tracking."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        v_init = torch.randn(5, 784)
        sampler.sample(v_init, 5)
        
        stats = sampler.get_solver_statistics()
        assert 'avg_solve_time' in stats
        assert 'success_rate' in stats
        assert 'total_solves' in stats
        assert stats['total_solves'] == 5
        assert stats['success_rate'] == 1.0  # All should succeed
    
    def test_qubo_solver_statistics_with_failures(self, mock_model, mock_solver, qubo_config):
        """Test solver statistics with some failures."""
        # Make solver fail on first attempt, succeed on retry
        mock_solver.solve.side_effect = [
            np.random.randint(0, 2, 784 + 64),  # Sample 1: Success
            Exception("Solver failed"),          # Sample 2: First attempt fails
            np.random.randint(0, 2, 784 + 64),  # Sample 2: Retry succeeds
        ]
        
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        v_init = torch.randn(2, 784)
        sampler.sample(v_init, 2)
        
        stats = sampler.get_solver_statistics()
        assert stats['total_solves'] == 2  # 2 successful solves (times recorded)
        # 2 successes out of 3 attempts
        assert abs(stats['success_rate'] - 2/3) < 0.01
    
    def test_qubo_reset_statistics(self, mock_model, mock_solver, qubo_config):
        """Test resetting solver statistics."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        v_init = torch.randn(5, 784)
        sampler.sample(v_init, 5)
        
        # Check that statistics exist
        stats = sampler.get_solver_statistics()
        assert stats['total_solves'] == 5
        
        # Reset statistics
        sampler.reset_statistics()
        
        # Check that statistics are reset
        stats = sampler.get_solver_statistics()
        assert stats['total_solves'] == 0
        assert stats['success_rate'] == 0.0
    
    def test_qubo_input_validation(self, mock_model, mock_solver, qubo_config):
        """Test input validation for QUBO sampler."""
        sampler = QUBOSampler(mock_model, mock_solver, qubo_config)
        
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            v_init = torch.randn(32, 100)  # Wrong visible dimension
            sampler.sample(v_init, 32)
        
        # Test with negative num_samples
        with pytest.raises(ValueError):
            v_init = torch.randn(32, 784)
            sampler.sample(v_init, -1)


class TestSamplerIntegration:
    """Integration tests for samplers with real components."""
    
    @pytest.fixture
    def real_model(self):
        """Create a real RBM model for integration testing."""
        return RBM(n_visible=28, n_hidden=16)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return torch.randint(0, 2, (8, 28)).float()
    
    def test_gibbs_sampler_integration(self, real_model, sample_data):
        """Test Gibbs sampler with real RBM model."""
        config = {
            'k_steps': 1,
            'persistent': False,
            'use_momentum': False,
            'momentum': 0.5,
            'temperature': 1.0,
            'seed': 42
        }
        
        sampler = GibbsSampler(real_model, config)
        
        # Test sampling
        v_samples, h_samples = sampler.sample(sample_data, 8)
        
        assert v_samples.shape == (8, 28)
        assert h_samples.shape == (8, 16)
        
        # Check that values are in valid range
        assert torch.all(v_samples >= 0) and torch.all(v_samples <= 1)
        assert torch.all(h_samples >= 0) and torch.all(h_samples <= 1)
    
    def test_qubo_sampler_integration(self, real_model, sample_data):
        """Test QUBO sampler with real RBM model (mocked solver)."""
        config = {
            'gumbel_scale': 1.0,
            'solver_timeout': 60.0,
            'max_retries': 3,
            'seed': 42
        }
        
        # Mock solver for integration test
        mock_solver = Mock()
        mock_solver.name = "TestSolver"
        mock_solver.solve.return_value = np.random.randint(0, 2, 28 + 16)
        
        sampler = QUBOSampler(real_model, mock_solver, config)
        
        # Test sampling
        v_samples, h_samples = sampler.sample(sample_data, 8)
        
        assert v_samples.shape == (8, 28)
        assert h_samples.shape == (8, 16)
        
        # Check that values are binary
        assert torch.all((v_samples == 0) | (v_samples == 1))
        assert torch.all((h_samples == 0) | (h_samples == 1))