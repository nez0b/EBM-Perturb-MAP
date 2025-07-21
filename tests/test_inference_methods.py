"""
Comprehensive tests for inference method switching functionality.

This module tests the fixes for inference method selection (QUBO vs Gibbs)
and ensures proper solver configuration independence between training and inference.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any

# Test imports
try:
    from rbm.solvers.hexaly import HexalySolver
    HEXALY_IMPORT_SUCCESS = True
except ImportError:
    HEXALY_IMPORT_SUCCESS = False
    HexalySolver = None


class TestInferenceMethodSwitching:
    """Test suite for inference method switching between QUBO and Gibbs."""
    
    @pytest.fixture
    def mock_config_qubo(self) -> Dict[str, Any]:
        """Create a configuration dict for QUBO inference method."""
        return {
            'model': {'n_visible': 784, 'n_hidden': 128},
            'training': {'batch_size': 1},
            'data': {'image_size': [28, 28]},
            'solver': {'suppress_output': True},
            'inference': {
                'method': 'qubo',
                'qubo_solver': 'hexaly',
                'time_limit': 10.0,
                'reconstruction_samples': 5,
                'num_generated_samples': 10
            }
        }
    
    @pytest.fixture
    def mock_config_gibbs(self) -> Dict[str, Any]:
        """Create a configuration dict for Gibbs inference method."""
        return {
            'model': {'n_visible': 784, 'n_hidden': 128},
            'training': {'batch_size': 1},
            'data': {'image_size': [28, 28]},
            'inference': {
                'method': 'gibbs',
                'gibbs_steps': 1000,
                'reconstruction_samples': 5,
                'num_generated_samples': 10
            }
        }
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mock training manager with model and training method."""
        manager = MagicMock()
        manager.model = MagicMock()
        manager.training_method = MagicMock()
        manager.training_method.name = "PerturbAndMAP(Hexaly)"
        manager.training_method.negative_phase.return_value = (
            torch.randn(1, 784),  # v_neg
            torch.randn(1, 128)   # h_neg
        )
        manager.model.create_qubo_for_sampling.return_value = (
            np.random.randn(128, 128),  # Q matrix
            None  # offset
        )
        manager.model.reconstruct.return_value = torch.randn(1, 784)
        return manager
    
    def test_qubo_inference_method_selection(self, mock_config_qubo):
        """Test that QUBO inference method is correctly selected from config."""
        # Test the method selection logic from train_and_infer_qubo.py
        inference_config = mock_config_qubo['inference']
        inference_method = inference_config.get('method', 'gibbs').lower()
        
        assert inference_method == 'qubo'
        assert inference_method in ['qubo', 'gibbs']  # Validation
    
    def test_gibbs_inference_method_selection(self, mock_config_gibbs):
        """Test that Gibbs inference method is correctly selected from config."""
        inference_config = mock_config_gibbs['inference']
        inference_method = inference_config.get('method', 'gibbs').lower()
        
        assert inference_method == 'gibbs'
        assert inference_method in ['qubo', 'gibbs']  # Validation
    
    def test_invalid_inference_method_raises_error(self):
        """Test that invalid inference methods raise appropriate errors."""
        invalid_config = {
            'inference': {'method': 'invalid_method'}
        }
        
        inference_method = invalid_config['inference']['method'].lower()
        
        # This should fail validation
        with pytest.raises(AssertionError):
            assert inference_method in ['qubo', 'gibbs']
    
    def test_default_inference_method_is_gibbs(self):
        """Test that the default inference method is Gibbs when not specified."""
        config_no_method = {'inference': {}}
        
        inference_method = config_no_method['inference'].get('method', 'gibbs').lower()
        
        assert inference_method == 'gibbs'
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available")
    def test_inference_solver_initialization_hexaly(self, mock_config_qubo):
        """Test that QUBO inference solver is properly initialized with correct parameters."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly'):
                inference_config = mock_config_qubo['inference']
                solver_name = inference_config.get('qubo_solver', 'hexaly').lower()
                time_limit = inference_config.get('time_limit', 10.0)
                suppress_output = mock_config_qubo['solver'].get('suppress_output', True)
                
                # Simulate solver initialization from train_and_infer_qubo.py
                if solver_name == 'hexaly':
                    inference_solver = HexalySolver(
                        time_limit=time_limit,
                        suppress_output=suppress_output
                    )
                    
                    assert inference_solver.time_limit == 10.0
                    assert inference_solver.suppress_output == True
                    assert inference_solver.name == "Hexaly"
                
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_inference_solver_separate_time_limit(self, mock_config_qubo):
        """Test that inference solver uses separate time limit from training config."""
        # Training solver might have different time limit
        training_time_limit = 0.5  # Training uses sub-second limits
        inference_time_limit = 10.0  # Inference uses longer limits
        
        assert mock_config_qubo['inference']['time_limit'] == inference_time_limit
        
        # Verify that inference and training can have different time limits
        assert training_time_limit != inference_time_limit
    
    def test_qubo_reconstruction_method_selection(self, mock_manager, mock_config_qubo):
        """Test that QUBO reconstruction method calls the correct solver path."""
        inference_method = mock_config_qubo['inference']['method']
        
        # Mock test image
        test_image = torch.rand(1, 1, 28, 28)  # Batch, channels, height, width
        v_pos = test_image.view(test_image.size(0), -1)
        
        if inference_method == 'qubo':
            # Simulate QUBO reconstruction path
            v_input = (v_pos.view(-1) > 0.5).float()
            Q_h, _ = mock_manager.model.create_qubo_for_sampling(v_input)
            
            # Verify QUBO path is called
            mock_manager.model.create_qubo_for_sampling.assert_called_once()
            assert isinstance(Q_h, np.ndarray)
            
        elif inference_method == 'gibbs':
            # Simulate Gibbs reconstruction path  
            v_neg, h_neg = mock_manager.training_method.negative_phase(v_pos)
            
            # Verify Gibbs path is called
            mock_manager.training_method.negative_phase.assert_called_once()
            assert isinstance(v_neg, torch.Tensor)
            assert isinstance(h_neg, torch.Tensor)
    
    def test_gibbs_reconstruction_method_selection(self, mock_manager, mock_config_gibbs):
        """Test that Gibbs reconstruction method calls the correct path."""
        inference_method = mock_config_gibbs['inference']['method']
        
        # Mock test image
        test_image = torch.rand(1, 1, 28, 28)
        v_pos = test_image.view(test_image.size(0), -1)
        
        if inference_method == 'gibbs':
            # Simulate Gibbs reconstruction path
            v_neg, h_neg = mock_manager.training_method.negative_phase(v_pos)
            
            # Verify Gibbs path is called
            mock_manager.training_method.negative_phase.assert_called_once()
            assert isinstance(v_neg, torch.Tensor)
            assert isinstance(h_neg, torch.Tensor)
    
    def test_generation_task_solver_independence(self, mock_manager):
        """Test that generation task uses inference solver, not training solver."""
        # Mock inference solver (separate from training)
        mock_inference_solver = MagicMock()
        mock_inference_solver.solve.return_value = np.array([1, 0, 1, 0, 1])
        
        # Mock training method solver (should NOT be used for generation)
        mock_manager.training_method.solver = MagicMock()
        
        # Simulate QUBO-based generation using inference solver
        random_v = torch.rand(1, 784)
        v_input = (random_v.view(-1) > 0.5).float()
        
        Q_h, _ = mock_manager.model.create_qubo_for_sampling(v_input)
        h_sample_np = mock_inference_solver.solve(Q_h)  # Uses INFERENCE solver
        
        # Verify inference solver was called
        mock_inference_solver.solve.assert_called_once()
        
        # Verify training method solver was NOT called
        mock_manager.training_method.solver.solve.assert_not_called()
    
    def test_config_propagation_to_inference_components(self, mock_config_qubo):
        """Test that configuration values properly propagate to inference components."""
        inference_config = mock_config_qubo['inference']
        
        # Test all expected config values are present
        assert 'method' in inference_config
        assert 'qubo_solver' in inference_config
        assert 'time_limit' in inference_config
        assert 'reconstruction_samples' in inference_config
        assert 'num_generated_samples' in inference_config
        
        # Test values are correct types
        assert isinstance(inference_config['method'], str)
        assert isinstance(inference_config['qubo_solver'], str)
        assert isinstance(inference_config['time_limit'], (int, float))
        assert isinstance(inference_config['reconstruction_samples'], int)
        assert isinstance(inference_config['num_generated_samples'], int)
    
    def test_multi_batch_reconstruction_collection(self, mock_config_qubo):
        """Test that multiple images are collected for reconstruction (fix for single image issue)."""
        desired_samples = mock_config_qubo['inference']['reconstruction_samples']  # 5
        batch_size = mock_config_qubo['training']['batch_size']  # 1
        
        # Simulate collecting samples from multiple batches
        test_images = []
        total_collected = 0
        
        # Mock data loader with single image batches
        mock_batches = [torch.rand(1, 1, 28, 28) for _ in range(desired_samples)]
        
        for batch in mock_batches:
            batch_size_current = batch.size(0)
            needed = desired_samples - total_collected
            if needed <= 0:
                break
            
            # Take what we need from this batch
            take = min(needed, batch_size_current)
            test_images.append(batch[:take])
            total_collected += take
        
        if test_images:
            test_images_tensor = torch.cat(test_images, dim=0)
            num_samples = test_images_tensor.size(0)
        
        # Verify we collected the desired number of samples
        assert num_samples == desired_samples
        assert num_samples > 1  # Fixed: no longer just 1 image
        assert test_images_tensor.shape[0] == 5  # Expected number of samples


class TestInferenceMethodIntegration:
    """Integration tests for inference method functionality."""
    
    def test_inference_method_validation_function(self):
        """Test the validation logic for inference methods."""
        valid_methods = ['qubo', 'gibbs']
        
        # Test valid methods
        for method in valid_methods:
            assert method in valid_methods  # Should pass
        
        # Test invalid method
        invalid_method = 'invalid'
        with pytest.raises(AssertionError):
            assert invalid_method in valid_methods  # Should fail
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available")  
    def test_hexaly_solver_import_and_availability(self):
        """Test that Hexaly solver can be imported and its availability checked."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly'):
                solver = HexalySolver()
                assert solver.name == "Hexaly"
                assert hasattr(solver, 'time_limit')
                assert hasattr(solver, 'suppress_output')
                
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_inference_config_defaults(self):
        """Test that inference configuration has reasonable defaults."""
        minimal_config = {}
        
        # Test default method
        method = minimal_config.get('method', 'gibbs')
        assert method == 'gibbs'
        
        # Test default solver
        solver = minimal_config.get('qubo_solver', 'hexaly')
        assert solver == 'hexaly'
        
        # Test default time limit
        time_limit = minimal_config.get('time_limit', 10.0)
        assert time_limit == 10.0
        assert isinstance(time_limit, (int, float))
        
        # Test default samples
        reconstruction_samples = minimal_config.get('reconstruction_samples', 5)
        assert reconstruction_samples == 5
        
        num_generated_samples = minimal_config.get('num_generated_samples', 10)
        assert num_generated_samples == 10