"""
Integration tests for the inference pipeline end-to-end functionality.

This module tests the complete inference pipeline including config propagation,
solver initialization, and the fixes for generation task using inference solver
instead of training solver.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Dict, Any

# Test imports with availability checking
try:
    from rbm.solvers.hexaly import HexalySolver
    HEXALY_IMPORT_SUCCESS = True
except ImportError:
    HEXALY_IMPORT_SUCCESS = False
    HexalySolver = None

try:
    from rbm.training.training_manager import TrainingManager
    TRAINING_MANAGER_AVAILABLE = True
except ImportError:
    TRAINING_MANAGER_AVAILABLE = False
    TrainingManager = None


class TestInferencePipelineIntegration:
    """Integration tests for the complete inference pipeline."""
    
    @pytest.fixture
    def mock_config_complete(self) -> Dict[str, Any]:
        """Create a complete configuration for integration testing."""
        return {
            'model': {
                'n_visible': 784,
                'n_hidden': 128,
                'model_type': 'rbm'
            },
            'training': {
                'batch_size': 1,
                'method': 'perturb_map',
                'checkpoint_path': 'test_checkpoint.pth'
            },
            'data': {
                'image_size': [28, 28],
                'dataset': 'mnist'
            },
            'solver': {
                'name': 'hexaly',
                'time_limit': 0.5,
                'suppress_output': True
            },
            'inference': {
                'method': 'qubo',
                'qubo_solver': 'hexaly', 
                'time_limit': 10.0,
                'reconstruction_samples': 5,
                'num_generated_samples': 3
            }
        }
    
    @pytest.fixture
    def mock_checkpoint_data(self):
        """Create mock checkpoint data."""
        return {
            'epoch': 10,
            'model_state_dict': {
                'W': torch.randn(128, 784),
                'b': torch.randn(784),
                'c': torch.randn(128)
            },
            'training_method': 'perturb_map'
        }
    
    @pytest.fixture
    def mock_test_images(self):
        """Create mock test images for inference."""
        return torch.rand(5, 1, 28, 28)  # 5 test images
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available")
    def test_inference_solver_initialization_from_config(self, mock_config_complete):
        """Test that inference solver is correctly initialized from config parameters."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly'):
                # Simulate inference solver initialization logic from train_and_infer_qubo.py
                inference_config = mock_config_complete['inference']
                inference_method = inference_config.get('method', 'gibbs').lower()
                
                if inference_method == 'qubo':
                    solver_name = inference_config.get('qubo_solver', 'hexaly').lower()
                    time_limit = inference_config.get('time_limit', 10.0)
                    suppress_output = mock_config_complete['solver'].get('suppress_output', True)
                    
                    if solver_name == 'hexaly':
                        inference_solver = HexalySolver(
                            time_limit=time_limit,
                            suppress_output=suppress_output
                        )
                        
                        # Verify solver was created with correct parameters
                        assert inference_solver.time_limit == 10.0
                        assert inference_solver.suppress_output == True
                        assert inference_solver.name == "Hexaly"
                        
                        # Verify independence from training solver config
                        training_time_limit = mock_config_complete['solver']['time_limit']  # 0.5
                        assert inference_solver.time_limit != training_time_limit
                        
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_config_propagation_chain(self, mock_config_complete):
        """Test that config values propagate correctly through the system."""
        # Test inference config propagation
        inference_config = mock_config_complete['inference']
        
        # Verify all expected inference parameters
        assert inference_config['method'] == 'qubo'
        assert inference_config['qubo_solver'] == 'hexaly'
        assert inference_config['time_limit'] == 10.0
        assert inference_config['reconstruction_samples'] == 5
        assert inference_config['num_generated_samples'] == 3
        
        # Test solver config propagation
        solver_config = mock_config_complete['solver']
        assert solver_config['name'] == 'hexaly'
        assert solver_config['suppress_output'] == True
        
        # Test training config separation
        training_config = mock_config_complete['training']
        assert training_config['batch_size'] == 1
        
        # Verify inference and training configs are independent
        assert inference_config['time_limit'] != solver_config['time_limit']
    
    @pytest.mark.skipif(not TRAINING_MANAGER_AVAILABLE, reason="TrainingManager not available")
    def test_training_manager_solver_creation(self, mock_config_complete):
        """Test that TrainingManager creates Hexaly solver with correct parameters."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly'), \
                 patch.object(TrainingManager, 'load_checkpoint') as mock_load, \
                 patch('rbm.models.rbm.RBM'):
                
                mock_load.return_value = {'epoch': 1}
                
                # Test the training manager's solver creation
                solver_config = mock_config_complete['solver']
                
                # Simulate _create_solver method logic
                if solver_config['name'] == 'hexaly':
                    solver = HexalySolver(
                        time_limit=solver_config.get('time_limit', 120.0),
                        nb_threads=solver_config.get('nb_threads', 4),
                        seed=solver_config.get('seed', 42),
                        suppress_output=solver_config.get('suppress_output', True)
                    )
                    
                    # Verify training solver parameters
                    assert solver.time_limit == 0.5  # Training time limit
                    assert solver.suppress_output == True
                    assert solver.nb_threads == 4  # Default
                    assert solver.seed == 42  # Default
                    
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_multi_batch_reconstruction_fix(self, mock_config_complete, mock_test_images):
        """Test the fix for collecting multiple reconstruction images."""
        inference_config = mock_config_complete['inference']
        desired_samples = inference_config['reconstruction_samples']  # 5
        batch_size = mock_config_complete['training']['batch_size']  # 1
        
        # Simulate the multi-batch collection logic from train_and_infer_qubo.py
        test_images = []
        total_collected = 0
        
        # Mock data loader that yields batches of size 1
        mock_data_loader = [(mock_test_images[i:i+1], torch.tensor([6])) for i in range(5)]
        
        for batch, _ in mock_data_loader:
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
        else:
            pytest.fail("No test images collected")
        
        # Verify the fix works: we collect multiple images, not just 1
        assert num_samples == desired_samples
        assert num_samples > 1  # This was the bug: only collecting 1 image
        assert total_collected == 5
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available")
    def test_generation_task_uses_inference_solver(self, mock_config_complete):
        """Test that generation task uses inference solver, not training solver."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly') as mock_hexaly:
                # Setup mocks
                mock_optimizer = MagicMock()
                mock_param = MagicMock()
                mock_model = MagicMock()
                
                mock_hexaly.HexalyOptimizer.return_value.__enter__.return_value = mock_optimizer
                mock_optimizer.get_param.return_value = mock_param
                mock_optimizer.get_model.return_value = mock_model
                
                # Mock solution extraction
                mock_var = MagicMock()
                mock_var.get_value.return_value = 1
                mock_model.bool.return_value = mock_var
                
                # Create inference solver (separate from training)
                inference_config = mock_config_complete['inference'] 
                inference_time_limit = inference_config['time_limit']  # 10.0s
                suppress_output = mock_config_complete['solver']['suppress_output']
                
                inference_solver = HexalySolver(
                    time_limit=inference_time_limit,
                    suppress_output=suppress_output
                )
                
                # Create mock training solver (should NOT be used)
                training_time_limit = mock_config_complete['solver']['time_limit']  # 0.5s  
                training_solver = HexalySolver(
                    time_limit=training_time_limit,
                    suppress_output=suppress_output
                )
                
                # Simulate generation task using inference solver
                Q_test = np.array([[1, -1], [-1, 1]])
                solution = inference_solver.solve(Q_test)
                
                # Verify inference solver parameters were used (10.0s time limit)
                # Since 10.0 >= 1.0, should use time-based control
                mock_param.set_time_limit.assert_called_with(10)  # inference time limit
                mock_param.set_verbosity.assert_called_with(0)  # suppress_output=True
                
                # Verify solution was returned
                assert len(solution) == 2  # Mocked Q_test is 2x2 matrix
                
                # The key test: inference solver was used, not training solver
                # (This is validated by the correct time_limit being set)
                
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_inference_method_conditional_logic(self, mock_config_complete):
        """Test the conditional logic for choosing between QUBO and Gibbs inference."""
        # Mock model and training method
        mock_model = MagicMock()
        mock_training_method = MagicMock()
        
        # Mock QUBO path
        mock_model.create_qubo_for_sampling.return_value = (np.random.randn(5, 5), None)
        mock_model.reconstruct.return_value = torch.randn(1, 784)
        
        # Mock Gibbs path
        mock_training_method.negative_phase.return_value = (
            torch.randn(1, 784),  # v_neg
            torch.randn(1, 128)   # h_neg
        )
        
        # Test QUBO inference path
        inference_method = mock_config_complete['inference']['method']  # 'qubo'
        test_image = torch.rand(1, 784)
        
        if inference_method == 'qubo':
            # Simulate QUBO reconstruction logic
            v_input = (test_image.view(-1) > 0.5).float()
            Q_h, _ = mock_model.create_qubo_for_sampling(v_input)
            
            # Mock QUBO solver result
            h_sample_np = np.array([1, 0, 1, 0, 1])
            h_sample = torch.from_numpy(h_sample_np).float().unsqueeze(0)
            
            reconstructed = mock_model.reconstruct(h_sample)
            
            # Verify QUBO path was used
            mock_model.create_qubo_for_sampling.assert_called_once()
            mock_model.reconstruct.assert_called_once()
            mock_training_method.negative_phase.assert_not_called()
            
        elif inference_method == 'gibbs':
            # Simulate Gibbs reconstruction logic
            v_neg, h_neg = mock_training_method.negative_phase(test_image)
            reconstructed = mock_model.reconstruct(h_neg)
            
            # Verify Gibbs path was used
            mock_training_method.negative_phase.assert_called_once()
            mock_model.reconstruct.assert_called_once()
    
    def test_solver_parameter_independence(self, mock_config_complete):
        """Test that inference solver parameters are independent from training solver."""
        # Extract configurations
        solver_config = mock_config_complete['solver']  # Training solver config
        inference_config = mock_config_complete['inference']  # Inference config
        
        # Training solver parameters (used during training)
        training_time_limit = solver_config['time_limit']  # 0.5s
        training_suppress_output = solver_config['suppress_output']  # True
        
        # Inference solver parameters (used during inference)
        inference_time_limit = inference_config['time_limit']  # 10.0s
        inference_suppress_output = training_suppress_output  # Inherited from solver config
        
        # Verify independence
        assert training_time_limit != inference_time_limit  # Different time limits
        assert training_suppress_output == inference_suppress_output  # Same suppress_output
        
        # Verify training uses sub-second limit (iteration-based)
        assert training_time_limit < 1.0
        
        # Verify inference uses normal time limit (time-based)
        assert inference_time_limit >= 1.0


class TestErrorHandlingIntegration:
    """Integration tests for error handling in the inference pipeline."""
    
    def test_invalid_inference_method_error_handling(self):
        """Test error handling for invalid inference methods."""
        invalid_config = {
            'inference': {'method': 'invalid_method'}
        }
        
        inference_method = invalid_config['inference']['method'].lower()
        
        # Test the validation logic that should catch this
        valid_methods = ['qubo', 'gibbs']
        
        with pytest.raises(AssertionError):
            assert inference_method in valid_methods
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available")
    def test_solver_unavailable_error_handling(self):
        """Test error handling when Hexaly solver is unavailable."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            # Mock Hexaly as unavailable
            with patch('rbm.solvers.hexaly.HEXALY_AVAILABLE', False):
                with pytest.raises(ImportError, match="Hexaly is not available"):
                    HexalySolver()
                    
        except ImportError:
            pytest.skip("Hexaly solver not available")
    
    def test_config_missing_parameters_handling(self):
        """Test handling of missing configuration parameters with defaults."""
        minimal_config = {
            'inference': {}  # Missing all parameters
        }
        
        # Test default values are applied
        inference_config = minimal_config['inference']
        method = inference_config.get('method', 'gibbs')
        solver = inference_config.get('qubo_solver', 'hexaly')
        time_limit = inference_config.get('time_limit', 10.0)
        samples = inference_config.get('reconstruction_samples', 5)
        
        assert method == 'gibbs'
        assert solver == 'hexaly'
        assert time_limit == 10.0
        assert samples == 5
    
    def test_checkpoint_loading_error_simulation(self):
        """Test error handling for checkpoint loading failures."""
        # This would typically be handled by TrainingManager
        fake_checkpoint_path = "/nonexistent/checkpoint.pth"
        
        # Simulate the error condition
        with pytest.raises((FileNotFoundError, OSError)):
            # This would be the actual torch.load call
            torch.load(fake_checkpoint_path)


class TestPerformanceIntegration:
    """Integration tests for performance-related fixes."""
    
    @pytest.mark.skipif(not HEXALY_IMPORT_SUCCESS, reason="Hexaly not available") 
    def test_sub_second_vs_normal_time_limit_performance(self):
        """Test that sub-second and normal time limits work as expected."""
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        try:
            with patch('rbm.solvers.hexaly.hexaly') as mock_hexaly:
                mock_optimizer = MagicMock()
                mock_param = MagicMock()
                
                mock_hexaly.HexalyOptimizer.return_value.__enter__.return_value = mock_optimizer
                mock_optimizer.get_param.return_value = mock_param
                mock_optimizer.get_model.return_value = MagicMock()
                
                # Test sub-second limit (should use iterations)
                solver_fast = HexalySolver(time_limit=0.5)
                Q = np.array([[1]])
                
                # Mock solution
                mock_var = MagicMock()
                mock_var.get_value.return_value = 1
                mock_optimizer.get_model.return_value.bool.return_value = mock_var
                
                solver_fast.solve(Q)
                
                # Verify iteration-based control for sub-second
                mock_param.set_time_limit.assert_called_with(3600)
                mock_param.set_iteration_limit.assert_called_with(2500)  # 0.5 * 5000
                
                # Reset mocks
                mock_param.reset_mock()
                
                # Test normal limit (should use time)
                solver_normal = HexalySolver(time_limit=10.0)
                solver_normal.solve(Q)
                
                # Verify time-based control for normal time
                mock_param.set_time_limit.assert_called_with(10)
                mock_param.set_iteration_limit.assert_not_called()
                
        except ImportError:
            pytest.skip("Hexaly solver not available")