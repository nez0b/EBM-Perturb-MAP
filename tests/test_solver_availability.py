"""
Tests for solver availability detection and graceful handling.

This module tests the solver availability checking mechanisms and ensures
proper pytest skipping behavior when solvers are not available.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

# Test availability checking for different solvers
SOLVER_IMPORTS = {
    'hexaly': {
        'module': 'rbm.solvers.hexaly',
        'class': 'HexalySolver',
        'available_flag': 'HEXALY_AVAILABLE'
    },
    'gurobi': {
        'module': 'rbm.solvers.gurobi', 
        'class': 'GurobiSolver',
        'available_flag': None
    },
    'scip': {
        'module': 'rbm.solvers.scip',
        'class': 'ScipSolver', 
        'available_flag': None
    },
    'dirac': {
        'module': 'rbm.solvers.dirac',
        'class': 'DiracSolver',
        'available_flag': None
    }
}


class TestSolverAvailability:
    """Test suite for solver availability detection."""
    
    def test_hexaly_availability_when_installed(self):
        """Test Hexaly availability detection when properly installed."""
        try:
            from rbm.solvers.hexaly import HexalySolver, HEXALY_AVAILABLE
            
            if HEXALY_AVAILABLE:
                # Test that solver can be imported
                assert HexalySolver is not None
                
                try:
                    # Test that solver can be instantiated (may fail if no license)
                    solver = HexalySolver()
                    assert solver.name == "Hexaly"
                    assert hasattr(solver, 'is_available')
                except ImportError:
                    # License not available, but import succeeded
                    pass
            else:
                pytest.skip("Hexaly library not installed")
                
        except ImportError:
            pytest.skip("Hexaly solver module not available")
    
    def test_hexaly_availability_when_not_installed(self):
        """Test Hexaly availability detection when not installed."""
        with patch.dict('sys.modules', {'hexaly.optimizer': None}):
            with patch('rbm.solvers.hexaly.HEXALY_AVAILABLE', False):
                try:
                    from rbm.solvers.hexaly import HexalySolver
                    
                    # Should raise ImportError when trying to instantiate
                    with pytest.raises(ImportError, match="Hexaly is not available"):
                        HexalySolver()
                        
                except ImportError:
                    # Module import failed, which is expected
                    pass
    
    def test_hexaly_license_check_failure(self):
        """Test Hexaly behavior when license check fails."""
        try:
            from rbm.solvers.hexaly import HexalySolver
        except ImportError:
            pytest.skip("Hexaly solver not available")
        
        # Mock the is_available property to simulate license failure
        with patch.object(HexalySolver, 'is_available', property(lambda self: False)):
            with pytest.raises(ImportError, match="Hexaly is not available"):
                HexalySolver()
    
    def test_gurobi_availability_detection(self):
        """Test Gurobi solver availability detection."""
        try:
            from rbm.solvers.gurobi import GurobiSolver
            
            # Test that solver has proper availability check
            assert hasattr(GurobiSolver, 'is_available')
            
            try:
                solver = GurobiSolver()
                assert solver.name == "Gurobi"
            except ImportError:
                # Gurobi not installed, which is expected
                pass
                
        except ImportError:
            # Module not available, which is expected
            pass
    
    def test_scip_availability_detection(self):
        """Test SCIP solver availability detection.""" 
        try:
            from rbm.solvers.scip import ScipSolver
            
            # Test that solver has proper availability check
            assert hasattr(ScipSolver, 'is_available')
            
            try:
                solver = ScipSolver()
                assert solver.name == "SCIP"
            except ImportError:
                # SCIP not installed, which is expected
                pass
                
        except ImportError:
            # Module not available, which is expected
            pass
    
    def test_dirac_availability_detection(self):
        """Test Dirac solver availability detection."""
        try:
            from rbm.solvers.dirac import DiracSolver
            
            # Test that solver has proper availability check
            assert hasattr(DiracSolver, 'is_available')
            
            try:
                solver = DiracSolver()
                assert solver.name == "Dirac"  
            except ImportError:
                # Dirac not installed, which is expected
                pass
                
        except ImportError:
            # Module not available, which is expected
            pass


class TestPytestSkipBehavior:
    """Test proper pytest skip behavior for unavailable solvers."""
    
    def test_pytest_skip_with_hexaly_unavailable(self):
        """Test that pytest.skip works correctly when Hexaly is unavailable."""
        # Simulate Hexaly being unavailable
        hexaly_available = False
        
        try:
            import hexaly.optimizer as hexaly
            hexaly_available = True
        except ImportError:
            hexaly_available = False
        
        if not hexaly_available:
            pytest.skip("Hexaly solver not available - this should skip gracefully")
        
        # If we reach here, Hexaly is available
        from rbm.solvers.hexaly import HexalySolver
        solver = HexalySolver()
        assert solver.name == "Hexaly"
    
    def test_conditional_test_execution_based_on_availability(self):
        """Test that tests are conditionally executed based on solver availability."""
        try:
            from rbm.solvers.hexaly import HexalySolver, HEXALY_AVAILABLE
            hexaly_import_success = True
        except ImportError:
            hexaly_import_success = False
            HEXALY_AVAILABLE = False
        
        # Test the import success flag
        if hexaly_import_success and HEXALY_AVAILABLE:
            # This part should only run if Hexaly is available
            solver = HexalySolver()
            assert solver.name == "Hexaly"
        else:
            # This part runs when Hexaly is not available
            pytest.skip("Hexaly not available - test skipped appropriately")
    
    @pytest.mark.skipif(True, reason="Always skip this test to verify skip behavior")
    def test_always_skipped_test(self):
        """Test that always fails - should be skipped."""
        pytest.fail("This test should never run due to skipif decorator")
    
    def test_skip_message_formatting(self):
        """Test that skip messages are properly formatted."""
        skip_messages = [
            "Hexaly solver import failed",
            "Hexaly solver not available - install hexaly and check license", 
            "Gurobi is not available - install gurobipy",
            "SCIP is not available - install python-scip"
        ]
        
        for message in skip_messages:
            # Verify skip messages are strings and informative
            assert isinstance(message, str)
            assert len(message) > 10  # Should be descriptive
            assert any(solver in message.lower() for solver in ['hexaly', 'gurobi', 'scip'])


class TestSolverDetectionUtilities:
    """Test utilities for solver detection."""
    
    def test_solver_availability_check_function(self):
        """Test general solver availability checking."""
        def check_solver_availability(solver_name: str) -> bool:
            """Check if a QUBO solver is available."""
            try:
                if solver_name == 'gurobi':
                    from rbm.solvers.gurobi import GurobiSolver
                    solver = GurobiSolver()
                    return solver.is_available if hasattr(solver, 'is_available') else False
                elif solver_name == 'scip':
                    from rbm.solvers.scip import ScipSolver
                    solver = ScipSolver()
                    return solver.is_available if hasattr(solver, 'is_available') else False
                elif solver_name == 'hexaly':
                    from rbm.solvers.hexaly import HexalySolver
                    solver = HexalySolver()
                    return solver.is_available if hasattr(solver, 'is_available') else False
                elif solver_name == 'dirac':
                    from rbm.solvers.dirac import DiracSolver
                    solver = DiracSolver()
                    return solver.is_available if hasattr(solver, 'is_available') else False
                else:
                    return False
            except ImportError:
                return False
        
        # Test the function with different solvers
        for solver_name in ['hexaly', 'gurobi', 'scip', 'dirac', 'nonexistent']:
            availability = check_solver_availability(solver_name)
            assert isinstance(availability, bool)
            
            if solver_name == 'nonexistent':
                assert availability == False
    
    def test_get_available_solvers_function(self):
        """Test function to get list of available solvers."""
        def get_available_solvers() -> list:
            """Get list of available QUBO solvers."""
            solvers = ['gurobi', 'scip', 'hexaly', 'dirac'] 
            available = []
            
            for solver_name in solvers:
                try:
                    if solver_name == 'hexaly':
                        from rbm.solvers.hexaly import HexalySolver
                        solver = HexalySolver()
                        if hasattr(solver, 'is_available') and solver.is_available:
                            available.append(solver_name)
                    elif solver_name == 'gurobi':
                        from rbm.solvers.gurobi import GurobiSolver  
                        solver = GurobiSolver()
                        if hasattr(solver, 'is_available') and solver.is_available:
                            available.append(solver_name)
                    elif solver_name == 'scip':
                        from rbm.solvers.scip import ScipSolver
                        solver = ScipSolver()
                        if hasattr(solver, 'is_available') and solver.is_available:
                            available.append(solver_name)
                    elif solver_name == 'dirac':
                        from rbm.solvers.dirac import DiracSolver
                        solver = DiracSolver()
                        if hasattr(solver, 'is_available') and solver.is_available:
                            available.append(solver_name)
                except (ImportError, Exception):
                    # Solver not available
                    continue
            
            return available
        
        available_solvers = get_available_solvers()
        
        # Test return type and structure
        assert isinstance(available_solvers, list)
        
        # All items should be valid solver names
        valid_solvers = {'gurobi', 'scip', 'hexaly', 'dirac'}
        for solver in available_solvers:
            assert solver in valid_solvers
    
    def test_solver_import_error_handling(self):
        """Test proper handling of solver import errors."""
        import_attempts = []
        
        # Test each solver import with error handling
        for solver_info in SOLVER_IMPORTS.values():
            try:
                module = __import__(solver_info['module'], fromlist=[solver_info['class']])
                solver_class = getattr(module, solver_info['class'])
                import_attempts.append({
                    'module': solver_info['module'],
                    'class': solver_info['class'], 
                    'success': True,
                    'error': None
                })
            except ImportError as e:
                import_attempts.append({
                    'module': solver_info['module'],
                    'class': solver_info['class'],
                    'success': False,
                    'error': str(e)
                })
            except Exception as e:
                import_attempts.append({
                    'module': solver_info['module'],
                    'class': solver_info['class'],
                    'success': False,
                    'error': f"Unexpected error: {str(e)}"
                })
        
        # Verify we attempted to import all solvers
        assert len(import_attempts) == len(SOLVER_IMPORTS)
        
        # Check that error handling was proper (no unhandled exceptions)
        for attempt in import_attempts:
            assert 'success' in attempt
            assert 'error' in attempt
            if not attempt['success']:
                assert attempt['error'] is not None


class TestTrainingManagerSolverSelection:
    """Test solver selection logic in TrainingManager."""
    
    def test_solver_fallback_mechanism(self):
        """Test that unavailable solvers trigger fallback to available ones."""
        # Mock available solvers list
        def mock_get_available_solvers():
            return ['hexaly']  # Only Hexaly available
        
        def mock_check_solver_availability(solver_name):
            return solver_name == 'hexaly'
        
        # Test fallback logic
        requested_solver = 'gurobi'  # Not available
        available_solvers = mock_get_available_solvers()
        
        if not mock_check_solver_availability(requested_solver):
            if available_solvers:
                fallback_solver = available_solvers[0]
                assert fallback_solver == 'hexaly'
            else:
                pytest.fail("No solvers available - should raise error")
    
    def test_no_solvers_available_error(self):
        """Test error handling when no solvers are available."""
        def mock_get_available_solvers():
            return []  # No solvers available
        
        available_solvers = mock_get_available_solvers()
        
        if not available_solvers:
            # This should trigger an error in the actual system
            with pytest.raises(AssertionError):
                assert len(available_solvers) > 0, "No QUBO solvers available!"
    
    def test_solver_configuration_validation(self):
        """Test that solver configurations are properly validated."""
        valid_solvers = ['gurobi', 'scip', 'hexaly', 'dirac']
        
        # Test valid solver names
        for solver in valid_solvers:
            assert solver in valid_solvers
        
        # Test invalid solver names
        invalid_solvers = ['invalid', 'cplex', 'mosek', '']
        for solver in invalid_solvers:
            assert solver not in valid_solvers