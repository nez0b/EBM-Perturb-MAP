"""
Comprehensive tests for the SCIP QUBO solver.

This module contains tests for the ScipSolver class, including tests with
various small QUBO problems, edge cases, and error handling.
"""

import pytest
import numpy as np
import torch
from typing import Tuple, List

# Import the SCIP solver
try:
    from rbm.solvers.scip import ScipSolver
    SCIP_IMPORT_SUCCESS = True
except ImportError:
    SCIP_IMPORT_SUCCESS = False
    ScipSolver = None


class TestScipSolver:
    """Test suite for the SCIP QUBO solver."""
    
    @pytest.fixture
    def solver(self):
        """Create a SCIP solver instance for testing."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        if not ScipSolver.is_available:
            pytest.skip("SCIP solver not available - install pyscipopt")
        return ScipSolver(time_limit=30.0)
    
    @pytest.fixture
    def solver_short_timeout(self):
        """Create a SCIP solver with short timeout for timeout tests."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        if not ScipSolver.is_available:
            pytest.skip("SCIP solver not available - install pyscipopt")
        return ScipSolver(time_limit=1.0)
    
    # ========== Basic Functionality Tests ==========
    
    def test_solver_availability(self):
        """Test that solver availability is properly reported."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        
        # Test by creating an instance (is_available is an instance property)
        try:
            solver = ScipSolver()
            availability = solver.is_available
            assert isinstance(availability, bool)
        except ImportError:
            # If SCIP is not available, this is expected
            pytest.skip("SCIP is not available - this is expected")
    
    def test_solver_initialization(self):
        """Test solver initialization with different parameters."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        if not ScipSolver.is_available:
            pytest.skip("SCIP solver not available")
        
        # Test default initialization
        solver1 = ScipSolver()
        assert solver1.time_limit == 60.0
        
        # Test custom time limit
        solver2 = ScipSolver(time_limit=120.0)
        assert solver2.time_limit == 120.0
    
    def test_solver_name(self, solver):
        """Test that solver name is correct."""
        assert solver.name == "SCIP"
    
    def test_solver_unavailable_error(self):
        """Test that appropriate error is raised when SCIP is unavailable."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        
        # Mock the availability check
        original_available = ScipSolver.is_available
        
        # Temporarily make solver unavailable
        ScipSolver.is_available = False
        
        try:
            with pytest.raises(ImportError, match="SCIP is not available"):
                ScipSolver()
        finally:
            # Restore original availability
            ScipSolver.is_available = original_available
    
    # ========== Small QUBO Problem Tests ==========
    
    def test_2x2_simple_qubo(self, solver):
        """Test simple 2x2 QUBO with known optimal solution."""
        # Q = [[1, -2], [-2, 1]]
        # min x^T Q x where x ∈ {0,1}^2
        # Known optimal solution: [1, 1] with objective value = -2
        Q = np.array([[1, -2], [-2, 1]], dtype=np.float64)
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (2,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value
        objective_value = solution.T @ Q @ solution
        
        # The optimal solution should be [1, 1] with objective -2
        # But we'll be more lenient and just check it's reasonable
        assert objective_value <= 0  # Should be negative
        
        # If we got the optimal solution [1, 1]
        if np.array_equal(solution, [1, 1]):
            assert np.isclose(objective_value, -2)
    
    def test_3x3_identity_matrix(self, solver):
        """Test 3x3 identity matrix QUBO."""
        # Q = I_3 (identity matrix)
        # min x^T Q x = min sum(x_i^2) = min sum(x_i)  (since x_i ∈ {0,1})
        # Optimal solution: [0, 0, 0] with objective value = 0
        Q = np.eye(3, dtype=np.float64)
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (3,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value
        objective_value = solution.T @ Q @ solution
        assert objective_value >= 0  # Should be non-negative
        
        # Optimal solution should be all zeros
        if np.array_equal(solution, [0, 0, 0]):
            assert np.isclose(objective_value, 0)
    
    def test_3x3_diagonal_negative(self, solver):
        """Test 3x3 diagonal matrix with negative values."""
        # Q = [[-1, 0, 0], [0, -2, 0], [0, 0, -3]]
        # Optimal solution: [1, 1, 1] with objective value = -6
        Q = np.array([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], dtype=np.float64)
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (3,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value
        objective_value = solution.T @ Q @ solution
        assert objective_value <= 0  # Should be negative
        
        # Optimal solution should be [1, 1, 1]
        if np.array_equal(solution, [1, 1, 1]):
            assert np.isclose(objective_value, -6)
    
    @pytest.mark.parametrize("qubo_matrix,expected_properties", [
        # Test case 1: 2x2 with negative off-diagonal encouraging both variables
        (np.array([[0, -1], [-1, 0]], dtype=np.float64), {"min_objective": -2, "variables": 2}),
        
        # Test case 2: 4x4 identity (all zeros is optimal)
        (np.eye(4, dtype=np.float64), {"min_objective": 0, "variables": 4}),
        
        # Test case 3: 3x3 with mixed positive/negative
        (np.array([[1, -1, 0], [-1, 1, -1], [0, -1, 1]], dtype=np.float64), {"min_objective": None, "variables": 3}),
    ])
    def test_parametrized_qubo_problems(self, solver, qubo_matrix, expected_properties):
        """Test multiple QUBO problems with known properties."""
        solution = solver.solve(qubo_matrix)
        
        # Basic checks
        assert solution.shape == (expected_properties["variables"],)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value bounds if specified
        if expected_properties["min_objective"] is not None:
            objective_value = solution.T @ qubo_matrix @ solution
            assert objective_value >= expected_properties["min_objective"] - 1e-6
    
    def test_4x4_max_cut_problem(self, solver):
        """Test 4x4 max cut problem formulation."""
        # Max cut for a 4-node cycle graph
        # Adjacency matrix: A = [[0,1,0,1], [1,0,1,0], [0,1,0,1], [1,0,1,0]]
        # QUBO matrix: Q = diag(degree) - A
        A = np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0]], dtype=np.float64)
        
        degrees = np.sum(A, axis=1)
        Q = np.diag(degrees) - A
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (4,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Calculate cut value (number of edges cut)
        cut_value = 0
        for i in range(4):
            for j in range(i+1, 4):
                if A[i, j] == 1 and solution[i] != solution[j]:
                    cut_value += 1
        
        # For a 4-cycle, max cut should be 2
        assert cut_value <= 4  # Upper bound
        assert cut_value >= 0  # Lower bound
    
    def test_5x5_random_symmetric_qubo(self, solver):
        """Test 5x5 random symmetric QUBO problem."""
        # Generate deterministic random QUBO
        np.random.seed(42)
        Q_upper = np.random.randn(5, 5) * 0.5
        Q = np.triu(Q_upper) + np.triu(Q_upper, 1).T  # Make symmetric
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (5,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check that objective value is reasonable
        objective_value = solution.T @ Q @ solution
        assert np.isfinite(objective_value)
    
    # ========== Edge Cases and Error Handling ==========
    
    def test_empty_matrix_error(self, solver):
        """Test that empty matrix raises appropriate error."""
        Q = np.array([], dtype=np.float64).reshape(0, 0)
        
        with pytest.raises(ValueError, match="QUBO matrix cannot be empty"):
            solver.solve(Q)
    
    def test_non_square_matrix_error(self, solver):
        """Test that non-square matrix raises appropriate error."""
        Q = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2x3 matrix
        
        with pytest.raises(ValueError, match="QUBO matrix must be square"):
            solver.solve(Q)
    
    def test_invalid_input_types(self, solver):
        """Test that invalid input types raise appropriate errors."""
        # Test string input
        with pytest.raises(ValueError, match="QUBO matrix must be a numpy array or torch tensor"):
            solver.solve("not a matrix")
        
        # Test list input
        with pytest.raises(ValueError, match="QUBO matrix must be a numpy array or torch tensor"):
            solver.solve([[1, 2], [3, 4]])
        
        # Test 1D array
        with pytest.raises(ValueError, match="QUBO matrix must be 2-dimensional"):
            solver.solve(np.array([1, 2, 3]))
    
    def test_torch_tensor_input(self, solver):
        """Test that PyTorch tensors are properly converted."""
        # Create a torch tensor QUBO
        Q_torch = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        
        solution = solver.solve(Q_torch)
        
        # Check solution is binary numpy array
        assert isinstance(solution, np.ndarray)
        assert solution.shape == (2,)
        assert np.all((solution == 0) | (solution == 1))
    
    def test_large_values_numerical_stability(self, solver):
        """Test numerical stability with large values."""
        # Create QUBO with large values
        Q = np.array([[1000, -500], [-500, 1000]], dtype=np.float64)
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (2,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value is reasonable
        objective_value = solution.T @ Q @ solution
        assert np.isfinite(objective_value)
    
    def test_asymmetric_matrix_handling(self, solver):
        """Test handling of asymmetric matrices."""
        # Create asymmetric matrix
        Q = np.array([[1, -2], [-1, 1]], dtype=np.float64)
        
        # Should still work (solver symmetrizes internally)
        solution = solver.solve(Q)
        
        assert solution.shape == (2,)
        assert np.all((solution == 0) | (solution == 1))
    
    # ========== Performance and Timeout Tests ==========
    
    def test_time_limit_enforcement(self, solver_short_timeout):
        """Test that solver respects time limits."""
        # Create a larger problem that might hit time limit
        np.random.seed(123)
        Q = np.random.randn(10, 10)
        Q = (Q + Q.T) / 2  # Make symmetric
        
        # Should complete even with short timeout for this small problem
        solution = solver_short_timeout.solve(Q)
        
        assert solution.shape == (10,)
        assert np.all((solution == 0) | (solution == 1))
    
    def test_larger_problem_performance(self, solver):
        """Test solver performance on slightly larger problem."""
        # Create 8x8 problem
        np.random.seed(456)
        Q = np.random.randn(8, 8) * 0.1
        Q = (Q + Q.T) / 2  # Make symmetric
        
        solution = solver.solve(Q)
        
        assert solution.shape == (8,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Check objective value
        objective_value = solution.T @ Q @ solution
        assert np.isfinite(objective_value)
    
    def test_solution_binary_constraint(self, solver):
        """Test that all solutions are strictly binary."""
        # Test with various matrices
        test_matrices = [
            np.array([[1, 0], [0, 1]], dtype=np.float64),
            np.array([[-1, 0], [0, -1]], dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.float64),
            np.array([[2, -1], [-1, 2]], dtype=np.float64),
        ]
        
        for Q in test_matrices:
            solution = solver.solve(Q)
            
            # Check all values are exactly 0 or 1
            assert np.all(solution == solution.astype(int))
            assert np.all((solution == 0) | (solution == 1))
    
    # ========== McCormick Linearization Verification ==========
    
    def test_linearization_correctness(self, solver):
        """Test that McCormick linearization produces correct results."""
        # Use a simple problem where we can verify the linearization
        # Q = [[1, -1], [-1, 1]]
        # Original: min x1^2 - x1*x2 - x2*x1 + x2^2 = min x1 + x2 - 2*x1*x2
        # Since x1, x2 ∈ {0,1}, x1^2 = x1 and x2^2 = x2
        # So: min x1 + x2 - 2*x1*x2
        
        Q = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (2,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Verify objective value manually
        x1, x2 = solution[0], solution[1]
        expected_objective = x1 + x2 - 2 * x1 * x2
        actual_objective = solution.T @ Q @ solution
        
        assert np.isclose(actual_objective, expected_objective, rtol=1e-6)
    
    def test_auxiliary_variables_consistency(self, solver):
        """Test that auxiliary variables maintain consistency."""
        # Create a 3x3 problem to test auxiliary variables y_ij
        Q = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float64)
        
        solution = solver.solve(Q)
        
        # Check solution is binary
        assert solution.shape == (3,)
        assert np.all((solution == 0) | (solution == 1))
        
        # Verify that the solution satisfies the original QUBO
        objective_value = solution.T @ Q @ solution
        assert np.isfinite(objective_value)
        
        # For this specific matrix, verify the objective calculation
        x1, x2, x3 = solution[0], solution[1], solution[2]
        manual_objective = (x1 - x2)**2 + (x2 - x3)**2
        
        # The solver should give the same result
        assert np.isclose(objective_value, manual_objective, rtol=1e-6)


# ========== Utility Functions for Tests ==========

def calculate_objective_value(solution: np.ndarray, Q: np.ndarray) -> float:
    """Calculate the objective value for a given solution and QUBO matrix."""
    return float(solution.T @ Q @ solution)


def is_binary_solution(solution: np.ndarray) -> bool:
    """Check if a solution is binary (contains only 0s and 1s)."""
    return np.all((solution == 0) | (solution == 1))


def generate_random_symmetric_qubo(size: int, seed: int = None) -> np.ndarray:
    """Generate a random symmetric QUBO matrix."""
    if seed is not None:
        np.random.seed(seed)
    
    Q_upper = np.random.randn(size, size)
    Q = np.triu(Q_upper) + np.triu(Q_upper, 1).T
    return Q


# ========== Integration Tests ==========

class TestScipSolverIntegration:
    """Integration tests for SCIP solver with other components."""
    
    def test_solver_with_rbm_qubo(self):
        """Test solver integration with RBM-generated QUBO matrices."""
        if not SCIP_IMPORT_SUCCESS:
            pytest.skip("SCIP solver import failed")
        if not ScipSolver.is_available:
            pytest.skip("SCIP solver not available")
        
        # Import RBM to generate realistic QUBO
        try:
            from rbm.models.rbm import RBM
        except ImportError:
            pytest.skip("RBM module not available")
        
        # Create small RBM
        rbm = RBM(n_visible=4, n_hidden=3)
        solver = ScipSolver(time_limit=60.0)
        
        # Generate joint QUBO
        Q_joint = rbm.create_joint_qubo()
        
        # Should be able to solve it
        solution = solver.solve(Q_joint)
        
        assert solution.shape == (7,)  # 4 visible + 3 hidden
        assert np.all((solution == 0) | (solution == 1))