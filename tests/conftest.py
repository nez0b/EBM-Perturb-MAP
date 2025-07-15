"""
Shared pytest fixtures and configuration for RBM tests.

This module provides common fixtures and utilities used across multiple test files.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Common test data for QUBO problems
SMALL_QUBO_PROBLEMS = {
    "2x2_simple": {
        "matrix": np.array([[1, -2], [-2, 1]], dtype=np.float64),
        "optimal_solution": np.array([1, 1]),
        "optimal_value": -2.0,
        "description": "Simple 2x2 QUBO with strong negative coupling"
    },
    "3x3_identity": {
        "matrix": np.eye(3, dtype=np.float64),
        "optimal_solution": np.array([0, 0, 0]),
        "optimal_value": 0.0,
        "description": "3x3 identity matrix - prefers all zeros"
    },
    "3x3_negative_diagonal": {
        "matrix": np.array([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], dtype=np.float64),
        "optimal_solution": np.array([1, 1, 1]),
        "optimal_value": -6.0,
        "description": "3x3 negative diagonal - prefers all ones"
    },
    "2x2_negative_coupling": {
        "matrix": np.array([[0, -1], [-1, 0]], dtype=np.float64),
        "optimal_solution": np.array([1, 1]),  # or [0, 0], both give same value
        "optimal_value": -2.0,
        "description": "2x2 with negative off-diagonal coupling"
    },
    "4x4_max_cut": {
        "matrix": np.array([[2, -1, -1, 0], [-1, 2, 0, -1], [-1, 0, 2, -1], [0, -1, -1, 2]], dtype=np.float64),
        "optimal_solution": None,  # Multiple optimal solutions
        "optimal_value": 0.0,  # Max cut value for 4-cycle
        "description": "4x4 Max Cut problem for cycle graph"
    }
}


@pytest.fixture
def small_qubo_problems():
    """Provide dictionary of small QUBO problems for testing."""
    return SMALL_QUBO_PROBLEMS


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def set_random_seed(random_seed):
    """Set random seed for numpy and torch."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return random_seed


@pytest.fixture
def symmetric_random_qubo():
    """Generate a symmetric random QUBO matrix."""
    def _generate(size: int, seed: int = 42, scale: float = 1.0):
        np.random.seed(seed)
        Q_upper = np.random.randn(size, size) * scale
        Q = np.triu(Q_upper) + np.triu(Q_upper, 1).T
        return Q
    return _generate


@pytest.fixture
def test_timeout():
    """Provide a reasonable timeout for tests."""
    return 30.0


@pytest.fixture
def short_timeout():
    """Provide a short timeout for timeout testing."""
    return 2.0


# Test utility functions as fixtures
@pytest.fixture
def qubo_utils():
    """Provide utility functions for QUBO testing."""
    class QUBOUtils:
        @staticmethod
        def calculate_objective(solution: np.ndarray, Q: np.ndarray) -> float:
            """Calculate objective value for a solution."""
            return float(solution.T @ Q @ solution)
        
        @staticmethod
        def is_binary(solution: np.ndarray) -> bool:
            """Check if solution is binary."""
            return np.all((solution == 0) | (solution == 1))
        
        @staticmethod
        def verify_solution(solution: np.ndarray, Q: np.ndarray) -> dict:
            """Verify a solution and return statistics."""
            return {
                "is_binary": QUBOUtils.is_binary(solution),
                "objective_value": QUBOUtils.calculate_objective(solution, Q),
                "solution_sum": np.sum(solution),
                "solution_norm": np.linalg.norm(solution)
            }
        
        @staticmethod
        def make_symmetric(Q: np.ndarray) -> np.ndarray:
            """Make a matrix symmetric."""
            return (Q + Q.T) / 2
        
        @staticmethod
        def enumerate_binary_solutions(n: int):
            """Enumerate all binary solutions for n variables."""
            for i in range(2**n):
                solution = np.zeros(n, dtype=int)
                for j in range(n):
                    solution[j] = (i >> j) & 1
                yield solution
        
        @staticmethod
        def find_optimal_solution(Q: np.ndarray):
            """Find optimal solution by enumeration (for small problems only)."""
            n = Q.shape[0]
            if n > 20:  # Too large for enumeration
                return None, None
            
            best_solution = None
            best_value = float('inf')
            
            for solution in QUBOUtils.enumerate_binary_solutions(n):
                value = QUBOUtils.calculate_objective(solution, Q)
                if value < best_value:
                    best_value = value
                    best_solution = solution.copy()
            
            return best_solution, best_value
    
    return QUBOUtils


# Parametrized fixtures for different solver configurations
@pytest.fixture(params=[10.0, 30.0, 60.0])
def solver_time_limit(request):
    """Parametrized time limits for solver testing."""
    return request.param


@pytest.fixture(params=[2, 3, 4, 5])
def matrix_size(request):
    """Parametrized matrix sizes for testing."""
    return request.param


# Mock fixtures for testing error conditions
@pytest.fixture
def invalid_qubo_matrices():
    """Provide invalid QUBO matrices for error testing."""
    return {
        "empty": np.array([], dtype=np.float64).reshape(0, 0),
        "non_square": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        "1d_array": np.array([1, 2, 3], dtype=np.float64),
        "string": "not a matrix",
        "list": [[1, 2], [3, 4]],
        "3d_array": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    }


# Performance test fixtures
@pytest.fixture
def large_qubo_matrix():
    """Generate a larger QUBO matrix for performance testing."""
    def _generate(size: int = 10, seed: int = 123):
        np.random.seed(seed)
        Q = np.random.randn(size, size) * 0.1
        return (Q + Q.T) / 2
    return _generate


# Integration test fixtures
@pytest.fixture
def mock_rbm_qubo():
    """Generate a QUBO matrix similar to what RBM would produce."""
    def _generate(n_visible: int = 4, n_hidden: int = 3, seed: int = 42):
        np.random.seed(seed)
        
        # Total variables
        n_total = n_visible + n_hidden
        Q = np.zeros((n_total, n_total))
        
        # Simulate RBM weight matrix
        W = np.random.randn(n_hidden, n_visible) * 0.1
        
        # Simulate bias terms (would be perturbed with Gumbel noise in real RBM)
        b = np.random.randn(n_visible) * 0.1
        c = np.random.randn(n_hidden) * 0.1
        
        # Fill QUBO matrix
        # Diagonal terms (biases)
        Q[np.arange(n_visible), np.arange(n_visible)] = -b
        Q[np.arange(n_visible, n_total), np.arange(n_visible, n_total)] = -c
        
        # Off-diagonal terms (weights)
        Q[:n_visible, n_visible:] = -W.T
        Q[n_visible:, :n_visible] = -W
        
        # Symmetrize
        Q = (Q + Q.T) / 2
        
        return Q
    
    return _generate


# Gurobi solver fixtures
@pytest.fixture
def gurobi_solver():
    """Provide a Gurobi solver instance if available."""
    try:
        from rbm.solvers.gurobi import GurobiSolver
        if GurobiSolver.is_available:
            return GurobiSolver(suppress_output=True, time_limit=30.0)
        else:
            pytest.skip("Gurobi solver not available")
    except ImportError:
        pytest.skip("Gurobi solver import failed")


@pytest.fixture
def gurobi_solver_verbose():
    """Provide a Gurobi solver with output enabled."""
    try:
        from rbm.solvers.gurobi import GurobiSolver
        if GurobiSolver.is_available:
            return GurobiSolver(suppress_output=False, time_limit=30.0)
        else:
            pytest.skip("Gurobi solver not available")
    except ImportError:
        pytest.skip("Gurobi solver import failed")


# SCIP solver fixtures
@pytest.fixture
def scip_solver():
    """Provide a SCIP solver instance if available."""
    try:
        from rbm.solvers.scip import ScipSolver
        if ScipSolver.is_available:
            return ScipSolver(time_limit=30.0)
        else:
            pytest.skip("SCIP solver not available")
    except ImportError:
        pytest.skip("SCIP solver import failed")


# Hexaly solver fixtures
@pytest.fixture
def hexaly_solver():
    """Provide a Hexaly solver instance if available."""
    try:
        from rbm.solvers.hexaly import HexalySolver
        if HexalySolver.is_available:
            return HexalySolver(time_limit=30.0, seed=42)
        else:
            pytest.skip("Hexaly solver not available")
    except ImportError:
        pytest.skip("Hexaly solver import failed")


@pytest.fixture
def hexaly_solver_multithreaded():
    """Provide a Hexaly solver with multiple threads."""
    try:
        from rbm.solvers.hexaly import HexalySolver
        if HexalySolver.is_available:
            return HexalySolver(time_limit=30.0, nb_threads=2, seed=42)
        else:
            pytest.skip("Hexaly solver not available")
    except ImportError:
        pytest.skip("Hexaly solver import failed")


# Solver comparison fixtures
@pytest.fixture
def both_solvers():
    """Provide both SCIP and Gurobi solvers if available."""
    solvers = {}
    
    # Try to get SCIP solver
    try:
        from rbm.solvers.scip import ScipSolver
        if ScipSolver.is_available:
            solvers["scip"] = ScipSolver(time_limit=30.0)
    except ImportError:
        pass
    
    # Try to get Gurobi solver
    try:
        from rbm.solvers.gurobi import GurobiSolver
        if GurobiSolver.is_available:
            solvers["gurobi"] = GurobiSolver(suppress_output=True, time_limit=30.0)
    except ImportError:
        pass
    
    if len(solvers) < 2:
        pytest.skip("Both SCIP and Gurobi solvers not available")
    
    return solvers


@pytest.fixture
def all_three_solvers():
    """Provide all three solvers (SCIP, Gurobi, Hexaly) if available."""
    solvers = {}
    
    # Try to get SCIP solver
    try:
        from rbm.solvers.scip import ScipSolver
        if ScipSolver.is_available:
            solvers["scip"] = ScipSolver(time_limit=30.0)
    except ImportError:
        pass
    
    # Try to get Gurobi solver
    try:
        from rbm.solvers.gurobi import GurobiSolver
        if GurobiSolver.is_available:
            solvers["gurobi"] = GurobiSolver(suppress_output=True, time_limit=30.0)
    except ImportError:
        pass
    
    # Try to get Hexaly solver
    try:
        from rbm.solvers.hexaly import HexalySolver
        if HexalySolver.is_available:
            solvers["hexaly"] = HexalySolver(time_limit=30.0, seed=42)
    except ImportError:
        pass
    
    if len(solvers) < 3:
        pytest.skip("All three solvers (SCIP, Gurobi, Hexaly) not available")
    
    return solvers


@pytest.fixture
def available_solvers():
    """Provide all available solvers."""
    solvers = {}
    
    # Try to get SCIP solver
    try:
        from rbm.solvers.scip import ScipSolver
        if ScipSolver.is_available:
            solvers["scip"] = ScipSolver(time_limit=30.0)
    except ImportError:
        pass
    
    # Try to get Gurobi solver
    try:
        from rbm.solvers.gurobi import GurobiSolver
        if GurobiSolver.is_available:
            solvers["gurobi"] = GurobiSolver(suppress_output=True, time_limit=30.0)
    except ImportError:
        pass
    
    # Try to get Hexaly solver
    try:
        from rbm.solvers.hexaly import HexalySolver
        if HexalySolver.is_available:
            solvers["hexaly"] = HexalySolver(time_limit=30.0, seed=42)
    except ImportError:
        pass
    
    # Try to get Dirac solver
    try:
        from rbm.solvers.dirac import DiracSolver
        if DiracSolver.is_available:
            solvers["dirac"] = DiracSolver(time_limit=30.0)
    except ImportError:
        pass
    
    if len(solvers) == 0:
        pytest.skip("No solvers available")
    
    return solvers


# Solver comparison utilities
@pytest.fixture
def solver_comparison_utils():
    """Provide utilities for comparing solver results."""
    class SolverComparisonUtils:
        @staticmethod
        def compare_solutions(sol1: np.ndarray, sol2: np.ndarray, Q: np.ndarray, 
                            tolerance: float = 1e-6) -> dict:
            """Compare two solutions on the same QUBO problem."""
            obj1 = float(sol1.T @ Q @ sol1)
            obj2 = float(sol2.T @ Q @ sol2)
            
            return {
                "solution1": sol1,
                "solution2": sol2,
                "objective1": obj1,
                "objective2": obj2,
                "objectives_equal": abs(obj1 - obj2) < tolerance,
                "solutions_equal": np.array_equal(sol1, sol2),
                "objective_difference": abs(obj1 - obj2),
                "both_binary": (np.all((sol1 == 0) | (sol1 == 1)) and 
                               np.all((sol2 == 0) | (sol2 == 1))),
                "both_same_shape": sol1.shape == sol2.shape
            }
        
        @staticmethod
        def validate_solver_solution(solver, Q: np.ndarray) -> dict:
            """Validate a solver's solution on a QUBO problem."""
            solution = solver.solve(Q)
            objective = float(solution.T @ Q @ solution)
            
            return {
                "solver_name": solver.name,
                "solution": solution,
                "objective": objective,
                "is_binary": np.all((solution == 0) | (solution == 1)),
                "shape": solution.shape,
                "dtype": solution.dtype,
                "solution_sum": int(np.sum(solution)),
                "is_finite": np.isfinite(objective)
            }
        
        @staticmethod
        def benchmark_solvers(solvers: dict, problems: list, num_runs: int = 3) -> dict:
            """Benchmark multiple solvers on multiple problems."""
            import time
            
            results = {}
            
            for solver_name, solver in solvers.items():
                results[solver_name] = {
                    "times": [],
                    "objectives": [],
                    "solutions": [],
                    "success_rate": 0
                }
                
                successful_runs = 0
                
                for problem in problems:
                    problem_times = []
                    problem_objectives = []
                    problem_solutions = []
                    
                    for run in range(num_runs):
                        try:
                            start_time = time.time()
                            solution = solver.solve(problem)
                            elapsed = time.time() - start_time
                            
                            if np.all((solution == 0) | (solution == 1)):
                                objective = float(solution.T @ problem @ solution)
                                problem_times.append(elapsed)
                                problem_objectives.append(objective)
                                problem_solutions.append(solution)
                                successful_runs += 1
                        except Exception:
                            pass
                    
                    if problem_times:
                        results[solver_name]["times"].append(np.mean(problem_times))
                        results[solver_name]["objectives"].append(np.mean(problem_objectives))
                        results[solver_name]["solutions"].append(problem_solutions[0])
                
                total_possible = len(problems) * num_runs
                results[solver_name]["success_rate"] = successful_runs / total_possible
            
            return results
        
        @staticmethod
        def validate_three_way_comparison(scip_sol: np.ndarray, gurobi_sol: np.ndarray, 
                                        hexaly_sol: np.ndarray, Q: np.ndarray) -> dict:
            """Validate and compare three-way solver results."""
            solutions = {"scip": scip_sol, "gurobi": gurobi_sol, "hexaly": hexaly_sol}
            objectives = {}
            
            for name, sol in solutions.items():
                objectives[name] = float(sol.T @ Q @ sol)
            
            # Check exact solver agreement (SCIP vs Gurobi)
            exact_agreement = abs(objectives["scip"] - objectives["gurobi"]) < 1e-6
            
            # Check heuristic solver quality (Hexaly vs exact solvers)
            hexaly_vs_scip = abs(objectives["hexaly"] - objectives["scip"])
            hexaly_vs_gurobi = abs(objectives["hexaly"] - objectives["gurobi"])
            
            return {
                "solutions": solutions,
                "objectives": objectives,
                "exact_solvers_agree": exact_agreement,
                "hexaly_quality_vs_scip": hexaly_vs_scip,
                "hexaly_quality_vs_gurobi": hexaly_vs_gurobi,
                "best_objective": min(objectives.values()),
                "best_solver": min(objectives.keys(), key=lambda k: objectives[k]),
                "objective_range": max(objectives.values()) - min(objectives.values()),
                "all_binary": all(np.all((sol == 0) | (sol == 1)) for sol in solutions.values())
            }
        
        @staticmethod
        def compare_solver_types(exact_solvers: dict, heuristic_solvers: dict, Q: np.ndarray) -> dict:
            """Compare exact vs heuristic solver types."""
            exact_results = {}
            heuristic_results = {}
            
            for name, solver in exact_solvers.items():
                sol = solver.solve(Q)
                exact_results[name] = {"solution": sol, "objective": float(sol.T @ Q @ sol)}
            
            for name, solver in heuristic_solvers.items():
                sol = solver.solve(Q)
                heuristic_results[name] = {"solution": sol, "objective": float(sol.T @ Q @ sol)}
            
            # Check exact solver consistency
            exact_objectives = [result["objective"] for result in exact_results.values()]
            exact_consistency = max(exact_objectives) - min(exact_objectives) < 1e-6
            
            # Check heuristic quality
            best_exact = min(exact_objectives) if exact_objectives else float('inf')
            heuristic_quality = {}
            for name, result in heuristic_results.items():
                heuristic_quality[name] = result["objective"] - best_exact
            
            return {
                "exact_results": exact_results,
                "heuristic_results": heuristic_results,
                "exact_consistency": exact_consistency,
                "heuristic_quality": heuristic_quality,
                "best_overall": min(exact_objectives + [r["objective"] for r in heuristic_results.values()]) if exact_objectives else min(r["objective"] for r in heuristic_results.values())
            }
    
    return SolverComparisonUtils


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "comparison: mark test as solver comparison test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.name or "rbm" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add comparison marker to solver comparison tests
        if "comparison" in item.name or "solver_comparison" in item.fspath.basename:
            item.add_marker(pytest.mark.comparison)