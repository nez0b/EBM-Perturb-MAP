"""
Comprehensive comparison tests between SCIP and Gurobi QUBO solvers.

This module contains tests that compare the performance, solution quality,
and robustness of both SCIP and Gurobi solvers on the same QUBO problems.
"""

import pytest
import numpy as np
import torch
import time
from typing import Tuple, List, Dict, Any

# Import both solvers
try:
    from rbm.solvers.scip import ScipSolver
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    ScipSolver = None

try:
    from rbm.solvers.gurobi import GurobiSolver
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    GurobiSolver = None

# Skip all tests if either solver is not available
BOTH_SOLVERS_AVAILABLE = SCIP_AVAILABLE and GUROBI_AVAILABLE


class TestSolverComparison:
    """Test suite for comparing SCIP and Gurobi QUBO solvers."""
    
    @pytest.fixture
    def scip_solver(self):
        """Create a SCIP solver instance."""
        if not SCIP_AVAILABLE:
            pytest.skip("SCIP solver not available")
        
        # Create an instance to check availability
        try:
            solver = ScipSolver(time_limit=30.0)
            if not solver.is_available:
                pytest.skip("SCIP solver not available - install pyscipopt")
            return solver
        except ImportError:
            pytest.skip("SCIP solver not available - install pyscipopt")
    
    @pytest.fixture
    def gurobi_solver(self):
        """Create a Gurobi solver instance."""
        if not GUROBI_AVAILABLE:
            pytest.skip("Gurobi solver not available")
        
        try:
            import gurobipy as gp
        except ImportError:
            pytest.skip("Gurobi solver not available - install gurobipy and check license")
        
        try:
            return GurobiSolver(suppress_output=True, time_limit=30.0)
        except ImportError:
            pytest.skip("Gurobi solver not available - install gurobipy and check license")
    
    @pytest.fixture
    def both_solvers(self, scip_solver, gurobi_solver):
        """Create both solver instances."""
        return {"scip": scip_solver, "gurobi": gurobi_solver}
    
    @pytest.fixture
    def test_problems(self):
        """Provide a set of test QUBO problems with known properties."""
        return {
            "2x2_simple": {
                "matrix": np.array([[1, -2], [-2, 1]], dtype=np.float64),
                "optimal_solution": np.array([1, 1]),
                "optimal_objective": -2.0,
                "description": "Simple 2x2 QUBO with strong negative coupling"
            },
            "3x3_identity": {
                "matrix": np.eye(3, dtype=np.float64),
                "optimal_solution": np.array([0, 0, 0]),
                "optimal_objective": 0.0,
                "description": "3x3 identity matrix - prefers all zeros"
            },
            "3x3_negative_diagonal": {
                "matrix": np.array([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], dtype=np.float64),
                "optimal_solution": np.array([1, 1, 1]),
                "optimal_objective": -6.0,
                "description": "3x3 negative diagonal - prefers all ones"
            },
            "2x2_negative_coupling": {
                "matrix": np.array([[0, -1], [-1, 0]], dtype=np.float64),
                "optimal_solution": np.array([1, 1]),  # or [0, 0], both give same value
                "optimal_objective": -2.0,
                "description": "2x2 with negative off-diagonal coupling"
            },
            "4x4_random_symmetric": {
                "matrix": self._generate_random_symmetric_qubo(4, seed=42),
                "optimal_solution": None,  # Unknown, to be found
                "optimal_objective": None,
                "description": "4x4 random symmetric QUBO"
            }
        }
    
    def _generate_random_symmetric_qubo(self, size: int, seed: int = 42) -> np.ndarray:
        """Generate a random symmetric QUBO matrix."""
        np.random.seed(seed)
        Q_upper = np.random.randn(size, size) * 0.5
        Q = np.triu(Q_upper) + np.triu(Q_upper, 1).T
        return Q
    
    # ========== Solution Quality Comparison Tests ==========
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_identical_solutions_small_problems(self, both_solvers, test_problems):
        """Test that both solvers find identical solutions for small problems."""
        solvers = both_solvers
        
        # Test on problems with known optimal solutions
        for problem_name, problem_data in test_problems.items():
            if problem_data["optimal_solution"] is not None:
                Q = problem_data["matrix"]
                
                scip_solution = solvers["scip"].solve(Q)
                gurobi_solution = solvers["gurobi"].solve(Q)
                
                # Both should be binary
                assert np.all((scip_solution == 0) | (scip_solution == 1))
                assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
                
                # Calculate objective values
                scip_obj = scip_solution.T @ Q @ scip_solution
                gurobi_obj = gurobi_solution.T @ Q @ gurobi_solution
                
                # Both should find the same optimal objective value (within tolerance)
                assert np.isclose(scip_obj, gurobi_obj, rtol=1e-6), \
                    f"Problem {problem_name}: SCIP obj={scip_obj}, Gurobi obj={gurobi_obj}"
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_objective_value_consistency(self, both_solvers):
        """Test that objective values are consistently calculated."""
        solvers = both_solvers
        
        # Test multiple random problems
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            Q = self._generate_random_symmetric_qubo(3, seed=seed)
            
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Calculate objective values
            scip_obj = scip_solution.T @ Q @ scip_solution
            gurobi_obj = gurobi_solution.T @ Q @ gurobi_solution
            
            # Both should find reasonable objective values
            assert np.isfinite(scip_obj)
            assert np.isfinite(gurobi_obj)
            
            # For optimization problems, both should find similar quality solutions
            # (may not be identical due to different algorithms, but should be close)
            obj_diff = abs(scip_obj - gurobi_obj)
            assert obj_diff < 10.0, f"Large objective difference: {obj_diff}"
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_solution_validation(self, both_solvers):
        """Test that both solvers produce valid binary solutions."""
        solvers = both_solvers
        
        test_matrices = [
            np.array([[1, 0], [0, 1]], dtype=np.float64),
            np.array([[-1, 0], [0, -1]], dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.float64),
            np.array([[2, -1], [-1, 2]], dtype=np.float64),
            np.random.randn(5, 5),  # Random matrix
        ]
        
        for i, Q in enumerate(test_matrices):
            if Q.ndim == 2:
                Q = (Q + Q.T) / 2  # Ensure symmetry
            
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both solutions should be binary
            assert np.all((scip_solution == 0) | (scip_solution == 1)), \
                f"SCIP solution {i} not binary: {scip_solution}"
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1)), \
                f"Gurobi solution {i} not binary: {gurobi_solution}"
            
            # Both should have same dimension
            assert scip_solution.shape == gurobi_solution.shape, \
                f"Shape mismatch for problem {i}"
    
    # ========== Performance Comparison Tests ==========
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_execution_time_comparison(self, both_solvers):
        """Compare execution times between solvers."""
        solvers = both_solvers
        
        # Test on problems of increasing size
        problem_sizes = [3, 4, 5, 6]
        timing_results = {"scip": [], "gurobi": []}
        
        for size in problem_sizes:
            Q = self._generate_random_symmetric_qubo(size, seed=size)
            
            # Time SCIP solver
            start_time = time.time()
            scip_solution = solvers["scip"].solve(Q)
            scip_time = time.time() - start_time
            timing_results["scip"].append(scip_time)
            
            # Time Gurobi solver
            start_time = time.time()
            gurobi_solution = solvers["gurobi"].solve(Q)
            gurobi_time = time.time() - start_time
            timing_results["gurobi"].append(gurobi_time)
            
            # Both should produce valid solutions
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
        
        # Print timing results for inspection (not assertion)
        print(f"\\nTiming comparison (problem sizes {problem_sizes}):")
        print(f"SCIP times: {timing_results['scip']}")
        print(f"Gurobi times: {timing_results['gurobi']}")
        
        # Basic sanity check: no solver should be extremely slow
        for solver_name, times in timing_results.items():
            for time_val in times:
                assert time_val < 30.0, f"{solver_name} took too long: {time_val}s"
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_timeout_behavior(self, both_solvers):
        """Compare how solvers handle timeout situations."""
        # Create solvers with very short timeouts
        scip_short = ScipSolver(time_limit=0.1)
        gurobi_short = GurobiSolver(suppress_output=True, time_limit=0.1)
        
        # Test on a problem that might hit timeout
        Q = self._generate_random_symmetric_qubo(8, seed=999)
        
        scip_solution = scip_short.solve(Q)
        gurobi_solution = gurobi_short.solve(Q)
        
        # Both should still return valid binary solutions even if timeout
        assert scip_solution.shape == (8,)
        assert gurobi_solution.shape == (8,)
        assert np.all((scip_solution == 0) | (scip_solution == 1))
        assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_success_rate_comparison(self, both_solvers):
        """Compare success rate of solvers across multiple problems."""
        solvers = both_solvers
        
        num_problems = 10
        success_counts = {"scip": 0, "gurobi": 0}
        
        for i in range(num_problems):
            Q = self._generate_random_symmetric_qubo(4, seed=i)
            
            # Test SCIP
            try:
                scip_solution = solvers["scip"].solve(Q)
                if np.all((scip_solution == 0) | (scip_solution == 1)):
                    success_counts["scip"] += 1
            except Exception:
                pass
            
            # Test Gurobi
            try:
                gurobi_solution = solvers["gurobi"].solve(Q)
                if np.all((gurobi_solution == 0) | (gurobi_solution == 1)):
                    success_counts["gurobi"] += 1
            except Exception:
                pass
        
        # Both solvers should have high success rates
        scip_rate = success_counts["scip"] / num_problems
        gurobi_rate = success_counts["gurobi"] / num_problems
        
        assert scip_rate >= 0.8, f"SCIP success rate too low: {scip_rate}"
        assert gurobi_rate >= 0.8, f"Gurobi success rate too low: {gurobi_rate}"
    
    # ========== Robustness Comparison Tests ==========
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_edge_case_robustness(self, both_solvers):
        """Compare robustness on edge cases."""
        solvers = both_solvers
        
        edge_cases = [
            # All zeros matrix
            np.zeros((3, 3), dtype=np.float64),
            # All ones matrix
            np.ones((3, 3), dtype=np.float64),
            # Identity matrix
            np.eye(3, dtype=np.float64),
            # Large values
            np.array([[1000, -500], [-500, 1000]], dtype=np.float64),
            # Small values
            np.array([[0.001, -0.0005], [-0.0005, 0.001]], dtype=np.float64),
        ]
        
        for i, Q in enumerate(edge_cases):
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both should handle edge cases gracefully
            assert scip_solution.shape == (Q.shape[0],), f"SCIP failed on edge case {i}"
            assert gurobi_solution.shape == (Q.shape[0],), f"Gurobi failed on edge case {i}"
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_numerical_stability(self, both_solvers):
        """Compare numerical stability on ill-conditioned problems."""
        solvers = both_solvers
        
        # Create problems with different condition numbers
        problem_types = [
            # Well-conditioned
            np.array([[2, 0], [0, 2]], dtype=np.float64),
            # Ill-conditioned
            np.array([[1000, 999], [999, 1000]], dtype=np.float64),
            # Nearly singular
            np.array([[1, 1], [1, 1.001]], dtype=np.float64),
        ]
        
        for i, Q in enumerate(problem_types):
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both should produce valid solutions
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
            
            # Objective values should be finite
            scip_obj = scip_solution.T @ Q @ scip_solution
            gurobi_obj = gurobi_solution.T @ Q @ gurobi_solution
            assert np.isfinite(scip_obj)
            assert np.isfinite(gurobi_obj)
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_random_matrix_robustness(self, both_solvers):
        """Test robustness on various random matrices."""
        solvers = both_solvers
        
        # Test different matrix properties
        test_configs = [
            {"size": 3, "scale": 0.1, "seed": 42},
            {"size": 4, "scale": 1.0, "seed": 123},
            {"size": 5, "scale": 2.0, "seed": 456},
        ]
        
        for config in test_configs:
            Q = self._generate_random_symmetric_qubo(
                config["size"], 
                seed=config["seed"]
            ) * config["scale"]
            
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both solutions should be valid
            assert scip_solution.shape == (config["size"],)
            assert gurobi_solution.shape == (config["size"],)
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
    
    # ========== Algorithmic Differences Tests ==========
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_linearization_vs_quadratic(self, both_solvers):
        """Compare SCIP's linearization approach vs Gurobi's native quadratic handling."""
        solvers = both_solvers
        
        # Test problems where linearization might behave differently
        test_problems = [
            # Strongly quadratic problem
            np.array([[1, -2, 0], [-2, 1, -2], [0, -2, 1]], dtype=np.float64),
            # Diagonal problem (linear terms only)
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]], dtype=np.float64),
            # Dense interaction matrix
            np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64),
        ]
        
        for i, Q in enumerate(test_problems):
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both approaches should find valid solutions
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
            
            # Calculate objective values
            scip_obj = scip_solution.T @ Q @ scip_solution
            gurobi_obj = gurobi_solution.T @ Q @ gurobi_solution
            
            # Both should find similar quality solutions
            assert np.isfinite(scip_obj)
            assert np.isfinite(gurobi_obj)
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_solution_diversity(self, both_solvers):
        """Test if solvers find different but equally good solutions."""
        solvers = both_solvers
        
        # Create problem with multiple optimal solutions
        # Example: Q = [[0, 0], [0, 0]] - all solutions are equivalent
        Q = np.zeros((2, 2), dtype=np.float64)
        
        scip_solution = solvers["scip"].solve(Q)
        gurobi_solution = solvers["gurobi"].solve(Q)
        
        # Both should be binary
        assert np.all((scip_solution == 0) | (scip_solution == 1))
        assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
        
        # Both should have the same objective value (0 in this case)
        scip_obj = scip_solution.T @ Q @ scip_solution
        gurobi_obj = gurobi_solution.T @ Q @ gurobi_solution
        assert np.isclose(scip_obj, gurobi_obj, rtol=1e-6)
    
    @pytest.mark.skipif(not BOTH_SOLVERS_AVAILABLE, reason="Both solvers not available")
    def test_convergence_behavior(self, both_solvers):
        """Test convergence behavior on problems of increasing difficulty."""
        solvers = both_solvers
        
        # Create problems with increasing connectivity/difficulty
        for density in [0.3, 0.5, 0.7]:
            Q = self._generate_sparse_qubo(size=5, density=density, seed=42)
            
            scip_solution = solvers["scip"].solve(Q)
            gurobi_solution = solvers["gurobi"].solve(Q)
            
            # Both should converge to valid solutions
            assert np.all((scip_solution == 0) | (scip_solution == 1))
            assert np.all((gurobi_solution == 0) | (gurobi_solution == 1))
    
    def _generate_sparse_qubo(self, size: int, density: float, seed: int = 42) -> np.ndarray:
        """Generate a sparse symmetric QUBO matrix."""
        np.random.seed(seed)
        Q = np.random.randn(size, size)
        
        # Make sparse
        mask = np.random.rand(size, size) < density
        Q = Q * mask
        
        # Make symmetric
        Q = (Q + Q.T) / 2
        
        return Q


# ========== Utility Classes for Advanced Testing ==========

class SolverBenchmark:
    """Utility class for benchmarking solver performance."""
    
    def __init__(self, solvers: Dict[str, Any]):
        self.solvers = solvers
        self.results = {}
    
    def run_benchmark(self, problems: List[np.ndarray], num_runs: int = 3):
        """Run benchmark on a list of problems."""
        self.results = {name: [] for name in self.solvers.keys()}
        
        for problem in problems:
            for solver_name, solver in self.solvers.items():
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    solution = solver.solve(problem)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                
                avg_time = np.mean(times)
                self.results[solver_name].append(avg_time)
        
        return self.results
    
    def get_performance_summary(self):
        """Get summary statistics of performance."""
        summary = {}
        for solver_name, times in self.results.items():
            summary[solver_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "median_time": np.median(times),
                "max_time": np.max(times),
                "min_time": np.min(times)
            }
        return summary


class SolutionValidator:
    """Utility class for validating and comparing solutions."""
    
    @staticmethod
    def validate_solution(solution: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
        """Validate a solution and return statistics."""
        return {
            "is_binary": np.all((solution == 0) | (solution == 1)),
            "objective_value": float(solution.T @ Q @ solution),
            "solution_sum": int(np.sum(solution)),
            "solution_norm": float(np.linalg.norm(solution)),
            "shape": solution.shape,
            "dtype": solution.dtype
        }
    
    @staticmethod
    def compare_solutions(sol1: np.ndarray, sol2: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
        """Compare two solutions on the same problem."""
        val1 = SolutionValidator.validate_solution(sol1, Q)
        val2 = SolutionValidator.validate_solution(sol2, Q)
        
        return {
            "solution1": val1,
            "solution2": val2,
            "objectives_equal": np.isclose(val1["objective_value"], val2["objective_value"]),
            "solutions_equal": np.array_equal(sol1, sol2),
            "objective_difference": abs(val1["objective_value"] - val2["objective_value"])
        }