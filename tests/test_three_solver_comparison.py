"""
Comprehensive three-way comparison tests between SCIP, Gurobi, and Hexaly QUBO solvers.

This module contains tests that compare the performance, solution quality,
and robustness of all three solvers on the same QUBO problems.
"""

import pytest
import numpy as np
import torch
import time
from typing import Tuple, List, Dict, Any

# Import all three solvers
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

try:
    from rbm.solvers.hexaly import HexalySolver
    HEXALY_AVAILABLE = True
except ImportError:
    HEXALY_AVAILABLE = False
    HexalySolver = None

# Skip all tests if fewer than 2 solvers are available
ALL_SOLVERS_AVAILABLE = SCIP_AVAILABLE and GUROBI_AVAILABLE and HEXALY_AVAILABLE
ANY_TWO_SOLVERS_AVAILABLE = sum([SCIP_AVAILABLE, GUROBI_AVAILABLE, HEXALY_AVAILABLE]) >= 2


class TestThreeSolverComparison:
    """Test suite for comparing SCIP, Gurobi, and Hexaly QUBO solvers."""
    
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
    def hexaly_solver(self):
        """Create a Hexaly solver instance."""
        if not HEXALY_AVAILABLE:
            pytest.skip("Hexaly solver not available")
        
        try:
            import hexaly.optimizer as hexaly
        except ImportError:
            pytest.skip("Hexaly solver not available - install hexaly and check license")
        
        try:
            return HexalySolver(time_limit=30.0, seed=42)
        except ImportError:
            pytest.skip("Hexaly solver not available - install hexaly and check license")
    
    @pytest.fixture
    def all_solvers(self, scip_solver, gurobi_solver, hexaly_solver):
        """Create all three solver instances."""
        return {"scip": scip_solver, "gurobi": gurobi_solver, "hexaly": hexaly_solver}
    
    @pytest.fixture
    def available_solvers(self):
        """Create all available solver instances."""
        solvers = {}
        
        # Try to get SCIP solver
        if SCIP_AVAILABLE:
            try:
                solver = ScipSolver(time_limit=30.0)
                if solver.is_available:
                    solvers["scip"] = solver
            except ImportError:
                pass
        
        # Try to get Gurobi solver
        if GUROBI_AVAILABLE:
            try:
                import gurobipy as gp
                solvers["gurobi"] = GurobiSolver(suppress_output=True, time_limit=30.0)
            except ImportError:
                pass
        
        # Try to get Hexaly solver
        if HEXALY_AVAILABLE:
            try:
                import hexaly.optimizer as hexaly
                solvers["hexaly"] = HexalySolver(time_limit=30.0, seed=42)
            except ImportError:
                pass
        
        if len(solvers) < 2:
            pytest.skip("At least two solvers needed for comparison")
        
        return solvers
    
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
            "4x4_max_cut": {
                "matrix": self._generate_max_cut_qubo(4),
                "optimal_solution": None,  # Multiple optimal solutions
                "optimal_objective": None,
                "description": "4x4 Max Cut problem for cycle graph"
            },
            "5x5_random_symmetric": {
                "matrix": self._generate_random_symmetric_qubo(5, seed=42),
                "optimal_solution": None,  # Unknown, to be found
                "optimal_objective": None,
                "description": "5x5 random symmetric QUBO"
            }
        }
    
    def _generate_random_symmetric_qubo(self, size: int, seed: int = 42) -> np.ndarray:
        """Generate a random symmetric QUBO matrix."""
        np.random.seed(seed)
        Q_upper = np.random.randn(size, size) * 0.5
        Q = np.triu(Q_upper) + np.triu(Q_upper, 1).T
        return Q
    
    def _generate_max_cut_qubo(self, size: int) -> np.ndarray:
        """Generate QUBO formulation for Max Cut on a cycle graph."""
        # Create adjacency matrix for cycle graph
        A = np.zeros((size, size))
        for i in range(size):
            A[i, (i + 1) % size] = 1
            A[(i + 1) % size, i] = 1
        
        # QUBO formulation: Q = diag(degree) - A
        degrees = np.sum(A, axis=1)
        Q = np.diag(degrees) - A
        return Q
    
    # ========== Solution Quality Comparison Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_three_way_solution_quality(self, all_solvers, test_problems):
        """Test that all three solvers find comparable solution quality."""
        solvers = all_solvers
        
        # Test on problems with known optimal solutions
        for problem_name, problem_data in test_problems.items():
            if problem_data["optimal_solution"] is not None:
                Q = problem_data["matrix"]
                
                solutions = {}
                objectives = {}
                
                for solver_name, solver in solvers.items():
                    solutions[solver_name] = solver.solve(Q)
                    objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
                    
                    # All should be binary
                    assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
                
                # All should find reasonable objective values
                for solver_name, obj in objectives.items():
                    assert np.isfinite(obj)
                
                # Check that all objectives are close to each other
                # (allowing for heuristic nature of Hexaly)
                obj_values = list(objectives.values())
                max_diff = max(obj_values) - min(obj_values)
                assert max_diff < 2.0, f"Large objective difference in {problem_name}: {objectives}"
    
    @pytest.mark.skipif(not ANY_TWO_SOLVERS_AVAILABLE, reason="At least two solvers needed")
    def test_available_solvers_comparison(self, available_solvers):
        """Test comparison with whatever solvers are available."""
        solvers = available_solvers
        
        # Test on a simple problem
        Q = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        
        solutions = {}
        objectives = {}
        
        for solver_name, solver in solvers.items():
            solutions[solver_name] = solver.solve(Q)
            objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
            
            # All should be binary
            assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
            assert np.isfinite(objectives[solver_name])
        
        # If we have both exact solvers (SCIP and Gurobi), they should agree
        if "scip" in solvers and "gurobi" in solvers:
            assert np.isclose(objectives["scip"], objectives["gurobi"], rtol=1e-6)
    
    # ========== Solver Type Comparison Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_exact_vs_heuristic_comparison(self, all_solvers):
        """Compare exact solvers (SCIP, Gurobi) vs heuristic solver (Hexaly)."""
        solvers = all_solvers
        
        # Test on problems where exact solution is known
        test_cases = [
            np.array([[1, 0], [0, 1]], dtype=np.float64),  # Optimal: [0, 0]
            np.array([[-1, 0], [0, -1]], dtype=np.float64),  # Optimal: [1, 1]
            np.array([[0, -1], [-1, 0]], dtype=np.float64),  # Optimal: [0, 0] or [1, 1]
        ]
        
        for Q in test_cases:
            solutions = {}
            objectives = {}
            
            for solver_name, solver in solvers.items():
                solutions[solver_name] = solver.solve(Q)
                objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
            
            # Exact solvers should agree
            assert np.isclose(objectives["scip"], objectives["gurobi"], rtol=1e-6)
            
            # Heuristic solver should find reasonable solution
            # (may not be optimal but should be close)
            hexaly_diff_scip = abs(objectives["hexaly"] - objectives["scip"])
            hexaly_diff_gurobi = abs(objectives["hexaly"] - objectives["gurobi"])
            
            assert hexaly_diff_scip < 2.0, f"Hexaly too far from SCIP: {hexaly_diff_scip}"
            assert hexaly_diff_gurobi < 2.0, f"Hexaly too far from Gurobi: {hexaly_diff_gurobi}"
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_algorithm_approach_differences(self, all_solvers):
        """Test different algorithmic approaches: linearization vs quadratic vs local search."""
        solvers = all_solvers
        
        # Test on problems with different characteristics
        test_problems = [
            # Strongly quadratic problem
            np.array([[1, -2, 0], [-2, 1, -2], [0, -2, 1]], dtype=np.float64),
            # Sparse problem
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]], dtype=np.float64),
            # Dense interaction matrix
            np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64),
        ]
        
        for i, Q in enumerate(test_problems):
            solutions = {}
            objectives = {}
            
            for solver_name, solver in solvers.items():
                solutions[solver_name] = solver.solve(Q)
                objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
                
                # All should produce valid solutions
                assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
                assert np.isfinite(objectives[solver_name])
            
            # All approaches should find reasonable solutions
            obj_values = list(objectives.values())
            max_diff = max(obj_values) - min(obj_values)
            assert max_diff < 5.0, f"Large algorithm difference on problem {i}: {objectives}"
    
    # ========== Performance Comparison Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_three_way_performance_comparison(self, all_solvers):
        """Compare execution times between all three solvers."""
        solvers = all_solvers
        
        # Test on problems of increasing size
        problem_sizes = [3, 4, 5, 6]
        timing_results = {name: [] for name in solvers.keys()}
        
        for size in problem_sizes:
            Q = self._generate_random_symmetric_qubo(size, seed=size)
            
            for solver_name, solver in solvers.items():
                start_time = time.time()
                solution = solver.solve(Q)
                elapsed = time.time() - start_time
                timing_results[solver_name].append(elapsed)
                
                # All should produce valid solutions
                assert np.all((solution == 0) | (solution == 1))
        
        # Print timing results for inspection
        print(f"\\nThree-way timing comparison (problem sizes {problem_sizes}):")
        for solver_name, times in timing_results.items():
            print(f"{solver_name} times: {times}")
        
        # Basic sanity check: no solver should be extremely slow
        for solver_name, times in timing_results.items():
            for time_val in times:
                assert time_val < 30.0, f"{solver_name} took too long: {time_val}s"
    
    @pytest.mark.skipif(not ANY_TWO_SOLVERS_AVAILABLE, reason="At least two solvers needed")
    def test_available_solvers_performance(self, available_solvers):
        """Test performance with whatever solvers are available."""
        solvers = available_solvers
        
        # Test on a moderate-sized problem
        Q = self._generate_random_symmetric_qubo(5, seed=123)
        
        timing_results = {}
        solutions = {}
        
        for solver_name, solver in solvers.items():
            start_time = time.time()
            solutions[solver_name] = solver.solve(Q)
            timing_results[solver_name] = time.time() - start_time
            
            # All should produce valid solutions
            assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
        
        # Print results
        print(f"\\nAvailable solvers performance:")
        for solver_name, time_val in timing_results.items():
            obj_val = solutions[solver_name].T @ Q @ solutions[solver_name]
            print(f"{solver_name}: {time_val:.3f}s, objective: {obj_val:.3f}")
    
    # ========== Robustness Comparison Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_three_way_robustness(self, all_solvers):
        """Compare robustness on edge cases across all solvers."""
        solvers = all_solvers
        
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
            solutions = {}
            objectives = {}
            
            for solver_name, solver in solvers.items():
                solutions[solver_name] = solver.solve(Q)
                objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
                
                # All should handle edge cases gracefully
                assert solutions[solver_name].shape == (Q.shape[0],)
                assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
                assert np.isfinite(objectives[solver_name])
    
    @pytest.mark.skipif(not ANY_TWO_SOLVERS_AVAILABLE, reason="At least two solvers needed")
    def test_numerical_stability_comparison(self, available_solvers):
        """Compare numerical stability across available solvers."""
        solvers = available_solvers
        
        # Create problems with different numerical properties
        problem_types = [
            # Well-conditioned
            np.array([[2, 0], [0, 2]], dtype=np.float64),
            # Ill-conditioned
            np.array([[1000, 999], [999, 1000]], dtype=np.float64),
            # Nearly singular
            np.array([[1, 1], [1, 1.001]], dtype=np.float64),
        ]
        
        for i, Q in enumerate(problem_types):
            solutions = {}
            objectives = {}
            
            for solver_name, solver in solvers.items():
                solutions[solver_name] = solver.solve(Q)
                objectives[solver_name] = solutions[solver_name].T @ Q @ solutions[solver_name]
                
                # All should produce valid solutions
                assert np.all((solutions[solver_name] == 0) | (solutions[solver_name] == 1))
                assert np.isfinite(objectives[solver_name])
    
    # ========== Solution Validation Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_solution_validation_consistency(self, all_solvers):
        """Test that all solvers produce solutions that can be validated consistently."""
        solvers = all_solvers
        
        # Test on various problem types
        test_matrices = [
            np.array([[1, 0], [0, 1]], dtype=np.float64),
            np.array([[-1, 0], [0, -1]], dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.float64),
            np.array([[2, -1], [-1, 2]], dtype=np.float64),
            self._generate_random_symmetric_qubo(4, seed=789),
        ]
        
        for Q in test_matrices:
            solutions = {}
            validation_results = {}
            
            for solver_name, solver in solvers.items():
                solutions[solver_name] = solver.solve(Q)
                validation_results[solver_name] = self._validate_solution(solutions[solver_name], Q)
                
                # All validations should pass
                assert validation_results[solver_name]["is_binary"]
                assert validation_results[solver_name]["correct_shape"]
                assert validation_results[solver_name]["finite_objective"]
    
    def _validate_solution(self, solution: np.ndarray, Q: np.ndarray) -> dict:
        """Validate a solution and return validation results."""
        return {
            "is_binary": np.all((solution == 0) | (solution == 1)),
            "correct_shape": solution.shape == (Q.shape[0],),
            "finite_objective": np.isfinite(solution.T @ Q @ solution),
            "objective_value": float(solution.T @ Q @ solution),
            "solution_sum": int(np.sum(solution)),
            "solution_norm": float(np.linalg.norm(solution))
        }
    
    # ========== Comparative Analysis Tests ==========
    
    @pytest.mark.skipif(not ALL_SOLVERS_AVAILABLE, reason="All three solvers not available")
    def test_solver_characteristics_analysis(self, all_solvers):
        """Analyze and compare solver characteristics across multiple problems."""
        solvers = all_solvers
        
        # Test on a variety of problem types
        test_problems = [
            ("small_diagonal", np.array([[-1, 0], [0, -2]], dtype=np.float64)),
            ("coupling", np.array([[1, -1], [-1, 1]], dtype=np.float64)),
            ("3x3_mixed", np.array([[1, -1, 0], [-1, 1, -1], [0, -1, 1]], dtype=np.float64)),
            ("random_4x4", self._generate_random_symmetric_qubo(4, seed=456)),
        ]
        
        results = {solver_name: {"objectives": [], "times": [], "success_rate": 0} 
                  for solver_name in solvers.keys()}
        
        for problem_name, Q in test_problems:
            for solver_name, solver in solvers.items():
                try:
                    start_time = time.time()
                    solution = solver.solve(Q)
                    elapsed = time.time() - start_time
                    
                    if np.all((solution == 0) | (solution == 1)):
                        objective = solution.T @ Q @ solution
                        results[solver_name]["objectives"].append(objective)
                        results[solver_name]["times"].append(elapsed)
                        results[solver_name]["success_rate"] += 1
                except Exception:
                    pass
        
        # Calculate success rates
        total_problems = len(test_problems)
        for solver_name in results.keys():
            results[solver_name]["success_rate"] /= total_problems
        
        # All solvers should have high success rates
        for solver_name, result in results.items():
            assert result["success_rate"] >= 0.75, f"{solver_name} low success rate: {result['success_rate']}"
        
        # Print comparative analysis
        print(f"\\nSolver characteristics analysis:")
        for solver_name, result in results.items():
            if result["objectives"]:
                avg_obj = np.mean(result["objectives"])
                avg_time = np.mean(result["times"])
                print(f"{solver_name}: avg_obj={avg_obj:.3f}, avg_time={avg_time:.3f}s, success_rate={result['success_rate']:.2f}")


# ========== Utility Classes for Three-Way Testing ==========

class ThreeSolverBenchmark:
    """Utility class for benchmarking three solvers."""
    
    def __init__(self, solvers: Dict[str, Any]):
        self.solvers = solvers
        self.results = {}
    
    def run_benchmark(self, problems: List[np.ndarray], num_runs: int = 3):
        """Run benchmark on a list of problems."""
        self.results = {name: {"times": [], "objectives": [], "solutions": []} 
                       for name in self.solvers.keys()}
        
        for problem in problems:
            for solver_name, solver in self.solvers.items():
                problem_times = []
                problem_objectives = []
                problem_solutions = []
                
                for _ in range(num_runs):
                    try:
                        start_time = time.time()
                        solution = solver.solve(problem)
                        elapsed = time.time() - start_time
                        
                        if np.all((solution == 0) | (solution == 1)):
                            objective = solution.T @ problem @ solution
                            problem_times.append(elapsed)
                            problem_objectives.append(objective)
                            problem_solutions.append(solution)
                    except Exception:
                        pass
                
                if problem_times:
                    self.results[solver_name]["times"].append(np.mean(problem_times))
                    self.results[solver_name]["objectives"].append(np.mean(problem_objectives))
                    self.results[solver_name]["solutions"].append(problem_solutions[0])
        
        return self.results
    
    def get_performance_summary(self):
        """Get comparative performance summary."""
        summary = {}
        for solver_name, data in self.results.items():
            if data["times"]:
                summary[solver_name] = {
                    "mean_time": np.mean(data["times"]),
                    "mean_objective": np.mean(data["objectives"]),
                    "std_time": np.std(data["times"]),
                    "std_objective": np.std(data["objectives"]),
                    "success_rate": len(data["times"]) / len(self.results[list(self.results.keys())[0]]["times"]) if self.results else 0
                }
        return summary


class ThreeSolverValidator:
    """Utility class for validating and comparing three-way solutions."""
    
    @staticmethod
    def validate_three_solutions(sol1: np.ndarray, sol2: np.ndarray, sol3: np.ndarray, 
                                Q: np.ndarray, names: List[str]) -> Dict[str, Any]:
        """Validate and compare three solutions on the same problem."""
        solutions = [sol1, sol2, sol3]
        objectives = [sol.T @ Q @ sol for sol in solutions]
        
        return {
            "solutions": {names[i]: solutions[i] for i in range(3)},
            "objectives": {names[i]: objectives[i] for i in range(3)},
            "all_binary": all(np.all((sol == 0) | (sol == 1)) for sol in solutions),
            "all_same_shape": all(sol.shape == solutions[0].shape for sol in solutions),
            "objective_range": max(objectives) - min(objectives),
            "best_solver": names[np.argmin(objectives)],
            "objective_agreement": max(objectives) - min(objectives) < 1e-6,
            "pairwise_differences": {
                f"{names[i]}_vs_{names[j]}": abs(objectives[i] - objectives[j])
                for i in range(3) for j in range(i+1, 3)
            }
        }