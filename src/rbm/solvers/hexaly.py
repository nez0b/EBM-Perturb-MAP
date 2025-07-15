"""
Hexaly QUBO solver implementation.

This module implements a QUBO solver using the Hexaly heuristic optimizer.
"""

import numpy as np
from typing import Union
import torch
from .base import QUBOSolver

try:
    import hexaly.optimizer as hexaly
    HEXALY_AVAILABLE = True
except ImportError:
    HEXALY_AVAILABLE = False
    hexaly = None


class HexalySolver(QUBOSolver):
    """
    QUBO solver using the Hexaly heuristic optimizer.
    
    This solver formulates the QUBO problem as a binary optimization problem
    and uses Hexaly's local search algorithms to find high-quality solutions.
    Note that Hexaly is a heuristic solver and does not guarantee optimal solutions.
    
    Args:
        time_limit (float): Time limit for optimization in seconds.
        nb_threads (int): Number of threads to use for optimization.
        seed (int): Random seed for reproducibility.
    """
    
    def __init__(self, time_limit: float = 60.0, nb_threads: int = 4, seed: int = 42):
        if not self.is_available:
            raise ImportError("Hexaly is not available. Please install hexaly and check license.")
        
        self.time_limit = time_limit
        self.nb_threads = nb_threads
        self.seed = seed
    
    @property
    def name(self) -> str:
        return "Hexaly"
    
    @property
    def is_available(self) -> bool:
        if not HEXALY_AVAILABLE:
            return False
        
        # Test if Hexaly license is available
        try:
            with hexaly.HexalyOptimizer() as optimizer:
                return True
        except Exception:
            return False
    
    def solve(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Solve the QUBO problem using Hexaly.
        
        Args:
            Q: The QUBO matrix.
            
        Returns:
            Binary solution vector.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
            RuntimeError: If Hexaly fails to solve the problem.
        """
        Q = self._validate_qubo_matrix(Q)
        n_vars = Q.shape[0]
        
        try:
            with hexaly.HexalyOptimizer() as optimizer:
                # Set parameters
                optimizer.get_param().set_time_limit(int(self.time_limit))
                optimizer.get_param().set_nb_threads(self.nb_threads)
                optimizer.get_param().set_seed(self.seed)
                
                # Create model
                model = optimizer.get_model()
                
                # Create binary variables
                x = [model.bool() for _ in range(n_vars)]
                
                # Build quadratic objective: minimize x^T Q x
                objective = model.sum()
                
                # Add diagonal terms
                for i in range(n_vars):
                    if Q[i, i] != 0:
                        objective += float(Q[i, i]) * x[i]
                
                # Add off-diagonal terms (both upper and lower triangular)
                for i in range(n_vars):
                    for j in range(i + 1, n_vars):
                        if Q[i, j] != 0:
                            # Since QUBO is x^T Q x, we need Q[i,j] + Q[j,i] coefficient
                            coeff = float(Q[i, j] + Q[j, i])
                            if coeff != 0:
                                objective += coeff * x[i] * x[j]
                
                # Set objective to minimize
                model.minimize(objective)
                
                # Close model
                model.close()
                
                # Solve
                optimizer.solve()
                
                # Extract solution
                solution = np.zeros(n_vars, dtype=np.int8)
                for i in range(n_vars):
                    solution[i] = int(x[i].get_value())
                
                return solution
                
        except Exception as e:
            error_msg = f"Hexaly solver failed to solve the problem: {str(e)}"
            raise RuntimeError(error_msg) from e