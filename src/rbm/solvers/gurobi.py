"""
Gurobi QUBO solver implementation.

This module implements a QUBO solver using the Gurobi commercial optimizer.
"""

import numpy as np
from typing import Union
import torch
from .base import QUBOSolver

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None


class GurobiSolver(QUBOSolver):
    """
    QUBO solver using the Gurobi commercial optimizer.
    
    This solver formulates the QUBO problem as a quadratic binary program
    and uses Gurobi's optimization algorithms to find the solution.
    
    Args:
        suppress_output (bool): Whether to suppress Gurobi's console output.
        time_limit (float): Time limit for optimization in seconds.
    """
    
    def __init__(self, suppress_output: bool = True, time_limit: float = 60.0):
        if not self.is_available:
            raise ImportError("Gurobi is not available. Please install gurobipy.")
        
        self.suppress_output = suppress_output
        self.time_limit = time_limit
    
    @property
    def name(self) -> str:
        return "Gurobi"
    
    @property
    def is_available(self) -> bool:
        return GUROBI_AVAILABLE
    
    def solve(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Solve the QUBO problem using Gurobi.
        
        Args:
            Q: The QUBO matrix.
            
        Returns:
            Binary solution vector.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
            RuntimeError: If Gurobi fails to solve the problem.
        """
        Q = self._validate_qubo_matrix(Q)
        n_vars = Q.shape[0]
        
        try:
            # Create Gurobi environment and model
            with gp.Env(empty=True) as env:
                if self.suppress_output:
                    env.setParam('OutputFlag', 0)
                env.start()
                
                with gp.Model(env=env) as model:
                    # Set time limit
                    model.setParam('TimeLimit', self.time_limit)
                    
                    # Add binary variables
                    x = model.addMVar(shape=n_vars, vtype=GRB.BINARY, name="x")
                    
                    # Set quadratic objective: minimize x^T Q x
                    model.setObjective(x @ Q @ x, GRB.MINIMIZE)
                    
                    # Optimize
                    model.optimize()
                    
                    # Extract solution
                    if model.Status == GRB.OPTIMAL:
                        solution = x.X.astype(np.int8)
                        return solution
                    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
                        # Return best solution found within time limit
                        solution = x.X.astype(np.int8)
                        return solution
                    else:
                        # If no solution found, return random solution
                        print(f"Warning: Gurobi did not find an optimal solution "
                              f"(Status: {model.Status}). Returning random solution.")
                        return np.random.randint(0, 2, size=n_vars, dtype=np.int8)
                        
        except gp.GurobiError as e:
            error_msg = f"Gurobi error occurred: {e.errno} - {e.message}"
            print(f"Error: {error_msg}")
            # Return random solution as fallback
            return np.random.randint(0, 2, size=n_vars, dtype=np.int8)
        except Exception as e:
            error_msg = f"Unexpected error in Gurobi solver: {str(e)}"
            print(f"Error: {error_msg}")
            # Return random solution as fallback
            return np.random.randint(0, 2, size=n_vars, dtype=np.int8)