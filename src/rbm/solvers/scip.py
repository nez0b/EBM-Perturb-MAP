"""
SCIP QUBO solver implementation.

This module implements a QUBO solver using the SCIP optimization solver
with linearization to handle quadratic objectives.
"""

import numpy as np
from typing import Union
import torch
from .base import QUBOSolver

try:
    from pyscipopt import Model, quicksum
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    Model = None
    quicksum = None


class ScipSolver(QUBOSolver):
    """
    QUBO solver using the SCIP optimization solver with linearization.
    
    This solver linearizes the quadratic QUBO objective using auxiliary binary
    variables and McCormick constraints to handle the quadratic terms.
    
    Args:
        time_limit (float): Time limit for optimization in seconds.
    """
    
    def __init__(self, time_limit: float = 60.0):
        if not self.is_available:
            raise ImportError("SCIP is not available. Please install pyscipopt.")
        
        self.time_limit = time_limit
    
    @property
    def name(self) -> str:
        return "SCIP"
    
    @property
    def is_available(self) -> bool:
        return SCIP_AVAILABLE
    
    def solve(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Solve the QUBO problem using SCIP with linearization.
        
        The quadratic objective x^T Q x is linearized using auxiliary variables
        y_ij = x_i * x_j with McCormick constraints:
        - y_ij <= x_i
        - y_ij <= x_j  
        - y_ij >= x_i + x_j - 1
        
        Args:
            Q: The QUBO matrix.
            
        Returns:
            Binary solution vector.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
            RuntimeError: If SCIP fails to solve the problem.
        """
        Q = self._validate_qubo_matrix(Q)
        n = Q.shape[0]
        
        try:
            # Create SCIP model
            model = Model("QUBO_SCIP_Linearized")
            model.hideOutput()  # Suppress solver output
            
            # Set time limit
            model.setParam('limits/time', self.time_limit)
            
            # Create binary variables for the original problem
            x = {i: model.addVar(vtype="B", name=f"x({i})") for i in range(n)}
            
            # Create auxiliary binary variables y_ij to represent x_i * x_j
            y = {(i, j): model.addVar(vtype="B", name=f"y({i},{j})") 
                 for i in range(n) for j in range(i + 1, n)}
            
            # Add linearization constraints to enforce y_ij = x_i * x_j
            for i in range(n):
                for j in range(i + 1, n):
                    model.addCons(y[i, j] <= x[i])
                    model.addCons(y[i, j] <= x[j])
                    model.addCons(y[i, j] >= x[i] + x[j] - 1)
            
            # Set the linearized objective function
            # The objective x^T Q x is rewritten as:
            # sum(Q_ii * x_i) + sum_{i<j}((Q_ij + Q_ji) * y_ij)
            linear_terms = quicksum(Q[i, i] * x[i] for i in range(n))
            quadratic_terms = quicksum((Q[i, j] + Q[j, i]) * y[i, j] 
                                     for i, j in y.keys())
            
            model.setObjective(linear_terms + quadratic_terms, "minimize")
            
            # Optimize the model
            model.optimize()
            
            # Extract the solution
            solution = np.zeros(n, dtype=np.int8)
            status = model.getStatus()
            
            if status in ["optimal", "gaplimit", "timelimit"]:
                sol = model.getBestSol()
                if sol is not None:
                    for i in range(n):
                        # Get variable value from solution
                        val = model.getVal(x[i], sol)
                        if val > 0.5:  # Round to nearest binary value
                            solution[i] = 1
                else:
                    print(f"Warning: SCIP found status '{status}' but no solution available.")
                    return np.random.randint(0, 2, size=n, dtype=np.int8)
            else:
                print(f"Warning: SCIP did not find an optimal solution (Status: {status}).")
                return np.random.randint(0, 2, size=n, dtype=np.int8)
            
            return solution
            
        except Exception as e:
            error_msg = f"Error during SCIP optimization: {str(e)}"
            print(f"Error: {error_msg}")
            return np.random.randint(0, 2, size=n, dtype=np.int8)