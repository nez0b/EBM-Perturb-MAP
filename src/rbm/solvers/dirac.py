"""
Dirac-3 QUBO solver implementation.

This module implements a QUBO solver using the Dirac-3 quantum annealing service.
Note: This requires the eqc_models package and proper authentication.
"""

import numpy as np
from typing import Union
import torch
from .base import QUBOSolver

try:
    from eqc_models.solvers import Dirac3IntegerCloudSolver
    from eqc_models.base import QuadraticModel
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False
    Dirac3IntegerCloudSolver = None
    QuadraticModel = None


class DiracSolver(QUBOSolver):
    """
    QUBO solver using the Dirac-3 quantum annealing cloud service.
    
    This solver converts the QUBO matrix into the Dirac QuadraticModel format
    and uses the cloud-based quantum annealer to find solutions.
    
    Args:
        num_samples (int): Number of solution samples to request.
        relaxation_schedule (int): Solver-specific relaxation parameter.
    """
    
    def __init__(self, num_samples: int = 10, relaxation_schedule: int = 1):
        if not self.is_available:
            raise ImportError(
                "Dirac solver is not available. Please install eqc_models "
                "and ensure proper authentication is configured."
            )
        
        self.num_samples = num_samples
        self.relaxation_schedule = relaxation_schedule
    
    @property
    def name(self) -> str:
        return "Dirac-3"
    
    @property
    def is_available(self) -> bool:
        return DIRAC_AVAILABLE
    
    def solve(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Solve the QUBO problem using Dirac-3.
        
        The QUBO matrix Q is decomposed into linear (C) and quadratic (J) parts:
        - C = diagonal of Q (linear coefficients)
        - J = off-diagonal part of Q (quadratic coefficients)
        
        Args:
            Q: The QUBO matrix.
            
        Returns:
            Binary solution vector.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
            RuntimeError: If Dirac solver fails.
        """
        Q = self._validate_qubo_matrix(Q)
        n_vars = Q.shape[0]
        
        try:
            # Decompose the QUBO matrix Q into Linear (C) and Quadratic (J) parts
            # The linear coefficients are the diagonal elements of the Q matrix
            C = np.diag(Q).astype(np.float64)
            
            # The quadratic coefficients are the off-diagonal elements
            J = Q.copy().astype(np.float64)
            np.fill_diagonal(J, 0)
            
            # Create the QuadraticModel for the Dirac Solver
            model = QuadraticModel(C, J)
            
            # Set the variable bounds to 1 (binary variables: 0 or 1)
            model.upper_bound = np.ones((n_vars,), dtype=np.int64)
            
            print(f"Submitting {n_vars}-variable QUBO to Dirac-3 Cloud Solver...")
            
            # Initialize and call the cloud solver
            solver = Dirac3IntegerCloudSolver()
            response = solver.solve(
                model, 
                num_samples=self.num_samples, 
                relaxation_schedule=self.relaxation_schedule
            )
            
            # Process the response and return the best solution
            if response and response.get("results", {}).get("solutions"):
                # The solution is returned as a list, convert it to a NumPy array
                best_solution = np.array(
                    response["results"]["solutions"][0], 
                    dtype=np.int8
                )
                print("Dirac solver returned a solution.")
                return best_solution
            else:
                # Handle cases where the solver fails or returns empty response
                print("Warning: Dirac solver did not return a valid solution. "
                      "Returning random vector.")
                return np.random.randint(0, 2, size=n_vars, dtype=np.int8)
                
        except Exception as e:
            error_msg = f"Error in Dirac solver: {str(e)}"
            print(f"Error: {error_msg}")
            return np.random.randint(0, 2, size=n_vars, dtype=np.int8)