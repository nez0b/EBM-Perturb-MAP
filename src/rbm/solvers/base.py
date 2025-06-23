"""
Base interface for QUBO solvers.

This module defines the abstract base class that all QUBO solvers must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import torch


class QUBOSolver(ABC):
    """
    Abstract base class for QUBO (Quadratic Unconstrained Binary Optimization) solvers.
    
    All QUBO solvers must implement the solve method to minimize x^T Q x
    where x is a binary vector and Q is the QUBO matrix.
    """
    
    @abstractmethod
    def solve(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Solve the QUBO problem: minimize x^T Q x.
        
        Args:
            Q: The QUBO matrix (square, symmetric).
            
        Returns:
            Binary solution vector.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
            RuntimeError: If the solver fails to find a solution.
        """
        pass
    
    def _validate_qubo_matrix(self, Q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Validate and convert QUBO matrix to numpy array.
        
        Args:
            Q: Input QUBO matrix.
            
        Returns:
            Validated numpy QUBO matrix.
            
        Raises:
            ValueError: If Q is not a valid QUBO matrix.
        """
        # Convert torch tensor to numpy if necessary
        if isinstance(Q, torch.Tensor):
            Q = Q.detach().cpu().numpy()
        
        if not isinstance(Q, np.ndarray):
            raise ValueError("QUBO matrix must be a numpy array or torch tensor")
        
        if Q.ndim != 2:
            raise ValueError("QUBO matrix must be 2-dimensional")
        
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("QUBO matrix must be square")
        
        if Q.shape[0] == 0:
            raise ValueError("QUBO matrix cannot be empty")
        
        return Q
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this solver."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this solver is available (dependencies installed, etc.)."""
        pass