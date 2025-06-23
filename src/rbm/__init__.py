"""
RBM Package - Restricted Boltzmann Machines with Perturb-and-MAP training.

This package implements RBM training using the Perturb-and-MAP methodology
with various QUBO solvers for the MAP optimization step.
"""

from .models.rbm import RBM
from .models.hybrid import CNN_RBM

__version__ = "0.1.0"
__all__ = ["RBM", "CNN_RBM"]