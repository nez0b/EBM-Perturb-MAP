"""
EBM training methods module.

This module provides different training methods for Energy-Based Models,
including Perturb-and-MAP, Contrastive Divergence, and future extensions.
"""

from .base import EBMTrainingMethod
from .contrastive_divergence import ContrastiveDivergenceTraining
from .perturb_map import PerturbMapTraining

__all__ = ['EBMTrainingMethod', 'ContrastiveDivergenceTraining', 'PerturbMapTraining']