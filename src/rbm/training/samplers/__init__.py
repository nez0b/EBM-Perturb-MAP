"""
Sampling strategies module.

This module provides different sampling strategies for EBM training,
including QUBO sampling, Gibbs sampling, and future extensions.
"""

from .base import AbstractSampler
from .gibbs_sampler import GibbsSampler
from .qubo_sampler import QUBOSampler

__all__ = ['AbstractSampler', 'GibbsSampler', 'QUBOSampler']