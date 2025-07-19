"""
EBM training module.

This module provides different training methods for Energy-Based Models,
including Perturb-and-MAP, Contrastive Divergence, and related utilities.
"""

from .trainer import Trainer
from .training_manager import TrainingManager
from .methods import EBMTrainingMethod, ContrastiveDivergenceTraining, PerturbMapTraining
from .samplers import AbstractSampler, GibbsSampler, QUBOSampler
from .comparison import MethodComparison, MetricsCalculator, ComparisonAnalyzer, ComparisonVisualizer

__all__ = [
    'Trainer',  # Original trainer for backwards compatibility
    'TrainingManager',
    'EBMTrainingMethod',
    'ContrastiveDivergenceTraining', 
    'PerturbMapTraining',
    'AbstractSampler',
    'GibbsSampler',
    'QUBOSampler',
    'MethodComparison',
    'MetricsCalculator',
    'ComparisonAnalyzer',
    'ComparisonVisualizer'
]