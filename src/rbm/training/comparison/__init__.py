"""
Training method comparison framework.

This module provides tools for comparing different EBM training methods,
including metrics collection, benchmarking, and visualization.
"""

from .framework import MethodComparison
from .metrics import MetricsCalculator, ComparisonAnalyzer, ComparisonVisualizer, MethodMetrics

__all__ = [
    'MethodComparison',
    'MetricsCalculator', 
    'ComparisonAnalyzer',
    'ComparisonVisualizer',
    'MethodMetrics'
]