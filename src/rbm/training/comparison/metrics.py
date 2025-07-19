"""
Metrics for comparing different EBM training methods.

This module provides metrics and evaluation functions for comparing
the performance of different EBM training methods like CD and P&M.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path
import json


@dataclass
class MethodMetrics:
    """Metrics for a single training method."""
    
    method_name: str
    hyperparameters: Dict[str, Any]
    training_time: float
    final_loss: float
    convergence_epoch: Optional[int]
    energy_gap_mean: float
    energy_gap_std: float
    reconstruction_error: float
    solver_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'hyperparameters': self.hyperparameters,
            'training_time': self.training_time,
            'final_loss': self.final_loss,
            'convergence_epoch': self.convergence_epoch,
            'energy_gap_mean': self.energy_gap_mean,
            'energy_gap_std': self.energy_gap_std,
            'reconstruction_error': self.reconstruction_error,
            'solver_metrics': self.solver_metrics
        }


class MetricsCalculator:
    """Calculator for EBM training metrics."""
    
    def __init__(self, convergence_threshold: float = 1e-4, 
                 convergence_patience: int = 5):
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
    
    def calculate_method_metrics(self, training_results: Dict[str, Any]) -> MethodMetrics:
        """
        Calculate comprehensive metrics for a training method.
        
        Args:
            training_results: Results from TrainingManager.train()
            
        Returns:
            MethodMetrics object with calculated metrics.
        """
        history = training_results['history']
        method_name = training_results['method']
        hyperparameters = training_results['hyperparameters']
        
        # Basic metrics
        training_time = training_results['total_time']
        final_loss = history[-1]['reconstruction_error'] if history else float('inf')
        
        # Convergence analysis
        convergence_epoch = self._find_convergence_epoch(history)
        
        # Energy gap analysis
        energy_gaps = []
        for epoch_metrics in history:
            if 'positive_energy' in epoch_metrics and 'negative_energy' in epoch_metrics:
                gap = epoch_metrics['positive_energy'] - epoch_metrics['negative_energy']
                energy_gaps.append(gap)
        
        energy_gap_mean = np.mean(energy_gaps) if energy_gaps else 0.0
        energy_gap_std = np.std(energy_gaps) if energy_gaps else 0.0
        
        # Reconstruction error
        reconstruction_error = final_loss
        
        # Solver-specific metrics (for P&M)
        solver_metrics = None
        if 'pm_' in method_name.lower():
            solver_metrics = self._extract_solver_metrics(history)
        
        return MethodMetrics(
            method_name=method_name,
            hyperparameters=hyperparameters,
            training_time=training_time,
            final_loss=final_loss,
            convergence_epoch=convergence_epoch,
            energy_gap_mean=energy_gap_mean,
            energy_gap_std=energy_gap_std,
            reconstruction_error=reconstruction_error,
            solver_metrics=solver_metrics
        )
    
    def _find_convergence_epoch(self, history: List[Dict[str, Any]]) -> Optional[int]:
        """Find the epoch where training converged."""
        if len(history) < self.convergence_patience:
            return None
        
        losses = [epoch['reconstruction_error'] for epoch in history]
        
        for i in range(self.convergence_patience, len(losses)):
            recent_losses = losses[i-self.convergence_patience:i]
            if np.std(recent_losses) < self.convergence_threshold:
                return i - self.convergence_patience + 1
        
        return None
    
    def _extract_solver_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract solver-specific metrics from training history."""
        solver_metrics = {}
        
        # Extract P&M solver metrics
        pm_keys = ['pm_avg_solve_time', 'pm_success_rate', 'pm_total_solves']
        for key in pm_keys:
            values = [epoch.get(key, 0) for epoch in history if key in epoch]
            if values:
                solver_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'final': values[-1]
                }
        
        return solver_metrics
    
    def calculate_efficiency_score(self, metrics: MethodMetrics) -> float:
        """
        Calculate an efficiency score combining quality and speed.
        
        Args:
            metrics: MethodMetrics object
            
        Returns:
            Efficiency score (higher is better)
        """
        # Quality component (inverse of reconstruction error)
        quality_score = 1.0 / (1.0 + metrics.reconstruction_error)
        
        # Speed component (inverse of training time)
        speed_score = 1.0 / (1.0 + metrics.training_time)
        
        # Convergence bonus
        convergence_bonus = 1.2 if metrics.convergence_epoch is not None else 1.0
        
        # Combined score
        efficiency_score = (quality_score * speed_score * convergence_bonus)
        
        return efficiency_score


class ComparisonAnalyzer:
    """Analyzer for comparing multiple EBM training methods."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
    
    def compare_methods(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple training methods.
        
        Args:
            training_results: Dictionary with method names as keys and 
                            TrainingManager results as values
            
        Returns:
            Comparison results dictionary
        """
        method_metrics = {}
        
        # Calculate metrics for each method
        for method_name, results in training_results.items():
            method_metrics[method_name] = self.metrics_calculator.calculate_method_metrics(results)
        
        # Perform comparison analysis
        comparison_results = {
            'method_metrics': method_metrics,
            'performance_ranking': self._rank_methods(method_metrics),
            'statistical_analysis': self._statistical_analysis(method_metrics),
            'recommendations': self._generate_recommendations(method_metrics)
        }
        
        return comparison_results
    
    def _rank_methods(self, method_metrics: Dict[str, MethodMetrics]) -> List[Tuple[str, float]]:
        """Rank methods by efficiency score."""
        scores = []
        for method_name, metrics in method_metrics.items():
            score = self.metrics_calculator.calculate_efficiency_score(metrics)
            scores.append((method_name, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _statistical_analysis(self, method_metrics: Dict[str, MethodMetrics]) -> Dict[str, Any]:
        """Perform statistical analysis of method performance."""
        analysis = {}
        
        # Training time comparison
        training_times = {name: metrics.training_time for name, metrics in method_metrics.items()}
        fastest_method = min(training_times, key=training_times.get)
        slowest_method = max(training_times, key=training_times.get)
        
        analysis['training_time'] = {
            'fastest': fastest_method,
            'slowest': slowest_method,
            'speedup': training_times[slowest_method] / training_times[fastest_method]
        }
        
        # Final loss comparison
        final_losses = {name: metrics.final_loss for name, metrics in method_metrics.items()}
        best_method = min(final_losses, key=final_losses.get)
        worst_method = max(final_losses, key=final_losses.get)
        
        analysis['final_loss'] = {
            'best': best_method,
            'worst': worst_method,
            'improvement': final_losses[worst_method] / final_losses[best_method]
        }
        
        # Convergence analysis
        converged_methods = [name for name, metrics in method_metrics.items() 
                           if metrics.convergence_epoch is not None]
        
        analysis['convergence'] = {
            'converged_methods': converged_methods,
            'convergence_rate': len(converged_methods) / len(method_metrics)
        }
        
        return analysis
    
    def _generate_recommendations(self, method_metrics: Dict[str, MethodMetrics]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Best overall method
        ranking = self._rank_methods(method_metrics)
        if ranking:
            best_method = ranking[0][0]
            recommendations.append(f"Best overall method: {best_method}")
        
        # Speed vs quality trade-offs
        training_times = {name: metrics.training_time for name, metrics in method_metrics.items()}
        final_losses = {name: metrics.final_loss for name, metrics in method_metrics.items()}
        
        fastest_method = min(training_times, key=training_times.get)
        best_quality_method = min(final_losses, key=final_losses.get)
        
        if fastest_method != best_quality_method:
            recommendations.append(f"For speed: use {fastest_method}")
            recommendations.append(f"For quality: use {best_quality_method}")
        
        # Convergence recommendations
        converged_methods = [name for name, metrics in method_metrics.items() 
                           if metrics.convergence_epoch is not None]
        
        if converged_methods:
            fastest_convergence = min(converged_methods, 
                                    key=lambda x: method_metrics[x].convergence_epoch)
            recommendations.append(f"Fastest convergence: {fastest_convergence}")
        
        return recommendations


class ComparisonVisualizer:
    """Visualizer for comparison results."""
    
    def __init__(self, output_dir: str = "./comparison_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        else:
            plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_training_curves(self, training_results: Dict[str, Dict[str, Any]], 
                           metric: str = 'reconstruction_error') -> None:
        """Plot training curves for multiple methods."""
        plt.figure(figsize=(12, 8))
        
        for method_name, results in training_results.items():
            history = results['history']
            epochs = [epoch['epoch'] for epoch in history]
            values = [epoch.get(metric, 0) for epoch in history]
            
            plt.plot(epochs, values, label=method_name, marker='o', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Training Progress: {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_{metric}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_method_comparison(self, method_metrics: Dict[str, MethodMetrics]) -> None:
        """Plot comparison of different methods."""
        methods = list(method_metrics.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training time comparison
        training_times = [method_metrics[m].training_time for m in methods]
        axes[0, 0].bar(methods, training_times, color='skyblue')
        axes[0, 0].set_title('Training Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Final loss comparison
        final_losses = [method_metrics[m].final_loss for m in methods]
        axes[0, 1].bar(methods, final_losses, color='lightcoral')
        axes[0, 1].set_title('Final Loss')
        axes[0, 1].set_ylabel('Reconstruction Error')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Energy gap comparison
        energy_gaps = [method_metrics[m].energy_gap_mean for m in methods]
        axes[1, 0].bar(methods, energy_gaps, color='lightgreen')
        axes[1, 0].set_title('Energy Gap (Mean)')
        axes[1, 0].set_ylabel('Energy Gap')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Convergence epoch comparison
        convergence_epochs = [method_metrics[m].convergence_epoch or 0 for m in methods]
        axes[1, 1].bar(methods, convergence_epochs, color='gold')
        axes[1, 1].set_title('Convergence Epoch')
        axes[1, 1].set_ylabel('Epoch')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_ranking(self, ranking: List[Tuple[str, float]]) -> None:
        """Plot efficiency ranking of methods."""
        methods = [item[0] for item in ranking]
        scores = [item[1] for item in ranking]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, scores, color=['gold', 'silver', 'brown'][:len(methods)])
        
        plt.title('Method Efficiency Ranking')
        plt.ylabel('Efficiency Score')
        plt.xlabel('Training Method')
        plt.xticks(rotation=45)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_ranking.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, comparison_results: Dict[str, Any], 
                       output_path: str = "comparison_report.json") -> None:
        """Generate a detailed comparison report."""
        
        # Convert MethodMetrics to dict for serialization
        serializable_results = {}
        for key, value in comparison_results.items():
            if key == 'method_metrics':
                serializable_results[key] = {
                    name: metrics.to_dict() for name, metrics in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Save report
        with open(self.output_dir / output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comparison report saved to {self.output_dir / output_path}")