"""
Comparison framework for EBM training methods.

This module provides a high-level framework for comparing different
EBM training methods (CD, P&M, etc.) in a systematic way.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import time
import copy
from pathlib import Path
import json

from ..training_manager import TrainingManager
from .metrics import ComparisonAnalyzer, ComparisonVisualizer, MethodMetrics


class MethodComparison:
    """
    Framework for comparing multiple EBM training methods.
    
    This class provides a systematic way to compare different training methods
    (CD, P&M, etc.) on the same dataset with consistent experimental setup.
    """
    
    def __init__(self, base_config: Dict[str, Any], output_dir: str = "./comparison_results"):
        """
        Initialize the comparison framework.
        
        Args:
            base_config: Base configuration that will be modified for each method
            output_dir: Directory to save comparison results
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer and visualizer
        self.analyzer = ComparisonAnalyzer()
        self.visualizer = ComparisonVisualizer(str(self.output_dir / "plots"))
        
        # Store results
        self.training_results = {}
        self.comparison_results = None
        
        print(f"Initialized MethodComparison with output directory: {self.output_dir}")
    
    def add_method_config(self, method_name: str, method_config: Dict[str, Any]) -> None:
        """
        Add a method configuration for comparison.
        
        Args:
            method_name: Name identifier for the method
            method_config: Method-specific configuration to merge with base config
        """
        # Merge method config with base config
        full_config = self._merge_configs(self.base_config, method_config)
        
        # Validate the configuration
        from ...utils.config import validate_config
        validated_config = validate_config(full_config)
        
        # Store the configuration
        self.method_configs = getattr(self, 'method_configs', {})
        self.method_configs[method_name] = validated_config
        
        print(f"Added method configuration: {method_name}")
    
    def run_comparison(self, data_loader: DataLoader, 
                      methods: Optional[List[str]] = None,
                      runs_per_method: int = 1,
                      save_checkpoints: bool = True) -> Dict[str, Any]:
        """
        Run the comparison experiment.
        
        Args:
            data_loader: DataLoader for training data
            methods: List of method names to compare (None for all configured methods)
            runs_per_method: Number of runs per method for statistical significance
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Comparison results dictionary
        """
        if not hasattr(self, 'method_configs'):
            raise ValueError("No method configurations added. Use add_method_config() first.")
        
        # Use all configured methods if none specified
        if methods is None:
            methods = list(self.method_configs.keys())
        
        print(f"Running comparison for methods: {methods}")
        print(f"Runs per method: {runs_per_method}")
        print("-" * 60)
        
        # Run training for each method
        for method_name in methods:
            if method_name not in self.method_configs:
                print(f"Warning: Method {method_name} not configured, skipping...")
                continue
            
            method_results = []
            
            # Run multiple times for statistical significance
            for run_idx in range(runs_per_method):
                print(f"\nRunning {method_name} - Run {run_idx + 1}/{runs_per_method}")
                
                # Create training manager
                config = copy.deepcopy(self.method_configs[method_name])
                
                # Update checkpoint path for this run
                if save_checkpoints:
                    checkpoint_path = self.output_dir / f"{method_name}_run_{run_idx + 1}.pth"
                    config['training']['checkpoint_path'] = str(checkpoint_path)
                
                try:
                    # Run training
                    training_manager = TrainingManager(config)
                    results = training_manager.train(data_loader)
                    
                    # Add run information
                    results['run_idx'] = run_idx
                    results['method_name'] = method_name
                    results['config'] = config
                    
                    method_results.append(results)
                    
                    print(f"  Run {run_idx + 1} completed: "
                          f"Final loss = {results['final_metrics'].get('reconstruction_error', 'N/A'):.4f}, "
                          f"Time = {results['total_time']:.2f}s")
                    
                except Exception as e:
                    print(f"  Run {run_idx + 1} failed: {e}")
                    continue
            
            # Store results for this method
            if method_results:
                self.training_results[method_name] = method_results
            else:
                print(f"Warning: No successful runs for method {method_name}")
        
        # Analyze results
        self._analyze_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save results
        self._save_results()
        
        print("\nComparison completed successfully!")
        return self.comparison_results
    
    def _merge_configs(self, base_config: Dict[str, Any], method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge method-specific config with base config."""
        merged = copy.deepcopy(base_config)
        
        for key, value in method_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _analyze_results(self) -> None:
        """Analyze comparison results."""
        if not self.training_results:
            print("No training results to analyze")
            return
        
        # For multiple runs, we need to aggregate results
        aggregated_results = {}
        
        for method_name, runs in self.training_results.items():
            if not runs:
                continue
            
            # For now, use the best run (lowest final loss)
            best_run = min(runs, key=lambda x: x['final_metrics'].get('reconstruction_error', float('inf')))
            aggregated_results[method_name] = best_run
        
        # Perform comparison analysis
        self.comparison_results = self.analyzer.compare_methods(aggregated_results)
        
        # Add multi-run statistics if available
        if any(len(runs) > 1 for runs in self.training_results.values()):
            self.comparison_results['multi_run_stats'] = self._calculate_multi_run_stats()
    
    def _calculate_multi_run_stats(self) -> Dict[str, Any]:
        """Calculate statistics across multiple runs."""
        stats = {}
        
        for method_name, runs in self.training_results.items():
            if len(runs) <= 1:
                continue
            
            # Extract metrics from all runs
            final_losses = [run['final_metrics'].get('reconstruction_error', float('inf')) 
                          for run in runs]
            training_times = [run['total_time'] for run in runs]
            
            stats[method_name] = {
                'num_runs': len(runs),
                'final_loss': {
                    'mean': float(np.mean(final_losses)),
                    'std': float(np.std(final_losses)),
                    'min': float(np.min(final_losses)),
                    'max': float(np.max(final_losses))
                },
                'training_time': {
                    'mean': float(np.mean(training_times)),
                    'std': float(np.std(training_times)),
                    'min': float(np.min(training_times)),
                    'max': float(np.max(training_times))
                }
            }
        
        return stats
    
    def _generate_visualizations(self) -> None:
        """Generate comparison visualizations."""
        if not self.comparison_results:
            return
        
        # Use best run for each method for visualization
        best_runs = {}
        for method_name, runs in self.training_results.items():
            if runs:
                best_run = min(runs, key=lambda x: x['final_metrics'].get('reconstruction_error', float('inf')))
                best_runs[method_name] = best_run
        
        # Generate plots
        self.visualizer.plot_training_curves(best_runs, 'reconstruction_error')
        self.visualizer.plot_training_curves(best_runs, 'positive_energy')
        self.visualizer.plot_training_curves(best_runs, 'negative_energy')
        
        self.visualizer.plot_method_comparison(self.comparison_results['method_metrics'])
        self.visualizer.plot_efficiency_ranking(self.comparison_results['performance_ranking'])
    
    def _save_results(self) -> None:
        """Save comparison results to files."""
        # Save detailed results
        results_file = self.output_dir / "comparison_results.json"
        
        # Convert to JSON-serializable format
        serializable_results = self._make_serializable(self.comparison_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary report
        self._save_summary_report()
        
        print(f"Results saved to: {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, MethodMetrics):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _save_summary_report(self) -> None:
        """Save a human-readable summary report."""
        report_file = self.output_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("EBM Training Methods Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance ranking
            f.write("Performance Ranking:\n")
            f.write("-" * 20 + "\n")
            for i, (method, score) in enumerate(self.comparison_results['performance_ranking']):
                f.write(f"{i+1}. {method}: {score:.3f}\n")
            f.write("\n")
            
            # Statistical analysis
            f.write("Statistical Analysis:\n")
            f.write("-" * 20 + "\n")
            stats = self.comparison_results['statistical_analysis']
            
            f.write(f"Fastest method: {stats['training_time']['fastest']}\n")
            f.write(f"Slowest method: {stats['training_time']['slowest']}\n")
            f.write(f"Speedup: {stats['training_time']['speedup']:.2f}x\n\n")
            
            f.write(f"Best quality method: {stats['final_loss']['best']}\n")
            f.write(f"Worst quality method: {stats['final_loss']['worst']}\n")
            f.write(f"Quality improvement: {stats['final_loss']['improvement']:.2f}x\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            for recommendation in self.comparison_results['recommendations']:
                f.write(f"• {recommendation}\n")
            
            # Method details
            f.write("\nDetailed Method Metrics:\n")
            f.write("-" * 30 + "\n")
            for method_name, metrics in self.comparison_results['method_metrics'].items():
                f.write(f"\n{method_name}:\n")
                f.write(f"  Training time: {metrics.training_time:.2f}s\n")
                f.write(f"  Final loss: {metrics.final_loss:.4f}\n")
                f.write(f"  Convergence epoch: {metrics.convergence_epoch}\n")
                f.write(f"  Energy gap: {metrics.energy_gap_mean:.4f} ± {metrics.energy_gap_std:.4f}\n")
                
                if metrics.solver_metrics:
                    f.write(f"  Solver success rate: {metrics.solver_metrics.get('pm_success_rate', {}).get('mean', 'N/A')}\n")
                    f.write(f"  Avg solve time: {metrics.solver_metrics.get('pm_avg_solve_time', {}).get('mean', 'N/A')}\n")
        
        print(f"Summary report saved to: {report_file}")
    
    def print_summary(self) -> None:
        """Print a summary of comparison results."""
        if not self.comparison_results:
            print("No comparison results available. Run comparison first.")
            return
        
        print("\n" + "=" * 50)
        print("COMPARISON RESULTS SUMMARY")
        print("=" * 50)
        
        # Performance ranking
        print("\nPerformance Ranking:")
        for i, (method, score) in enumerate(self.comparison_results['performance_ranking']):
            print(f"  {i+1}. {method}: {score:.3f}")
        
        # Recommendations
        print("\nRecommendations:")
        for recommendation in self.comparison_results['recommendations']:
            print(f"  • {recommendation}")
        
        print("\n" + "=" * 50)


# Import numpy for statistics
import numpy as np