# Modular EBM Training Framework

## Overview

This document describes the modular Energy-Based Model (EBM) training framework implemented for comparing different training methods, specifically Contrastive Divergence (CD) and Perturb-and-MAP (P&M). The framework provides a flexible, extensible architecture for implementing and comparing various EBM training algorithms.

## Architecture

The framework is built around the **Strategy Pattern**, allowing different training methods to be easily swapped and compared. The architecture consists of four main components:

### 1. Training Methods (`src/rbm/training/methods/`)
- **Abstract Base Class**: `EBMTrainingMethod` defines the interface for all training methods
- **Concrete Implementations**: 
  - `ContrastiveDivergenceTraining`: Implements CD with k-step Gibbs sampling
  - `PerturbMapTraining`: Implements P&M with QUBO optimization
- **Shared Functionality**: Positive phase and gradient computation are shared across methods

### 2. Samplers (`src/rbm/training/samplers/`)
- **Abstract Base Class**: `AbstractSampler` defines the interface for sampling strategies
- **Concrete Implementations**:
  - `GibbsSampler`: Implements Gibbs sampling for CD (supports CD-k and Persistent CD)
  - `QUBOSampler`: Implements QUBO optimization for P&M with retry logic
- **Extensible Design**: Easy to add new sampling strategies (e.g., Langevin sampling)

### 3. Training Manager (`src/rbm/training/training_manager.py`)
- **Orchestration**: High-level interface for training with different methods
- **Configuration-Driven**: Method selection and parameters controlled via YAML/JSON config
- **Monitoring**: Training progress tracking, checkpointing, and metrics collection

### 4. Comparison Framework (`src/rbm/training/comparison/`)
- **Systematic Comparison**: Framework for comparing multiple training methods
- **Metrics Collection**: Comprehensive metrics including convergence, efficiency, and solver performance
- **Visualization**: Automated generation of comparison plots and reports
- **Statistical Analysis**: Performance ranking and statistical significance testing

## Key Features

### Modular Design
- **Decoupled Components**: Each component can be developed and tested independently
- **Strategy Pattern**: Easy to add new training methods and sampling strategies
- **Configuration-Driven**: Method selection and parameters controlled via config files

### Comprehensive Comparison
- **Multi-Method Support**: Compare CD, P&M, and future methods in a single framework
- **Performance Metrics**: Training time, convergence rate, reconstruction error, energy gaps
- **Automated Reporting**: Generate detailed comparison reports with visualizations

### Extensibility
- **Easy to Add Methods**: New training methods only need to implement `negative_phase()`
- **Flexible Sampling**: New sampling strategies can be added by implementing `AbstractSampler`
- **Configurable Parameters**: All method-specific parameters controlled via configuration

### Testing and Validation
- **Comprehensive Test Suite**: Unit tests for all components with >90% coverage
- **Integration Testing**: End-to-end tests for method switching and comparison
- **Configuration Validation**: Automatic validation of configuration parameters

## Usage Examples

### Basic Training with CD
```python
from rbm.training import TrainingManager
from rbm.utils.config import load_config

# Load configuration
config = load_config('configs/contrastive_divergence.yaml')

# Create training manager
trainer = TrainingManager(config)

# Train the model
results = trainer.train(data_loader)
```

### Basic Training with P&M
```python
from rbm.training import TrainingManager
from rbm.utils.config import load_config

# Load configuration
config = load_config('configs/perturb_map.yaml')

# Create training manager
trainer = TrainingManager(config)

# Train the model
results = trainer.train(data_loader)
```

### Method Comparison
```python
from rbm.training.comparison import MethodComparison

# Initialize comparison framework
comparison = MethodComparison(base_config, output_dir="./results")

# Add method configurations
comparison.add_method_config('CD-1', cd_config)
comparison.add_method_config('P&M-Gurobi', pm_config)

# Run comparison
results = comparison.run_comparison(data_loader, runs_per_method=3)

# Print summary
comparison.print_summary()
```

## Configuration Structure

### Training Method Selection
```yaml
training:
  method: contrastive_divergence  # or 'perturb_map'
  epochs: 20
  learning_rate: 0.01
  batch_size: 64
  optimizer: sgd
```

### Contrastive Divergence Parameters
```yaml
cd_params:
  k_steps: 1              # Number of Gibbs steps
  persistent: false       # Use Persistent CD
  use_momentum: false     # Use momentum in sampling
  momentum: 0.5           # Momentum coefficient
  temperature: 1.0        # Sampling temperature
  seed: 42               # Random seed
```

### Perturb-and-MAP Parameters
```yaml
pm_params:
  gumbel_scale: 1.0       # Gumbel noise scale
  solver_timeout: 60.0    # QUBO solver timeout
  max_retries: 3          # Max solver retries
  seed: 42               # Random seed
```

## File Structure

```
src/rbm/training/
├── __init__.py                    # Main module exports
├── training_manager.py            # Training orchestration
├── methods/                       # Training methods
│   ├── __init__.py
│   ├── base.py                   # Abstract base class
│   ├── contrastive_divergence.py # CD implementation
│   └── perturb_map.py            # P&M implementation
├── samplers/                      # Sampling strategies
│   ├── __init__.py
│   ├── base.py                   # Abstract base class
│   ├── gibbs_sampler.py          # Gibbs sampling
│   └── qubo_sampler.py           # QUBO sampling
└── comparison/                    # Comparison framework
    ├── __init__.py
    ├── framework.py              # Main comparison logic
    ├── metrics.py                # Metrics and visualization
    └── example_comparison.py     # Usage example
```

## Configuration Files

Pre-configured examples are provided in `configs/`:
- `contrastive_divergence.yaml`: CD-1 configuration
- `persistent_cd.yaml`: PCD-3 with momentum
- `perturb_map.yaml`: P&M with Gurobi solver
- `scip.yaml`: SCIP solver configuration

## Testing

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Categories
```bash
python run_tests.py --pattern "test_training_methods"
python run_tests.py --pattern "test_samplers"
python run_tests.py --pattern "test_config"
```

### Run with Coverage
```bash
python run_tests.py --coverage
```

## Implementation Details

### Strategy Pattern Implementation
The framework uses the Strategy Pattern to separate the "what" (training algorithm) from the "how" (negative phase sampling). This allows:
- **Shared positive phase**: Both CD and P&M use the same positive phase computation
- **Interchangeable negative phase**: Different sampling strategies can be easily swapped
- **Extensible design**: New methods only need to implement the negative phase

### Negative Phase Abstraction
All training methods implement the same interface:
```python
def negative_phase(self, v_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate negative phase samples."""
    pass
```

### Sampling Strategy Abstraction
All samplers implement the same interface:
```python
def sample(self, v_init: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples from the model distribution."""
    pass
```

## Performance Considerations

### CD Implementation
- **Efficient Gibbs sampling**: Vectorized operations for batch processing
- **Persistent chains**: Optional persistent negative particles for faster convergence
- **Momentum support**: Momentum-based sampling for improved mixing

### P&M Implementation
- **QUBO optimization**: Efficient QUBO matrix construction and solving
- **Retry logic**: Automatic retry on solver failures with fallback to random sampling
- **Adaptive timeouts**: Dynamic solver timeout adjustment based on performance

### Comparison Framework
- **Parallel execution**: Multiple training runs can be executed in parallel
- **Efficient metrics**: Optimized metrics computation with incremental updates
- **Memory management**: Efficient handling of large training histories

## Extension Points

### Adding New Training Methods
1. Create new class inheriting from `EBMTrainingMethod`
2. Implement `negative_phase()` method
3. Add method-specific configuration validation
4. Register method in `TrainingManager._create_training_method()`

### Adding New Sampling Strategies
1. Create new class inheriting from `AbstractSampler`
2. Implement `sample()` method
3. Add sampler-specific configuration options
4. Use sampler in existing or new training methods

### Adding New Metrics
1. Extend `MetricsCalculator` with new metric computation
2. Add metric to `MethodMetrics` dataclass
3. Update visualization in `ComparisonVisualizer`
4. Add metric to comparison reports

## Best Practices

### Configuration Management
- Use YAML for human-readable configurations
- Validate all configurations before training
- Use default values for optional parameters
- Document all configuration options

### Testing
- Write unit tests for all new components
- Test edge cases and error conditions
- Use mocking for external dependencies
- Maintain high test coverage (>90%)

### Performance Optimization
- Profile code to identify bottlenecks
- Use vectorized operations where possible
- Implement efficient data structures
- Monitor memory usage during training

## Future Enhancements

### Planned Features
1. **Langevin Sampling**: Add Langevin dynamics as a sampling strategy
2. **Advanced Metrics**: Add more sophisticated convergence diagnostics
3. **Distributed Training**: Support for multi-GPU and distributed training
4. **Hyperparameter Optimization**: Automatic hyperparameter tuning
5. **More Solvers**: Support for additional QUBO solvers

### Experimental Features
1. **Variational Methods**: Integration with variational inference
2. **Neural Samplers**: Learning-based sampling strategies
3. **Adaptive Methods**: Methods that adapt parameters during training
4. **Ensemble Methods**: Combining multiple training methods

## Conclusion

The modular EBM training framework provides a flexible, extensible platform for implementing and comparing different EBM training methods. The clean architecture, comprehensive testing, and detailed documentation make it easy to use, extend, and maintain.

The framework successfully demonstrates that CD and P&M can be implemented within a unified architecture while maintaining their distinct characteristics and performance profiles. This foundation enables future research into hybrid methods, novel sampling strategies, and advanced training techniques.