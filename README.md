# RBM with Perturb-and-MAP

A Python package for training Restricted Boltzmann Machines (RBMs) using the Perturb-and-MAP methodology with various QUBO solvers.

## Overview

This project implements RBM training using the "Perturb-and-MAP" approach, which uses Gumbel's trick to perturb energy functionals and then solves MAP optimization problems to obtain unbiased Boltzmann-distributed samples. The MAP optimization is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solved using various solvers including Gurobi, SCIP, and Dirac-3.

## Features

- **Multiple QUBO Solvers**: Support for Gurobi, SCIP, and Dirac-3 quantum annealing
- **Modular Architecture**: Clean separation of models, solvers, training, and inference
- **Configuration Management**: YAML-based experiment configuration
- **Comprehensive Inference**: Reconstruction, generation, and denoising capabilities
- **Visualization Tools**: Built-in plotting for results analysis
- **Command Line Interface**: Easy-to-use scripts for training and inference

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd RBM

# Install in development mode
pip install -e .
```

### Solver Dependencies

The package supports multiple QUBO solvers as optional dependencies:

```bash
# For Gurobi solver (requires license)
pip install -e .[gurobi]

# For SCIP solver
pip install -e .[scip]

# For Dirac-3 quantum annealer
pip install -e .[dirac]

# Install all solvers
pip install -e .[all]

# Development dependencies
pip install -e .[dev]
```

## Quick Start

### Using the Command Line Interface

1. **Training an RBM**:
```bash
python experiments/train_rbm.py --config mnist_digit6 --epochs 20
```

2. **Running Inference**:
```bash
python experiments/run_inference.py checkpoint.pth --config mnist_digit6 --task both
```

### Using the Python API

```python
import torch
from rbm.models.rbm import RBM
from rbm.solvers.gurobi import GurobiSolver
from rbm.training.trainer import Trainer
from rbm.utils.config import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load("default")

# Create model and solver
model = RBM(n_visible=784, n_hidden=64)
solver = GurobiSolver()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
trainer = Trainer(model, solver, optimizer, config)
results = trainer.train(data_loader)
```

## Project Structure

```
RBM/
├── src/rbm/                    # Main package
│   ├── models/                 # Model implementations
│   │   ├── rbm.py             # Core RBM model
│   │   └── hybrid.py          # CNN-RBM hybrid models
│   ├── solvers/               # QUBO solver implementations
│   │   ├── base.py            # Abstract solver interface
│   │   ├── gurobi.py          # Gurobi solver
│   │   ├── scip.py            # SCIP solver
│   │   └── dirac.py           # Dirac-3 solver
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Main training orchestration
│   │   └── metrics.py         # Training metrics and logging
│   ├── inference/             # Inference utilities
│   │   ├── reconstruction.py  # Image reconstruction
│   │   └── generation.py      # Sample generation
│   ├── data/                  # Data loading utilities
│   │   └── mnist.py           # MNIST data handling
│   └── utils/                 # Utility modules
│       ├── config.py          # Configuration management
│       └── visualization.py   # Plotting functions
├── experiments/               # Training and inference scripts
│   ├── train_rbm.py          # Training script
│   └── run_inference.py      # Inference script
├── configs/                   # Configuration files
│   ├── default.yaml          # Default configuration
│   └── mnist_digit6.yaml     # Digit 6 experiment config
├── notebooks/                 # Demo notebooks
│   └── demo_rbm_package.ipynb # Package usage demo
└── tests/                     # Unit tests (to be added)
```

## Configuration

Experiments are configured using YAML files in the `configs/` directory. Here's an example configuration:

```yaml
model:
  n_visible: 784
  n_hidden: 64
  model_type: rbm

training:
  epochs: 10
  learning_rate: 0.01
  batch_size: 64
  checkpoint_every: 5

data:
  dataset: mnist
  digit_filter: null  # null for all digits, integer for specific digit
  image_size: [28, 28]

solver:
  name: gurobi
  time_limit: 60.0
  suppress_output: true

inference:
  gibbs_steps: 1000
  num_generated_samples: 10
```

## QUBO Solvers

### Gurobi (Commercial)
- **Installation**: Requires Gurobi license and `gurobipy` package
- **Performance**: Excellent for small to medium problems
- **Usage**: Default choice for most experiments

### SCIP (Open Source)
- **Installation**: `pip install pyscipopt`
- **Performance**: Good open-source alternative
- **Usage**: Uses linearization to handle quadratic objectives

### Dirac-3 (Quantum Annealing)
- **Installation**: Requires `eqc-models` package and cloud authentication
- **Performance**: Experimental quantum annealing approach
- **Usage**: For research into quantum optimization

## Examples

### Training on MNIST Digit 6

```bash
python experiments/train_rbm.py \
    --config mnist_digit6 \
    --epochs 20 \
    --solver gurobi \
    --output-dir ./results/digit6
```

### Reconstruction and Generation

```bash
python experiments/run_inference.py \
    ./results/digit6/checkpoint.pth \
    --config mnist_digit6 \
    --task both \
    --output-dir ./results/digit6/inference
```

### Custom Configuration

Create a new configuration file:

```python
from rbm.utils.config import ConfigManager

config_manager = ConfigManager()

# Create custom config
config = {
    'model': {'n_visible': 196, 'n_hidden': 32},
    'training': {'epochs': 5, 'learning_rate': 0.001},
    'solver': {'name': 'scip'}
}

config_manager.save(config, "my_experiment")
```

## Research Applications

This implementation is particularly useful for:

- **Quantum-Classical Hybrid Algorithms**: Exploring quantum annealing for ML training
- **QUBO Optimization Research**: Comparing different optimization approaches
- **Generative Modeling**: Training RBMs for image generation and reconstruction
- **Energy-Based Models**: Research into energy-based learning approaches

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rbm_perturb_map,
  title={RBM with Perturb-and-MAP: QUBO-based Training for Restricted Boltzmann Machines},
  author={RBM Research Team},
  year={2024},
  url={https://github.com/your-repo/rbm-perturb-map}
}
```

## Acknowledgments

- The Perturb-and-MAP methodology implementation
- QUBO solver integrations (Gurobi, SCIP, Dirac-3)
- PyTorch deep learning framework