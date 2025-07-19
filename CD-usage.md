# Contrastive Divergence Training and Inference Usage Guide

This document provides comprehensive usage instructions for training and running inference with RBM models using Contrastive Divergence (CD) methodology.

## Overview

The CD implementation provides:
- **Training Script**: `experiments/train_rbm_cd.py` - Train RBMs using Contrastive Divergence
- **Inference Script**: `experiments/run_inference_cd.py` - Run inference with CD-trained models
- **Modular Framework**: Uses TrainingManager with ContrastiveDivergenceTraining method
- **Gibbs Sampling**: Pure CD methodology without QUBO solvers

## Training with Contrastive Divergence

### Basic Usage

```bash
# Basic CD training with default configuration
python experiments/train_rbm_cd.py

# Train with specific configuration file
python experiments/train_rbm_cd.py --config-file configs/contrastive_divergence.yaml

# Train with custom parameters
python experiments/train_rbm_cd.py --epochs 20 --learning-rate 0.01 --batch-size 64

# Resume training from checkpoint
python experiments/train_rbm_cd.py --resume --output-dir ./previous_results
```

### CD-Specific Parameters

```bash
# CD-1 (standard Contrastive Divergence)
python experiments/train_rbm_cd.py --k-steps 1

# CD-5 (5-step Contrastive Divergence)
python experiments/train_rbm_cd.py --k-steps 5

# Persistent Contrastive Divergence (PCD)
python experiments/train_rbm_cd.py --persistent

# Combined CD-3 with PCD
python experiments/train_rbm_cd.py --k-steps 3 --persistent
```

### Figure 6 Experiments

```bash
# Train on digit 6 only (figure 6 experiment)
python experiments/train_rbm_cd.py --figure6

# Train on any specific digit
python experiments/train_rbm_cd.py --digit-filter 8

# Figure 6 with custom parameters
python experiments/train_rbm_cd.py --figure6 --epochs 30 --k-steps 3
```

### Complete Training Example

```bash
# Comprehensive CD training with all options
python experiments/train_rbm_cd.py \
    --config contrastive_divergence \
    --epochs 25 \
    --learning-rate 0.005 \
    --batch-size 128 \
    --k-steps 3 \
    --persistent \
    --figure6 \
    --output-dir ./my_cd_results \
    --seed 123
```

### Training Output

The training script produces:
- **Checkpoint**: `rbm_cd_checkpoint.pth` - Model weights and training state
- **Training Summary**: `training_summary.txt` - Detailed training metrics
- **Epoch Logs**: `cd_training.log` - CSV format epoch-by-epoch metrics
- **Figures Directory**: `figures/` - Training plots and visualizations
- **Console Output**: Real-time training progress with CD-specific metrics

Example training output:
```
=== RBM Training with Contrastive Divergence ===
Configuration: contrastive_divergence
Model: rbm
Visible units: 784
Hidden units: 64
Training method: contrastive_divergence
Epochs: 20
Learning rate: 0.01
Batch size: 64
CD-k steps: 1
Persistent CD: False
Digit filter: 6
--------------------------------------------------
Loaded training data: 5918 samples
Initialized ContrastiveDivergence(k=1) with k_steps=1, persistent=False

Starting training...
Epoch 1/20 | Loss: 0.2307 | Pos Energy: -0.6398 | Neg Energy: 24.4932 | Time: 0.02s
Epoch 2/20 | Loss: 0.2150 | Pos Energy: -7.1563 | Neg Energy: 24.3786 | Time: 0.02s
...
```

#### Epoch Logging

The training script creates detailed CSV logs in `cd_training.log`:
```csv
2025-07-16 12:17:15 | epoch,reconstruction_error,positive_energy,negative_energy,epoch_time,cd_k_steps,cd_persistent,cd_acceptance_rate
2025-07-16 12:17:15 | 1,0.232988,2.580720,24.607950,0.0231,1.0,0.0,0.250658
2025-07-16 12:17:15 | 2,0.220673,1.089336,25.078592,0.0200,1.0,0.0,0.500428
2025-07-16 12:17:15 | 3,0.212058,-2.580127,25.290792,0.0183,1.0,0.0,0.499578
...
```

#### Resume Training

To resume training from a checkpoint:
```bash
# Resume from default checkpoint location
python experiments/train_rbm_cd.py --resume

# Resume from specific directory
python experiments/train_rbm_cd.py --resume --output-dir ./my_results

# Resume with additional epochs
python experiments/train_rbm_cd.py --resume --epochs 30
```

## Inference with CD-Trained Models

### Basic Usage

```bash
# Run all inference tasks
python experiments/run_inference_cd.py path/to/rbm_cd_checkpoint.pth

# Run specific task
python experiments/run_inference_cd.py checkpoint.pth --task reconstruction
python experiments/run_inference_cd.py checkpoint.pth --task generation
python experiments/run_inference_cd.py checkpoint.pth --task interpolation
```

### CD-Specific Parameters

```bash
# Adjust reconstruction quality (k-step Gibbs sampling)
python experiments/run_inference_cd.py checkpoint.pth --task reconstruction --k-steps 3

# Adjust generation quality (more Gibbs steps)
python experiments/run_inference_cd.py checkpoint.pth --task generation --gibbs-steps 2000

# Quick inference for testing
python experiments/run_inference_cd.py checkpoint.pth --gibbs-steps 100
```

### Figure 6 Model Inference

```bash
# Inference with figure 6 trained model
python experiments/run_inference_cd.py figure6_checkpoint.pth --digit-filter 6

# Generate digit 6 samples
python experiments/run_inference_cd.py figure6_checkpoint.pth --task generation --digit-filter 6
```

### Complete Inference Example

```bash
# Comprehensive inference with all options
python experiments/run_inference_cd.py \
    my_cd_results/rbm_cd_checkpoint.pth \
    --config contrastive_divergence \
    --task all \
    --k-steps 3 \
    --gibbs-steps 1500 \
    --digit-filter 6 \
    --output-dir ./inference_results \
    --seed 456
```

### Inference Output

The inference script produces:
- **Reconstruction Plots**: `cd_reconstruction_clean.png`, `cd_reconstruction_denoising.png`
- **Generation Plots**: `cd_generation_samples.png`
- **Interpolation Plots**: `cd_interpolation.png`
- **Console Metrics**: Reconstruction errors, generation times, and quality metrics

Example inference output:
```
=== CD-RBM Inference ===
Checkpoint: my_cd_results/rbm_cd_checkpoint.pth
Training method: ContrastiveDivergence(k=1)
Checkpoint epoch: 20
Final reconstruction error: 0.167296
Model loaded: 784 visible, 64 hidden units

=== Running CD Reconstruction Task (k=3) ===
Reconstructing clean images...
Denoising noisy images...
Clean reconstruction error: 0.153837
Denoising reconstruction error: 0.165237

=== Running CD Generation Task (1500 steps) ===
Generating 10 samples with 1500 Gibbs steps...
Generation completed in 0.15s
Average time per sample: 0.015s
```

## Configuration Files

### Default CD Configuration

The `configs/contrastive_divergence.yaml` file contains:

```yaml
# Model configuration
model:
  n_visible: 784
  n_hidden: 64
  model_type: rbm

# Training parameters
training:
  epochs: 20
  learning_rate: 0.01
  batch_size: 64
  optimizer: sgd
  method: contrastive_divergence

# CD-specific parameters
cd_params:
  k_steps: 1          # CD-k steps
  persistent: false   # Enable PCD
  sampling_mode: stochastic
  temperature: 1.0
  seed: 42

# Data configuration
data:
  dataset: mnist
  digit_filter: null  # Set to 0-9 for specific digit
  image_size: [28, 28]
  data_root: ./data

# Inference parameters
inference:
  gibbs_steps: 1000
  num_generated_samples: 10
  reconstruction_samples: 5
```

### Custom Configuration

Create your own configuration file:

```yaml
# my_cd_config.yaml
model:
  n_visible: 784
  n_hidden: 128      # Larger hidden layer

training:
  epochs: 50
  learning_rate: 0.005
  batch_size: 128
  method: contrastive_divergence

cd_params:
  k_steps: 5         # CD-5
  persistent: true   # PCD
  sampling_mode: stochastic

data:
  digit_filter: 6    # Figure 6 experiment
```

Use with:
```bash
python experiments/train_rbm_cd.py --config-file my_cd_config.yaml
```

## Key Differences from P&M Training

| Aspect | Contrastive Divergence | Perturb-and-MAP |
|--------|----------------------|-----------------|
| **Methodology** | Gibbs sampling | QUBO optimization |
| **Negative Phase** | k-step Gibbs chains | QUBO solver |
| **Sampling** | Stochastic/probabilistic | Deterministic MAP |
| **Dependencies** | Pure PyTorch | Requires QUBO solvers |
| **Speed** | Fast, lightweight | Slower (solver overhead) |
| **Quality** | Depends on k-steps | Depends on solver quality |

## Performance Tuning

### Training Performance

```bash
# Fast training (less accurate)
python experiments/train_rbm_cd.py --k-steps 1 --batch-size 128

# High quality training (slower)
python experiments/train_rbm_cd.py --k-steps 5 --persistent --batch-size 32

# Balanced training
python experiments/train_rbm_cd.py --k-steps 3 --batch-size 64
```

### Inference Performance

```bash
# Fast inference
python experiments/run_inference_cd.py checkpoint.pth --k-steps 1 --gibbs-steps 500

# High quality inference
python experiments/run_inference_cd.py checkpoint.pth --k-steps 5 --gibbs-steps 2000
```

## Common Use Cases

### 1. Standard MNIST Training

```bash
# Train standard RBM on MNIST
python experiments/train_rbm_cd.py --epochs 20 --k-steps 1

# Resume if interrupted
python experiments/train_rbm_cd.py --resume --epochs 30

# Run inference
python experiments/run_inference_cd.py outputs/rbm_cd_checkpoint.pth
```

### 2. Figure 6 Digit Modeling

```bash
# Train on digit 6
python experiments/train_rbm_cd.py --figure6 --epochs 30 --output-dir ./digit6_results

# Resume digit 6 training if needed
python experiments/train_rbm_cd.py --resume --figure6 --output-dir ./digit6_results --epochs 50

# Generate digit 6 samples
python experiments/run_inference_cd.py digit6_results/rbm_cd_checkpoint.pth --task generation --digit-filter 6
```

### 3. High-Quality Generation

```bash
# Train with PCD
python experiments/train_rbm_cd.py --k-steps 3 --persistent --epochs 50

# Generate with many Gibbs steps
python experiments/run_inference_cd.py checkpoint.pth --task generation --gibbs-steps 3000
```

### 4. Denoising Experiments

```bash
# Train denoising model
python experiments/train_rbm_cd.py --epochs 25 --k-steps 2

# Test denoising capability
python experiments/run_inference_cd.py checkpoint.pth --task reconstruction --k-steps 5
```

## Troubleshooting

### Common Issues

1. **Training too slow**: Reduce `--k-steps` or increase `--batch-size`
2. **Poor sample quality**: Increase `--k-steps`, use `--persistent`, or train longer
3. **Memory issues**: Reduce `--batch-size` or model size
4. **Convergence problems**: Adjust `--learning-rate` or use different optimizer

### Monitoring Training

- Watch reconstruction error: should decrease over time
- Monitor energy gap: positive - negative energy
- Check acceptance rates: should be reasonable (~0.3-0.7)
- Review `cd_training.log` for detailed epoch-by-epoch metrics
- Use `--resume` to continue interrupted training sessions

### Debugging

```bash
# Quick test run
python experiments/train_rbm_cd.py --epochs 1 --batch-limit 2

# Verbose inference
python experiments/run_inference_cd.py checkpoint.pth --gibbs-steps 100 --task generation
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple configurations
for k in 1 3 5; do
    python experiments/train_rbm_cd.py --k-steps $k --output-dir ./results_k$k
done
```

### Custom Experiments

```bash
# Compare CD-k values
python experiments/train_rbm_cd.py --k-steps 1 --output-dir ./cd1_results
python experiments/train_rbm_cd.py --k-steps 5 --output-dir ./cd5_results

# Compare with/without PCD
python experiments/train_rbm_cd.py --k-steps 3 --output-dir ./cd3_results
python experiments/train_rbm_cd.py --k-steps 3 --persistent --output-dir ./pcd3_results
```

## Best Practices

1. **Start Simple**: Begin with CD-1, then experiment with higher k-steps
2. **Monitor Metrics**: Watch reconstruction error and energy gaps in `cd_training.log`
3. **Use PCD Carefully**: PCD can be unstable, monitor training closely
4. **Save Checkpoints**: Training can be interrupted and resumed with `--resume`
5. **Experiment with Digits**: Figure 6 experiments often show clearer patterns
6. **Quality vs Speed**: Balance k-steps and Gibbs steps based on needs
7. **Reproducibility**: Always set `--seed` for consistent results
8. **Long Training**: Use `--resume` for multi-session training on complex models

## References

- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence
- Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient
- Carreira-Perpiñán, M. Á., & Hinton, G. E. (2005). On contrastive divergence learning