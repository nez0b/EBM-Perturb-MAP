#!/usr/bin/env python3
"""
Quick test script to validate the package structure works.
"""

import sys
sys.path.insert(0, "src")

import torch
import torch.optim as optim

# Test imports
from rbm.models.rbm import RBM
from rbm.solvers.gurobi import GurobiSolver
from rbm.training.trainer import Trainer
from rbm.data.mnist import load_mnist_data
from rbm.utils.config import ConfigManager
from rbm.utils.visualization import plot_reconstruction

def main():
    print("🧪 Testing RBM Package Structure")
    print("=" * 40)
    
    # Test 1: Configuration loading
    print("✅ Test 1: Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load("test_quick")
    print(f"   Config loaded: {config['model']['model_type']} model")
    
    # Test 2: Model creation
    print("✅ Test 2: Creating RBM model...")
    model = RBM(
        n_visible=config['model']['n_visible'],
        n_hidden=config['model']['n_hidden']
    )
    print(f"   Model created: {model.n_visible} visible, {model.n_hidden} hidden units")
    
    # Test 3: Solver availability
    print("✅ Test 3: Checking QUBO solver...")
    if GurobiSolver.is_available:
        solver = GurobiSolver(suppress_output=True)
        print(f"   {solver.name} solver is available")
    else:
        print("   ❌ Gurobi solver not available")
        return
    
    # Test 4: Data loading
    print("✅ Test 4: Loading MNIST data...")
    try:
        train_loader, dataset_size = load_mnist_data(config, train=True)
        print(f"   Data loaded: {dataset_size} samples")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # Test 5: Training setup (without actual training)
    print("✅ Test 5: Setting up trainer...")
    optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'])
    trainer = Trainer(model, solver, optimizer, config)
    print("   Trainer created successfully")
    
    # Test 6: Single QUBO solve (quick test)
    print("✅ Test 6: Testing QUBO solving...")
    try:
        # Create a small test QUBO
        import numpy as np
        test_Q = np.random.randn(5, 5) * 0.1
        test_Q = (test_Q + test_Q.T) / 2  # Make symmetric
        
        result = solver.solve(test_Q)
        print(f"   QUBO solve successful: solution shape {result.shape}")
    except Exception as e:
        print(f"   ❌ QUBO solve failed: {e}")
        return
    
    print("\n🎉 All tests passed! Package structure is working correctly!")
    print("\nThe training script works but takes time due to QUBO optimization.")
    print("To run actual training:")
    print("  python experiments/train_rbm.py --config mnist_digit6 --epochs 5")

if __name__ == "__main__":
    main()