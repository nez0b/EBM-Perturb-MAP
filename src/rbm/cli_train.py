#!/usr/bin/env python3
"""
CLI entry point for RBM training.
"""

import sys
from pathlib import Path

# Add experiments to path for imports
experiments_path = Path(__file__).parent.parent.parent / "experiments"
sys.path.insert(0, str(experiments_path))

from train_rbm import main

if __name__ == "__main__":
    main()