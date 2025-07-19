#!/usr/bin/env python3
"""
Test runner for EBM training framework.

This script runs all unit tests for the modular EBM training system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_pattern="test_*.py", verbose=False, coverage=False):
    """
    Run tests using pytest.
    
    Args:
        test_pattern: Pattern for test files to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    
    # Get the project root directory
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    # Add src to Python path
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append(str(tests_dir))
    
    # Add test pattern
    cmd.extend(["-k", test_pattern.replace("test_", "").replace(".py", "")])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=rbm",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "-x",          # Stop on first failure
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run tests
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for EBM training framework")
    parser.add_argument(
        "--pattern", "-p", 
        default="test_*.py",
        help="Pattern for test files to run (default: test_*.py)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true",
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--install-deps", "-i", 
        action="store_true",
        help="Install test dependencies first"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            "pip", "install", "-r", "tests/requirements.txt"
        ], check=True)
        print("Dependencies installed successfully!")
        print("-" * 60)
    
    # Run tests
    success = run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())