"""
Main entry point for running ScalarGrad as a module.

Usage: python -m scalargrad
"""

from . import __version__, __author__
from .config import logger

def main():
    """Main function for command-line interface."""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                       ScalarGrad                         ║
║     A Production-Grade Autograd Engine                   ║
╚═══════════════════════════════════════════════════════════╝

Version: {__version__}
Author: {__author__}

ScalarGrad is a scalar-valued automatic differentiation engine
with neural network capabilities.

For usage examples, see:
  - examples/basic_operations.py
  - examples/neural_network.py

Documentation: https://github.com/yourusername/scalargrad

Quick Start:
  >>> from scalargrad import Scalar
  >>> a = Scalar(2.0)
  >>> b = Scalar(3.0)
  >>> c = a * b + b ** 2
  >>> c.backward()
  >>> print(f"dc/da = {{a.grad}}, dc/db = {{b.grad}}")

For help: https://github.com/yourusername/scalargrad#readme
""")
    
    logger.info("ScalarGrad loaded successfully!")
    

if __name__ == "__main__":
    main()
