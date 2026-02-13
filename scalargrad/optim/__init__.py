"""
Optimization module for ScalarGrad.
"""

from .optimizer import Optimizer
from .optimizers import SGD, Adam

__all__ = ['Optimizer', 'SGD', 'Adam']
