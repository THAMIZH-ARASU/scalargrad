"""
ScalarGrad: A Production-Grade Scalar-Valued Autograd Engine
=============================================================

A professional implementation of automatic differentiation with neural network
capabilities, visualization tools, and training utilities.

Author: Thamizharasu Saravanan
License: MIT
Version: 1.0.3
"""

__version__ = '1.0.3'
__author__ = 'Thamizharasu Saravanan'
__license__ = 'MIT'

# Core components
from .core import Scalar
from .config import Config, LogLevel, config, logger

# Neural network components
from .nn import Module, Neuron, Layer, MLP

# Optimizers
from .optim import Optimizer, SGD, Adam

# Loss functions
from .loss import Loss, MSELoss, SVMLoss

# Training utilities
from .training import Trainer, TrainingMetrics

# Visualization
from .visualization import Visualizer

__all__ = [
    # Core
    'Scalar',
    'Config',
    'LogLevel',
    'config',
    'logger',
    
    # Neural Networks
    'Module',
    'Neuron',
    'Layer',
    'MLP',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'Adam',
    
    # Loss Functions
    'Loss',
    'MSELoss',
    'SVMLoss',
    
    # Training
    'Trainer',
    'TrainingMetrics',
    
    # Visualization
    'Visualizer',
]
