"""
Base Module class for neural network components.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core import Scalar


class Module(ABC):
    """
    Abstract base class for all neural network modules.
    
    Provides common interface for parameters, gradient zeroing, and training mode.
    """
    
    def __init__(self):
        """Initialize module."""
        self.training = True
    
    @abstractmethod
    def parameters(self) -> List[Scalar]:
        """
        Return list of all trainable parameters.
        
        Returns:
            List of Scalar parameters
        """
        pass
    
    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            p.grad = 0.0
    
    def train(self) -> None:
        """Set module to training mode."""
        self.training = True
    
    def eval(self) -> None:
        """Set module to evaluation mode."""
        self.training = False
    
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return len(self.parameters())
