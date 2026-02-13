"""
Base optimizer class.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core import Scalar


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, parameters: List[Scalar]):
        """
        Initialize optimizer.
        
        Args:
            parameters: List of Scalar parameters to optimize
        """
        self.parameters = parameters
    
    @abstractmethod
    def step(self) -> None:
        """Perform one optimization step."""
        pass
    
    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters:
            p.grad = 0.0
