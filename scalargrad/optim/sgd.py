"""
SGD Optimizer implementations for ScalarGrad.
"""

import math
from typing import List

from ..core import Scalar
from .optimizer import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(
        self,
        parameters: List[Scalar],
        lr: float = 0.01,
        momentum: float = 0.0
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            momentum: Momentum coefficient
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0] * len(parameters)
    
    def step(self) -> None:
        """Update parameters using SGD with momentum."""
        for i, p in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
            p.data += self.velocities[i]