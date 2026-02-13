"""
Optimizer implementations for ScalarGrad.
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


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(
        self,
        parameters: List[Scalar],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            eps: Numerical stability constant
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(parameters)  # First moment
        self.v = [0.0] * len(parameters)  # Second moment
        self.t = 0  # Time step
    
    def step(self) -> None:
        """Update parameters using Adam."""
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Compute bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
