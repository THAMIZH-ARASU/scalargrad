import random
from typing import List, Optional, Union

from ..core import Scalar
from .module import Module


class Neuron(Module):
    """
    Single neuron with weighted inputs and optional activation.
    
    Implements: output = activation(w·x + b)
    
    Attributes:
        w: Weight parameters
        b: Bias parameter
        activation: Activation function to apply
    """
    
    def __init__(
        self,
        nin: int,
        activation: str = 'relu',
        init_scale: float = 1.0
    ):
        """
        Initialize neuron.
        
        Args:
            nin: Number of input features
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'linear')
            init_scale: Scale for weight initialization
        """
        super().__init__()
        self.w = [Scalar(random.uniform(-1, 1) * init_scale) for _ in range(nin)]
        self.b = Scalar(0)
        
        # Set activation function
        activations = {
            'relu': lambda x: x.relu(),
            'tanh': lambda x: x.tanh(),
            'sigmoid': lambda x: x.sigmoid(),
            'linear': lambda x: x
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_fn = activations[activation]
        self.activation = activation
    
    def __call__(self, x: List[Scalar]) -> Scalar:
        """
        Forward pass through neuron.
        
        Args:
            x: List of input Scalars
            
        Returns:
            Output Scalar
        """
        # Compute weighted sum: w·x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        return self.activation_fn(activation)
    
    def parameters(self) -> List[Scalar]:
        """Return neuron parameters."""
        return self.w + [self.b]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Neuron(in={len(self.w)}, activation={self.activation})"