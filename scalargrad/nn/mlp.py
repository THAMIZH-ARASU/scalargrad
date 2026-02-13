import random
from typing import List, Optional, Union

from ..core import Scalar
from .module import Module
from .layer import Layer

class MLP(Module):
    """
    Multi-Layer Perceptron (feedforward neural network).
    
    Attributes:
        layers: List of Layer objects
    """
    
    def __init__(
        self,
        nin: int,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        init_scale: float = 1.0
    ):
        """
        Initialize MLP.
        
        Args:
            nin: Number of input features
            layer_sizes: List of layer output sizes
            activations: List of activation functions (default: ReLU for hidden, linear for output)
            init_scale: Scale for weight initialization
        """
        super().__init__()
        
        # Default activations: ReLU for hidden layers, linear for output
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 1) + ['linear']
        
        assert len(activations) == len(layer_sizes), \
            "Number of activations must match number of layers"
        
        # Build layers
        sizes = [nin] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation=activations[i], init_scale=init_scale)
            for i in range(len(layer_sizes))
        ]
    
    def __call__(self, x: List[Scalar]) -> Union[Scalar, List[Scalar]]:
        """
        Forward pass through network.
        
        Args:
            x: List of input Scalars
            
        Returns:
            Network output
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Scalar]:
        """Return all network parameters."""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MLP(layers={len(self.layers)}, params={self.num_parameters()})"
