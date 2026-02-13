import random
from typing import List, Optional, Union

from ..core import Scalar
from .module import Module
from .neuron import Neuron


class Layer(Module):
    """
    Layer of neurons with parallel computation.
    
    Attributes:
        neurons: List of Neuron objects
    """
    
    def __init__(
        self,
        nin: int,
        nout: int,
        activation: str = 'relu',
        init_scale: float = 1.0
    ):
        """
        Initialize layer.
        
        Args:
            nin: Number of input features
            nout: Number of output neurons
            activation: Activation function for neurons
            init_scale: Scale for weight initialization
        """
        super().__init__()
        self.neurons = [
            Neuron(nin, activation=activation, init_scale=init_scale)
            for _ in range(nout)
        ]
    
    def __call__(self, x: List[Scalar]) -> Union[Scalar, List[Scalar]]:
        """
        Forward pass through layer.
        
        Args:
            x: List of input Scalars
            
        Returns:
            Single Scalar if nout=1, else List of Scalars
        """
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self) -> List[Scalar]:
        """Return all layer parameters."""
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Layer(neurons={len(self.neurons)}, {self.neurons[0]})"