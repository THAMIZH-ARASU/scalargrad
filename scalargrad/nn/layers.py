"""
Neural network layer implementations.
"""

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
