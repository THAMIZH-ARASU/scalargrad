"""
Tests for neural network components.
"""

import pytest
from scalargrad import Scalar, Neuron, Layer, MLP


class TestNeuron:
    """Test Neuron class."""
    
    def test_neuron_creation(self):
        """Test neuron initialization."""
        neuron = Neuron(nin=3)
        assert len(neuron.w) == 3
        assert neuron.b is not None
    
    def test_neuron_forward(self):
        """Test neuron forward pass."""
        neuron = Neuron(nin=2, activation='linear')
        inputs = [Scalar(1.0), Scalar(2.0)]
        output = neuron(inputs)
        assert isinstance(output, Scalar)
    
    def test_neuron_parameters(self):
        """Test neuron parameters."""
        neuron = Neuron(nin=3)
        params = neuron.parameters()
        assert len(params) == 4  # 3 weights + 1 bias


class TestLayer:
    """Test Layer class."""
    
    def test_layer_creation(self):
        """Test layer initialization."""
        layer = Layer(nin=3, nout=4)
        assert len(layer.neurons) == 4
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = Layer(nin=2, nout=3)
        inputs = [Scalar(1.0), Scalar(2.0)]
        outputs = layer(inputs)
        assert isinstance(outputs, list)
        assert len(outputs) == 3
    
    def test_layer_single_output(self):
        """Test layer with single output."""
        layer = Layer(nin=2, nout=1)
        inputs = [Scalar(1.0), Scalar(2.0)]
        output = layer(inputs)
        assert isinstance(output, Scalar)
    
    def test_layer_parameters(self):
        """Test layer parameters."""
        layer = Layer(nin=2, nout=3)
        params = layer.parameters()
        assert len(params) == 3 * (2 + 1)  # 3 neurons * (2 weights + 1 bias)


class TestMLP:
    """Test MLP class."""
    
    def test_mlp_creation(self):
        """Test MLP initialization."""
        mlp = MLP(nin=3, layer_sizes=[4, 5, 1])
        assert len(mlp.layers) == 3
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(nin=2, layer_sizes=[4, 1])
        inputs = [Scalar(1.0), Scalar(2.0)]
        output = mlp(inputs)
        assert isinstance(output, Scalar)
    
    def test_mlp_parameters(self):
        """Test MLP parameters."""
        mlp = MLP(nin=2, layer_sizes=[3, 1])
        params = mlp.parameters()
        # Layer 1: 3 neurons * (2 weights + 1 bias) = 9
        # Layer 2: 1 neuron * (3 weights + 1 bias) = 4
        # Total: 13
        assert len(params) == 13
    
    def test_mlp_custom_activations(self):
        """Test MLP with custom activations."""
        mlp = MLP(
            nin=2,
            layer_sizes=[3, 1],
            activations=['tanh', 'sigmoid']
        )
        assert mlp.layers[0].neurons[0].activation == 'tanh'
        assert mlp.layers[1].neurons[0].activation == 'sigmoid'
