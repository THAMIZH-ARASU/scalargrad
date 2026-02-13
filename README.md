<div align="center">
  <img src="assets/scalargrad_banner.png" alt="ScalarGrad Banner" width="800"/>
</div>

# ScalarGrad

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal scalar-valued automatic differentiation (autograd) engine with neural network capabilities, built from scratch in Python.

## Features

- **🔥 Core Autograd Engine**: Reverse-mode automatic differentiation for scalar values
- **🧠 Neural Networks**: Full-featured MLP implementation with customizable architectures
- **📊 Optimizers**: SGD with momentum and Adam optimizer
- **📉 Loss Functions**: MSE and SVM loss implementations
- **🎨 Visualization**: Computation graph and training metric visualization
- **⚡ Zero Dependencies**: Core functionality requires no external libraries
- **🛠️ Production-Ready**: Comprehensive logging, error handling, and configuration management

## Installation

### From PyPI (when published)

```bash
pip install scalargrad
```

### From Source

```bash
git clone https://github.com/THAMIZH-ARASU/scalargrad.git
cd scalargrad
pip install -e .
```

### Optional Dependencies

For visualization features:

```bash
pip install scalargrad[viz]      # For computation graph visualization
pip install scalargrad[plot]     # For training plots and decision boundaries
pip install scalargrad[all]      # Install all optional dependencies
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Autograd Example

```python
from scalargrad import Scalar

# Create scalar values
a = Scalar(2.0, label='a')
b = Scalar(3.0, label='b')

# Build computation graph
c = a * b + b ** 2
c.label = 'c'

# Compute gradients
c.backward()

print(f"c = {c.data}")       # c = 15.0
print(f"dc/da = {a.grad}")   # dc/da = 3.0
print(f"dc/db = {b.grad}")   # dc/db = 8.0
```

### Neural Network Training

```python
from scalargrad import MLP, Adam, MSELoss, Trainer

# Create a neural network
model = MLP(
    nin=2,                              # 2 input features
    layer_sizes=[16, 16, 1],            # Hidden layers and output
    activations=['relu', 'relu', 'linear']
)

# Setup training
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = MSELoss()
trainer = Trainer(model, optimizer, loss_fn)

# Training data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]  # XOR problem

# Train the model
history = trainer.fit(X, y, epochs=100)
```

## Architecture

The package is organized into the following modules:

```
scalargrad/
├── core.py           # Core Scalar class and autograd engine
├── config.py         # Configuration and logging
├── nn/               # Neural network components
│   ├── module.py     # Base Module class
│   └── layers.py     # Neuron, Layer, MLP implementations
├── optim/            # Optimization algorithms
│   ├── optimizer.py  # Base Optimizer class
│   └── optimizers.py # SGD, Adam implementations
├── loss.py           # Loss functions
├── training.py       # Training utilities
└── visualization.py  # Visualization tools
```

## API Reference

### Core Components

#### `Scalar`

The fundamental building block representing a scalar value with automatic differentiation.

```python
from scalargrad import Scalar

x = Scalar(5.0, label='x')
y = x ** 2 + 3 * x + 1
y.backward()  # Compute gradients
```

**Supported Operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Activation functions: `relu()`, `tanh()`, `sigmoid()`
- Mathematical: `exp()`, `log()`

#### `Module`

Base class for all neural network components.

```python
from scalargrad.nn import Module

class CustomLayer(Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters
    
    def __call__(self, x):
        # Forward pass
        pass
    
    def parameters(self):
        # Return trainable parameters
        pass
```

### Neural Networks

#### `MLP` (Multi-Layer Perceptron)

```python
from scalargrad import MLP

model = MLP(
    nin=3,                           # Input features
    layer_sizes=[16, 16, 1],         # Layer sizes
    activations=['relu', 'relu', 'linear'],  # Activations per layer
    init_scale=1.0                   # Weight initialization scale
)
```

### Optimizers

#### `SGD` (Stochastic Gradient Descent)

```python
from scalargrad import SGD

optimizer = SGD(
    parameters=model.parameters(),
    lr=0.01,          # Learning rate
    momentum=0.9      # Momentum coefficient
)
```

#### `Adam`

```python
from scalargrad import Adam

optimizer = Adam(
    parameters=model.parameters(),
    lr=0.001,         # Learning rate
    beta1=0.9,        # First moment decay
    beta2=0.999,      # Second moment decay
    eps=1e-8          # Numerical stability
)
```

### Loss Functions

#### `MSELoss` (Mean Squared Error)

```python
from scalargrad import MSELoss

loss_fn = MSELoss()
loss = loss_fn(predictions, targets)
```

#### `SVMLoss` (Support Vector Machine Loss)

```python
from scalargrad import SVMLoss

loss_fn = SVMLoss()
loss = loss_fn(predictions, targets)  # targets should be -1 or 1
```

### Training

#### `Trainer`

```python
from scalargrad import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)

history = trainer.fit(
    X=training_data,
    y=training_labels,
    epochs=100,
    batch_size=32,      # None for full batch
    verbose=True
)
```

### Visualization

#### Computation Graph

```python
from scalargrad import Visualizer

# Visualize computation graph
graph = Visualizer.draw_graph(output_scalar, format='svg')
graph.render('computation_graph', view=True)
```

#### Training History

```python
# Plot training metrics
Visualizer.plot_training_history(trainer.history)
```

#### Decision Boundary

```python
# Plot decision boundary (for 2D data)
Visualizer.plot_decision_boundary(model, X, y)
```

## Examples

Check the `examples/` directory for complete examples:

- `basic_operations.py`: Demonstrates basic autograd functionality
- `neural_network.py`: Full neural network training example

Run examples:

```bash
python examples/basic_operations.py
python examples/neural_network.py
```

## Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=scalargrad --cov-report=html
```

## Publishing to PyPI

1. **Build the package:**

```bash
python -m build
```

2. **Upload to TestPyPI (optional):**

```bash
python -m twine upload --repository testpypi dist/*
```

3. **Upload to PyPI:**

```bash
python -m twine upload dist/*
```

## Configuration

Global configuration can be modified:

```python
from scalargrad import config, LogLevel

config.log_level = LogLevel.DEBUG
config.seed = 42
config.gradient_clip = 1.0  # Clip gradients
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- PyTorch's autograd system
- Modern deep learning frameworks

## Citation

If you use ScalarGrad in your research, please cite:

```bibtex
@software{scalargrad2024,
  title = {ScalarGrad: A Production-Grade Scalar-Valued Autograd Engine},
  author = {Thamizharasu Saravanan},
  year = {2026},
  url = {https://github.com/THAMIZH-ARASU/scalargrad}
}
```

## Support

For questions and support:
- Open an issue on [GitHub](https://github.com/THAMIZH-ARASU/scalargrad/issues)
- Check the [documentation](https://github.com/THAMIZH-ARASU/scalargrad#readme)

---

**Made with ❤️ for the deep learning community**