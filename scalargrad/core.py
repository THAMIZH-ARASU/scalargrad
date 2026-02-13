"""
Core Scalar class and automatic differentiation engine.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Set, Callable, Union

from .config import config, logger


class Scalar:
    """
    Represents a scalar value in the computational graph with automatic differentiation.
    
    This class implements reverse-mode automatic differentiation (backpropagation)
    for scalar operations. Each Scalar maintains its value, gradient, and connections
    to parent nodes in the computation graph.
    
    Attributes:
        data: The scalar value
        grad: The gradient of this scalar with respect to some output
        _backward: Function to compute gradients for parent nodes
        _prev: Set of parent Scalar objects
        _op: String representation of the operation that created this Scalar
        label: Optional label for visualization
    """
    
    def __init__(
        self,
        data: float,
        _children: Tuple[Scalar, ...] = (),
        _op: str = '',
        label: str = ''
    ):
        """
        Initialize a Scalar value.
        
        Args:
            data: The numerical value
            _children: Tuple of parent Scalars in the computation graph
            _op: Operation that produced this Scalar
            label: Optional label for identification and visualization
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Scalar] = set(_children)
        self._op = _op
        self.label = label
        
        logger.debug(f"Created Scalar: {self}")
    
    def __repr__(self) -> str:
        """String representation of the Scalar."""
        label_str = f"'{self.label}'" if self.label else "unlabeled"
        return f"Scalar({label_str}, data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other: Union[Scalar, float]) -> Scalar:
        """Addition operation."""
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other: Union[Scalar, float]) -> Scalar:
        """Multiplication operation."""
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other: Union[int, float]) -> Scalar:
        """Power operation."""
        assert isinstance(other, (int, float)), "Power must be int or float"
        out = Scalar(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self) -> Scalar:
        """Negation operation."""
        return self * -1
    
    def __radd__(self, other: Union[Scalar, float]) -> Scalar:
        """Reverse addition."""
        return self + other
    
    def __sub__(self, other: Union[Scalar, float]) -> Scalar:
        """Subtraction operation."""
        return self + (-other)
    
    def __rsub__(self, other: Union[Scalar, float]) -> Scalar:
        """Reverse subtraction."""
        return other + (-self)
    
    def __rmul__(self, other: Union[Scalar, float]) -> Scalar:
        """Reverse multiplication."""
        return self * other
    
    def __truediv__(self, other: Union[Scalar, float]) -> Scalar:
        """Division operation."""
        return self * (other ** -1)
    
    def __rtruediv__(self, other: Union[Scalar, float]) -> Scalar:
        """Reverse division."""
        return other * (self ** -1)
    
    def relu(self) -> Scalar:
        """
        Rectified Linear Unit activation function.
        
        Returns:
            Scalar with ReLU applied: max(0, x)
        """
        out = Scalar(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self) -> Scalar:
        """
        Hyperbolic tangent activation function.
        
        Returns:
            Scalar with tanh applied
        """
        t = math.tanh(self.data)
        out = Scalar(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self) -> Scalar:
        """
        Sigmoid activation function.
        
        Returns:
            Scalar with sigmoid applied: 1 / (1 + exp(-x))
        """
        s = 1 / (1 + math.exp(-self.data))
        out = Scalar(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self) -> Scalar:
        """
        Exponential function.
        
        Returns:
            Scalar with exp applied
        """
        out = Scalar(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self) -> Scalar:
        """
        Natural logarithm function.
        
        Returns:
            Scalar with log applied
        """
        assert self.data > 0, "log requires positive input"
        out = Scalar(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self) -> None:
        """
        Compute gradients for all Scalars in the computation graph.
        
        Uses topological sort to ensure gradients are computed in the correct order.
        Implements reverse-mode automatic differentiation (backpropagation).
        """
        # Build topological order
        topo: List[Scalar] = []
        visited: Set[Scalar] = set()
        
        def build_topo(v: Scalar) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output to 1
        self.grad = 1.0
        
        # Backpropagate gradients
        for node in reversed(topo):
            node._backward()
            
            # Apply gradient clipping if configured
            if config.gradient_clip is not None:
                node.grad = max(min(node.grad, config.gradient_clip), -config.gradient_clip)
        
        logger.info(f"Backward pass completed for {len(topo)} nodes")
    
    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        self.grad = 0.0
