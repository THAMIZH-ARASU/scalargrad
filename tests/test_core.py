"""
Tests for core Scalar class and autograd functionality.
"""

import pytest
from scalargrad import Scalar


class TestScalarBasicOps:
    """Test basic scalar operations."""
    
    def test_addition(self):
        """Test addition operation."""
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = a + b
        assert c.data == 5.0
    
    def test_multiplication(self):
        """Test multiplication operation."""
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = a * b
        assert c.data == 6.0
    
    def test_power(self):
        """Test power operation."""
        a = Scalar(2.0)
        b = a ** 3
        assert b.data == 8.0
    
    def test_negation(self):
        """Test negation operation."""
        a = Scalar(5.0)
        b = -a
        assert b.data == -5.0
    
    def test_subtraction(self):
        """Test subtraction operation."""
        a = Scalar(5.0)
        b = Scalar(3.0)
        c = a - b
        assert c.data == 2.0
    
    def test_division(self):
        """Test division operation."""
        a = Scalar(6.0)
        b = Scalar(2.0)
        c = a / b
        assert c.data == 3.0


class TestScalarGradients:
    """Test gradient computation."""
    
    def test_simple_gradient(self):
        """Test gradient for simple operation."""
        a = Scalar(2.0)
        b = a * 3.0
        b.backward()
        assert a.grad == 3.0
    
    def test_chain_rule(self):
        """Test chain rule application."""
        a = Scalar(2.0)
        b = a ** 2
        c = b * 3.0
        c.backward()
        # dc/da = dc/db * db/da = 3 * 2*a = 3 * 4 = 12
        assert a.grad == 12.0
    
    def test_addition_gradient(self):
        """Test gradient for addition."""
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0
    
    def test_complex_graph(self):
        """Test gradient for complex computation graph."""
        a = Scalar(-4.0)
        b = Scalar(2.0)
        c = a + b
        d = a * b + b ** 3
        e = c - d
        f = e ** 2
        f.backward()
        assert a.grad != 0  # Should have non-zero gradient


class TestScalarActivations:
    """Test activation functions."""
    
    def test_relu_positive(self):
        """Test ReLU with positive input."""
        a = Scalar(5.0)
        b = a.relu()
        assert b.data == 5.0
    
    def test_relu_negative(self):
        """Test ReLU with negative input."""
        a = Scalar(-5.0)
        b = a.relu()
        assert b.data == 0.0
    
    def test_tanh(self):
        """Test tanh activation."""
        a = Scalar(0.0)
        b = a.tanh()
        assert abs(b.data - 0.0) < 1e-6
    
    def test_sigmoid(self):
        """Test sigmoid activation."""
        a = Scalar(0.0)
        b = a.sigmoid()
        assert abs(b.data - 0.5) < 1e-6
    
    def test_exp(self):
        """Test exponential function."""
        a = Scalar(0.0)
        b = a.exp()
        assert abs(b.data - 1.0) < 1e-6
    
    def test_log(self):
        """Test logarithm function."""
        a = Scalar(1.0)
        b = a.log()
        assert abs(b.data - 0.0) < 1e-6


class TestScalarZeroGrad:
    """Test gradient zeroing."""
    
    def test_zero_grad(self):
        """Test gradient reset."""
        a = Scalar(2.0)
        b = a * 3.0
        b.backward()
        assert a.grad != 0
        a.zero_grad()
        assert a.grad == 0.0
