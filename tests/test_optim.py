"""
Tests for optimizer implementations.
"""

import pytest
from scalargrad import Scalar, SGD, Adam


class TestSGD:
    """Test SGD optimizer."""
    
    def test_sgd_creation(self):
        """Test SGD initialization."""
        params = [Scalar(1.0), Scalar(2.0)]
        optimizer = SGD(params, lr=0.01)
        assert optimizer.lr == 0.01
        assert len(optimizer.parameters) == 2
    
    def test_sgd_step(self):
        """Test SGD optimization step."""
        param = Scalar(1.0)
        optimizer = SGD([param], lr=0.1)
        
        # Simulate gradient
        param.grad = 0.5
        
        initial_value = param.data
        optimizer.step()
        
        # Should decrease by lr * grad
        assert param.data < initial_value
    
    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        param = Scalar(1.0)
        optimizer = SGD([param], lr=0.1, momentum=0.9)
        
        param.grad = 0.5
        optimizer.step()
        
        assert optimizer.velocities[0] != 0


class TestAdam:
    """Test Adam optimizer."""
    
    def test_adam_creation(self):
        """Test Adam initialization."""
        params = [Scalar(1.0), Scalar(2.0)]
        optimizer = Adam(params, lr=0.001)
        assert optimizer.lr == 0.001
        assert len(optimizer.parameters) == 2
    
    def test_adam_step(self):
        """Test Adam optimization step."""
        param = Scalar(1.0)
        optimizer = Adam([param], lr=0.1)
        
        # Simulate gradient
        param.grad = 0.5
        
        initial_value = param.data
        optimizer.step()
        
        # Should update parameters
        assert param.data != initial_value
        assert optimizer.t == 1
    
    def test_adam_moments(self):
        """Test Adam moment updates."""
        param = Scalar(1.0)
        optimizer = Adam([param], lr=0.1)
        
        param.grad = 0.5
        optimizer.step()
        
        # First and second moments should be updated
        assert optimizer.m[0] != 0
        assert optimizer.v[0] != 0


class TestOptimizerZeroGrad:
    """Test gradient zeroing."""
    
    def test_zero_grad(self):
        """Test gradient reset."""
        params = [Scalar(1.0), Scalar(2.0)]
        params[0].grad = 1.0
        params[1].grad = 2.0
        
        optimizer = SGD(params)
        optimizer.zero_grad()
        
        assert params[0].grad == 0.0
        assert params[1].grad == 0.0
