"""
Tests for loss functions.
"""

import pytest
from scalargrad import Scalar, MSELoss, SVMLoss


class TestMSELoss:
    """Test MSE loss function."""
    
    def test_mse_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        loss_fn = MSELoss()
        predictions = [Scalar(1.0), Scalar(2.0), Scalar(3.0)]
        targets = [1.0, 2.0, 3.0]
        
        loss = loss_fn(predictions, targets)
        assert abs(loss.data) < 1e-6
    
    def test_mse_calculation(self):
        """Test MSE calculation."""
        loss_fn = MSELoss()
        predictions = [Scalar(1.0), Scalar(2.0)]
        targets = [2.0, 3.0]
        
        loss = loss_fn(predictions, targets)
        # MSE = ((1-2)^2 + (2-3)^2) / 2 = (1 + 1) / 2 = 1.0
        assert abs(loss.data - 1.0) < 1e-6


class TestSVMLoss:
    """Test SVM loss function."""
    
    def test_svm_perfect_prediction(self):
        """Test SVM with perfect predictions."""
        loss_fn = SVMLoss()
        predictions = [Scalar(2.0), Scalar(-2.0)]
        targets = [1.0, -1.0]
        
        loss = loss_fn(predictions, targets)
        assert loss.data >= 0  # SVM loss is always non-negative
    
    def test_svm_calculation(self):
        """Test SVM loss calculation."""
        loss_fn = SVMLoss()
        predictions = [Scalar(0.5)]
        targets = [1.0]
        
        loss = loss_fn(predictions, targets)
        # Loss = max(0, 1 + (-1 * 0.5)) = max(0, 0.5) = 0.5
        assert abs(loss.data - 0.5) < 1e-6
