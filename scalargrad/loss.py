"""
Loss functions for ScalarGrad.
"""

from abc import ABC, abstractmethod
from typing import List

from .core import Scalar


class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def __call__(self, predictions: List[Scalar], targets: List[float]) -> Scalar:
        """Compute loss."""
        pass


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def __call__(self, predictions: List[Scalar], targets: List[float]) -> Scalar:
        """
        Compute MSE loss.
        
        Args:
            predictions: List of predicted Scalars
            targets: List of target values
            
        Returns:
            MSE loss as Scalar
        """
        losses = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
        return sum(losses, Scalar(0)) * (1.0 / len(losses))


class SVMLoss(Loss):
    """SVM max-margin loss."""
    
    def __call__(self, predictions: List[Scalar], targets: List[float]) -> Scalar:
        """
        Compute SVM hinge loss.
        
        Args:
            predictions: List of predicted Scalars
            targets: List of target values (-1 or 1)
            
        Returns:
            SVM loss as Scalar
        """
        losses = [(Scalar(1) + (-target * pred)).relu() 
                  for pred, target in zip(predictions, targets)]
        return sum(losses, Scalar(0)) * (1.0 / len(losses))
