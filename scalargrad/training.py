"""
Training utilities for ScalarGrad.
"""

from dataclasses import dataclass
from typing import List, Optional

from .core import Scalar
from .nn import Module
from .optim import Optimizer
from .loss import Loss
from .config import logger


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    loss: float
    accuracy: float = 0.0
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


class Trainer:
    """
    Trainer class for managing the training loop.
    
    Provides utilities for training, validation, and metric tracking.
    """
    
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Loss
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer instance
            loss_fn: Loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history: List[TrainingMetrics] = []
    
    def train_epoch(
        self,
        X: List[List[float]],
        y: List[float],
        batch_size: Optional[int] = None
    ) -> TrainingMetrics:
        """
        Train for one epoch.
        
        Args:
            X: Input data
            y: Target values
            batch_size: Batch size (None for full batch)
            
        Returns:
            TrainingMetrics for this epoch
        """
        self.model.train()
        
        # Convert to batches
        if batch_size is None:
            batch_size = len(X)
        
        total_loss = 0.0
        correct = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            
            # Forward pass
            inputs = [[Scalar(x) for x in row] for row in batch_X]
            predictions = [self.model(x) for x in inputs]
            
            # Ensure predictions is a list of Scalars
            if not isinstance(predictions[0], list):
                predictions = predictions
            else:
                # Handle multi-output case
                predictions = [p[0] if isinstance(p, list) else p for p in predictions]
            
            # Compute loss
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.data
            for pred, target in zip(predictions, batch_y):
                if (pred.data > 0) == (target > 0):
                    correct += 1
        
        avg_loss = total_loss / (len(X) // batch_size)
        accuracy = correct / len(X)
        
        return TrainingMetrics(epoch=len(self.history), loss=avg_loss, accuracy=accuracy)
    
    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        epochs: int,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> List[TrainingMetrics]:
        """
        Train the model.
        
        Args:
            X: Input data
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            List of training metrics
        """
        for epoch in range(epochs):
            metrics = self.train_epoch(X, y, batch_size)
            self.history.append(metrics)
            
            if verbose and (epoch % max(1, epochs // 10) == 0):
                logger.info(f"Epoch {epoch}: Loss={metrics.loss:.4f}, Acc={metrics.accuracy:.2%}")
        
        return self.history
