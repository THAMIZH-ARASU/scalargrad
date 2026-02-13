"""
Visualization utilities for ScalarGrad.
"""

import math
import warnings
from typing import Set, Tuple, Any, List

from .core import Scalar
from .nn import Module
from .training import TrainingMetrics
from .config import logger

# Visualization imports
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    warnings.warn("Graphviz not available. Install with: pip install graphviz")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib numpy")


class Visualizer:
    """Utilities for visualizing computation graphs and training results."""
    
    @staticmethod
    def trace(root: Scalar) -> Tuple[Set[Scalar], Set[Tuple[Scalar, Scalar]]]:
        """
        Build computation graph topology.
        
        Args:
            root: Root Scalar to trace from
            
        Returns:
            Tuple of (nodes, edges) in the computation graph
        """
        nodes, edges = set(), set()
        
        def build(v: Scalar) -> None:
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        
        build(root)
        return nodes, edges
    
    @staticmethod
    def draw_graph(root: Scalar, format: str = 'svg', rankdir: str = 'LR') -> Any:
        """
        Draw computation graph using Graphviz.
        
        Args:
            root: Root Scalar of the graph
            format: Output format ('svg', 'png', etc.)
            rankdir: Graph direction ('LR' or 'TB')
            
        Returns:
            Graphviz Digraph object
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz not available. Install with: pip install graphviz")
        
        nodes, edges = Visualizer.trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
        
        for n in nodes:
            label = n.label if n.label else ''
            dot.node(
                name=str(id(n)),
                label=f"{{{label} | data {n.data:.4f} | grad {n.grad:.4f}}}",
                shape='record'
            )
            
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
    
    @staticmethod
    def plot_training_history(history: List[TrainingMetrics], figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot training history using matplotlib.
        
        Args:
            history: List of TrainingMetrics
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib numpy")
        
        epochs = [m.epoch for m in history]
        losses = [m.loss for m in history]
        accuracies = [m.accuracy for m in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(epochs, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, accuracies, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(
        model: Module,
        X: 'np.ndarray',
        y: 'np.ndarray',
        resolution: float = 0.1,
        figsize: Tuple[int, int] = (10, 8),
        max_points: int = 10000
    ) -> None:
        """
        Plot decision boundary for 2D classification.
        
        Args:
            model: Trained model
            X: Input data (n_samples, 2)
            y: Labels
            resolution: Grid resolution (higher = faster, lower quality)
            figsize: Figure size
            max_points: Maximum number of grid points to evaluate
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib numpy")
        
        import numpy as np
        
        # Setup marker generator and color map
        markers = ('o', 's', '^', 'v', '<')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = plt.cm.RdYlBu
        
        # Plot decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Adjust resolution if needed to limit total points
        n_x1_points = int((x1_max - x1_min) / resolution)
        n_x2_points = int((x2_max - x2_min) / resolution)
        total_points = n_x1_points * n_x2_points
        
        if total_points > max_points:
            # Adjust resolution to meet max_points constraint
            scale_factor = math.sqrt(total_points / max_points)
            resolution = resolution * scale_factor
            logger.warning(f"Adjusted resolution to {resolution:.3f} to limit points to {max_points}")
        
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        
        # Predict on grid with batch processing for memory efficiency
        grid_points = np.c_[xx1.ravel(), xx2.ravel()]
        total_grid_points = len(grid_points)
        
        # Process in batches to reduce memory usage
        batch_size = 500
        Z_list = []
        
        print(f"Evaluating {total_grid_points} points for decision boundary...")
        
        for i in range(0, total_grid_points, batch_size):
            batch = grid_points[i:i+batch_size]
            inputs = [[Scalar(x) for x in row] for row in batch]
            predictions = [model(x) for x in inputs]
            batch_Z = [1 if p.data > 0 else -1 for p in predictions]
            Z_list.extend(batch_Z)
            
            # Clear Scalar objects to free memory
            del inputs, predictions
            
            # Progress indicator
            if (i + batch_size) % 2000 == 0 or (i + batch_size) >= total_grid_points:
                progress = min(100, int((i + batch_size) / total_grid_points * 100))
                print(f"Progress: {progress}%")
        
        Z = np.array(Z_list)
        Z = Z.reshape(xx1.shape)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        # Plot data points
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=f'Class {cl}',
                edgecolor='black'
            )
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
