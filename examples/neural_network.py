"""
Example: Neural Network Training

This example demonstrates training a neural network on synthetic data.
"""

import warnings
warnings.filterwarnings('ignore')

from scalargrad import MLP, Adam, SVMLoss, Trainer, Visualizer, config, LogLevel

# Check for optional dependencies
try:
    from sklearn.datasets import make_moons
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")


def main():
    if not SKLEARN_AVAILABLE:
        print("This example requires scikit-learn. Install with: pip install scikit-learn")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE: Neural Network Training")
    print("="*60)
    
    # Configure
    config.log_level = LogLevel.INFO
    config.seed = 42
    
    # Generate synthetic moon dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    y = y * 2 - 1  # Convert to {-1, 1}
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    # Create model
    model = MLP(
        nin=2,
        layer_sizes=[16, 16, 1],
        activations=['relu', 'relu', 'linear']
    )
    
    print(f"\nModel: {model}")
    print(f"Parameters: {model.num_parameters()}")
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = SVMLoss()
    trainer = Trainer(model, optimizer, loss_fn)
    
    # Train
    print("\nTraining...")
    history = trainer.fit(X.tolist(), y.tolist(), epochs=50, verbose=False)
    
    # Print final metrics
    final_metrics = history[-1]
    print(f"\nFinal Results:")
    print(f"  Loss: {final_metrics.loss:.4f}")
    print(f"  Accuracy: {final_metrics.accuracy:.2%}")
    
    # Visualize results
    if PLOTTING_AVAILABLE:
        print("\nGenerating visualizations...")
        Visualizer.plot_training_history(history)
        Visualizer.plot_decision_boundary(model, X, y)
    else:
        print("\nSkipping visualization (matplotlib not available)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
