"""
Example: Basic Operations and Autograd

This example demonstrates basic scalar operations and automatic differentiation.
"""

from scalargrad import Scalar

def main():
    print("\n" + "="*60)
    print("EXAMPLE: Basic Operations and Autograd")
    print("="*60)
    
    # Create scalars
    a = Scalar(-4.0, label='a')
    b = Scalar(2.0, label='b')
    
    # Build computation graph
    c = a + b
    c.label = 'c'
    d = a * b + b ** 3
    d.label = 'd'
    e = c - d
    e.label = 'e'
    f = e ** 2
    f.label = 'f'
    g = f / 2.0
    g.label = 'g'
    g = g + 10.0 / f
    
    print(f"\nFinal value: {g}")
    print("\nComputing gradients...")
    g.backward()
    
    print(f"dg/da = {a.grad:.4f}")
    print(f"dg/db = {b.grad:.4f}")
    
    # Demonstrate activation functions
    print("\n" + "="*60)
    print("Activation Functions")
    print("="*60)
    
    x = Scalar(0.5, label='x')
    
    relu_out = x.relu()
    print(f"\nReLU({x.data}) = {relu_out.data:.4f}")
    
    tanh_out = x.tanh()
    print(f"tanh({x.data}) = {tanh_out.data:.4f}")
    
    sigmoid_out = x.sigmoid()
    print(f"sigmoid({x.data}) = {sigmoid_out.data:.4f}")
    
    exp_out = x.exp()
    print(f"exp({x.data}) = {exp_out.data:.4f}")
    
    log_out = x.log()
    print(f"log({x.data}) = {log_out.data:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
