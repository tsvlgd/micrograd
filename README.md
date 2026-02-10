# Micrograd 🧠

A tiny **autograd engine** with a bite! Implements backpropagation (reverse-mode automatic differentiation) over a dynamically built computational graph (DAG) and a small neural networks library on top of it with a PyTorch-like API.

Both components are incredibly small (~100 lines for the autograd engine and ~50 lines for the neural network library), making this an excellent resource for understanding how deep learning libraries work under the hood. The DAG operates over scalar values, decomposing each neuron operation into individual arithmetic operations and ReLU activations.

---

## Table of Contents

- [Micrograd 🧠](#micrograd-)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
    - [Key Files](#key-files)
  - [Getting Started](#getting-started)
    - [Basic Usage: Computing Gradients](#basic-usage-computing-gradients)
    - [Neural Network Example](#neural-network-example)
  - [API Reference](#api-reference)
    - [Value Class](#value-class)
    - [Neural Network Classes](#neural-network-classes)
  - [Examples](#examples)
  - [Testing](#testing)
    - [Prerequisites](#prerequisites)
    - [Running Tests](#running-tests)
  - [Educational Resources](#educational-resources)
    - [Key Concepts](#key-concepts)
  - [License](#license)
  - [References](#references)

---

## Features

✨ **Core Capabilities:**
- **Automatic Differentiation**: Reverse-mode autodiff for scalar computations
- **Dynamic Computation Graphs**: DAG is built on-the-fly during forward pass
- **Elementary Operations**: Addition, multiplication, power, ReLU activation
- **Neural Network Components**: Neurons, layers, and modules with PyTorch-like interface
- **Gradient Computation**: Full backpropagation support with automatic gradient calculation
- **Minimal Codebase**: Lightweight implementation perfect for learning

---

## Installation

Install micrograd via pip:

```bash
pip install micrograd
```

---

## Project Structure

```
micrograd/
├── README.md                      # Project documentation
├── dag.png                        # Computational graph visualization example
├── src/                           # Source code directory
│   ├── __init__.py               # Package initialization
│   ├── engine.py                 # Core autograd engine (Value class)
│   └── nn.py                     # Neural network components (Module, Neuron, Layer)
├── notebooks/                    # Educational notebooks
│   └── autograd_build_along.ipynb # Step-by-step autograd implementation
└── test/                         # Test suite
    └── test_engine.py            # Unit tests with PyTorch verification
```

### Key Files

**`engine.py`** - Autograd Engine
- `Value`: Core class representing scalar values in the computation graph
- Supports: addition, multiplication, exponentiation, ReLU activation
- Implements automatic differentiation via the `backward()` method

**`nn.py`** - Neural Network Library
- `Module`: Base class for all neural network components
- `Neuron`: Single neuron with weights, bias, and optional ReLU activation
- `Layer`: Collection of neurons producing multiple outputs often used as building block for MLPs

**`test_engine.py`** - Test Suite
- Validates correctness against PyTorch implementations
- Tests forward and backward pass consistency

---

## Getting Started

### Basic Usage: Computing Gradients

Here's a simple example showing how to use the autograd engine:

```python
from micrograd.engine import Value

# Create scalar values
a = Value(-4.0)
b = Value(2.0)

# Build a computation graph
c = a + b
d = a * b + b**3

# More complex computations
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()

# Final result
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

# Forward pass (automatic)
print(f'{g.data:.4f}')  # Output: 24.7041

# Backward pass (compute gradients)
g.backward()

# Access gradients
print(f'{a.grad:.4f}')  # Output: 138.8338 (dg/da)
print(f'{b.grad:.4f}')  # Output: 645.5773 (dg/db)
```

**What's happening:**
1. Values are wrapped in the `Value` class to track the computation graph
2. Operations (`+`, `*`, `**`, `relu()`) build the DAG automatically
3. `backward()` traverses the graph and computes gradients via chain rule
4. Gradients are accumulated in the `.grad` attribute of each Value

---

### Neural Network Example

Build and use a simple neuron:

```python
from micrograd.engine import Value
from micrograd.nn import Neuron

# Create a neuron that takes 2 inputs
neuron = Neuron(2)

# Create input values
x = [Value(1.0), Value(-2.0)]

# Forward pass through the neuron
output = neuron(x)

# The neuron applies: output = ReLU(w1*x1 + w2*x2 + b)
print(output.data)

# Backward pass
output.backward()

# Access gradients of parameters
for i, w in enumerate(neuron.parameters()):
    print(f'Gradient of parameter {i}: {w.grad}')
```

---

## API Reference

### Value Class

**Constructor:**
```python
Value(data, _children=(), _op="")
```
- `data`: Scalar numeric value
- `_children`: Tuple of parent nodes in the graph
- `_op`: String representing the operation

**Methods:**
- `backward()`: Compute gradients via backpropagation
- `relu()`: Apply ReLU activation

**Attributes:**
- `data`: The scalar value
- `grad`: Accumulated gradient (initialized to 0.0)

**Supported Operations:**
```python
a + b          # Addition
a * b          # Multiplication  
a ** n         # Power (n must be int/float)
a.relu()       # ReLU activation: max(0, a)
-a             # Negation (via multiplication by -1)
a - b          # Subtraction (via a + (-b))
a / b          # Division (via a * (b ** -1))
```

### Neural Network Classes

**Module** - Base class for all components
```python
zero_grad()      # Reset all parameter gradients to 0
parameters()     # Return list of learnable parameters
```

**Neuron** - Single artificial neuron
```python
Neuron(nin, nonlin=True, dropout=0.0)
```
- `nin`: Number of input connections
- `nonlin`: Use ReLU activation (default: True)
- `dropout`: Dropout rate (0.0 to 1.0)

**Layer** - Collection of neurons
```python
Layer(nin, nout, **kwargs)
```
- `nin`: Number of inputs per neuron
- `nout`: Number of neurons in layer

---

## Examples

See the [notebooks](notebooks/) directory for comprehensive examples:

- **`autograd_build_along.ipynb`**: Step-by-step guide building an autograd engine from scratch

---

## Testing

Micrograd includes a test suite that validates correctness against PyTorch implementations.

### Prerequisites

Install PyTorch (used as reference for validation):
```bash
pip install torch
```

### Running Tests

```bash
python -m pytest
```

The test suite includes:
- **`test_sanity_check()`**: Basic forward/backward pass validation
- **`test_more_ops()`**: Complex multi-operation computation graphs

Each test verifies that both the forward pass and backward pass produce identical results to PyTorch.

---

## Educational Resources

This project is ideal for learning:

- 📚 **How automatic differentiation works** - See how PyTorch computes gradients
- 🔗 **Computational graphs** - Understand DAGs and topological sorting
- 🧮 **Backpropagation algorithm** - Implement the chain rule efficiently
- 🤖 **Neural network fundamentals** - Build neurons and layers from scratch
- 💻 **Python metaprogramming** - Operator overloading and functional patterns

### Key Concepts

**Forward Pass**: Computation flows forward through the DAG, building the graph structure

**Backward Pass**: Gradients flow backward using the chain rule:
$$\frac{\partial L}{\partial x} = \sum_{\text{paths}} \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$$

**Computational Graph**: A DAG where nodes are Values and edges represent operations

---

## License

Open source - feel free to use for learning and educational purposes.

---

## References

Inspired by modern autograd engines like PyTorch, TensorFlow, and JAX. Perfect companion to understanding deep learning frameworks!
