# PINN-AUTOGRAD

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Dependencies - None](https://img.shields.io/badge/dependencies-numpy%20-blue)

A lightweight, dependency-free (purely numpy) Physics-Informed Neural Network (PINN) framework built completely from scratch using a custom reverse-mode automatic differentiation engine.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [PINN Module](#pinn-module)
    - 4.1. [PINN Algorithm](#pinn-algorithm)
    - 4.2. [PINN Class](#pinn-class)
    - 4.3. [PINN Features](#pinn-features)
    - 4.4. [PINN Examples](#pinn-examples)
5. [AUTOGRAD Module](#autograd-module)
    - 5.1. [Autograd Algorithm](#autograd-algorithm)
    - 5.2. [Var Class](#var-class)
    - 5.3. [Autograd Features](#autograd-features)
    - 5.4. [Autograd Examples](#autograd-examples)
6. [Resources & References](#resources--references)
7. [License](#license)


## Features

- **Zero dependencies** -- Built entirely on `numpy`, no torch/jax required
- **Matrix native** -- Batch support out of the box, no scalar limitations
- **$n^{th}$ order derivatives** -- $\frac{d^2u}{dx^2}$, $\frac{d^3u}{dx^3}$ for differential equation solving
- **PINN ready** -- Clean `.diff()` API for derivatives
- **Supports PDE** -- Partial Derivatives possible
- **Extensible** -- Add any custom differentiable operator via `ufunc`/`bifunc`
- **Adam / SGD** -- Optimizers built in, custom optimizers via kwargs
- **Save/Load** -- Persist trained models with numpy `.npz`


# Installation

```bash
pip install pinn-autograd
```

# Quickstart

```python
from PINN import Var, PINN
import numpy as np
```


# PINN Module 
---


## PINN Algorithm

Physics Informed Neural Network is build on top of Neural Network with two additional things:

- Custom Loss function:  $L_{Total} = L_{Data}+\lambda\cdot L_{Physics}$  where $\lambda$= importance of physics loss
- Collocation Points: Unlabeled data points sampled within the physical domain that enable the PINN to train on the physics loss even when the true target value at those points is completely unknown.


## PINN class

An out-of-the-box class designed to easily instantiate and optimize physics-informed layouts.
```python
# Instantiate with specified architecture and layer activations
>>> PINN(layers=[1,20,20,1], act=["ReLU","ReLU","softplus"])

# Train with physics residuals, boundary constraints, and collocation points
>>> PINN.train(phy_eq, train_x, train_y, collocation_x, epoch, lr)

# Infer over your test domain fields
>>> PINN.predict(x_test)

# Benchmark utility to train a standard data-only neural network
>>> PINN.train_nn(train_x, train_y, epoch, lr) 

# Save trained model 
>>> PINN.save('pinn_model_1') # saves to pinn_model_1.npz
```


## PINN Features

Extensive Features build on top of autograd system
- He and Xavier initializers
- Native ADAM and SGD optimizers
- Built-in ReLU,tanh and softplus activation functions
- Custom differential equation using Var class
- Dynamic configuration via kwargs.
- train_nn option to benchmark against standard data-only neural network.
- save/load model


## PINN Examples



#### Nuclear Decay 

```python
# Nuclear decay:  dN/dt + 3*N = 0
def decay(model,t):
    N=model.forward(t)
    N_t=N.diff(t)
    res=N_t + 3*N
    return res**2

# Boundary condition/Data
t=Var(data = np.array([0,2,4]), name = 't')
N=Var(data = 10*np.exp(-3*t.data), name = 'N')

# Collocation points
t_coll=Var(data = np.linspace(0,4,500), name = 'collocation t')
N_coll=10*np.exp(-3*t_coll.data) # true data

# PINN model
pinn_model=PINN([1,20,20,1],['softplus','softplus'])
# Training
pinn_model.train(decay,t,N,t_coll,epochs=600,lr=0.01,lambda_p=0.90)

output = pinn_model.predict(t_coll).data
pinn_model.save('PINN_decay') #saves model in `PINN_decay.npz`
```


#### Wave 

```python
# wave equation: d2u/dt2 - d2u/dx2 = 0 ; u(x,t) ; c=1
def wave(model, xt):
    x,t = xt.col(0),xt.col(1)
    xt = Var.stack(x,t) # reconstructing to include x,t in computational graph
    u = model.forward(xt)
    
    u_x, u_t = u.diff(x), u.diff(t)
    u_xx, u_tt = u_x.diff(x), u_t.diff(t)
    
    res = u_tt - u_xx
    return res**2


x,t = np.linspace(0,1,20).reshape(-1,1),np.linspace(0,1,20).reshape(-1,1)

# Boundary condition
x0,x1 = np.zeros_like(x), np.ones_like(x) # x=0 and x=1
t0 = np.zeros_like(t) # time = 0

xt_bc_1 = np.hstack([x1,t]) # x=1, t=0,1
xt_bc_2 = np.hstack([x0,t]) # x=0, t=0,1

u0 = np.zeros_like(x) # u=0 for  x=0 and 1, t=0,1 

u_init = np.sin(np.pi * x) # u=sin(x*pi) for  x=0,1 , t=0
xt_init =  np.hstack([x,t0]) 

xt_bc = Var(data = np.vstack([xt_init,xt_bc_1,xt_bc_2]), name = 'xt boundary points')
u_bc = Var(data = np.vstack([u_init,u0,u0]), name = 'u boundary points')

# Collocation points
t_mesh, x_mesh = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))

xt_coll = np.hstack([x_mesh.flatten().reshape(-1, 1), t_mesh.flatten().reshape(-1, 1)])
xt_coll = Var(xt_coll)
x, t  = Var(xt_coll).col(0), Var(xt_coll).col(1) 

wave_pinn = pinn([2, 20,20,20, 1], ['softplus','softplus','softplus'])

wave_pinn.train(wave_eq, xt_bc, u_bc , xt_coll, 
                    epochs=1000, lr=0.01, lambda_p=0.9)
wave_pinn.save('PINN_wave') #saves model in `PINN_wave.npz`
```


# AUTOGRAD Module
---


## Autograd Algorithm


The AUTOGRAD engine treats every mathematical operation as a node in a dynamic directed acyclic graph (DAG).

- Forward Pass: As calculations are performed, the engine builds the graph in real time. Each Var node stores its raw numeric values (.data) and a reference to its creator function (parent).

- Local Gradients: Along with the data, the engine immediately computes and stores the local gradient; the derivative of that specific operation's output with respect to its immediate inputs.

- Backward Pass (Reverse-Mode AD): When evaluating gradients, the engine traverses the graph in reverse, starting from the output. It systematically multiplies the incoming upstream gradient by the stored local gradient at each node to accumulate the exact global gradient across the entire network via the chain rule.

Example:
```
    x = a + b
    y = k * j
    z = x * y
    
    During forward pass we store all the parent and its child
    
    x: parent  a,b: child
    y: parent  k,j: child
    z: parent  x,y: child
    
                            a───┐             
                              + ├────x───┐    
                            b───┘        │    
                                       * ├───z
                            k───┐        │    
                              * ├────y───┘    
                            j───┘             					                
    
    We calculate local gradeints
    
    dx/da = 1   dx/db = 1
    dy/dk = j   dy/dj = k
    dz/dx = y   dz/dy = x
    
    To get global gradient we multiply local gradients (chain rule)
    
    dz/da = dz/dx * dx/da = y * 1 = y
    dz/dj = dz/dy * dy/dj = x * k
    dz/dk = dz/dy * dy/dk = x * j
``` 


## Var class

Var class uses Autograd algorithm and defines forward and backward pass function for any differentiable operation.
It has many predefined functions such as :
```python
>>> Var.sin()
>>> Var.exp()
>>> Var1 @ Var2 # matrix multiplication
>>> Var1 * Var2
>>> Var.ReLU() # ReLU activation function
>>> Var1.concat(Var2) # give new matrix of [Var1,Var2]
>>> Var.mean() 

```


## Autograd Features


- Build custom differentiable functions using bifunc/ufunc
- $n^{th}$ order derivative supported
- Supports partial derivatives via `Var.col()`,`Var.stack()`
- Clean `.diff()` API


### bifunc/ufunc

Var class has two main function that let's you create any differentiable function
- bifunc: defines a binary function using two input, forward and backward pass logic. 
- ufunc: defines unary function using one input,forward and backward pass logic.

```python
# BINARY FUNCTION
def bifunc(self,other,forward,backward_self,backward_other)->'Var':
    """
    creates binary function
    
    Parameters:
    ----------
    self,other:Var
    forward: function(self,other) -> numpy.ndarray|list|int|float :
		     binary operation on self.data, other.data
		     
    backward_self: function(self,other,parent) -> Var :
			 returns (local gradient of self) * (parent.grad) 
			 
    backward_other: function(self,other,parent) -> Var :
		     returns (local gradient of other) * (parent.grad)
    
    > parent.grad (upstream parent gradient) is required to follow the chain rule during backward pass
    
    Returns:
    -------
        parent: Var
            output after performing operation on self,other
    
    Example:
    -------
        >>> def mul(var1, var2):
        ...     # x = v1 * v2 , dx/dv1 = v2 , dx/dv2 = v1
        ...     return var1.bifunc(
        ...         var2,
        ...         forward = lambda v1, v2: v1.data * v2.data,
        ...         backward_self = lambda v1, v2, x: v2 * x.grad,
        ...         backward_other = lambda v1, v2, x: v1 * x.grad)
        >>> data = mul(Var(4), Var(2)) # 4*2
    """
        
# UNARY FUNCTION    
def ufunc(self, forward, backward,**kwargs) -> 'Var':
    """
    creates unary function
    
    Parameters:
    ----------
    self: Var
    forward:  function(data) -> numpy.ndarray|list|int|float : 
				apply forward function on self.data
				
    backward: function(self,parent) -> Var :
			    returns (local gradeint of self) * (parent.grad)
    
    > parent.grad (upstream parent gradient) is required to follow the chain rule during backward pass
    
    Returns:
    -------
        parent: Var
            output after performing operation on self
     
    Example:
    -------
        >>> # x = exp(v1) , dx/dv1 = x
        >>> exp = lambda x: x.ufunc(
        ...               forward = lambda v1 : np.exp(v1.data), 
        ...                backward = lambda v1,x: x * x.grad)
        >>> data = exp(Var(42))  
    """
```


## Example

```python
x = Var(2*np.pi, 'x')
y = x.sin().exp()         # exp(sin(2π)) = exp(0) = 1
dy_dx = y.diff(x)         # cos(2π)*exp(sin(2π)) = 1*1 = 1
d2y_dx2 = dy_dx.diff(x)   # cos²(2π)*exp(sin(2π)) - sin(2π)*exp(sin(2π)) = 1 - 0 = 1

print(dy_dx)    # 1.0
print(d2y_dx2)  # 1.0
```
```python
# Partial derivative
x = Var(np.linspace(1,10,20))
t = Var(np.linspace(1,10,20))
xt = Var.stack(x,t) # [x,t] matrix

xt_square = xt**2  # [x**2,t**]
dxt_square_dx = xt_square.diff(x) # 2*x
dxt_square_dt = xt_square.diff(t) # 2*t

print(dxt_square_dx)
print(dxt_square_dt)
```


# References
---
- **An Expert's Guide to Training Physics-informed Neural Networks**: Sifan Wang et. al. , [arXiv:2308.08468](https://arxiv.org/abs/2308.08468)

* **Automatic Differentiation**: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). *Automatic differentiation in machine learning: a survey.* ,[	arXiv:1502.05767](https://doi.org/10.48550/arXiv.1502.05767)

* [Micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd) – A foundational educational resource used to understand reverse-mode automatic differentiation graph building.

* Youtube series by **StatQuest with Josh Starmer** on [Neural Networks/ Deep Learning](https://youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&si=g4RJc4cDU0YQw2-X)


# License
---
This project is open-source and licensed under the terms of the **MIT License**.
