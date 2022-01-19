---
layout: page
title: Automatic Differentiation
---

### What is automatic differentiation?
Deep learning frameworks expedite this work by automatically calculating derivatives, i.e., automatic differentiation. In practice, based on our designed model the system builds a computational graph, tracking which data combined through which operations to produce the output. Automatic differentiation enables the system to subsequently backpropagate gradients. Here, backpropagate simply means to trace through the computational graph, filling in the partial derivatives with respect to each parameter.

### How do you compute dy/dx where `y = 2 * torch.dot(x, x)`?
```
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
y = 2 * torch.dot(x, x)
y.backward()
x.grad
assert x.grad == 4 * x
```

### How do you do backward for non-scalar variables?
```
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

### How do you detach computation?
Sometimes, we wish to move some calculations outside of the recorded computational graph. For example, say that y was calculated as a function of x, and that subsequently z was calculated as a function of both y and x. Now, imagine that we wanted to calculate the gradient of z with respect to x, but wanted for some reason to treat y as a constant, and only take into account the role that x played after y was calculated.

Here, we can detach y to return a new variable u that has the same value as y but discards any information about how y was computed in the computational graph. In other words, the gradient will not flow backwards through u to x. Thus, the following backpropagation function computes the partial derivative of z = u * x with respect to x while treating u as a constant, instead of the partial derivative of z = x * x * x with respect to x.

```
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
assert x.grad == u
```

### How do you compute `a.grad` given a python function `f(a)`?
```
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

