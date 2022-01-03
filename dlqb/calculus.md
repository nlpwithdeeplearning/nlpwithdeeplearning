---
layout: page
title: Calculus
---

### What are the two kinds of calculus?
1. Integral Calculus
1. Differential Calculus

### How can are the two key concerns of the task of fitting models?
1. optimization: the process of fitting our models to observed data
1. generalization: the mathematical principles and practitioners’ wisdom that guide as to how to produce models whose validity extends beyond the exact set of data examples used to train them

### What is the derivative f' of f: R -> R?
A derivative can be interpreted as the instantaneous rate of change of a function with respect to its variable. It is also the slope of the tangent line to the curve of the function.

f'(x) = limit((f(x+h)-f(x))/h, h, 0)

### What is the derivative of C?
0

### What is the derivative of x^n?
n*x^(n-1)

### What is the derivative of e^x?
e^x

### What is the derivative of ln(x)?
1/x

### What is the derivative of C*f(x)?
C*derivative(f(x))

### What is the derivative of f(x)+g(x)?
derivative(f(x)) + derivative(g(x))

### What is the derivative of f(x)*g(x)?
f(x)*derivative(g(x)) + derivative(f(x))*g(x)

### What is the derivative of f(x)/g(x)?
(g(x)*derivative(f(x)) - f(x)*derivative(g(x)))/g(x)**2

### Plot the function  𝑢=𝑓(𝑥)  and its tangent line  𝑦=2𝑥−3  at  𝑥=1 , where the coefficient  2  is the slope of the tangent line
```
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

### What is a partial derivative?
Let  𝑦=𝑓(𝑥1,𝑥2,…,𝑥𝑛)  be a function with  𝑛  variables. The partial derivative of  𝑦  with respect to its  𝑖th  parameter  𝑥𝑖  is limit((f(x1,..x_i +h, ...) - f(x1,...x_i,..))/h, h, 0). We simply treat  𝑥1,…,𝑥𝑖−1,𝑥𝑖+1,…,𝑥𝑛  as constants and calculate the derivative of  𝑦  with respect to  𝑥𝑖 .

### What is a gradient?
We concatenate partial derivatives of a multivariate function with respect to all its variables to obtain the gradient vector of the function.

Let  𝐱  be an  𝑛 -dimensional vector, the following rules are often used when differentiating multivariate functions:

For all  𝐀∈ℝ𝑚×𝑛 ,  ∇𝐱𝐀𝐱=𝐀⊤ ,

For all  𝐀∈ℝ𝑛×𝑚 ,  ∇𝐱𝐱⊤𝐀=𝐀 ,

For all  𝐀∈ℝ𝑛×𝑛 ,  ∇𝐱𝐱⊤𝐀𝐱=(𝐀+𝐀⊤)𝐱 ,

∇𝐱‖𝐱‖2=∇𝐱𝐱⊤𝐱=2𝐱 .

Similarly, for any matrix  𝐗 , we have  ∇𝐗‖𝐗‖2𝐹=2𝐗 .

### What is the chain rule?
The chain rule enables us to differentiate composite functions.

Suppose that functions  𝑦=𝑓(𝑢)  and  𝑢=𝑔(𝑥)  are both differentiable, then the chain rule states that 𝑑𝑦/𝑑𝑥=𝑑𝑦/𝑑𝑢 * 𝑑𝑢/𝑑𝑥

Suppose that the differentiable function  𝑦  has variables  𝑢1,𝑢2,…,𝑢𝑚 , where each differentiable function  𝑢𝑖  has variables  𝑥1,𝑥2,…,𝑥𝑛 . Note that  𝑦  is a function of  𝑥1,𝑥2,…,𝑥𝑛 . Then the chain rule gives: 𝑑𝑦/𝑑𝑥𝑖 = 𝑑𝑦/𝑑𝑢1 * 𝑑𝑢1/𝑑𝑥𝑖 + 𝑑𝑦/𝑑𝑢2 * 𝑑𝑢2/𝑑𝑥𝑖 +⋯+ 𝑑𝑦/𝑑𝑢𝑚 * 𝑑𝑢𝑚/𝑑𝑥𝑖