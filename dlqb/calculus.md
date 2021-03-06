---
layout: page
title: Calculus
---

### What are the two kinds of calculus?
1. Integral Calculus
1. Differential Calculus

### How can are the two key concerns of the task of fitting models?
1. optimization: the process of fitting our models to observed data
1. generalization: the mathematical principles and practitionersβ wisdom that guide as to how to produce models whose validity extends beyond the exact set of data examples used to train them

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

### Plot the function  π’=π(π₯)  and its tangent line  π¦=2π₯β3  at  π₯=1 , where the coefficient  2  is the slope of the tangent line
```
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

### What is a partial derivative?
Let  π¦=π(π₯1,π₯2,β¦,π₯π)  be a function with  π  variables. The partial derivative of  π¦  with respect to its  πth  parameter  π₯π  is limit((f(x1,..x_i +h, ...) - f(x1,...x_i,..))/h, h, 0). We simply treat  π₯1,β¦,π₯πβ1,π₯π+1,β¦,π₯π  as constants and calculate the derivative of  π¦  with respect to  π₯π .

### What is a gradient?
We concatenate partial derivatives of a multivariate function with respect to all its variables to obtain the gradient vector of the function.

Let  π±  be an  π -dimensional vector, the following rules are often used when differentiating multivariate functions:

For all  πββπΓπ ,  βπ±ππ±=πβ€ ,

For all  πββπΓπ ,  βπ±π±β€π=π ,

For all  πββπΓπ ,  βπ±π±β€ππ±=(π+πβ€)π± ,

βπ±βπ±β2=βπ±π±β€π±=2π± .

Similarly, for any matrix  π , we have  βπβπβ2πΉ=2π .

### What is the chain rule?
The chain rule enables us to differentiate composite functions.

Suppose that functions  π¦=π(π’)  and  π’=π(π₯)  are both differentiable, then the chain rule states that ππ¦/ππ₯=ππ¦/ππ’ * ππ’/ππ₯

Suppose that the differentiable function  π¦  has variables  π’1,π’2,β¦,π’π , where each differentiable function  π’π  has variables  π₯1,π₯2,β¦,π₯π . Note that  π¦  is a function of  π₯1,π₯2,β¦,π₯π . Then the chain rule gives: ππ¦/ππ₯π = ππ¦/ππ’1 * ππ’1/ππ₯π + ππ¦/ππ’2 * ππ’2/ππ₯π +β―+ ππ¦/ππ’π * ππ’π/ππ₯π