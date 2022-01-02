---
layout: page
title: Data Manipulation
---

### Create a vector of evenly spaced values, starting at 0 (included) and ending at n (not included).
`x = torch.arange(12, dtype=torch.float32)`

### Access a tensor’s shape (the length along each axis).
`x.shape`

### Know the total number of elements in a tensor
`x.numel()`

### Transform tensor x from a row vector with shape (12,) to a matrix with shape (3, 4)
`x = x.reshape(3, 4)`
or
`x.reshape(-1, 4)`
or
`x.reshape(3, -1)`

### Create a tensor with all elements set to 0 and a shape of (2, 3, 4)
`torch.zeros((2, 3, 4))`

### Create a tensor with all elements set to 1 and a shape of (2, 3, 4)
`torch.ones((2, 3, 4))`

### Create a tensor with shape (3, 4) with each of its elements randomly sampled from a standard Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1
`torch.randn(3, 4)`

### Create a tensor by supplying a list of lists
`torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])`

### The common standard arithmetic operators (+, -, *, /, and **)
`x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation`

### unary operators like exponentiation
`torch.exp(x)`

### Show what happens when we concatenate two matrices along rows (axis 0, the first element of the shape) vs. columns (axis 1, the second element of the shape)
`torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)`

### Sometimes, we want to construct a binary tensor via logical statements. Show an example
`X == Y`

### Summing all the elements in the tensor yields a tensor with only one element.
`X.sum()`

### What is broadcasting mechanism?
Under certain conditions, even when shapes differ, we can still perform elementwise operations by invoking the broadcasting mechanism. This mechanism works in the following way: First, expand one or both arrays by copying elements appropriately so that after this transformation, the two tensors have the same shape. Second, carry out the elementwise operations on the resulting arrays.

### If `a = torch.arange(3).reshape((3, 1))` and `b = torch.arange(2).reshape((1, 2))`, what is `a+b`?
tensor([[0, 1],
        [1, 2],
        [2, 3]])

### How do you select the last element or select the second and the third element?
`X[-1], X[1:3]`

### What does `X[1, 2] = 9` do?
Writes elements of a matrix by specifying indices.

### What does `X[0:2, :] = 12` do?
If we want to assign multiple elements the same value, we simply index all of them and then assign them the value.