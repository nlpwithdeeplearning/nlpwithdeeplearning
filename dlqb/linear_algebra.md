---
layout: page
title: Linear Algebra
---

### What is a scalar?
A scalar is represented by a tensor with just one element. Formally, we call values consisting of just one numerical quantity scalars.

### What is a vector?
You can think of a vector as simply a list of scalar values. We call these values the elements (entries or components) of the vector.

### What is the length or the dimensionality of the vector?
A vector is just an array of numbers. And just as every array has a length, so does every vector. In math notation, if we want to say that a vector  𝐱  consists of  𝑛  real-valued scalars, we can express this as  𝐱 ∈ ℝ^𝑛. The length of a vector is commonly called the dimension of the vector.

As with an ordinary Python array, we can access the length of a tensor by calling Python’s built-in len() function.

When a tensor represents a vector (with precisely one axis), we can also access its length via the .shape attribute. The shape is a tuple that lists the length (dimensionality) along each axis of the tensor. For tensors with just one axis, the shape has just one element.

### What is dimensionality?
we use the dimensionality of a vector or an axis to refer to its length, i.e., the number of elements of a vector or an axis. However, we use the dimensionality of a tensor to refer to the number of axes that a tensor has. In this sense, the dimensionality of some axis of a tensor will be the length of that axis.

### What is a matrix?
Just as vectors generalize scalars from order zero to order one, matrices generalize vectors from order one to order two. Matrices, which we will typically denote with bold-faced, capital letters (e.g.,  𝐗 ,  𝐘 , and  𝐙 ), are represented in code as tensors with two axes.

### What is a square matrix?
when a matrix has the same number of rows and columns, its shape becomes a square; thus, it is called a square matrix.

### What is transpose?
When we exchange a matrix’s rows and columns, the result is called the transpose of the matrix. In Pytorch, `A.T`

### What is a symmetric matrix?
As a special type of the square matrix, a symmetric matrix  𝐀  is equal to its transpose. `A == A.T`

### What are tensors as algebraic objects?
Just as vectors generalize scalars, and matrices generalize vectors, we can build data structures with even more axes. Tensors (“tensors” in this subsection refer to algebraic objects) give us a generic way of describing  𝑛 -dimensional arrays with an arbitrary number of axes. Vectors, for example, are first-order tensors, and matrices are second-order tensors. Tensors are denoted with capital letters of a special font face (e.g.,  𝖷 ,  𝖸 , and  𝖹 ) and their indexing mechanism is similar to that of matrices.

### Create a tensor with the shape (2, 3, 4) and has numbers 0 through 23.
`torch.arange(24).reshape(2, 3, 4)`

### What is Hadamard product?
Elementwise multiplication of two matrices is called their Hadamard product (math notation  ⊙ ). In Pytorch, it is `A * B`

### What is ∑ or sum() in Pytorch?
In mathematical notation, we express sums using the  ∑  symbol. In code, we can just call the function for calculating the sum. `x.sum()`

### By default, invoking the function for calculating the sum reduces a tensor along all its axes to a scalar. We can also specify the axes along which the tensor is reduced via summation. Show an example.
To reduce the row dimension (axis 0) by summing up elements of all the rows, we specify axis=0 when invoking the function. Since the input matrix reduces along axis 0 to generate the output vector, the dimension of axis 0 of the input is lost in the output shape.

`A.sum(axis=0)`

Specifying axis=1 will reduce the column dimension (axis 1) by summing up elements of all the columns. Thus, the dimension of axis 1 of the input is lost in the output shape.

`A.sum(axis=1)`

Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.

### What is mean?
A related quantity is the mean, which is also called the average. We calculate the mean by dividing the sum by the total number of elements. In code, we could just call the function for calculating the mean on tensors of arbitrary shape.

`A.mean(), A.sum() / A.numel()`
`A.mean(axis=0), A.sum(axis=0) / A.shape[0]`

### Sometimes it can be useful to keep the number of axes unchanged when invoking the function for calculating the sum or mean. Show how it can be done.
`A.sum(axis=1, keepdims=True)`
For an example application, since sum_A still keeps its two axes after summing each row, we can divide A by sum_A with broadcasting.
`A / sum_A`

### Calculate the cumulative sum of elements of A along some axis, say axis=0 (row by row)
`A.cumsum(axis=0)`

### What is dot product?
Given two vectors  𝐱,𝐲∈ℝ𝑑 , their dot product  𝐱⊤𝐲  (or  ⟨𝐱,𝐲⟩ ) is a sum over the products of the elements at the same position:  transpose(x)*y.

`torch.dot(x, y)` or `torch.sum(x * y)`

### What is matrix-vector product?
We can think of multiplication by a matrix  𝐀∈ℝ𝑚×𝑛  as a transformation that projects vectors from  ℝ𝑛  to  ℝ𝑚 . These transformations turn out to be remarkably useful. For example, we can represent rotations as multiplications by a square matrix. As we will see in subsequent chapters, we can also use matrix-vector products to describe the most intensive calculations required when computing each layer in a neural network given the values of the previous layer.

`torch.mv(A, x)`

### What is matrix-matrix multiplication?
We can think of the matrix-matrix multiplication  𝐀𝐁  as simply performing  𝑚  matrix-vector products and stitching the results together to form an  𝑛×𝑚  matrix. In the following snippet, we perform matrix multiplication on A and B. Here, A is a matrix with 5 rows and 4 columns, and B is a matrix with 4 rows and 3 columns. After multiplication, we obtain a matrix with 5 rows and 3 columns.

`torch.mm(A, B)`

### What is a norm?
Informally, the norm of a vector tells us how big a vector is. In linear algebra, a vector norm is a function  𝑓  that maps a vector to a scalar, satisfying a handful of properties:
1. if we scale all the elements of a vector by a constant factor  𝛼 , its norm also scales by the absolute value of the same constant factor: 𝑓(𝛼𝐱)=|𝛼|𝑓(𝐱)
1. triangle inequality: 𝑓(𝐱+𝐲)≤𝑓(𝐱)+𝑓(𝐲)
1. norm must be non-negative: 𝑓(𝐱)≥0
1. the smallest norm is achieved and only achieved by a vector consisting of all zeros: ∀𝑖,[𝐱]𝑖=0⇔𝑓(𝐱)=0

### 