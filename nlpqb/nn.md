---
layout: page
title: Neural Networks
---
### Why are neural networks called so?
Their origins lies in McCulloch-Pitts neuron, a simplified model of the human neuron as a kind of computing element that could be described in terms of propositional logic.

### Why is feedforward network called so?
Because the computation proceeds iteratively from one layer of units to the next.

### Why is deep learning called so?
Because modern networks are often deep (have many layers).

### How is `z` computed?
`x.w + b`

### What are some activation functions?
```
sigmoid(z) = 1/(1 + e^-z)
tanh(z) = (e^z - e^-z)/(e^z + e^-z)
ReLU(z) = max(z, 0)
```

### Why is ReLU preferred over sigmoid and tanh?
At very high values of z, sigmoid(z) and tanh(z) are saturated (extremely close to 1) and so the derivatives are close to zero. This creates a problem called vanishing gradient. ReLU doesn't have this problem; their derivative is 1 even at very high values of z.

### What is perceptron?
Perceptron is a very simple neural unit that has a binary output and does not have a non-linear activation function.

### What is the XOR problem?
XOR is not a lineatly separable function and so can't be calculated by a single perceptron. It needs a layered network of units.

### Why is multi-layer perceptrons a misnomer for feedforward networks?
Perceptrons are purely linear while modern feedforward networks are made up of non-linear activation functions.

### Give the formulae for two-layer neural network
```
a0 = x
z1 = a0.W1 + b1
a1 = g1(z1)
z2 = a1.W2 + b2
a2 = g2(z2)
yhat = a2
```

### For a text with n input work tokens w0, w1, ..., and wn-1, what is the equation for sentiment classifier with two layers?
```
x = [e_w0; e_w1; e_w2; ...; e_wn-1]
h = g(x.W + b)
z = h.U
y = softmax(z)
```

### What are the key concepts involved in calculating gradient for neural networks?
computation graphs, chain rule, error backpropagation (backward differentiation on a computation graph)

### What is one of the most important ways to regularize neural networks?
Dropout

### How do neural language models use neural networks?
Neural language models use neural network as a probabilistic classifier to compute the probability of the next word given the previous n words. They can use pretrained embeddings or can learn embeddings from scratch in the process of language modeling.

### What is fully connected feedforward network?
In a fully connected, feedforward network, each unit in layer i is connected to each unit in layer i+1 and there are no cycles.

### Where does the power of neural networks come from?
The power of neural networks comes from the ability of early layers to learn representations that can be utilized by later layers in the network.