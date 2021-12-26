---
layout: page
title: Sequence Processing
---

### What are two deep learning architectures for sequence processing?
Recurrent neural networks and transformers.

### What is an Elman network?
An Elman network is the same as Simple recurrent neural network. The hidden layer includes a recurrent connection as part of its input. The activation value of the hidden layer depends on the current input as well as the activation value of the hidden layer from the previous time step.

### What is backpropagation through time?
We process the sequence in reverse, computing the required gradients as we go, computing and saving the error term for use in hidden layer for each step backward in time.

### How can RNN be used for language models?
RNN language models process the input sequence one word at a time, attempting to predict the next word from the current word and the previous state.

### Why don't the RNNs have the limited context problem that n-gram models have?
The hidden state can in principle represent information about all of the preceeding words all the way back to the beginning of the sequence.

### What is teacher forcing in language modeling?
The idea that we always give the model the correct history sequence to predict the next word (rather than feeding the model its best case from the previous time step) is called the teacher forcing.

### What is weight tying in RNN-based language models?
Weight tying is a method that use a single set of embeddings at the input and softmax layers. This improves perplexity and also reduces the number of parameters required to learn.

### What are the different tasks in which RNN can be applied?
Using the plain RNN, we can do language modeling, sequence classification tasks like sentiment analysis and topic classification, sequence labeling tasks like part of speech tagging, and text generation tasks.

Using encoder-decoder approaches, we can do summarization, machine translation, and question answering.

### How is RNN used for sequence labeling (e.g., part of speech tagging)?
Pre-trained word embeddings serve as inputs and a softmax layer provides a probability distribution over the sequence tags (e.g., part of speech tags) as output at each time step.

### How is RNN used for sequence classification?
The final hidden state from the RNN is used as the input to a feedforward network that performs the classification.

### What is end-to-end training?
The training regimen that uses the loss from a downstream application to adjust the weights all the way through the network.

### How is pooling function used for classification?
Pooling function is used on all hidden states as an alternative to just using the last hidden state as input for feedforward network that does the actual classification.

### What is autoregressive generation?
Using a language model to incrementally generate words by repeatedly sampling the next word conditioned on previous choices.

### What are stacked RNNs?
Stacked RNNs consist of multiple networks where the output of one layer serves as the input to a subsequent layer.

### What is a bidirectional RNN?
A bidirectional RNN consists of two independent RNNs, one where the input is processed from the start to the end, and the other from the end to the start. We concatenate the representations computed by these two networks for each token so that both left and right contexts are captured at each point of time.

### What is a problem with Simple RNNs?
There are two main problems:
1. the hidden layers, and by extension the weights that determine the values in the hidden layer, are being asked to perform two different tasks simultaneously. They need to provide information useful for the current decision. They also need to be useful for updating and carrying forward information required for future decisions.
1. while backpropagating through time, the hidden layers are subject to repeated multiplication over the length of the sequence. This eventually drives gradients to zero causing the vanishing gradients problem.

### How do LSTMs address the issues with Simple RNNs?
LSTMs add an explicit context layer to the architecture through the use of specialized neural units that make use of gates to control the flow of information into and out of the units that comprise the network layers.

### What is a forget gate?
The purpose of the forget gate is to delete information from the context that is no longer needed.
f_t = sigmoid(h_(t-1).U_f + x_t.W_f)
k_t = elementwise_multiplication(c_(t-1), f_t)

### How do we compute the information useful for current decision in LSTM?
g_t = tanh(h_(t-1).U_g + x_t.W_g)

### How do we use add gate in LSTM?
Add gate is used to select information to add to the current context.
i_t = sigmoid(h_(t-1).U_i + x_t . W_i)
j_t = elementwise_multiplication(g_t, i_t)
c_t = j_t + k_t

### How do we use output gate in LSTM?
The output gate decides what information is required for current hidden state.
o_t = sigmoid(h_(t-1).U_o + x_t.W_o)
h_t = elementwise_multiplication(o_t, tanh(c_t))

### What are some problems with LSTM?
1) Passing information through an extended series of recurrent connections leads to information loss and difficulties in training
1) The inherently sequential nature of recurrent networks makes it hard to computation in parallel.

### What is self-attention?
Self-attention allows a network to directly extract and use information from arbitrarily large contexts without the need to pass it through intermediate recurrent connections in RNNs.
```
Q = X . WQ
K = X . WK
V = X . WV
SelfAttention(Q, K, V) = softmax(Q*transpose(K)/sqrt(d_k)) * V
```

### What is layer norm?
```
mu = sum(x_i, i = 0 to d_(h-1))/d_h, where d_h is the dimensionality of the hidden layer
sigma = sqrt(sum((x_i - mu)^2m i = 0 to d_h - 1)/d_h)
xhat = (x - mu)/sigma
LayerNorm = gamma*xhat + beta
```

### What is a transformer block?
```
z = LayerNorm(x + SelfAttention(x))
y = LayerNorm(z + FFNN(z))
```

### What is multihead attention?
```
Q_i = X.WQ_i
K_i = X.WK_i
V_i = X.WV_i
head_i = SelfAttention(Q, K, V)
MultiHeadAttn(X) = elementwise_add(head_0, head_1, ..., head_(n-1)) . W_o
```

### How do you model word order?
We model word order by combining input embeddings with positional embeddings specific to each position in an input sequence.

### What are the different tasks in which RNN can be applied?
Using the plain transformer, we can do language modeling, sequence classification tasks like sentiment analysis and topic classification, sequence labeling tasks like part of speech tagging, and text generation tasks.

Using encoder-decoder approaches, we can do summarization, machine translation, and question answering.

### How do we apply transformers for summarization?
Append a summary to each full-length article in a corpus with a unique marker separating the two. Train them as long sentences in a langauge model training set up. Use teacher forcing.

### What is finetuning followed by pretraining?
First train a transformer language model on a large corpus of text, in a normal self-supervised way.

Then add a linear or feedforward layer on top that we finetune on a smaller dataset hand-labeled with part-of-speech or sentiment labels.