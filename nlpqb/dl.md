---
layout: page
title: Deep Learning Fundamentals
---

## Attention
Attention is a technique for attending to different parts of an input vector to capture long-term dependencies. Within the context of NLP, traditional sequence-to-sequence models compressed the input sequence to a fixed-length context vector, which hindered their ability to remember long inputs such as sentences. In contrast, attention creates shortcuts between the context vector and the entire source input. Below you will find a continuously updating list of attention based building blocks used in deep learning.

### Multi-Head Attention
Multi-head Attention is a module for attention mechanisms which runs through an attention mechanism several times in parallel. The independent attention outputs are then concatenated and linearly transformed into the expected dimension. Intuitively, multiple attention heads allows for attending to parts of the sequence differently (e.g. longer-term dependencies versus shorter-term dependencies). Note that scaled dot-product attention is most commonly used in this module, although in principle it can be swapped out for other types of attention mechanism.

### Scaled Dot-Product Attention
Scaled dot-product attention is an attention mechanism where the dot products are scaled down by sqrt(d_k).

## Optimization Algorithms

### Adam
Adam is an adaptive learning rate optimization algorithm that utilises both momentum and scaling, combining the benefits of RMSProp and SGD w/th Momentum. The optimizer is designed to be appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.

### Stochastic Gradient Descent
Stochastic Gradient Descent is an iterative optimization technique that uses minibatches of data to form an expectation of the gradient, rather than the full gradient using all available data. SGD reduces redundancy compared to batch gradient descent - which recomputes gradients for similar examples before each parameter update - so it is usually much faster.

### RMSProp
RMSProp is an unpublished adaptive learning rate optimizer proposed by Geoff Hinton. The motivation is that the magnitude of gradients can differ for different weights, and can change during learning, making it hard to choose a single global learning rate. RMSProp tackles this by keeping a moving average of the squared gradient and adjusting the weight updates by this magnitude.

### SGD with Momentum
SGD with Momentum is a stochastic optimization method that adds a momentum term to regular stochastic gradient descent.

## Regularization
Regularization strategies are designed to reduce the test error of a machine learning algorithm, possibly at the expense of training error. Many different forms of regularization exist in the field of deep learning. Below you can find a constantly updating list of regularization strategies.

### Dropout
Dropout is a regularization technique for neural networks that drops a unit (along with connections) at training time with a specified probability p. At test time, all units are present, but with weights scaled by p. The idea is to prevent co-adaptation, where the neural network becomes too reliant on particular connections, as this could be symptomatic of overfitting. Intuitively, dropout can be thought of as creating an implicit ensemble of neural networks.

### Weight Decay or L2 Regularization
Weight Decay, or L2 Regularization, is a regularization technique applied to the weights of a neural network. We minimize a loss function compromising both the primary loss function and a penalty on the L2 Norm of the weights: L_new(w) = L_original(w) + lamda*transpose(w)*w, where lamda is a value determining the strength of the penalty (encouraging smaller weights). Weight decay can be incorporated directly into the weight update rule, rather than just implicitly by defining it through to objective function. Often weight decay refers to the implementation where we specify it directly in the weight update rule (whereas L2 regularization is usually the implementation which is specified in the objective function).

### Attention Dropout
Attention Dropout is a type of dropout used in attention-based architectures, where elements are randomly dropped out of the softmax in the attention equation. For example, for scaled-dot product attention, we would drop elements from the first term.

### Label Smoothing
**Label Smoothing** is a regularization technique that introduces noise for the labels. This accounts for the fact that datasets may have mistakes in them, so maximizing the likelihood of $\log{p}\left(y\mid{x}\right)$ directly can be harmful. Assume for a small constant $\epsilon$, the training set label $y$ is correct with probability $1-\epsilon$ and incorrect otherwise. Label Smoothing regularizes a model based on a softmax with $k$ output values by replacing the hard $0$ and $1$ classification targets with targets of $\frac{\epsilon}{k-1}$ and $1-\epsilon$ respectively.

### Entropy Regularization
Entropy Regularization is a type of regularization used in reinforcement learning. For on-policy policy gradient based methods like A3C, the same mutual reinforcement behaviour leads to a highly-peaked pi(a | s) towards a few actions or action sequences, since it is easier for the actor and critic to overoptimise to a small portion of the environment. To reduce this problem, entropy regularization adds an entropy term to the loss to promote action diversity

### Early Stopping
Early Stopping is a regularization technique for deep neural networks that stops training when parameter updates no longer begin to yield improves on a validation set. In essence, we store and update the current best parameters during training, and when parameter updates no longer yield an improvement (after a set number of iterations) we stop training and use the last best parameters. It works as a regularizer by restricting the optimization procedure to a smaller volume of parameter space.

### L1 Regularization
**$L_{1}$ Regularization** is a regularization technique applied to the weights of a neural network. We minimize a loss function compromising both the primary loss function and a penalty on the $L\_{1}$ Norm of the weights:

$$L\_{new}\left(w\right) = L\_{original}\left(w\right) + \lambda{||w||}\_{1}$$

where $\lambda$ is a value determining the strength of the penalty. In contrast to weight decay, $L_{1}$ regularization promotes sparsity; i.e. some parameters have an optimal value of zero.

## Skip Connection Blocks
Skip Connection Blocks are building blocks for neural networks that feature skip connections. These skip connections 'skip' some layers allowing gradients to better flow through the network.

### Residual Block
Residual Blocks are skip-connection blocks that learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. They were introduced as part of the ResNet architecture. The original mapping is recast to F(x) + x.

## Activation Functions
Activation functions are functions that we apply in neural networks after (typically) applying an affine transformation combining weights and input features. They are typically non-linear functions. The rectified linear unit, or ReLU, has been the most popular in the past decade, although the choice is architecture dependent and many alternatives have emerged in recent years. In this section, you will find a constantly updating list of activation functions.

### Rectified Linear Units
Rectified Linear Units, or ReLUs, are a type of activation function that are linear in the positive dimension, but zero in the negative dimension. The kink in the function is the source of the non-linearity. Linearity in the positive dimension has the attractive property that it prevents non-saturation of gradients (contrast with sigmoid activations), although for half of the real line its gradient is zero.

f(x) = max(0, x)

### Sigmoid Activation
Sigmoid Activations are a type of activation function for neural networks:
f(x) = 1/(1 + e^-x)

Some drawbacks of this activation that have been noted in the literature are: sharp damp gradients during backpropagation from deeper hidden layers to inputs, gradient saturation, and slow convergence.

### Tanh Activation
Tanh Activation is an activation function used for neural networks:
f(x) = (e^x + e^-x)/(e^x + e^-x)

Historically, the tanh function became preferred over the sigmoid function as it gave better performance for multi-layer neural networks. But it did not solve the vanishing gradient problem that sigmoids suffered, which was tackled more effectively with the introduction of ReLU activations.

### Leaky ReLU
Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient is determined before training, i.e. it is not learnt during training. This type of activation function is popular in tasks where we we may suffer from sparse gradients, for example training generative adversarial networks.

### Parameterized ReLU
A Parametric Rectified Linear Unit, or PReLU, is an activation function that generalizes the traditional rectified unit with a slope for negative values.

## Distributed Methods
This section contains a compilation of distributed methods for scaling deep learning to very large models. There are many different strategies for scaling training across multiple devices, including:

* Data Parallel : for each node we use the same model parameters to do forward propagation, but we send a small batch of different data to each node, compute the gradient normally, and send it back to the main node. Once we have all the gradients, we calculate the weighted average and use this to update the model parameters.

* Model Parallel : for each node we assign different layers to it. During forward propagation, we start in the node with the first layers, then move onto the next, and so on. Once forward propagation is done we calculate gradients for the last node, and update model parameters for that node. Then we backpropagate onto the penultimate node, update the parameters, and so on.

* Additional methods including Hybrid Parallel, Auto Parallel, and Distributed Communication.

## Self-Supervised Learning
Self-Supervised Learning refers to a category of methods where we learn representations in a self-supervised way (i.e without labels). These methods generally involve a pretext task that is solved to learn a good representation and a loss function to learn with. Below you can find a continuously updating list of self-supervised methods.

## Contrastive Predictive Coding
Contrastive Predictive Coding (CPC) learns self-supervised representations by predicting the future in latent space by using powerful autoregressive models. The model uses a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples.

## Normalization
Normalization layers in deep learning are used to make optimization easier by smoothing the loss surface of the network.

### Layer Normalization
Unlike batch normalization, Layer Normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases. It works well for RNNs and improves both the training time and the generalization performance of several existing RNN models. More recently, it has been used with Transformer models.

We compute the layer normalization statistics over all the hidden units in the same layer as follows:

$$ \mu^{l} = \frac{1}{H}\sum^{H}\_{i=1}a\_{i}^{l} $$

$$ \sigma^{l} = \sqrt{\frac{1}{H}\sum^{H}\_{i=1}\left(a\_{i}^{l}-\mu^{l}\right)^{2}}  $$

where $H$ denotes the number of hidden units in a layer.

### Batch Normalization
Batch Normalization aims to reduce internal covariate shift, and in doing so aims to accelerate the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and variances of layer inputs. Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows for use of much higher learning rates without the risk of divergence. Furthermore, batch normalization regularizes the model and reduces the need for Dropout.

We apply a batch normalization layer as follows for a minibatch $\mathcal{B}$:

$$ \mu\_{\mathcal{B}} = \frac{1}{m}\sum^{m}\_{i=1}x\_{i} $$

$$ \sigma^{2}\_{\mathcal{B}} = \frac{1}{m}\sum^{m}\_{i=1}\left(x\_{i}-\mu\_{\mathcal{B}}\right)^{2} $$

$$ \hat{x}\_{i} = \frac{x\_{i} - \mu\_{\mathcal{B}}}{\sqrt{\sigma^{2}\_{\mathcal{B}}+\epsilon}} $$

$$ y\_{i} = \gamma\hat{x}\_{i} + \beta = \text{BN}\_{\gamma, \beta}\left(x\_{i}\right) $$

Where $\gamma$ and $\beta$ are learnable parameters.

## Loss Functions
Loss Functions are used to frame the problem to be optimized within deep learning.

### Focal Loss
A Focal Loss function addresses class imbalance during training in tasks like object detection. Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. It is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases. Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples.

Formally, the Focal Loss adds a factor $(1 - p\_{t})^\gamma$ to the standard cross entropy criterion. Setting $\gamma>0$ reduces the relative loss for well-classified examples ($p\_{t}>.5$), putting more focus on hard, misclassified examples. Here there is tunable *focusing* parameter $\gamma \ge 0$. 

$$ {\text{FL}(p\_{t}) = - (1 - p\_{t})^\gamma \log\left(p\_{t}\right)} $$

### Triplet Loss
The goal of **Triplet loss**, in the context of Siamese Networks, is to maximize the joint probability among all score-pairs i.e. the product of all probabilities. By using its negative logarithm, we can get the loss formulation as follows:

$$
L\_{t}\left(\mathcal{V}\_{p}, \mathcal{V}\_{n}\right)=-\frac{1}{M N} \sum\_{i}^{M} \sum\_{j}^{N} \log \operatorname{prob}\left(v p\_{i}, v n\_{j}\right)
$$

where the balance weight $1/MN$ is used to keep the loss with the same scale for different number of instance sets.

### Connectionist Temporal Classification Loss
A Connectionist Temporal Classification Loss, or CTC Loss, is designed for tasks where we need alignment between sequences, but where that alignment is difficult - e.g. aligning each character to its location in an audio file. It calculates a loss between a continuous (unsegmented) time series and a target sequence. It does this by summing over the probability of possible alignments of input to target, producing a loss value which is differentiable with respect to each input node. The alignment of input to target is assumed to be “many-to-one”, which limits the length of the target sequence such that it must be <= the input length.

## Semi-Supervised Learning Methods
Semi-Supervised Learning methods leverage unlabelled data as well as labelled data to increase performance on machine learning tasks.

## Neural Architecture Search
Neural Architecture Search methods are search methods that seek to learn architectures for machine learning tasks, including the underlying build blocks. Neural Architecture Search (NAS) learns a modular architecture which can be transferred from a small dataset to a large dataset. The method does this by reducing the problem of learning best convolutional architectures to the problem of learning a small convolutional cell. The cell can then be stacked in series to handle larger images and more complex datasets.

### DARTS
Differentiable Architecture Search (DART) is a method for efficient architecture search. The search space is made continuous so that the architecture can be optimized with respect to its validation set performance through gradient descent.

### Differentiable Neural Architecture Search

## Feedforward Networks
Feedforward Networks are a type of neural network architecture which rely primarily on dense-like connections.

### Dense Connections
**Dense Connections**, or **Fully Connected Connections**, are a type of layer in a deep neural network that use a linear operation where every input is connected to every output by a weight. This means there are $n\_{\text{inputs}}*n\_{\text{outputs}}$ parameters, which can lead to a lot of parameters for a sizeable network.

$$h\_{l} = g\left(\textbf{W}^{T}h\_{l-1}\right)$$

where $g$ is an activation function.

### Position-Wise Feed-Forward Layer
Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that applies to the last dimension, which means the same dense layers are used for each position item in the sequence, so called position-wise.

### Feedforward Network
A Feedforward Network, or a Multilayer Perceptron (MLP), is a neural network with solely densely connected layers. This is the classic neural network architecture of the literature. It consists of inputs x passed through units h (of which there can be many layers) to predict a target y. Activation functions are generally chosen to be non-linear to allow for flexible functional approximation.

### Linear Layer
A Linear Layer is a projection XW + b.

## Adversarial Training
Adversarial Training methods use adversarial techniques to improve generalization (and the quality of representations learnt during training). Adversarial techniques are also sometimes used in the context of generative models with a generator and a discriminator. Below you can find a continuously updating list of adversarial training methods.

## Clustering
Clustering methods cluster a dataset so that similar datapoints are located in the same group.

### k-Means Clustering
**k-Means Clustering** is a clustering algorithm that divides a training set into $k$ different clusters of examples that are near each other. It works by initializing $k$ different centroids {$\mu\left(1\right),\ldots,\mu\left(k\right)$} to different values, then alternating between two steps until convergence:

(i) each training example is assigned to cluster $i$ where $i$ is the index of the nearest centroid $\mu^{(i)}$

(ii) each centroid $\mu^{(i)}$ is updated to the mean of all training examples $x^{(j)}$ assigned to cluster $i$.

### Spectral Clustering
Spectral clustering has attracted increasing attention due to the promising ability in dealing with nonlinearly separable datasets [15], [16]. In spectral clustering, the spectrum of the graph Laplacian is used to reveal the cluster structure. The spectral clustering algorithm mainly consists of two steps: 1) constructs the low dimensional embedded representation of the data based on the eigenvectors of the graph Laplacian, 2) applies k-means on the constructed low dimensional data to obtain the clustering result.

###  Self-Organizing Map
The Self-Organizing Map (SOM), commonly also known as Kohonen network (Kohonen 1982, Kohonen 2001) is a computational method for the visualization and analysis of high-dimensional data, especially experimentally acquired information.

## Learning Rate Schedules
Learning Rate Schedules refer to schedules for the learning rate during the training of neural networks.

### Linear Warmup With Linear Decay
Linear Warmup With Linear Decay is a learning rate schedule in which we increase the learning rate linearly for n updates and then linearly decay afterwards.

### Cosine Annealing
Cosine Annealing is a type of learning rate schedule that has the effect of starting with a large learning rate that is relatively rapidly decreased to a minimum value before being increased rapidly again. The resetting of the learning rate acts like a simulated restart of the learning process and the re-use of good weights as the starting point of the restart is referred to as a "warm restart" in contrast to a "cold restart" where a new set of small random numbers may be used as a starting point.

### Linear Warmup With Cosine Annealing
Linear Warmup With Cosine Annealing is a learning rate schedule where we increase the learning rate linearly for n updates and then anneal according to a cosine schedule afterwards.

## Interpretability
Interpretability Methods seek to explain the predictions made by neural networks by introducing mechanisms to enduce or enforce interpretability. For example, LIME approximates the neural network with a locally interpretable model.

### LIME
LIME, or Local Interpretable Model-Agnostic Explanations, is an algorithm that can explain the predictions of any classifier or regressor in a faithful way, by approximating it locally with an interpretable model. It modifies a single data sample by tweaking the feature values and observes the resulting impact on the output. It performs the role of an "explainer" to explain predictions from each data sample. The output of LIME is a set of explanations representing the contribution of each feature to a prediction for a single sample, which is a form of local interpretability.

Interpretable models in LIME can be, for instance, linear regression or decision trees, which are trained on small perturbations (e.g. adding noise, removing words, hiding parts of the image) of the original model to provide a good local approximation.

### Shapley Additive Explanations
SHAP, or SHapley Additive exPlanations, is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. Shapley values are approximating using Kernel SHAP, which uses a weighting kernel for the approximation, and DeepSHAP, which uses DeepLift to approximate them.

## Domain Adaptation

### Domain Adaptive Neighborhood Clustering via Entropy Optimization (DANCE)
Domain Adaptive Neighborhood Clustering via Entropy Optimization (DANCE) is a self-supervised clustering method that harnesses the cluster structure of the target domain using self-supervision. This is done with a neighborhood clustering technique that self-supervises feature learning in the target. At the same time, useful source features and class boundaries are preserved and adapted with a partial domain alignment loss that the authors refer to as entropy separation loss. This loss allows the model to either match each target example with the source, or reject it as unknown.

### Source Hypothesis Transfer
Source Hypothesis Transfer, or SHOT, is a representation learning framework for unsupervised domain adaptation. SHOT freezes the classifier module (hypothesis) of the source model and learns the target-specific feature extraction module by exploiting both information maximization and self-supervised pseudo-labeling to implicitly align representations from the target domains to the source hypothesis.

## Output Functions
Output functions are layers used towards the end of a network to transform to the desired form for a loss function. For example, the softmax relies on logits to construct a conditional probability. Below you can find a continuously updating list of output functions.

### Softmax
The **Softmax** output function transforms a previous layer's output into a vector of probabilities. It is commonly used for multiclass classification.  Given an input vector $x$ and a weighting vector $w$ we have:

$$ P(y=j \mid{x}) = \frac{e^{x^{T}w_{j}}}{\sum^{K}_{k=1}e^{x^{T}wk}} $$

## AutoML
AutoML methods are used to automatically solve machine learning tasks without needing the user to specify or experiment with architectures, hyperparameters and other settings.

### Minimum Description Length
Minimum Description Length provides a criterion for the selection of models, regardless of their complexity, without the restrictive assumption that the data form a sample from a 'true' distribution.

## Initialization
Initialization methods are used to initialize the weights in a neural network.

### Kaiming Initialization
Kaiming Initialization, or He Initialization, is an initialization method for neural networks that takes into account the non-linearity of activation functions, such as ReLU activations. A proper initialization method should avoid reducing or magnifying the magnitudes of input signals exponentially. That is, a zero-centered Gaussian with standard deviation of sqrt(2/n_l). Biases are initialized at 0.

### Xavier Initialization
**Xavier Initialization**, or **Glorot Initialization**, is an initialization scheme for neural networks. Biases are initialized be 0 and the weights $W\_{ij}$ at each layer are initialized as:

$$ W\_{ij} \sim U\left[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right] $$

Where $U$ is a uniform distribution and $n$ is the size of the previous layer (number of columns in $W$).

## Non-Parametric Regression
Non-Parametric Regression methods are a type of regression where we use non-parametric methods to approximate the functional form of the relationship.

### Gaussian Process
Gaussian Processes are non-parametric models for approximating functions. They rely upon a measure of similarity between points (the kernel function) to predict the value for an unseen point from training data. The models are fully probabilistic so uncertainty bounds are baked in with the model.

### Support Vector Machine
A Support Vector Machine, or SVM, is a non-parametric supervised learning model. For non-linear classification and regression, they utilise the kernel trick to map inputs to high-dimensional feature spaces. SVMs construct a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

### k-Nearest Neighbors
k-Nearest Neighbors is a clustering-based algorithm for classification and regression. It is a a type of instance-based learning as it does not attempt to construct a general internal model, but simply stores instances of the training data. Prediction is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

## Representation Learning

### Vision-and-Language BERT
Vision-and-Language BERT (ViLBERT) is a BERT-based model for learning task-agnostic joint representations of image content and natural language. ViLBERT extend the popular BERT architecture to a multi-modal two-stream model, processing both visual and textual inputs in separate streams that interact through co-attentional transformer layers.

## Fine-Tuning
Fine-Tuning methods in deep learning take existing trained networks and 'fine-tune' them to a new task so that information contained in the weights can be repurposed. Below you can find a continuously updating list of fine-tuning methods.

### Discriminative Fine-Tuning
Discriminative Fine-Tuning is a fine-tuning strategy that is used for ULMFiT type models. Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune each layer with different learning rates.

## Working Memory Models
Working Memory Models aim to supplement neural networks with a memory module to increase their capability for memorization and allowing them to more easily perform tasks such as retrieving and copying information.

### Memory Network
A Memory Network provides a memory component that can be read from and written to with the inference capabilities of a neural network model. The motivation is that many neural networks lack a long-term memory component, and their existing memory component encoded by states and weights is too small and not compartmentalized enough to accurately remember facts from the past (RNNs for example, have difficult memorizing and doing tasks like copying).

## Meta-Learning Algorithms
Meta-Learning methods are methods that learn to learn. An example is few-shot meta-learning methods which aim to quickly adapt to a new task with only a few datapoints.

### Model-Agnostic Meta-Learning
MAML, or Model-Agnostic Meta-Learning, is a model and task-agnostic algorithm for meta-learning that trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task.

## Approximate Inference
Approximate Inference methods are used within the context of Bayesian inference to approximate (intractable) posteriors. The most popular category were Markov Chain Monte Carlo methods; more recently variational methods have become popular.

### Markov Chain Monte Carlo
The golden standard for uncertainty quantification and Bayesian inference.

### Approximate Bayesian Computation
Class of methods in Bayesian Statistics where the posterior distribution is approximated over a rejection scheme on simulations because the likelihood function is intractable.

Different parameters get sampled and simulated. Then a distance function is calculated to measure the quality of the simulation compared to data from real observations. Only simulations that fall below a certain threshold get accepted

### Metropolis Hastings
Metropolis-Hastings is a Markov Chain Monte Carlo (MCMC) algorithm for approximate inference. It allows for sampling from a probability distribution where direct sampling is difficult - usually owing to the presence of an intractable integral.

## Dimensionality Reduction
Dimensionality Reduction methods transform data from a high-dimensional space into a low-dimensional space so that the low-dimensional space retains the most important properties of the original data.

### Principal Components Analysis
Principle Components Analysis (PCA) is an unsupervised method primary used for dimensionality reduction within machine learning. PCA is calculated via a singular value decomposition (SVD) of the design matrix, or alternatively, by calculating the covariance matrix of the data and performing eigenvalue decomposition on the covariance matrix. The results of PCA provide a low-dimensional picture of the structure of the data and the leading (uncorrelated) latent factors determining variation in the data.

### Linear Discriminant Analysis
Linear discriminant analysis (LDA), normal discriminant analysis (NDA), or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics, pattern recognition, and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.

### Absolute Position Encodings
**Absolute Position Encodings** are a type of position embeddings for Transformer-based models where positional encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d\_{model}$ as the embeddings, so that the two can be summed. In the original implementation, sine and cosine functions of different frequencies are used:

$$ \text{PE}\left(pos, 2i\right) = \sin\left(pos/10000^{2i/d\_{model}}\right) $$

$$ \text{PE}\left(pos, 2i+1\right) = \cos\left(pos/10000^{2i/d\_{model}}\right) $$

## Affinity Functions
Affinity Functions are pairwise functions used to represent a relationship between two entities. They were used in the context of non-local neural networks. Below you can find a list of different affinity functions.

### Embedded Gaussian Affinity
**Embedded Gaussian Affinity** is a type of affinity or self-similarity function between two points $\mathbf{x\_{i}}$ and $\mathbf{x\_{j}}$ that uses a Gaussian function in an embedding space:

$$ f\left(\mathbf{x\_{i}}, \mathbf{x\_{j}}\right) = e^{\theta\left(\mathbf{x\_{i}}\right)^{T}\phi\left(\mathbf{x\_{j}}\right)} $$

Here $\theta\left(x\_{i}\right) = W\_{θ}x\_{i}$ and $\phi\left(x\_{j}\right) = W\_{φ}x\_{j}$ are two embeddings.

## Hyperparameter Search
Hyperparameter Search methods are used to search for hyperparameters during the training stage of a neural network. Below you can find a continuously updating list of (specialized) hyperparameter search methods.

### Random Search
Random Search replaces the exhaustive enumeration of all combinations by selecting them randomly. This can be simply applied to the discrete setting described above, but also generalizes to continuous and mixed spaces. It can outperform Grid search, especially when only a small number of hyperparameters affects the final performance of the machine learning algorithm. In this case, the optimization problem is said to have a low intrinsic dimensionality. Random Search is also embarrassingly parallel, and additionally allows the inclusion of prior knowledge by specifying the distribution from which to sample.

### Dynamic Algorithm Configuration
Dynamic algorithm configuration (DAC) is capable of generalizing over prior optimization approaches, as well as handling optimization of hyperparameters that need to be adjusted over multiple time-steps.

### Symbolic rule learning
Symbolic rule learning methods find regularities in data that can be expressed in the form of 'if-then' rules based on symbolic representations of the data.