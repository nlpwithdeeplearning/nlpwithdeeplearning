---
layout: page
title: Introduction
---

### What is machine learning?
Machine learning studies how computer systems can leverage experience (often data) to improve performance at specific tasks. It combines ideas from statistics, data mining, and optimization. Often, it is used as a means of implementing AI solutions.

### What is representational learning?
As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. Deep learning is multi-level representation learning through learning many layers of transformations.

### What distinguishes deep learning from machine learning?
Deep learning replaces not only the shallow models at the end of traditional machine learning pipelines, but also the labor-intensive process of feature engineering.

### What made deep learning ubiquitous today?
Much of the recent progress in deep learning has been triggered by an abundance of data arising from cheap sensors and Internet-scale applications, and by significant progress in computation, mostly through GPUs. Whole system optimization is a key component in obtaining high performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

### What are the four components of a machine learning problem?
1. The data that we can learn from.
1. A model of how to transform the data.
1. An objective function that quantifies how well (or badly) the model is doing.
1. An algorithm to adjust the model’s parameters to optimize the objective function.

### What are the different kinds of machine learning problems?
1. Supervised Learning: Supervised learning addresses the task of predicting labels given input features. Each feature–label pair is called an example. Sometimes, when the context is clear, we may use the term examples to refer to a collection of inputs, even when the corresponding labels are unknown. Our goal is to produce a model that maps any input to a label prediction.
1. Unsupervised and Self-Supervised Learning: The boss might just hand you a giant dump of data and tell you to do some data science with it! This sounds vague because it is. We call this class of problems unsupervised learning, and the type and number of questions we could ask is limited only by our creativity. As a form of unsupervised learning, self-supervised learning leverages unlabeled data to provide supervision in training, such as by predicting some withheld part of the data using other parts.
1. Reinforcement Learning: If you are interested in using machine learning to develop an agent that interacts with an environment and takes actions, then you are probably going to wind up focusing on reinforcement learning. Reinforcement learning gives a very general statement of a problem, in which an agent interacts with an environment over a series of time steps. At each time step, the agent receives some observation from the environment and must choose an action that is subsequently transmitted back to the environment via some mechanism (sometimes called an actuator). Finally, the agent receives a reward from the environment.

### What are the different applications of supervised learning?
1. Regression: What makes a problem a regression is actually the output. Say that you are in the market for a new home. You might want to estimate the fair market value of a house, given some features. The label, the price of sale, is a numerical value. When labels take on arbitrary numerical values, we call this a regression problem. Our goal is to produce a model whose predictions closely approximate the actual label values.
1. Classification: In classification, we want our model to look at features, e.g., the pixel values in an image, and then predict which category (formally called class), among some discrete set of options, an example belongs.
    1. Binary classification: The simplest form of classification is when there are only two classes, a problem which we call binary classification. For example, our dataset could consist of images of animals and our labels might be the classes  {cat,dog}.
    1. Multiclass classification: When we have more than two possible classes, we call the problem multiclass classification. Common examples include hand-written character recognition  {0,1,2,...9,a,b,c,...}.
    1. Hierarchical classification: Hierarchies assume that there exist some relationships among the many classes. So not all errors are equal—if we must err, we would prefer to misclassify to a related class rather than to a distant class. In the case of animal classification, it might not be so bad to mistake a poodle (a dog breed) for a schnauzer (another dog breed), but our model would pay a huge penalty if it confused a poodle for a dinosaur.
1. Tagging or Multi-label classification: The problem of learning to predict classes that are not mutually exclusive is called multi-label classification. Auto-tagging problems are typically best described as multi-label classification problems. Think of the tags people might apply to posts on a technical blog, e.g., “machine learning”, “technology”, “gadgets”, “programming languages”, “Linux”, “cloud computing”, “AWS”. A typical article might have 5–10 tags applied because these concepts are correlated.
1. Search: Sometimes we do not just want to assign each example to a bucket or to a real value. The goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results is most relevant for a particular user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. Nowadays, search engines use machine learning and behavioral models to obtain query-dependent relevance scores.
1. Recommender systems: Recommender systems are another problem setting that is related to search and ranking. The problems are similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on personalization to specific users in the context of recommender systems. Many of the problems about how to deal with censoring, incentives, and feedback loops, are important open research questions.
1. Sequence Learning: They require a model to either ingest sequences of inputs or to emit sequences of outputs (or both). Specifically, sequence to sequence learning considers problems where input and output are both variable-length sequences, such as machine translation and transcribing text from the spoken speech. Examples include:
    1. Tagging and Parsing
    1. Automatic Speech Recognition
    1. Text to Speech
    1. Machine Translation

### What are the roots of machine learning?
1. Even in the middle ages, mathematicians had a keen intuition of estimates. For instance, the geometry book of Jacob Köbel (1460–1533) illustrates averaging the length of 16 adult men’s feet to obtain the average foot length.
1. Humans have held the desire to analyze data and to predict future outcomes for long and much of natural science has its roots in this. For instance, the Bernoulli distribution is named after Jacob Bernoulli (1655–1705), and the Gaussian distribution was discovered by Carl Friedrich Gauss (1777–1855). He invented, for instance, the least mean squares algorithm, which is still used today for countless problems from insurance calculations to medical diagnostics.
1. Statistics really took off with the collection and availability of data. One of its titans, Ronald Fisher (1890–1962), contributed significantly to its theory and also its applications in genetics. Many of his algorithms (such as linear discriminant analysis) and formula (such as the Fisher information matrix) are still in frequent use today. In fact, even the Iris dataset that Fisher released in 1936 is still used sometimes to illustrate machine learning algorithms.
1. A second influence for machine learning came from information theory by Claude Shannon (1916–2001) and the theory of computation via Alan Turing (1912–1954). Turing posed the question “can machines think?” in his famous paper Computing Machinery and Intelligence [Turing, 1950].
1. Another influence can be found in neuroscience and psychology. After all, humans clearly exhibit intelligent behavior. It is thus only reasonable to ask whether one could explain and possibly reverse engineer this capacity. One of the oldest algorithms inspired in this fashion was formulated by Donald Hebb (1904–1985). In his groundbreaking book The Organization of Behavior [Hebb & Hebb, 1949], he posited that neurons learn by positive reinforcement.

### What are some pioneering ideas in deep learning?
1. Novel methods for capacity control, such as dropout [Srivastava et al., 2014], have helped to mitigate the danger of overfitting.
1. Attention mechanisms solved a second problem that had plagued statistics for over a century: how to increase the memory and complexity of a system without increasing the number of learnable parameters. Researchers found an elegant solution by using what can only be viewed as a learnable pointer structure [Bahdanau et al., 2014].
1. Multi-stage designs, e.g., via the memory networks [Sukhbaatar et al., 2015] and the neural programmer-interpreter [Reed & DeFreitas, 2015] allowed statistical modelers to describe iterative approaches to reasoning.
1. The crucial innovation in generative adversarial networks [Goodfellow et al., 2014] was to replace the sampler by an arbitrary algorithm with differentiable parameters. These are then adjusted in such a way that the discriminator (effectively a two-sample test) cannot distinguish fake from real data. Through the ability to use arbitrary algorithms to generate data, it opened up density estimation to a wide variety of techniques.
1. In many cases, a single GPU is insufficient to process the large amounts of data available for training. Over the past decade the ability to build parallel and distributed training algorithms has improved significantly. One of the key challenges in designing scalable algorithms is that the workhorse of deep learning optimization, stochastic gradient descent, relies on relatively small minibatches of data to be processed. At the same time, small batches limit the efficiency of GPUs. Hence, training on 1024 GPUs with a minibatch size of, say 32 images per batch amounts to an aggregate minibatch of about 32000 images. Recent work, first by Li [Li, 2017], and subsequently by [You et al., 2017] and [Jia et al., 2018] pushed the size up to 64000 observations, reducing training time for the ResNet-50 model on the ImageNet dataset to less than 7 minutes. For comparison—initially training times were measured in the order of days.
1. The ability to parallelize computation has also contributed quite crucially to progress in reinforcement learning, at least whenever simulation is an option. This has led to significant progress in computers achieving superhuman performance in Go, Atari games, Starcraft, and in physics simulations (e.g., using MuJoCo). See e.g., [Silver et al., 2016] for a description of how to achieve this in AlphaGo. In a nutshell, reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e., whenever it is possible to try out lots of things to learn how they relate to each other. Simulation provides such an avenue.
1. Deep learning frameworks have played a crucial role in disseminating ideas. The third generation of tools, namely imperative tools for deep learning, was arguably spearheaded by Chainer, which used a syntax similar to Python NumPy to describe models. This idea was adopted by both PyTorch, the Gluon API of MXNet, and Jax.

### What are some success stories of deep learning?
1. Intelligent assistants
1. ability to recognize speech accurately
1. Object recognition
1. impressive progress in games and the fact that advanced algorithms played a crucial part in them
1. advent of self-driving cars and trucks