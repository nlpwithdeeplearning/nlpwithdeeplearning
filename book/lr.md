---
layout: page
title: Logistic Regression
---

### What does the discriminative model compute?
It directly computes `P(c|d)` where c is the class label and d is the data.

### What are the components of a probabilistic machine learning classifier?
1. feature representation
1. classification function such as sigmoid or softmax
1. objective function such as cross-entropy loss function
1. learning algorithm such as stochastic gradient descent

### What is sigmoid(z)?
1/(1 + e^-z)

### What is a decision boundary?
Decision boundary is the threshold of probability for the binary classification task.

### What is P(y=1) in logistic regression?
sigmoid(w.x + b)

### What are feature interactions?
Feature interactions are combination features from more simplistic features often using feature template or abstract specification of features in the absence of automated feature design using representation learning.

### How do you choose between logistic regression and naive bayes classifiers?
Naive Bayes has overly strong conditional independence assumptions. Thus, when the features correlate a lot with each other, logistic regressions will assign a more accurate probability, especially for large documents or datasets. For very small datasets or short documents, naive bayes could work better and it is also very fast to train as there is no optimization step.

### What is the cross entropy loss, L_CE(y_hat, y)?
`-y*log(y_hat) - (1-y)*log(1-y_hat)`

### Provide pseudocode for implementation of stochastic gradient descent?
```
def stochastic_gradient_descent(L, f, x, y):
    theta = initialize_parameters()
    while true:
        for (x_i, y_i) in shuffle((x, y)):
            y_hat_i = f(x_i, theta)
            gradient = delta_theta(L(y_hat_i, y_i))
            theta = theta - learning_rate * gradient
    return theta 
```

### What is batch training?
In batch training, we compute the gradient over the entire dataset before updating the parameters.

### What is mini-batch training?
We compute gradient on a group of m examples (e.g, 512 or 1024) in parallel, and then update the parameters.

### What is overfitting?
If the weights for features are set so that they perfectly fit the training data including modeling nosiy factor that just accidentially correlate with the class, the model can't generalize to unseen test data. This is called overfitting.

### What is the L2 or ridge regularized objective function?
theta_hat = argmax(sum(logP(y_i | x_i), i= 0 to m-1), theta) - alpha*sum(theta_j^2, j=0 to n-1)

### What is the L1 or lasso regularized objective function?
theta_hat = argmax(sum(log(P(y_i|x_i), i=0 to m-1)), theta) - alpha*sum(|theta_j|, j=0 to n-1)

### What is multinomial logistic regression or softmax regression or maxent classifier?
The target range over more than two classes and we use soft function to compute probability p(y = c|x).
softmax(z_i) = exp(z_i)/sum(exp(z_j), j=0 to k-1)

### What is the cross entropy in multinominal logistic regression?
L_CE(y_hat, y)  = -log(y_hat_k) # where k is the correct class
                = -log(exp(w_k.x + b_k)/sum(exp(w_j.x + b_j), j=0 to K-1)) # where K is the total number of classes