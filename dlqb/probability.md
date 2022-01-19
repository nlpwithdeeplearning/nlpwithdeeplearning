---
layout: page
title: Probability
---

### What is multinomial distribution?
The distribution that assigns probabilities to a number of discrete choices is called the multinomial distribution.

### Draw a sample dice in pytorch
```
import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

### What is central limit theorem?
The central limit theorem states that if you have a population with mean μ and standard deviation σ and take sufficiently large random samples from the population with replacement , then the distribution of the sample means will be approximately normally distributed with the population mean and standard deviation.

### Write some pseudo code to demonstrate central limit theorem
```
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.gca().set_xlabel('Groups of experiments')
plt.gca().set_ylabel('Estimated probability')
plt.legend();
```

### What are the axioms of probability?
1. For any event A, its probability is never negative, i.e.,  𝑃(A)≥0 ;
1. Probability of the entire sample space is  1 , i.e.,  𝑃(S)=1 ;
1. For any countable sequence of events  A1,A2,…  that are mutually exclusive ( Ai∩Aj=∅  for all  𝑖≠𝑗 ), the probability that any happens is equal to the sum of their individual probabilities

### What is a random variable?
A random variable can be pretty much any quantity and is not deterministic. It could take one value among a set of possibilities in a random experiment. Note that there is a subtle difference between discrete random variables, like the sides of a die, and continuous ones, like the weight and the height of a person. There is little point in asking whether two people have exactly the same height.

### What is joint probability?
Given any values  𝑎  and  𝑏 , the joint probability lets us answer, what is the probability that  𝐴=𝑎  and  𝐵=𝑏  simultaneously.

### What is conditional probability?
Note that for any values  𝑎  and  𝑏 ,  𝑃(𝐴=𝑎,𝐵=𝑏)≤𝑃(𝐴=𝑎) . This has to be the case, since for  𝐴=𝑎  and  𝐵=𝑏  to happen,  𝐴=𝑎  has to happen and  𝐵=𝑏  also has to happen (and vice versa). Thus,  𝐴=𝑎  and  𝐵=𝑏  cannot be more likely than  𝐴=𝑎  or  𝐵=𝑏  individually. This brings us to an interesting ratio:  0≤𝑃(𝐴=𝑎,𝐵=𝑏)/𝑃(𝐴=𝑎)≤1 . We call this ratio a conditional probability and denote it by  𝑃(𝐵=𝑏∣𝐴=𝑎) : it is the probability of  𝐵=𝑏 , provided that  𝐴=𝑎  has occurred.

### What is Bayes theorem?
𝑃(𝐴∣𝐵) = 𝑃(𝐵∣𝐴)𝑃(𝐴)/𝑃(𝐵)

### What is Marginalization?
It is the operation of determining  𝑃(𝐵)  from  𝑃(𝐴,𝐵) . We can see that the probability of  𝐵  amounts to accounting for all possible choices of  𝐴  and aggregating the joint probabilities over all of them: 𝑃(𝐵)=∑𝑃(𝐴,𝐵).

### What is Independence?
Two random variables  𝐴  and  𝐵  being independent means that the occurrence of one event of  𝐴  does not reveal any information about the occurrence of an event of  𝐵 . In this case  𝑃(𝐵∣𝐴)=𝑃(𝐵) . Statisticians typically express this as  𝐴⊥𝐵 . From Bayes’ theorem, it follows immediately that also  𝑃(𝐴∣𝐵)=𝑃(𝐴) . In all the other cases we call  𝐴  and  𝐵  dependent.

Likewise, two random variables  𝐴  and  𝐵  are conditionally independent given another random variable  𝐶  if and only if  𝑃(𝐴,𝐵∣𝐶)=𝑃(𝐴∣𝐶)𝑃(𝐵∣𝐶) . This is expressed as  𝐴⊥𝐵∣𝐶 .

### What is Expectation?
The expectation (or average) of the random variable  𝑋  is denoted as: 𝐸[𝑋]=∑𝑥𝑥𝑃(𝑋=𝑥).
 
When the input of a function  𝑓(𝑥)  is a random variable drawn from the distribution  𝑃  with different values  𝑥 , the expectation of  𝑓(𝑥)  is computed as: 𝐸𝑥∼𝑃[𝑓(𝑥)]=∑𝑥𝑓(𝑥)𝑃(𝑥).

### What are Variance and Standard Deviation?
In many cases we want to measure by how much the random variable  𝑋  deviates from its expectation. This can be quantified by the variance: Var[𝑋]=𝐸[(𝑋−𝐸[𝑋])**2]=𝐸[𝑋**2]−𝐸[𝑋]**2.
 
Its square root is called the standard deviation. The variance of a function of a random variable measures by how much the function deviates from the expectation of the function, as different values  𝑥  of the random variable are sampled from its distribution: Var[𝑓(𝑥)]=𝐸[(𝑓(𝑥)−𝐸[𝑓(𝑥)])**2].

