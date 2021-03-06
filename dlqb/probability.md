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
The central limit theorem states that if you have a population with mean Î¼ and standard deviation Ï and take sufficiently large random samples from the population with replacement , then the distribution of the sample means will be approximately normally distributed with the population mean and standard deviation.

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
1. For any event A, its probability is never negative, i.e.,  ð(A)â¥0 ;
1. Probability of the entire sample space is  1 , i.e.,  ð(S)=1 ;
1. For any countable sequence of events  A1,A2,â¦  that are mutually exclusive ( Aiâ©Aj=â  for all  ðâ ð ), the probability that any happens is equal to the sum of their individual probabilities

### What is a random variable?
A random variable can be pretty much any quantity and is not deterministic. It could take one value among a set of possibilities in a random experiment. Note that there is a subtle difference between discrete random variables, like the sides of a die, and continuous ones, like the weight and the height of a person. There is little point in asking whether two people have exactly the same height.

### What is joint probability?
Given any values  ð  and  ð , the joint probability lets us answer, what is the probability that  ð´=ð  and  ðµ=ð  simultaneously.

### What is conditional probability?
Note that for any values  ð  and  ð ,  ð(ð´=ð,ðµ=ð)â¤ð(ð´=ð) . This has to be the case, since for  ð´=ð  and  ðµ=ð  to happen,  ð´=ð  has to happen and  ðµ=ð  also has to happen (and vice versa). Thus,  ð´=ð  and  ðµ=ð  cannot be more likely than  ð´=ð  or  ðµ=ð  individually. This brings us to an interesting ratio:  0â¤ð(ð´=ð,ðµ=ð)/ð(ð´=ð)â¤1 . We call this ratio a conditional probability and denote it by  ð(ðµ=ðâ£ð´=ð) : it is the probability of  ðµ=ð , provided that  ð´=ð  has occurred.

### What is Bayes theorem?
ð(ð´â£ðµ) = ð(ðµâ£ð´)ð(ð´)/ð(ðµ)

### What is Marginalization?
It is the operation of determining  ð(ðµ)  from  ð(ð´,ðµ) . We can see that the probability of  ðµ  amounts to accounting for all possible choices of  ð´  and aggregating the joint probabilities over all of them: ð(ðµ)=âð(ð´,ðµ).

### What is Independence?
Two random variables  ð´  and  ðµ  being independent means that the occurrence of one event of  ð´  does not reveal any information about the occurrence of an event of  ðµ . In this case  ð(ðµâ£ð´)=ð(ðµ) . Statisticians typically express this as  ð´â¥ðµ . From Bayesâ theorem, it follows immediately that also  ð(ð´â£ðµ)=ð(ð´) . In all the other cases we call  ð´  and  ðµ  dependent.

Likewise, two random variables  ð´  and  ðµ  are conditionally independent given another random variable  ð¶  if and only if  ð(ð´,ðµâ£ð¶)=ð(ð´â£ð¶)ð(ðµâ£ð¶) . This is expressed as  ð´â¥ðµâ£ð¶ .

### What is Expectation?
The expectation (or average) of the random variable  ð  is denoted as: ð¸[ð]=âð¥ð¥ð(ð=ð¥).
 
When the input of a function  ð(ð¥)  is a random variable drawn from the distribution  ð  with different values  ð¥ , the expectation of  ð(ð¥)  is computed as: ð¸ð¥â¼ð[ð(ð¥)]=âð¥ð(ð¥)ð(ð¥).

### What are Variance and Standard Deviation?
In many cases we want to measure by how much the random variable  ð  deviates from its expectation. This can be quantified by the variance: Var[ð]=ð¸[(ðâð¸[ð])**2]=ð¸[ð**2]âð¸[ð]**2.
 
Its square root is called the standard deviation. The variance of a function of a random variable measures by how much the function deviates from the expectation of the function, as different values  ð¥  of the random variable are sampled from its distribution: Var[ð(ð¥)]=ð¸[(ð(ð¥)âð¸[ð(ð¥)])**2].

