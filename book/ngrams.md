---
layout: page
title: N-grams
---

### For which NLP applications is it important to model probability of next word or of the whole sentence?
Speech recognition uses such probability to output those sequences which are likely. Spelling and grammar correction find and correct errors based on how unlikely a word or sequence is. Machine translation also uses probabilities to suggest translations that fluent speakers are more likely to use. Augmentative and alternative communication (AAC) systems also use such models to to suggest words to be spoken.

### What are language models?
Language models are models that assign probabilities to sequences of words.

### What is an n-gram?
N-gram is a sequence of `n` words. N-gram also refers to the n-gram language model that assigns probability to the n-gram.

### How do we use chain rule of probabilty to the sequence of words w_0, w_1, ..., w_n-1?
P(w_0:n-1) = P(w_0)*P(w_1|w_0)*P(w_2|w_0:1)*...*P(w_n-1|w_0:n-2)

### What is a bigram model in terms of chain rule approximation?
The bigram model approximates the probability of a word given all the previous words P(w_i|w_0:i-1) to the conditional probability given the preceeding word P(w_i|w_i-1).

### What is the markov assumption in the bigram model?
The bigram model makes the markov assumption that the probability of a word depends only on the previous word.

### What is the general equation for n-gram language model approximation of the next word in a sequence?
P(w_i|w_0:i-1) ~ P(w_i|w_i-n+1:i-1)

### What is the maximum likelihood estimate (MLE) for bigram probabilities?
P(w_i|w_i-1) = C(w_i-1,i)/C(w_i-1)
This is obtained by Counting the number of bigrams C(w_i-1,i) and normalizing the counts to be between 0 and 1 by dividing it with the count of the previous word C(w_i-1).

### What is the formula for the general case of MLE n-gram parameter estimation?
We use the relative frequency of counts here: 
P(w_i|w_i-n+1:i-1) = C(w_i-n+1,i)/C(w_i-n+1,i-1)

### Why do we represent and compute language model probabilities in log format?
We use log probabilities because as the probabilities are less than or equal to 1, the more of them we multuple, the smaller it becomes. This would result in numerical underflow. In the log space though, these numbers are not as small and they can be added instead of multiplying. When we need to report the probabilities, we can take the exp of the logprob.

### What is extrinsic evaluation of a model?
Extrinsic evaluation is the measuring the performance of the model by embedding it in an application and using the metrics of the application. E.g., language models can be extrinsically evaluated by embedding them in speech recognition.

### What is intrinsic evaluation for a language model?
Intrinsic evaluation measures the quality of the model independent of any application. We choose a metric, train the model on training set, tune the hyperparameters on the dev set, and evaluate on the test set. The test set has to be large enough to give the statistical power to measure a statistically significant difference between two potential models.

### What is perplexity?
Perplexity is the intrinsic metric we use for evaluating language models. It is the inverse probability normalized by the number of words (`N`) in the test set. For the general case, it is:
PP(W) = P(w_0,N-1) ^ -1/N

For the bigram language model, it is:
PP(W) = pi(1/P(w_i|w_i-1), i in [0, N-1]) ^ 1/N

Lower perplexity means better language model.

### What is perplexity in terms of weighted average branching factor?
Perplexity can be thought of also as `weighted average branching factor` of a language, where the branching factor is the number of possible next words that can follow any word.

### Is the perplexity lower for unigram LM, bigram LM, or trigram LM?
The perplexity is lowest for trigram LM followed by bigram LM. The more information we have about the word sequence, the better is the LM probability estimation, and hence the lower the perplexity.

### What is sampling from a language model?
Sampling from a language model means to generate sentences according to how likely they are predicted by the model.

### What are three problems with probabilities predicted by language models?
Genre variations, dialect variations, and sparsity

### what are unknown words or out of vocabulary words and how do we deal with them?
In an open vocabulary system, the test set can potentially contain words that are not in the training set. To model robustness with these unknown words, we choose a vocabulary in advance, map any word in the training set that does not belong to the vocabulary to <UNK> and finally model the probabilities for <UNK>.

### How do we handle words that are not unknown but appear in the test set in new contexts?
Because these contexts or sequences are new, the language model assigns zero probability to them. To prevent this, we do smoothing or discounting. What this is it discounts some probability mass from more frequent events and assign that to events that are never seen. Some approaches to smoothing are Laplace (add-one) smoothing, add-k smoothing, stupid backoff, and Kneser-Ney smoothing.

### If P(w_i) = c_i/N, what is P_Laplace(w_i)?
(c_i + 1)/(N + V)

### If P(w_i) = c_i/N, what is P_Add-k(w_i|w_i-1)?
(C(w_i-1,i) + k)/(C(w_i-1) + kV)

### What is the relationship between perplexity (PP) and cross-entropy (H?
PP(W) = 2^H(W) 