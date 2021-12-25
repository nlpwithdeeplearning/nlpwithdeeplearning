---
layout: page
title: Parts of Speech and Named Entities
---

### What are the eight main parts of speech in English?
noun, verb, pronoun, preposition, adverb, conjunction, participle, and article.

### What is named entity?
Named entity is anything that can be referred to with a proper name and sometimes extensions.

### What are the useful clues to sentence structure and meaning?
Parts of speech and named entities

### What is part of speech tagging?
Taking a sequence of word tokens and assigning each of them a part of speech like noun or verb.

### What is named entity recognition (NER)?
Assigning words or phrases tags like person, location, or organization.

### What is sequence labeling?
Tasks in which we assign to each word token x_i in an input word sequence, a label y_i so that the output sequence Y has the same length as the input sequence X.

### What are the 17 parts of speech in the Universal Dependencies tagset?
1. Open Class
    1. Adjective
    1. Adverb
    1. Noun
    1. Verb
    1. Proper Noun
    1. Interjection
1. Closed Class Words
    1. Adposition (preposition and postposition)
    1. Auxiliary
    1. Coordinating Conjunction
    1. Determiner
    1. Numeral
    1. Particle
    1. Pronoun
    1. Subordinating Conjunction
1. Other
    1. Punctuation
    1. Symbols like $ or emoji
    1. Other

### What is the goal of POS-tagging?
Words are ambiguous and have more than one possible part of speech. The goal is to find the correct tag for the situation. For example, `book` can be a verb (book that flight) or a nound (I want that book).

### What is the accuracy and SOTA of part of speech algorithms?
97% is the accuracy performance across 15 language from Universal Dependencies (UD) treebank. Accuracies on various English treebanks are also 97% for various algorithms (HMMs, CRFs, BERT) and English.

### What is Most Frequent Class Baseline?
It is assigning each token to the class it occurred in most often in the training set.
It is good practice to compare new classifiers against a baseline at least as good as the most frequent class baseline.

### What is BIO tagging?
In BIO tagging, we label any token that begins a span of interest with the label B, tokens that occur inside a span are tagged with an I, and any tokens outside of any span of interest are labeled O.

### What is a markov chain?
A markov chain is a model that tells us something about the probabilities of sequences of random variables knows as states. Each state can take on values from some set.
Formally, it contains
Q = q_0, q_1, q_Nminus1  # A set of N states
A                     # A transition probability matrix
pi = pi_0, pi_1, ..., pi_Nminus1 # pi_i is the probability that the markov chain will stat in state i

### What is Markov assumption?
When predicting the future, the past doesn't matter, only the present.
P(q_i=a|q_0...q_iminus1) = P(q_i=a|q_iminus1)

### What are the components of a HMM?
Q = q_0, q_1, q_Nminus1  # A set of N states
A                     # A transition probability matrix
pi = pi_0, pi_1, ..., pi_Nminus1 # pi_i is the probability that the markov chain will stat in state i
O = o_0o_1...o_Tminus1  # sequence of T observations drawn from vocabulary
B = b_i(o_t)  # a sequence of observation likelihoods, also called emission probabilities

### What is the Viterbi algorithm?
Viterbi algorithm is the decoding algorithm for HMMs. It used dynamic programming to find the optimal sequence of tags given an observation sequence and an HMM. It outputs the state path through the HMM that assigns maximum likelihood to the observation sequence.

### What is conditional random field (CRF)?
CRF is a discriminative sequence model based on log-linear models.

### How does HMM compute the best tag sequence?
HMM relies on Bayes' rule and the likelihood P(X|Y).

Yhat = argmax_Y p(Y | X)
    = argmax_Y p(X | Y) * p(Y)
    = argmax_Y pi(p(x_i | y_i)) * pi(p(y_i | y_i-1))

### How does CRF compute the best tag sequence?
CRF computes the posterior p(Y|X) directly.

