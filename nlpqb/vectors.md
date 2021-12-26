---
layout: page
title: Vector Semantics
---

### What is distributional hypothesis?
Words that occur in similar contexts tend to have similar meanings.

### What are embeddings?
Vector representations of meaning of words. They are of two types: a) static embeddings (e.g., word2vec), and b) contextualized embeddings (e.g., BERT).

### What is a word sense?
Each lemma can have multiple meanings. Each of these aspects of meanings is called a word sense.

### What is synonymy?
Two words are synonymous if they can substitute each other in any sentence without changing the truth conditions or propositional meaning.

### What is the principle of contrast?
Principle of contrast states that a difference in linguistic form is always associated with some difference in meaning.

### What is a semantic field?
A semantic field is a set of words which cover a particular semantic domain and bear structured relations with each other.

### What are topic models?
Topic models apply unsupervised learning on large sets of texts to induce sets of associated words from text.

### What is a semantic frame?
A semantic frame is a set of words that denote perspectives or participants in a particular type of event.

### What is connotation?
Connotation refers to the aspects of a word's meaning that are related to a writer or reader's emotions, sentiment, opinions, or evaluations.

### What are the three important dimensions of affective meaning?
1. Valence: the pleasantness of stimulus
1. Arousal: the intensity of emotion provoked by the stimulus
1. Dominance: the degree of control exerted by the stimulus

### What is vector semantics?
Vector semantics is the idea that word meaning could be represented as a point in space. 

### What are two commonly used models for vector semantics?
* Tf-idf - results in very long vectors that are sparse
* word2vec - static, short, dense vectors

### What is a term-document matrix?
In a term-document matrix, each row represents a word in the vocabulary and each column represents a document from the corpora.

### What is vector space model?
In a vector space model, each document is represented as a count vector.

### What is a vector?
Vector is a list or array of numbers.

### What is vector space?
Vector space is a collection of vectors, characterized by their dimension.

### How do we use term-document matrices in information retrieval?
The term-document matrices are used to find similar documents. Two documents are similar if their column vector tend to be similar.

### What is information retrieval?
Information retrieval is the task of finding the document `d` from the `D` documents in some collection that best matches a query `q`.

### What is a word in term-document matrix?
row vector

### What are some alternatives to term-document matrix?
1. term-term matrix, also called word-word matrix
1. term-context matrix

### What is cosine similarity?
cosine(v, w) = v . w / (|V| * |W|) = v/|V| . w/|W|

### What is the formula for tf-idf?
tf_t,d = log_10(count(t, d) + 1)
idf_t = log_10(N/df_t)
w_t,d = tf_t,d * idf_t

### What is the formula for positive pointwise mutual information (PPMI)?
PPMI(w, c) = max(log_2(P(w,c)/(P(w)*P(c))), 0)

### What is self-supervision?
Self-supervision, used e.g. in word2vec, avoids the need for any sort of hand-labeled supervision signal by using e.g. the next word in the running text as the supervision signal.

### What is the intuition of skip-gram?
1. Treat the target word and a neighboring context context word as positive examples
1. Randomly sample other words in vocabulary to get negative samples
1. Use logistic regression to train a classifier to distinguish these two cases
1. Use the learned weights as embeddings

### What are some alternatives to word2vec?
fasttext, GloVe

### What is t-SNE?
t-SNE is a projection method to visualize embeddings.

### What are the different types of similarity or association?
1. first-order co-occurrence or syntagmatic association
1. second-order co-occurrence or paradigmatic association

### What is a parallelogram model?
Parallelogram model is used for analogy problem. For example, the semantic vector for `vine` can be found by subtracting `apple` from `tree` and adding `grape`.

### What is allocational harm?
Allocational harm is when a system allocates resources (jobs or credit) unfairly to different groups. For example, algorithms that use embeddings as part of a search for hiring potential programmers or doctors might incorrectly downweight documents with women's names.

### What is bias amplification?
Word embeddings don't just reflect the bias in data but they amplify it. Gendered terms become more gendered in embedding space than in input text and actual labor employment statistics.

### What is debiasing?
Debiasing is effort to remove biases, for example by developing a tranformation of embedding space that removes gender stereotypes by preserves definitional gender or changing the training procedure. This is an open problem.

### How do we evaluate vector models?
1. extrinsic evaluations on NLP tasks
1. computing the correlation between an algorithm's word similarity scores and word similarity ratings assigned by humans