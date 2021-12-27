---
layout: page
title: Lexicons for Sentiment, Affect, and Connotation
---

### What are the different kinds of affective states?
1. emotions
1. moods
1. attitudes (includes sentiment)
1. interpersonal stance
1. personality

### How can emotions be represented?
Emotion can be represented by fixed atomic units often called basic emotions, or as points in space defined by dimensions like valence and arousal.

### What are connotational aspects?
Words have connotational aspects related to affective states. This connotational aspect of word meaning can be represented in lexicons.

### How can affective lexicons be built?
Affective lexicons can be built by hand, using crowd sourcing to label the affective content of each word. Lexicons can also be built with semi-supervised approaches such as bootstrapping from seed words using similarity metrics like embedding cosine. Lexicons can also be learned in a fully supervised manner, when a convenient training signal can be found in the worls, such as ratings assigned by users on a review site.

### How can words in a lexicon be assigned weights?
1. Various functions of word counts in training texts
1. Ratio metrics like log odds ratio informative Dirichlet prior

### How can Affect be detected?
1. standard supervised text classification using all the words or bigrams
1. Lexicons and rule-based classifier
1. Connotation frames