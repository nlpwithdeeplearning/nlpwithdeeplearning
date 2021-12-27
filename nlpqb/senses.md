---
layout: page
title: Word Senses and Wordnet
---

### What is a word sense?
A word sense is the locus of word meaning; definitions and meaning relations are defined at the level of the word sense rather than wordforms.

### What is polysemous?
Having many senses.

### What are the different kinds of relations between senses?
Relations between senses include synonymy, antonymy, meronymy, and taxonomic relations hyponymy and hypernymy.

### What is WordNet?
WordNet is a large database of lexical relations for English, and WordNets exist for a variety of languages.

### What is Word-sense disambiguation (WSD)?
WSD is the task of determining the correct sense of a word in context. Supervised approaches make use of a corpus of sentences in which individual words (lexical sample task) or all words (all-words task) are hand-labeled with senses from a resource like WordNet. SemCor is the largest corpus with WordNet-labeled senses.

### What is the standard algorithm for WSD?
The standard supervised algorithm for WSD is nearest neighbors with contextual embeddings. Feature-based algorithms using parts of speech and embeddings of words in the context of the target word also work well. An important baseline for WSD is the most frequent sense, equivalent, in WordNet, to take the first sense. Another baseline is a knowledge-based WSD algorithm called the Lesk algorithm which chooses the sense whose dictionary definition shares the most words with the target word’s neighborhood.

### What is word sense induction?
Word sense induction is the task of learning word senses unsupervised.