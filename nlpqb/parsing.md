---
layout: page
title: Constituency Parsing
---

### What is structural ambiguity?
Structural ambiguity occurs when the grammar can assign more than one parse to a sentence. Two common kinds are attachment ambiguity and bracketing ambiguity.

### What are the common sources of structural ambiguity?
1. PP-attachment
1. coordination ambiguity
1. noun-phrase bracketing ambiguity

### What is CKY?
CKY uses dynamic programming to efficiently parse ambiguous sentences from a table of partial parses. It restricts to CNF, compactly represents all possible parses, and doesn't choose a single best parse.

### How do we convert a CFG grammar to CNF?
1. Copy all conforming rules to the new grammar unchanged.
1. Convert terminals within rules to dummy non-terminals.
3. Convert unit productions.
4. Make all rules binary and add them to new grammar.

### How can we choose a single parse from all possible parses?
Neural constituency parsers

### How do span-based neural constituency parsers work?
Span-based neural constituency parsers train a neural network to assign a score to each constituent, and then use a modified version of CKY to combine these constituent scores to find the best-scoring parse tree.

### How do you compute scores for a span in a neural constituency parser?
1. map to subwords
1. encode the subwords
1. get word embeddings (e.g., use the embedding of the last subword)
1. get fence embeddings by using right half of previous word and left half of next word
1. get span vector by subtracting the fence embedding vectors
1. Use MLP to get a score

### What is the CKY variant for integrating span scores into a parse?
```
s_best(i, i+1) = max_l s(i, i+1, l)
s_best(i, j) = max_l s(i, j, l) + max_k[s_best(i, k) + s_best(k, j)]
```

### What is supertagging?
Supertagging is the equivalent of POS tagging in highly lexicalized grammar frameworks such as CCG. The tags are very grammatically rich and dictate much of the derivation for a sentence.

### What are the metrics used for parsing?
Labeled recall, labeled precision, and cross-brackets.

### What is the PARSEVAL metics?
It measures how much the constituents in the hypothesis parse tree look like the constituents in hand-labeled reference parse. For this, it uses the labeled recall and labeled precision.

### What is labeled recall?
Number of correct constituents in hypothesis parse/Number of correct constituents in reference parse

### What is labeled precision?
Number of correct constituents in hypothesis parse/Number of total constituents in hypothesis parse

### What is cross-brackets?
The number of constituents for which the reference parse has a bracketing such as ((A B) C) but the hypothesis parse has a bracketing such as (A (B C)).

### What is chunking?
Chunking is the process of identifying and classifying the flat, non-overlapping segments of a sentence that constitute the basic non-recursive phrases corresponding to the major content-word POS: noun phrases, verb phrases, adjective phrases, and prepositional phrases.

### What are two methods for identifying shallow syntactic constituents in a text?
Partial parsing and chunking. They are solved by sequence models using BIO notation.

### What are some applications of parse trees?
1. grammar checking
1. semantic analysis
1. question answering
