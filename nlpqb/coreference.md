---
layout: page
title: Coreference resolution
---

### What is coreference resolution?
The task of linking together mentions in text which corefer, i.e. refer to the same discourse entity in the discourse model, resulting a set of coreference chains (also called clusters or entities).

### What can coreference mentions be?
Mentions can be definite NPs or indefinite NPs, pronouns (including zero pronouns) or names.

### What is the surface form of an entity mention linked to?
Its information status (new, old, or inferrable), and how accessible or salient the entity is.

### Are all NPs referring expressions?
Some NPs are not referring expressions, such as pleonastic *it* in *It is raining*.

### Provide examples of corpora for human-labeled coreference annotations
OntoNotes for English, Chinese, and Arabic, ARRAU for English, and AnCora for Spanish and Catalan

### What can coreference resolution start with?
Mention detection can start with all nouns and named entities and then use anaphoricity classifiers or referentiality classifiers to filter out non-mentions.

### What are the three common architectures for coreference using feature-based or neural classifiers?
1. mention-pair
1. mention-rank
1. entity-based

### How do modern coreference systems work?
Modern coreference systems tend to be end-to-end, performing mention detection and coreference in a single end-to-end architecture. Algorithms learn representations for text spans and heads, and learn to compare anaphor spans with candidate antecedent spans.

### How are coreference systems evaluated?
Coreference systems are evaluated by comparing with gold entity labels using precision/recall metrics like MUC, B3, CEAF, BLANC, or LEA.

### What are Winograd Schema Challenge problems?
The Winograd Schema Challenge problems are difficult coreference problems that seem to require world knowledge or sophisticated reasoning to solve.

### How can we exhibit gender bias?
Coreference systems exhibit gender bias which can be evaluated using datasets like Winobias and GAP.