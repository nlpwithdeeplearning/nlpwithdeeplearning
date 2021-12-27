---
layout: page
title: Dependency Parsing
---

### What are dependency grammars?
Dependency grammars describe the syntactic structure of a sentence solely in terms of directed binary grammatical relations between the words. We also called this a typed dependency structure because the labels from the heads to dependents are drawn from a fixed inventory.

### What is free word order?
Some languages have a more flexible word order. Constituency grammars would need additional rules for each possible place. A dependency grammar approach abtracts away from word order information.

### Provide examples of core Universal Dependency relations.
Clausal Argument Relations
1. NSUBJ (Nominal subject): **United** *canceled* the flight.
1. DOBJ (Direct object): United *diverted* the **flight** to Reno.
1. IOBJ (Indirect object): We *booked* **her** the flight to Miami.
Nominal Modifier Relations
1. NMOD (Nominal modifier): We took the **morning** *flight*.
1. AMOD (Adjectival modifier): Book the **cheapest** *flight*.
1. NUMMOD (Numeric modifier): Before the storm JetBlue canceled **1000** *flights*.
1. APPOS (Appositional modifier): *United*, a **unit** of UAL, matched the fares.
1. DET (Determiner): **Which** *flight* was delayed?
1. CASE (Prepositions, postpositions and other case markers): Book the flight **through** *Houston*.
Other Notable Relations
1. CONJ (Conjunct): We *flew* to Denver and **drove** to Steamboat.
1. CC (Coordinating conjunction): We flew to Denver **and** *drove* to Steamboat.

### What is a dependency tree?
Dependency tree is a directed graph that satisfies the following constraints:
1. There is a single designated root node that has no incoming arcs
1. With the exception of the root node, each vertex has exactly one incoming arc
1. There is a unique path from the root node to each vertex in V

### What is projectivity?
An arc from a head to a dependent is said to be projective if there is a path from the head to every word that lies between the head and the dependent in the sentence.

### Give an example of a nonprojective dependency?
In the sentence, "JetBlue canceled our flight this morning which was already late.", the dependencies are:
```
root(canceled)
nsubj(canceled, JetBlue)
dobj(canceled, flight)
mod(canceled, morning)
det(flight, our)
det(morning, this)
nmod(flight, was)
case(was, which)
mod(was, alte)
adv(late, already)
```

The arc `nmod(flight, was)` is non-projective since there is no path from *flight* to the intervening words *this* and *morning*.

### Which approaches are not okay and which are okay for parsing sentences with non-projective dependencies?
For training data, most English dependency treebanks were automatically derived from phrase-structure treebanks and such trees are guaranteed to be projective. So, we need other sources of data. Transitiona based approaches can only produce projective trees. Only the more flexible graph-based parsing approaches help with non-projective trees.

### What are some rules that can be applied to phrase-structure treebanks to get dependency treebanks?
1. Mark the head child of each node in a phrase structure, using the appropriate head rules.
1. In the dependency structure, make the head of each non-head child depend on the head of the head-child.

### What is transition-based parsing?
Transition-based parsing draws on shift-reduce parsing. It has a stack on which we build the parse, a buffer of tokens to be parsed, and a parser which takes actions on the parse via a predictor called an oracle.

### What are the three transition operators that will operate on the top two elements of the stack?
1. Left arc (for each of the dependency relations being used)
1. Right arc (for each of the dependency relations being used)
1. Shift

### How do we create training data for dependency parsing?
1. Choose LEFTARC if it produces a correct head-dependent relation given the reference parse and the current configuration,
1. Otherwise, choose RIGHTARC if (1) it produces a correct head-dependent relation given the reference parse and (2) all of the dependents of the word at the top of the stack have already been assigned,
1. Otherwise, choose SHIFT.

### How we train a neural classifier for dependency parsing?
The standard architecture is to pass the sentence through an encoder, then take the presentation of the top 2 words on the stack and the first word of the buffer, concatenate them, and present to a feedforward network that predicts the transition to take.

### What is an alternative transition system?
Arc eager

### How can we explore alternative decision sequences?
Using beam search

### What are graph-based methods for creating dependency structures based on?
Maximum spanning tree methods

### How do we evaluate dependency parsers?
Labeled and unlabeled accuracy