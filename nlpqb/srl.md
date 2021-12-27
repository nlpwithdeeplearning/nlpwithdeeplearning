---
layout: page
title: Semantic Role Labeling
---

### What are semantic roles?
Semantic roles are abstract models of the role an argument plays in the event described by the predicate.

### What are thematic roles?
Thematic roles are a model of semantic roles based on a single finite list of roles. Other semantic role models include per-verb semantic role lists and proto-agent/proto-patient, both of which are implemented in PropBank, and per-frame role lists, implemented in FrameNet.

### What is semantic role labeling?
Semantic role labeling is the task of assigning semantic role labels to the constituents of a sentence. The task is generally treated as a supervised machine learning task, with models trained on PropBank or FrameNet. Algorithms generally start by parsing a sentence and then automatically tag each parse tree node with a semantic role. Neural models map straight from words end-to-end.

### What are semantic selectional restrictions?
Semantic selectional restrictions allow words (particularly predicates) to post constraints on the semantic properties of their argument words. Selectional preference models (like selectional association or simple conditional probability) allow a weight or probability to be assigned to the association between a predicate and an argument word or class.