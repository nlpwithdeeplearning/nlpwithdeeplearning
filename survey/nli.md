---
layout: page
title: Natural Language Inference
---

**Natural language inference (NLI)** is the task of determining whether a "hypothesis" is 
true (entailment), false (contradiction), or undetermined (neutral) given a "premise".

Example:

| Premise | Label | Hypothesis |
| --- | ---| --- |
| A man inspects the uniform of a figure in some East Asian country. | contradiction | The man is sleeping. |
| An older and younger man smiling. | neutral  | Two men are smiling and laughing at the cats playing on the floor. |
| A soccer game with multiple males playing. | entailment | Some men are playing a sport. |

Approaches used for NLI include earlier symbolic and statistical approaches to more recent deep learning approaches. Benchmark datasets used for NLI include SNLI, MultiNLI, SciTail.

Papers
    1. T5
    1. BERT

## Cross-Lingual Natural Language Inference
Using data and models available for one language for which ample such resources are available (e.g., English) to solve a natural language inference task in another, commonly more low-resource, language.

## Visual Entailment
Visual Entailment (VE) - is a task consisting of image-sentence pairs whereby a premise is defined by an image, rather than a natural language sentence as in traditional Textual Entailment tasks. The goal is to predict whether the image semantically entails the text.

# Next
[Visual Question Answering](survey/visualqa.md)