---
layout: page
title: Semantic Textual Similarity
---

Semantic textual similarity deals with determining how similar two pieces of texts are. This can take the form of assigning a score from 1 to 5. Related tasks are paraphrase or duplicate identification.
## Datasets
    1. STS Benchmark

# Paraphrase Identification
The goal of Paraphrase Identification is to determine whether a pair of sentences have the same meaning.
1. Datasets
    1. Quora Question Pairs
1. Papers
    1. 2021 - Entailment as Few-Shot Learner: EFL, that can turn small LMs into better few-shot learners. The key idea of this approach is to reformulate potential NLP task into an entailment one, and then fine-tune the model with as little as 8 examples. We further demonstrate our proposed method can be: (i) naturally combined with an unsupervised contrastive learning-based data augmentation method; (ii) easily extended to multilingual few-shot learning. A systematic evaluation on 18 standard NLP tasks demonstrates that this approach improves the various existing SOTA few-shot learning methods by 12\%, and yields competitive few-shot performance with 500 times larger models, such as GPT-3.

# Cross-Lingual Semantic Textual Similarity
    1. Datasets
        1. Interpretable STS
    1. Papers
        1. 2021 EMNLP - Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders: Mirror-BERT converts pretrained language models into effective universal text encoders without any supervision, in 20-30 seconds. It is an extremely simple, fast, and effective contrastive learning technique. It relies on fully identical or slightly modified string pairs as positive (i.e., synonymous) fine-tuning examples, and aims to maximise their similarity during identity fine-tuning.

# Next
[Dialogue](/survey/dialogue.md)