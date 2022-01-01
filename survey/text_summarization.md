---
layout: page
title: Text Summarization
---

Shortening a set of data computationally, to create a summary that represents the most important or relevant information within the original content. Automatic Document Summarization is the task of rewriting a document into its shorter form while still retaining its important content. The most popular two paradigms are extractive approaches and abstractive approaches. Extractive approaches generate summaries by extracting parts of the original document (usually sentences), while abstractive methods may generate new words or phrases which are not in the original document. 

## Abstractive Text Summarization
Abstractive Text Summarization is the task of generating a short and concise summary that captures the salient ideas of the source text. The generated summaries potentially contain new phrases and sentences that may not appear in the source text.

## Extractive Text Summarization
Given a document, selecting a subset of the words or sentences which best represents a summary of the document.

## Multi-Document Summarization
Multi-Document Summarization is a process of representing a set of documents with a short piece of text by capturing the relevant information and filtering out the redundant information.

## Datasets
    1. Pubmed
    1. CNN/Daily Mail
    1. Reddit
    1. NEWSROOM
    1. KP20k
    1. Sentence Compression
    1. WikiHow
    1. LCSTS
    1. Multi-New
    1. New York Times Annotated Corpus

 ## Papers
    1. 2017 ACL - Get To The Point: Summarization with Pointer-Generator Networks - Neural sequence-to-sequence models have provided a viable new approach for abstractive text summarization (meaning they are not restricted to simply selecting and rearranging passages from the original text). However, these models have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways. First, we use a hybrid pointer-generator network that can copy words from the source text via pointing, which aids accurate reproduction of information, while retaining the ability to produce novel words through the generator. Second, we use coverage to keep track of what has been summarized, which discourages repetition. We apply our model to the CNN / Daily Mail summarization task, outperforming the current abstractive state-of-the-art by at least 2 ROUGE points.
    1. 2020 ACL - BART
    1. 2020 ICML - PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization - Recent work pre-training Transformers with self-supervised objectives on large text corpora has shown great success when fine-tuned on downstream NLP tasks including text summarization. However, pre-training objectives tailored for abstractive text summarization have not been explored. Furthermore there is a lack of systematic evaluation across diverse domains. In this work, we propose pre-training large Transformer-based encoder-decoder models on massive text corpora with a new self-supervised objective. In PEGASUS, important sentences are removed/masked from an input document and are generated together as one output sequence from the remaining sentences, similar to an extractive summary. We evaluated our best PEGASUS model on 12 downstream summarization tasks spanning news, science, stories, instructions, emails, patents, and legislative bills. Experiments demonstrate it achieves state-of-the-art performance on all 12 downstream datasets measured by ROUGE scores. Our model also shows surprising performance on low-resource summarization, surpassing previous state-of-the-art results on 6 datasets with only 1000 examples. Finally we validated our results using human evaluation and show that our model summaries achieve human performance on multiple datasets.

# Next
[Named Entity Recognition](/survey/ner.md)