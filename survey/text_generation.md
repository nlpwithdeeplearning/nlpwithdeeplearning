---
layout: page
title: Text Generation
---

Text generation is the task of generating text with the goal of appearing indistinguishable to human-written text. This task if more formally known as "natural language generation" in the literature.

Text generation can be addressed with Markov processes or deep generative models like LSTMs. Recently, some of the most advanced methods for text generation include BART, GPT and other GAN-based approaches. Text generation systems are evaluated either through human ratings or automatic evaluation metrics like METEOR, ROUGE, and BLEU.

## Dialogue Generation
Dialogue Generation is a fundamental component for real-world virtual assistants such as Siri and Alexa. It is the text generation task that automatically generate a response given a post by the user.

## Data-to-Text Generation
Data-to-text generation is the task of generating text from a data source.

## Multi-Document Summarization
Multi-Document Summarization is a process of representing a set of documents with a short piece of text by capturing the relevant information and filtering out the redundant information. Two prominent approaches to Multi-Document Summarization are extractive and abstractive summarization. Extractive summarization systems aim to extract salient snippets, sentences or passages from documents, while abstractive summarization systems aim to concisely paraphrase the content of the documents.

## Text Style Transfer
Transfer Text from one Style to Another
### Word Attribute Transfer
Changing a word's attribute, such as its gender.

## Datasets
    1. BookCorpus
    1. DailyDialog
    1. PERSONA-CHAT
    1. Billion Word Benchmark
    1. COCO Captions
    1. VQG
    1. Sentence Compression
    1. E2E
    1. MTNT
    1. WritingPrompts

## Papers
    1. BART: BART is a denoising autoencoder for pretraining sequence-to-sequence models. It is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Transformer-based neural machine translation architecture. It uses a standard seq2seq/NMT architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). This means the encoder's attention mask is fully visible, like BERT, and the decoder's attention mask is causal, like GPT2.
    1. GPT-3

# Next
[Word Embeddings](/survey/word_embeddings.md)