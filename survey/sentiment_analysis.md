---
layout: page
title: Sentiment Analysis
---

Sentiment analysis is the task of classifying the polarity of a given text. For instance, a text-based tweet can be categorized into either "positive", "negative", or "neutral". Given the text and accompanying labels, a model can be trained to predict the correct sentiment.

Sentiment analysis techniques can be categorized into machine learning approaches, lexicon-based approaches, and even hybrid methods. Some subcategories of research in sentiment analysis include: multimodal sentiment analysis, aspect-based sentiment analysis, fine-grained opinion analysis, language specific sentiment analysis.

More recently, deep learning techniques, such as RoBERTa and T5, are used to train high-performing sentiment classifiers that are evaluated using metrics like F1, recall, and precision. To evaluate sentiment analysis systems, benchmark datasets like SST, GLUE, and IMDB movie reviews are used.

### Datasets
    1. SST: The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.Each phrase is labelled as either negative, somewhat negative, neutral, somewhat positive or positive. The corpus with all 5 labels is referred to as SST-5 or SST fine-grained. Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.
    1. GLUE: General Language Understanding Evaluation (GLUE) benchmark is a collection of nine natural language understanding tasks, including single-sentence tasks CoLA and SST-2, similarity and paraphrasing tasks MRPC, STS-B and QQP, and natural language inference tasks MNLI, QNLI, RTE and WNLI.
    1. IMDB movie review: The IMDb Movie Reviews dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. No more than 30 reviews are included per movie. The dataset contains additional unlabeled data.

### Papers
    1. 2019 - BERT
    1. 2017 EACL - fastText - Bag of Tricks for Efficient Text Classification: fastText embeddings exploit subword information to construct word embeddings. Representations are learnt of character n-grams, and words represented as the sum of the n-gram vectors. This extends the word2vec type models with subword information. This helps the embeddings understand suffixes and prefixes. Once a word is represented using character n-grams, a skipgram model is trained to learn the embeddings. Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. We can train fastText on more than one billion words in less than ten minutes using a standard multicore~CPU, and classify half a million sentences among~312K classes in less than a minute.
    1. 2015 NeurIPS - Character-level Convolutional Networks for Text Classification: We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.

### Aspect-Based Sentiment Analysis
Aspect-based sentiment analysis is the task of identifying fine-grained opinion polarity towards a specific aspect associated with a given target.

### Multimodal Sentiment Analysis
Multimodal sentiment analysis is the task of performing sentiment analysis with multiple data sources - e.g. a camera feed of someone's face and their recorded speech.

### Aspect Sentiment Triplet Extraction
Aspect Sentiment Triplet Extraction (ASTE) is the task of extracting the triplets of target entities, their associated sentiment, and opinion spans explaining the reason for the sentiment.

### Twitter Sentiment Analysis
Twitter sentiment analysis is the task of performing sentiment analysis on tweets from Twitter.


# Next
[Text Generation](/survey/text_generation.md)