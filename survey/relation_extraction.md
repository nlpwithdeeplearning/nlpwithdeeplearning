---
layout: page
title: Relation Extraction
---

Relation Extraction is the task of predicting attributes and relations for entities in a sentence. For example, given a sentence “Barack Obama was born in Honolulu, Hawaii.”, a relation classifier aims at predicting the relation of “bornInCity”. Relation Extraction is the key component for building relation knowledge graphs, and it is of crucial significance to natural language processing applications such as structured search, sentiment analysis, question answering, and summarization.

1. Subtasks
    1. Relation Classification: Relation Classification is the task of identifying the semantic relation holding between two nominal entities in text.
        1. Datasets
            1. FewRel
            1. TACRED
        1. Papers
            1. 2019 - Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing: Classifying semantic relations between entity pairs in sentences is an important task in Natural Language Processing (NLP). Most previous models for relation classification rely on the high-level lexical and syntactic features obtained by NLP tools such as WordNet, dependency parser, part-of-speech (POS) tagger, and named entity recognizers (NER). In addition, state-of-the-art neural models based on attention mechanisms do not fully utilize information of entity that may be the most crucial features for relation classification. To address these issues, we propose a novel end-to-end recurrent neural model which incorporates an entity-aware attention mechanism with a latent entity typing (LET) method. Our model not only utilizes entities and their latent types as features effectively but also is more interpretable by visualizing attention mechanisms applied to our model and results of LET. Experimental results on the SemEval-2010 Task 8, one of the most popular relation classification task, demonstrate that our model outperforms existing state-of-the-art models without any high-level features.
            1. 2019 - Enriching Pre-trained Language Model with Entity Information for Relation Classification
    1. Joint Entity and Relation Extraction
        1. Datasets
            1. WebNLG
            1. DocRED
            1. SciERC
            1. ACE 2005
            1. RadGraph
        1. Papers
            1. 2021 EMNLP - REBEL: Relation Extraction By End-to-end Language generation: Extracting relation triplets from raw text is a crucial task in Information Extraction, enabling multiple applications such as populating or validating knowledge bases, factchecking, and other downstream tasks. However, it usually involves multiple-step pipelines that propagate errors or are limited to a small number of relation types. To overcome these issues, we propose the use of autoregressive seq2seq models. Such models have previously been shown to perform well not only in language generation, but also in NLU tasks such as Entity Linking, thanks to their framing as seq2seq tasks. In this paper, we show how Relation Extraction can be simplified by expressing triplets as a sequence of text and we present REBEL, a seq2seq model based on BART that performs end-to-end relation extraction for more than 200 different relation types. We show our model's flexibility by fine-tuning it on an array of Relation Extraction and Relation Classification benchmarks, with it attaining state-of-the-art performance in most of them.
    1. Relationship Extraction (Distant Supervised)
        1. Datasets
            1. New York Times Annotated Corpus
        1. Papers
            1. 2021 ACL - KGPool: Dynamic Knowledge Graph Context Selection for Relation Extraction: We present a novel method for relation extraction (RE) from a single sentence, mapping the sentence and two given entities to a canonical fact in a knowledge graph (KG). Especially in this presumed sentential RE setting, the context of a single sentence is often sparse. This paper introduces the KGPool method to address this sparsity, dynamically expanding the context with additional facts from the KG. It learns the representation of these facts (entity alias, entity descriptions, etc.) using neural methods, supplementing the sentential context. Unlike existing methods that statically use all expanded facts, KGPool conditions this expansion on the sentence. We study the efficacy of KGPool by evaluating it with different neural models and KGs (Wikidata and NYT Freebase). Our experimental evaluation on standard datasets shows that by feeding the KGPool representation into a Graph Neural Network, the overall method is significantly more accurate than state-of-the-art methods.
        1. Metrics
            1. P @ 10%
            1. P @ 30%
            1. AUC
            1. Average Precision
    1. Dialog Relationship Extraction
        1. Datasets
            1. DialogRE
            1. DDRel
        1. Papers
            1. 2020 ACL - Dialogue-Based Relation Extraction: We present the first human-annotated dialogue-based relation extraction (RE) dataset DialogRE, aiming to support the prediction of relation(s) between two arguments that appear in a dialogue. We further offer DialogRE as a platform for studying cross-sentence RE as most facts span multiple sentences. We argue that speaker-related information plays a critical role in the proposed task, based on an analysis of similarities and differences between dialogue-based and traditional RE tasks. Considering the timeliness of communication in a dialogue, we design a new metric to evaluate the performance of RE methods in a conversational setting and investigate the performance of several representative RE methods on DialogRE. Experimental results demonstrate that a speaker-aware extension on the best-performing model leads to gains in both the standard and conversational evaluation settings. DialogRE is available at https://dataset.org/dialogre/.

# Next
[Reading Comprehension](/survey/comprehension.md)