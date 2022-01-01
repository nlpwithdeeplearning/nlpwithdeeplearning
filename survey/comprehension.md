---
layout: page
title: Reading Comprehension
---

Most current question answering datasets frame the task as reading comprehension where the question is about a paragraph or document and the answer often is a span in the document.

Some specific tasks of reading comprehension include multi-modal machine reading comprehension and textual machine reading comprehension, among others. In the literature, machine reading comprehension can be divide into four categories: cloze style, multiple choice, span prediction, and free-form answer.

Benchmark datasets used for testing a model's reading comprehension abilities include MovieQA, ReCoRD, and RACE, among others.

The Machine Reading group at UCL also provides an overview of reading comprehension tasks: https://uclnlp.github.io/ai4exams/data.html

Subtasks
    1. Machine Reading Comprehension
        1. Datasets
            1. RACE
        1. Papers
            1. 2020 - Improving Machine Reading Comprehension with Single-choice Decision and Transfer Learning: Multi-choice Machine Reading Comprehension (MMRC) aims to select the correct answer from a set of options based on a given passage and question. Due to task specific of MMRC, it is non-trivial to transfer knowledge from other MRC tasks such as SQuAD, Dream. In this paper, we simply reconstruct multi-choice to single-choice by training a binary classification to distinguish whether a certain answer is correct. Then select the option with the highest confidence score. We construct our model upon ALBERT-xxlarge model and estimate it on the RACE dataset. During training, We adopt AutoML strategy to tune better parameters. Experimental results show that the single-choice is better than multi-choice. In addition, by transferring knowledge from other kinds of MRC tasks, our model achieves a new state-of-the-art results in both single and ensemble settings.
    1. Multi-Hop Reading Comprehension
        1. Datasets
            1. MedHop
        1. Papers
            1. 2021 EMNLP - Summarize-then-Answer: Generating Concise Explanations for Multi-hop Reading Comprehension: How can we generate concise explanations for multi-hop Reading Comprehension (RC)? The current strategies of identifying supporting sentences can be seen as an extractive question-focused summarization of the input text. However, these extractive explanations are not necessarily concise i.e. not minimally sufficient for answering a question. Instead, we advocate for an abstractive approach, where we propose to generate a question-focused, abstractive summary of input paragraphs and then feed it to an RC system. Given a limited amount of human-annotated abstractive explanations, we train the abstractive explainer in a semi-supervised manner, where we start from the supervised model and then train it further through trial and error maximizing a conciseness-promoted reward function. Our experiments demonstrate that the proposed abstractive explainer can generate more compact explanations than an extractive explainer with limited supervision (only 2k instances) while maintaining sufficiency.
    1. Logical Reasoning Reading Comprehension
        1. Datasets
            1. ReClor
        1. Papers
            1. ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning: Reading Comprehension dataset requiring logical reasoning (ReClor) extracted from standardized graduate admission examinations. As earlier studies suggest, human-annotated datasets usually contain biases, which are often exploited by models to achieve high accuracy without truly understanding the text. In order to comprehensively evaluate the logical reasoning ability of models on ReClor, we propose to identify biased data points and separate them into EASY set while the rest as HARD set. Empirical results show that state-of-the-art models have an outstanding ability to capture biases contained in the dataset with high accuracy on EASY set. However, they struggle on HARD set with poor performance near that of random guess, indicating more research is needed to essentially enhance the logical reasoning ability of current models.

# Next
[Natural Language Inference](/survey/nli.md)