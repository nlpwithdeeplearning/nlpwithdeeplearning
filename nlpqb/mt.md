---
layout: page
title: Machine Translation
---
### What is machine translation?
The use of computers to translate from one language to another.

### What is localization?
The task of adapting content or a product to a particular language community, for example through the use of computer-aided translation followed by post-editing.

### What is encoder-decoder?
An encoder-decoder network also known as sequence to sequence network is an architecture that can be implemented with RNNs or Transformers. It is used for mapping a sequence of input words or tokens to a sequence of tags that are not merely direct mappings from individual words.

### What are some applications of encoder-decoder models?
1. Machine translation where we map from source language to target language
1. Summarization where we map from long text to summary
1. dialogue where we map from what the user said to what our dialogue system should respond
1. semantic parsing where we map from a string of words to a semantic representation such as meaning representation language (MRL)

### What is a linguistic typology?
The study of the systematic cross-linguistics similarities and differencies.

### How do you classify languages by their word order typology?
German, French, English, and Mandarin are all SVO (Subject-Verb-Object) languages.

Hindi and Japanese are SOV languages.

Irish and Arabic are VSO languages.

### Which languages have prepositions and postpositions?
SVO and VSO languages have prepositions.

SOV languages have postpositions.

### What is a lexical gap?
Lexical gap is when no word or phrase, short of an explanatory footnote, can express the exact meaning of a word in another language.

### How do verb-framed languages and satellite-framed languages differ?
Verb-framed languages such as Spanish mark the direction of motion on the verb leaving the satellites to mark the manner of motion.

Satellite-framed languages such as English mark the direction of motion on the satellite leave the verb to mark the manner of motion.

### What are isolating vs polysynthetic languages?
Isolating languages such as Vietnamese and Cantonese have one morpheme per word.

Polysynthetic languages such as Siberian Yupik ("Eskimo") have very morphermes per word corresponding to a whole sentence in English.

### What are agglutinative vs fusion languages?
Agglutinative languages such as Turkish have relatively clean boundaries between morphemes.

Fusion languages such as Russian have a fusion of distinct morphological categories such as instrumental, singular, and first declension. For example, a single affix may conflate multiple morphemes.

### What are pro-drop languages?
Languages that can omit pronouns.

### What is referential density?
The density of pronouns used in a language.

### What are cold vs hot languages?
Cold languages are referentially sparse languages like Chinese or Japanese that require the hearer to do more inferential work to recover antecedents.

Hot languages are referentially dense. They make it explicit and easier for the hearer.

### What are the three components of an encoder-decoder network?
1. An encoder that accepts an input sequence and generates a corresponding sequence of contextualized representations
2. A context vector, c, which is a function of the contextualized representations from the encoder and acts as the input to decoder.
3. A decoder that accepts the context vector and generates an arbitrary length sequence of hidden states from which a corresponding sequence of output states can be obtained.

### What is the bottleneck for the RNN encoder-decoder network and how is it solved?
The final hidden state of the encoder must represent absolutely everything about the meaning of the source text. The attention mechanism allows decoder to get the information from all the hidden states of the encoder, not just the hidden state.

### What is greedy decoding?
Choosing the single most probable token to generate at each step is called greedy decoding.

### What is beam search?
Instead of choosing the best token to generate at each timestep, we keep `k` possible tokens at each time step. This fixed size memory footprint `k` is called beam width.

### What is cross-attention?
```
Q = H_dec_(i-1) * WQ
K = H_enc * WK
V = H_enc * WV
CrossAttention(Q, K, V) = softmax(Q*transpose(K)/sqrt(d_k)) * V
```

### What is the transformer block for the encoder and the decoder?
The encoder block is a typical tranformer block.
The decoder block has an extract cross-attention layer between self-attention layer and feedforward layer.

### Describe the wordpiece algorithms that is used for tokenization in machine translation in addition to BPE?
1. Initialize the wordpiece lexicon with characters
1. Repeat until there are V wordpieces (typically 8K to 32K)
    1. Train an n-gram language model on the training corpus, using the current set of wordpieces.
    1. Consider the set of possible new wordpieces made by concatenaing two wordpieces from the current lexicon. Choose the one new wordpiece that most increases the language model probability of the training corpus.

### Name some parallel corpora (bitexts) available for machine translation?
Europarl corpus, OpenSubtitles corpus, and ParaCrawl corpus

### What is backtranslation?
Backtranslation is a way of creating synthetic bitexts by training an intermediate target-to-source MT system on small bitext to translate the monolingual target data to the source language.

### What is Monte Carlo search or Monte Carlo decoding?
In Monte Carlo decoding, at each timestep, instead of always generating the word with the highest softmax probability, we roll a weighted die, and use it to choose the next word according to its softmax probability.

### What are the two dimensions to measure MT?
1. Adequacy (also called faithfulness or fidelity)
1. Fluency

### How is chrF calculated for automatic evaluation of MT?
We first calculate chrP (character precision) and chrR (character recall) and use their weighted harmonic mean to get chrF.

chrP is the percentage of character 1-grams, 2-grams, ..., k-grams in the hypothesis that occur in the reference, averaged.

chrP is the percentage of character 1-grams, 2-grams, ..., k-grams in the reference that occur in the hypothesis, averaged.