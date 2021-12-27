---
layout: page
title: Words
---

## Questions
### Question: What is a corpus?
Answer: A corpus (plural corpora) is a computer-readable collection of text or speech. For example, the Brown corpus is a million-word collection of samples from 500 texts of different genres (e.g., newspaper, fiction, nonfiction, and academic).

### Question: What is an utterance?
Answer: An utterance is the spoken correlate of a sentence.

### Question: What is the Switchboard corpus?
Answer: The Switchboard corpus is a collection of 2430 American English telephone conversations between strangers. They total 240 hours of speech and about 3 million words.

### Question: What is a disfluency?
Answer: A disfluency is a broken-off word (fragment) or a filler or filled pause (e.g., uh and um) in an utterance.

### Question: What is a lemma?
Answer: A lemma is a set of lexical forms having the same stem, the same major part-of-speech, and the same word sense. Dictionary entries or boldface forms roughly correspond to the lemmas; however, some lemmas have multiple boldface forms.

### Wordform: What is a wordform?
Answer: The wordform is the full inflected or derived form of the word.

### What are word types?
Answer: Word types are the set of words in the vocabulary or the number of distinct words in a corpus. Typically the vocabulary of the corpus is represented by V and the number of word types would then be |V|.

### What are word tokens?
Answer: Word tokens are the total number of words, typically represented by N. For example, the Shakespeare corpus has 31 thousand word types and 884 thousand word tokens.

### What is Herdan's law or Heaps' law?
The larger is the corpus, the more are the word types. The formula for this is:
$ |V| = kN^\beta $

### What is the construction in African American Language (AAL) corresponding to "I don't" in Mainstream American English (MAE)?
iont

### What is code switching?
Code switching is the phenomenon of speakers or writes using multiple languages in a single communicative act.

### What is a datasheet or data statement?
A datasheet or a data statement specifies properties of a dataset like:
* Motivation: Why was the data or corpus collected? Who collected it? Who funded it?
* Situation: What was the situation in which the text was written or spoken? E.g., what were the guidelines provided to them?
* Language variety: What language and locale was the data in?
* Speaker demographics: e.g., age, gender, nationality
* Collection process
* Annotation process
* Permissions/licensing

### What does this unix command do: tr -sc 'A-Za-z' '\n' < sh.txt
It tokenizes the words by replacing every non-alphabetic character with a new line.

### What does this unix command do: tr -sc 'A-Za-z' '\n' < sh.txt | sort | uniq -c
It sorts the tokens in alphabetical order and counts them.

### What does this unix command do: tr -sc 'A-Za-z' '\n' < sh.txt | tr A-Z a-z | sort | uniq -c
It sorts the lower-cased tokens in alphabetical order and counts them.

### tr -sc 'A-Za-z' '\n' < sh.txt | tr A-Z a-z | sort | uniq -c | sort -n -r
It sorts the lower-cased tokens in sh.txt in the order of their frequency.

### What is tokenization?
Tokenization is the task of segmenting running text into work tokens.

### What is a clitic contraction?
Clitic contractions are words such as `it's` that can be further tokenized to `it` and `is`.

### What is Penn Treebank tokenization standard?
Penn Treebank tokenization standard is a standard that is released by Linguistic Data Consortium (LDC). It specifies that the clitics should be separated out (`it's` becomes `it` and `'s`), the hyphenated words should be together and all punctuation should be separated out.

### What does this pattern of words represent - ([A-Z]\.)+
Abbreviations

### What does this pattern of words represent - \w+(-\W+)*
Words with optional internal hyphens

### What does this pattern of words represent - \$?\d+(\.\d+)?%?
Currency and percentages

### What does this pattern of words represent - \.\.\.
Ellipsis

### What does this pattern of words represent - [][.,;"'?():-_]
Punctuations marks, etc.

## What is a hanzi?
Hanzi are the characters used to compose words in Chinese.

### What is a morpheme?
A part of a word such as character (e.g., hanzi in chinese) that has a single unit of meaning.

### Does chinese NLP work better with word input or character input?
Chinese NLP can work better with character input since characters themselves are at a reasonable semantic level and the most word standards result in huge vocabulary with rare words?

### Does Japanese and Thai NLP work better with word input or character input?
For Japanese and Thai, we need to operate at a word level. So, special algorithms for `word segmentation` are required using neural sequence models.

### What is the unknown word problem in statistical NLP?
If the training corpus contains words such as `high`, `old`, and `older`, but not `higher`, then the trained model will not know what to do when `higher` is encountered in production.

### What are three widely used algorithms for token segmentation?
Byte-pair encoding, unigram language modeling, and WordPiece. All of them have a token leaner that learns the tokens/vocabulary from training corpus and a model that takes a raw sentences and segments into tokens from the vocabulary.

### What is byte pair encoding?
Byte pair encoding has a token learning phase first that initializes the vocabulary just with set of all individual characters. It then merges two symbols (A, B) that are most frequently adjacent, adds a new symbol (AB), and replaces every adjacent A and B in the corpus with AB. This continues until `k` (hyperparameter) new token are created. 
It then has a token parser step which first segments each word of input raw sentence into characters, and then it applies all the merge rules it learned greedily. Thus, if a word `older` is in the training corpus, the entire word is used for the token. If the word `higher` is not in the token but the words `old`, `older`, and `high`, it would be tokenized into `high` (because the characters `h`, `i`, `g`, and `h` occur together in training corpus) and `er` (because `e` and `r` occur together in training corpus).

### What is word normalization?
Word normalization is the task of mapping word tokens (e.g., United States, USA, United States, America, US, and U.S.A.) to a canonical format. Word normalization is useful for information retrieval and information extraction.

### What is case folding?
Case folding is mapping everything to lower case so that we can generalize for tasks such as information retrieval and speech recognition. Case folding can be detrimental for sentiment analyis, text classification, information extraction, and machine translation because of the semantic and syntactic information present in case.

### What is lemmatization?
Lemmatization is the task of identifying common roots of the words even when them might have surface diffences; e.g., ran and run; am, are, and is.

### What is morphological parsing?
Morphology is the study of the way words are built up from morphemes, the smaller meaning bearing units. There are two broad categories of morphemes - stems and affixes. Stems (e.g., `dog` in `dogs`) provide the central meaning. Affixes provide additional meaning (e.g., `s` in `dogs`).

### What is stemming?
Stemming is the naive version of morphological parsing. For example, Porter stemmer is a widely used stemming algorithm. It is based on a cascade of rules.

### What is sentence segmentation?
Sentence segmentation is segmenting text into sentences using cues like punctuation, abbreviation, and capitalization. It can be based on a separate machine learning model or some rules on tokenization output. An example rule is that sentence ends with an optional quote or bracket followed by punctuation unless that punctuation is already part of another token (e.g., abbreviation, number).

### What is minimum edit distance?
Minimum edit distance between two strings is the minimum number of editing operations (namely insertion, deletion and substitution) needed to perfectly align or match them.

### What is dynamic programming?
Dynamic programming is a tabular approach to solve problems by combining solutions to smaller problems. It is used in edit distance, Viterbi, and CKY parsing.

### Write pseudo code for minimum edit distance
```
def edit_distance(str0:str, str1: str) -> int:
    len0 = len(str0)
    len1 = len(str1)

    # initialize the table, table[len0][len1] is the final output
    table = [[0 for j in range(len1+1)] for i in range(len0+1)]
    
    for i in range(len0+1):
        for j in range(len1+1):
            # if str0 is empty, all of str1 needs to be inserted
            if i==0:
                table[i][j] = j
            # likewise for str1
            elif j==0:
                table[i][j] = i
            # if corresponding characters are matching, then go back diagonally
            elif str0[i-1] == str1[j-1]:
                table[i][j] = table[i-1][j-1]
            # this is either an insertion (move right) or deletion (up) or swap (diagonal) then based on which path is shorter
            else:
                table[i][j] = 1 + min( # 1 -> equal cost assumed
                    table[i][j-1], # right arrow = insert
                    table[i-1][j], # up arrow = delete
                    table[i-1][j-1], # diagonal = replace
                )
    
    return table[len0][len1]

if __name__ == '__main__':
    str0 = 'hare'
    str1 = 'hari'
    ed = edit_distance(str0, str1)
    print(ed)
```
