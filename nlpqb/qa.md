---
layout: page
title: Question Answering
---

### What are the two major paradigms of question answering?
1. Information-retrieval based QA: sometimes called open domain QA. They rely on the vast amount of text on the web or in collections of scientific papers like PubMed. Given a user question, IR is used to find relevant passages. Then neural reading comprehension algorithms read these retrieved passages and draw an answer directly from spans of text.
1. Knowledge based QA: A system builds a semantic representation of the query. These meaning representations are then used to query databases of facts.

### What are factoid questions?
Questions that can be answered with simple facts expressed in short texts.

### What are the different kins of question answering?
1. Factoid question answering
1. Long-form question answering
1. Community question answering
1. Answering questions on human exams

### What is information retrieval?
IR is the field encompassing the retrieval of all manner of media based on user information needs. The resulting IR system is often called a search engine.

### What is ad hoc retrieval?
The task in which a user poses a query to a retrieval system, which then returns an ordered set of documents from some collection. A document refers to whatever unit of text the system indexes and retrieves. A collection refers to a set of documents being used to satisy user requests. A term refers to a word in a collection, but it may also include phrases. Finally, a query represents a user's information need expressed as a set of terms.

### What is term weight?
Term weight is a better alternative to raw word counts used in IR. Two kinds of term weights are: 1) tf-idf and 2) BM25. BM25 is slightly more powerful.

### What is the BM25 score formula of a document `d` given a query `q`?
`sum(log(N/df_t) * tf_(t,d)/(k(1 -b + b*|d|/|d_avg|) + tf_(t,d)) , t in q)`

### What is a stop list?
 The list of high-frequency words to be removed.

### What is an inverted index?
An inverted index is a data structure we use for making search efficient and storing document frequency and count convenient. It consists of two parts, a dictionary and the postings. The dictionary is a lost of terms. Each term in the dictionary points to a postings list, which is a list of document IDs associated with each term.

### What is a common way to visualize precision and recall?
Precision-recall curve

### What is interpolated precision?
maximum precision value achieved at any level of recall at or above the one we’re calculating

### What is mean average precision?
```
AP = sum(Precision_r(d), d in R_r)/|R_r|
MAP = sum(AP(q), q in Q)/|Q|
```

### What is vocabulary mismatch problem?
The user posing a query might not guess exactly what words the writer of the answer might have used to discuss the issue. Tf-idf or BM25 only work if there is exact overlap of words between the query and document.

### What is the bi-encoder approach?
We use two separate encoder models, one to encode the one query and the other to encode the document. The dot product of the vectors coming out of them are used as the relevance score. 

### What is the dominant paradigm for IR-based QA?
The dominant paradigm for IR-based QA is the retrieve and read model. In the first stage of this 2-stage model we retrieve relevant passages from a text collection, usually using a search engine. In the second stage, a neural reading comprehension algorithm passes over each passage and finds spans that are likely to answer the question.

### What is the reading comprehension task?
Reading comprehension systems are given a factoid question q and a passage p that could contain the answer, and return an answer s (or perhaps declare that there is no answer in the passage, or in some setups make a choice from a set of possible answers).

### What is open domain QA?
They are given a factoid question and a large document collection (such as Wikipedia or a crawl of the web) and return an answer, usually a span of text extracted from a document.

### What are some IR-based QA datasets?
1. Stanford Question Answering Dataset (SQuAD) consists of passages from Wikipedia and associated questions whose answers are spans from the passage
1. Squad 2.0 in addition adds some questions that are designed to be unanswerable
1. HotpotQA dataset was created by showing crowd workers multiple context documents and asked to come up with questions that require reasoning about all of the documents
1. TriviaQA dataset contains 94K questions written by trivia enthusiasts, together with supporting documents from Wikipedia and the web resulting in 650K question-answer-evidence triples
1. Natural Questions dataset incorporates real Natural Questions anonymized queries to the Google search engine
1. TyDi QA dataset contains 204K question-answer pairs from 11 typologically diverse languages, including Arabic, Bengali, Kiswahili, Russian, and Thai

### What is answer span extraction?
The answer extraction task is commonly modeled by span labeling: identifying in the passage a span (a continuous string of text) that constitutes an answer. Neural span algorithms for reading comprehension are given a question q of n tokens q1,...,qn and a passage p of m tokens p1,..., pm. Their goal is thus to compute the probability P(a|q, p) that each possible span a is the answer.

### What is a standard baseline algorithm for reading comprehension?
A standard baseline algorithm for reading comprehension is to pass the question and passage to any encoder like BERT, as strings separated with a [SEP] token, resulting in an encoding token embedding for every passage token p_i. 

We’ll also need to add a linear layer that will be trained in the fine-tuning phase to predict the start and end position of the span. We’ll add two new special vectors: a span-start embedding S and a span-end embedding E, which will be learned in fine-tuning.

To get a span-start probability for each output token `p_i`, we compute the dot product between S and the output token and then use a softmax to normalize over all tokens.

We do the analogous thing to compute a span-end probability.

The score of a candidate span from position i to j is `S . p_i + E . p_j` , and the highest scoring span in which j ≥ i is chosen is the model prediction.

The training loss for fine-tuning is the negative sum of the log-likelihoods of the correct start and end positions for each instance.

### What is entity linking?
Entity linking is the task of associating a mention in text with the representation of some real-world entity in an ontology.

### What is wikification?
The most common ontology for factoid question-answering is Wikipedia, since Wikipedia is often the source of the text that answers the question. In this usage, each unique Wikipedia page acts as the unique id for a particular entity. This task of deciding which Wikipedia page corresponding to an individual is being referred to by a text mention has its own name: wikification

### What are the two stages of entity linking?
mention detection and mention disambiguation

### What is an example wikification algorithm?
TagMe

### What is an example neural graph-based linking algorithm?
ELQ linking algorithm that uses biencoders.

### What are two common paradigms for knowledge-based question answering?
1. Graph-based QA
1. Semantic parsing

### What are some datasets for QA using semantic parsing?
1. GeoQuery
1. Drop
1. ATIS

### How can transformers be used for QA?
The T5 system is an encoder-decoder architecture. In pretraining, it learns to fill in masked spans of task (marked by <M>) by generating the missing spans (separated by <M>) in the decoder. It is then fine-tuned on QA datasets, given the question, without adding any additional context or passages.

### What are the four broad stages of Watson QA that won the Jeopardy! challenge?
The 4 broad stages of Watson QA: (1) Question Processing, (2) Candidate Answer Generation, (3) Candidate Answer Scoring, and (4) Answer Merging and Confidence Scoring.

### How do we evaluate factoid question answering?
mean reciprocal rank = MRR = sum(1/rank_i)/|Q|