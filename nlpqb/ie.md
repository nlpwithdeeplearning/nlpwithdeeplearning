---
layout: page
title: Information Extraction
---

### What is information extraction?
Information extraction (IE) turns unstructured information embedded in texts into structured data, for example for populating a relational database to enable further processing.

### What is relation extraction?
Finding and classifying semantic relations among entities mentioned in a text, for example to populate a relational database or knowledge graphs.

### What is event extraction?
The task of event extraction is to identify mentions of events in texts. Events are to be classified as actions, states, reporting events (say, report, tell, explain), perception events, and so on. The aspect, tense, and modality of each event also needs to be extracted.

### What is a lexico-syntactic oe Hearst pattern for finding hyponyms?
NP_0 such as NP_1{,NP_2,...,(and|or)NP_i}, i >= 1

### How do we build a relation extraction model?
Through a linear layer on the top of an encoder (e.g., BERT), with the subject and object entities replaced in the input by their NER tags.

### How do we do semisupervised relation extraction via bootstrapping?
```
gather a set of seed tuples that have relation R
while true:
    find sentences that contain entities in tuples
    generalize the context between and around entities in sentences to get new patterns
    use the new patterns to grep for more tuples
    add to the tuples
```

### What is distant supervision for relation extraction?
The distant supervision method combines the advantages of bootstrapping with supervised learning. Instead of just a handful of seeds, distant supervision uses a large database to acquire a huge number of seed examples, creates lots of noisy pattern features from all these examples and then combines them in a supervised classifier.
```
for R in relations:
    for (e0, e1) with relation R in database:
        find sentences in text that contain e0 and e1
        add new training data to existing observations
train supervised classifier on observations
```

### How are relation extraction systems evaluated?
For supervised systems, we use test sets with human-annotated, gold standard relations, and compute precision, recall, and F-measure.

For semi-supervised and unsupervised with no access to tests, we calculate approximate precision on a random sample of relations from the output.

### How can relations among entities be extracted?
1. Pattern-based approaches
1. Supervised learning approaches when annotated training data is available
1. Lightly supervised bootstrapping methods when small numbers of seed tuples or seed patterns are available
1. Distant supervision when a database of relations is available
1. Unsupervised or Open IE methods

### How can we do reasoning about time?
By detection and normalization of temporal expressions. Temporal events can be detected and ordered in time using sequence models and classifiers trained on temporally and event-labeled data like the TimeBank corpus.

### How do template-filling expressions work?
They recognize stereotypical situations in texts and assign elements from the text to roles represented as fixed sets of slots.

### What are temporal expressions?
Temporal expressions are those that refer to absolute points in time, relative times, durations, and sets of these. They have temporal lexical triggers (noun, proper noun, adjective or adverb).

### What is TimeML?
TimeML is an annotation scheme in which temporal expressions are annotated with an XML tag, TIMEX3.

### What is temporal normalization?
Temporal normalization is the process of mapping a temporal expression to either a specific point of time or to a duration.

### What is a fully qualified date expression?
Fully qualified date expressions contain a year, month, and day in some conventional form.

### What is temporal anchor?
Fully qualified temporal expressions are fairly rare in real texts. Most temporal expressions, e.g. in news articles, are incomplete and are only implicitly anchored, often with respect to the dateline of the article. We refer to this dateline as the document’s temporal anchor.

### What is the TimeBank corpus?
The TimeBank corpus consists of 183 news articles selected from a variety of sources, including the Penn TreeBank and PropBank collections. Each article in the TimeBank corpus has had the temporal expressions and event mentions in them explicitly annotated in the TimeML annotation. In addition to temporal expressions and events, the TimeML annotation provides temporal links between events and temporal expressions that specify the nature of the relation between them.

### What is the machine learning approach to template filling?
The task is generally modeled by training two separate supervised systems. The first system decides whether the template is present in a particular sentence. This task is called template recognition. The second system has the job of role-filler extraction. A separate classifier is trained to detect each role (LEAD-AIRLINE, AMOUNT, and so on). This can be a
binary classifier that is run on every noun-phrase in the parsed input sentence, or a sequence model run over sequences of words.