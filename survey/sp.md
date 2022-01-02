---
layout: page
title: Semantic Parsing
---

Semantic Parsing is the task of transducing natural language utterances into formal meaning representations. The target meaning representations can be defined according to a wide variety of formalisms. This include linguistically-motivated semantic representations that are designed to capture the meaning of any sentence such as λ-calculus or the abstract meaning representations. Alternatively, for more task-driven approaches to Semantic Parsing, it is common for meaning representations to represent executable programs such as SQL queries, robotic commands, smart phone instructions, and even general-purpose programming languages like Python and Java.

## Datasets
1. Spider: Spider is a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 Yale students. The goal of the Spider challenge is to develop natural language interfaces to cross-domain databases. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables covering 138 different domains. In Spider 1.0, different complex SQL queries and databases appear in train and test sets. To do well on it, systems must generalize well to not only new SQL queries but also new database schemas.

1. ATIS: The ATIS (Airline Travel Information Systems) is a dataset consisting of audio recordings and corresponding manual transcripts about humans asking for flight information on automated airline travel inquiry systems. The data consists of 17 unique intent categories. The original split contains 4478, 500 and 893 intent-labeled reference utterances in train, development and test set respectively.

1. WikiTableQuestions: WikiTableQuestions is a question answering dataset over semi-structured tables. It is comprised of question-answer pairs on HTML tables, and was constructed by selecting data tables from Wikipedia that contained at least 8 rows and 5 columns. Amazon Mechanical Turk workers were then tasked with writing trivia questions about each table. WikiTableQuestions contains 22,033 questions. The questions were not designed by predefined templates but were hand crafted by users, demonstrating high linguistic variance. Compared to previous datasets on knowledge bases it covers nearly 4,000 unique column headers, containing far more relations than closed domain datasets and datasets for querying knowledge bases. Its questions cover a wide range of domains, requiring operations such as table lookup, aggregation, superlatives (argmax, argmin), arithmetic operations, joins and unions.

1. WebQuestionsSP (WebQuestions Semantic Parses Dataset): The WebQuestionsSP dataset contains full semantic parses in SPARQL queries for 4,737 questions, and “partial” annotations for the remaining 1,073 questions for which a valid parse could not be formulated or where the question itself is bad or needs a descriptive answer. This release also includes an evaluation script and the output of the STAGG semantic parsing system when trained using the full semantic parses. More detail can be found in the document and labeling instructions included in this release, as well as the paper.

1. WikiSQL: WikiSQL consists of a corpus of 87,726 hand-annotated SQL query and natural language question pairs. These SQL queries are further split into training (61,297 examples), development (9,145 examples) and test sets (17,284 examples). It can be used for natural language inference tasks related to relational databases.

1. AMR Bank: The AMR Bank is a set of English sentences paired with simple, readable semantic representations. Version 3.0 released in 2020 consists of 59,255 sentences. Each AMR is a single rooted, directed graph. AMRs include PropBank semantic roles, within-sentence coreference, named entities and types, modality, negation, questions, quantities, and so on.

## Papers
1. PICARD: Large pre-trained language models for textual data have an unconstrained output space; at each decoding step, they can produce any of 10,000s of sub-word tokens. When fine-tuned to target constrained formal languages like SQL, these models often generate invalid code, rendering it unusable. We propose PICARD (code and trained models available at https://github.com/ElementAI/picard), a method for constraining auto-regressive decoders of language models through incremental parsing. PICARD helps to find valid output sequences by rejecting inadmissible tokens at each decoding step. On the challenging Spider and CoSQL text-to-SQL translation tasks, we show that PICARD transforms fine-tuned T5 models with passable performance into state-of-the-art solutions.

1. ÚFAL at MRP 2020: Permutation-invariant Semantic Parsing in PERIN: We present PERIN, a novel permutation-invariant approach to sentence-to-graph semantic parsing. PERIN is a versatile, cross-framework and language independent architecture for universal modeling of semantic structures. Our system participated in the CoNLL 2020 shared task, Cross-Framework Meaning Representation Parsing (MRP 2020), where it was evaluated on five different frameworks (AMR, DRG, EDS, PTG and UCCA) across four languages. PERIN was one of the winners of the shared task. The source code and pretrained models are available at https://github.com/ufal/perin.

## Semantic Dependency Parsing
Identify semantic relationships between words in a text using a graph representation.

## UCCA Parsing
UCCA (Abend and Rappoport, 2013) is a semantic representation whose main design principles are ease of annotation, cross-linguistic applicability, and a modular architecture. UCCA represents the semantics of linguistic utterances as directed acyclic graphs (DAGs), where terminal (childless) nodes correspond to the text tokens, and non-terminal nodes to semantic units that participate in some super-ordinate relation. Edges are labeled, indicating the role of a child in the relation the parent represents. UCCA’s foundational layer mostly covers predicate-argument structure, semantic heads and inter-Scene relations. UCCA distinguishes primary edges, corresponding to explicit relations, from remote edges that allow for a unit to participate in several super-ordinate relations. Primary edges form a tree in each layer, whereas remote edges enable reentrancy, forming a DAG.

## DRS Parsing
Discourse Representation Structures (DRS) are formal meaning representations introduced by Discourse Representation Theory. DRS parsing is a complex task, comprising other NLP tasks, such as semantic role labeling, word sense disambiguation, co-reference resolution and named entity tagging. Also, DRSs show explicit scope for certain operators, which allows for a more principled and linguistically motivated treatment of negation, modals and quantification, as has been advocated in formal semantics. Moreover, DRSs can be translated to formal logic, which allows for automatic forms of inference by third parties.

# Next
