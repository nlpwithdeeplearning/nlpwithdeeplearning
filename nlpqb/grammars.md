---
layout: page
title: Constituency Grammars
---
### What is syntax?
Syntax refers to the way words are arranged together.

### What is syntactic constituency?
Syntactic constituency is the idea that groups of words can behave as single units or constituents.

### What is context-free grammar?
Context-free grammar is the most widely used formal system for modeling constituent structure in English and other natural languages. It consists of:
1. a set of rules or productions, each of which expresses the ways that symbols of the language can be grouped and ordered together, and 
2. a lexicon of words and symbols. 

### What are terminal symbols?
The symbols that correspond to words in the language.

### What are non-terminals?
The symbols that express abstractions over the terminals.

### What is a parse tree?
Parse tree shows the derivation of the string of words through the sequence of rule expansions starting with the root at the top.

### Give an example for a parse tree in bracketed notation.
[S [NP [Pro I]] [VP [V prefer] [NP [Det a] [Nom [N morning] [Nom [N flight]]]]]]

### What are grammatical sentences?
Sentences (strings of words) that can be derived by a grammar. These sentences are also said to be in the formal language defined by that grammar.

### What are ungrammatical sentences?
Sentences that cannot be derived by a given formal grammar are not in the language defined by that grammar and are referred to as ungrammatical.

### What is generative grammar?
The use of formal languages to model natural languages.

### What is the structure of sentences with declarative structure?
They have a subject noun phrase (NP) followed by a verb phrase.

`S -> NP VP`

### What is the structure of sentences with imperative structure?
They often begin with a verb phrase (VP) and have no subject.

`S -> VP`

### What is the structure of sentences with yes-no question?
They begin with an auxillary verb, followed by a subject NP, followed by a VP.

`S -> Aux NP VP`

### What are wh-structures?
One of their constituents is a wh-phrase, that is, one that includes a wh-word (who, whose, when, where, what, which, how, why).

### What is the wh-subject-question structure?
It is identical to the declarative structure except that the first noun phrase contains some wh-word.

`S -> Wh-NP VP`

### What is the wh-non-subject-question structure?
The wh-phrase is not the subject of the sentence and so the sentence include another subject with auxiliary appearing before it just as in yes-no question structures.

`S -> Wh-NP Aux NP VP`

### What is an example of a long-standing dependencies?
In `S -> Wh-NP Aux NP VP`, there is a long-standing dependency between Wh-NP and VP. For example, consider `what pizza do you have?`

### What is a clause?
Forming a complete thought.

### What is a noun phrase?
The noun phrase consists of a head (the central noun in the noun phrase) along with various modifiers such as determiner, cardinal numbers, ordinal numbers, quantifiers, adjectives, and predeterminers before the head noun, and postmodifiers such as prepositional phrases, non-finite clauses, and relative clauses. 
```
Det -> a | the | this | those | any | some
        | NP 's
Nominal -> Noun 
Nominal -> Nominal PP
GerundV -> being | arriving | leaving | ...
GerundVP -> GerundV (NP | PP | '' | NP PP)
Nominal -> Nominal GerundVP
Nominal -> Nominal RelClause
RelClause -> (who | that)VP
NP -> Pronoun
    | Proper-Noun
    | Det Nominal
```

### What is a verb phrase?
Here is a simple set of rules:
```
VP -> Verb
    | Verb NP
    | Verb NP PP
    | Verb PP
```
Verb phrases can also be significantly complicated than this. For example, they can be a verb followed by by an entire embedded sentence (sentential complements). 

### What do the subjects in English agree with the main verb in?
person and number

### What can verbs be subcategorized into?
The types of complements they expect. For example, transitive verbs take a direct object NP, while intransitive verbs don't.

### What is Coordination?
Conjoining phrase types with conjunctions like and, or, and but to form larger constructions of the same type.

### What is a treebank?
A syntactially annotated corpus of sentence paired with corresponding parse tree.

### What is a head?
The head is the word in the phrase that is grammatically the most important.

### What is Chomsky normal form?
A context free grammar is in Chomsky normal form (CNF) if it is `epsilon`-free and if in addition each production is either of the form `A -> B C` or `A -> a`. Any CFG can be converted to CNF.

### What is Chomsky-adjunction?
Generation of a symbol A with a potentially infinite sequence of symbols B with a rule of the form `A -> A B`

### What are lexicalized grammars?
Lexicalized grammars place more emphasis on the structure of the lexicon, lessening the burden on pure phrase-structure rules. Combinatorial categorical grammar (CCG) is an important computationally relevant lexicalized approach.