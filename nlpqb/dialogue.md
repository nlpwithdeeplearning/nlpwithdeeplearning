---
layout: page
title: Chatbots and Dialogue Systems
---

### Why is conversation or dialogue important?
Conversation or dialogue is the first kind of language we learn as children, and for most of us, it is the kind of language we most commonly indulge in, whether we are order some food, buying some grocery, participating in meetings, talking with family, making travel arrangements, or complaining about weather.

### What are dialogue systems or conversational agents?
Dialogue systems or conversational agents communicate with users in natural language (text, speech, or both), and fall into two classes:
1. Task-oriented dialogue: They converse with users to help complete tasks. Examples include Siri, Alexa, Google Now/Home, Cortana, etc. Example applications include giving directions, controlling appliances, finding restaurants, making calls, answering questions on corporate websites, interfacing with robots, and doing social good (e.g., robot lawyer).
1. Chatbots: These are systems designed for extended conversations, set up to mimic the unstructured conversations or 'chats' characteristic of human-human interaction, mainly for entertainment, but also for practical purposes like making task-oriented agents more natural.

### What is a turn?
A dialogue is a sequence of turns, each a single contribution from one speaker to the dialogue. A turn can consist of a sentence, although it might be as short as a single word or as long as multiple sentences.

### What is endpointing?
Spoken dialogue systems must detect whether a user is done speaking, so they can process the utterance and respond. This is task is called enpointing or endpoint detection. This can be quite challenging because of noise and because people often pause in the middle of turns.

### What are speech acts or dialog acts?
Each utterance in a dialogue is a kind of action being performed by the speaker. These actions are commonly called speech acts or dialog acts.

### What are the four major classes of dialog acts?
1. Constatives
1. Directives
1. Commissives
1. Acknowledgments

### What is grounding?
Like all collective acts, it is important for the participants to establish what they both agree on, called the common ground. Speakers do this by what is known as **grounding** each other's utterances.

### What is conversational analysis?
Conversational analysis is the field that discusses the local structure between speech acts.

### Provide some examples of adjacency pairs?
1. Questions set up an expectation for an Answer.
1. Proposals are followed by Acceptance (or Rejection).
1. Compliments often give rise to Downplayers.

### What is a side sequence?
Side sequence are the subdialogs that separate the dialogue acts corresponding to the adjacency pairs.

### What is the correction subdialogue?
Correction subdialogue is a side sequence that interrupts the prior discourse for the purpose of correcting a previously provided information.

### What is a clarification question side sequence?
Clarification question side sequence forms a subdialog between a Request and a Response. This is especially common in dialogue systems where speech recognition errors cause the system to have to ask for clarifications or repetitions.

### What are presequences?
Preseqeunces are a subdialog before the actual adjacency pair.

### What is initiative?
Sometimes a conversation is completely controlled by one participant. For example a reporter interviewing a chef might ask questions, and the chef responds. We say that the reporter in this case has the conversational initiative.

### What is mixed initiative?
In normal human-human dialogue, the initiative shifts back and forth between the participants, as they sometimes answer questions, sometimes ask them, sometimes take the conversations in new directions, sometimes not. We call such interactions mixed initiative.

### What is user initiative?
Mixed initiative is very difficult for dialogue systems. It's much easier to design dialogue systems to be passive responders. The initiative lies completely with the user. Such systems are called user initiative.

### Why can a system initiative architecture be frustrating?
Such systems can ask a question and give the users no opportunity to do anything until they answer it.

### What is implicature?
Implicature means a particular class of licensed inferences guided by set of maxims such as the maxim of relevance.

### What are the subtle characteristics of human conversations?
1. turns
1. speech acts
1. grounding
1. dialogue structure
1. initiative
1. implicature

### What are some datasets available for corpus-based chatbots?
1. transcripts of natural spoken conversational corpora such as Switchboard, CALLHOME, and CALLFRIEND
1. movie dialogues
1. crowdsourced corpora such as Topical-Chat and EmpatheticDialogues
1. pseudo-conversations from social media platforms such as Twitter, Reddit, and Weibo
1. Knowledge sources such as Wikipedia
1. User data

### What are retrieval methods for chatbots?
We use information retrieval to grab a response from some corpus that is appropriate given the dialogue context. We choose a response by using a finding the turn in the corpus whose encoding has the highest dot-product with the user’s turn.

### What are generation methods for chatbots?
We use a language model or encoder-decoder to generate the response given the dialogue context. We use an encoder-decoder to generate the response. The encoder sees the entire dialogue context.

### What is GUS?
GUS is a very old frame-based architecture for task-based dialogue. It was introduced in 1977 for travel planning. A frame is a kind of knowledge structure representing the kinds of intentions the system can extract from user sentences, and consists of a collection of slots, each of which can take a set of possible values. This set of frames is also called domain ontology. Types in GUS have hierarchical structure.

### How does the control structure work for frame-based dialogue?
1. The system’s goal is to fill the slots in the frame with the fillers the user intends, and then perform the relevant action for the user. To do this, the system asks questions of the user filling any slot that the user specifies.
1. If a user’s response fills multiple slots, the system fills all the relevant slots, and then continues asking questions to fill the remaining slots, skipping questions associated with filled slots.
1. Condition-action rules are attached to slots. The rules might automatically fill the slot in a related frame (e.g., Destination in plane booking frame is the same as StayLocation in hotel booking frame).
1. The system must be able to disambiguate which slot of which frame a given input is supposed to fill and then switch dialogue control to the that frame.

### What are the components of language understanding in GUS?
1. domain classification
1. intent determination
1. slot filling

### What is semantic grammar?
A semantic grammar is a context-free grammar in which the left-hand side of each rule corresponds to the semantic entities being expressed.

### What does the ASR component do in a dialogue system?
The ASR component takes audio input from a phone or other device and outputs a transcribed string of words. The ASR component can also be made dependent on the dialogue state. The ASR language model can be constrained to assign high probability to slot values that are expected. This can be done by training the language model on the utterances used at that state or by hand-writing finite-state or context free grammars for such sentences.

### What is restrictive grammar?
A language model that is completely dependent on dialogue state.

### What does the natural language generation module work in frame-based systems?
NLP component produces the utterances that the system says to the user. Frame-based systems tend to use template-based generation, in which all or most of the words in the sentence to be uttered to the user are prespecified by the dialogue designer. Sentences created by these templates are often called prompts. Templates might be completely fixed or can include some variables that are filled in by the generator.

### Why is grounding important even in template-based generation and how do we do it?
The system needs to demonstrate it understood when the user denies a request. Otherwise, without the acknowledgement, the user will not know that the system has understood. The use of *Okay* at the beginning of the response prompt add grounding to the templated response, making it much more natural.

### What are the components of a dialogue state or belief state architectures for task-oriented dialogue?
1. Automatic Speech Recognition
1. Spoken Language Understanding
1. Dialog State Tracker
1. Dialog Policy
1. Natural Language Generation
1. Text to Speech

### What does the NLU component of dialogue-state architecture do?
Extracts slot fillers from the user's utterance using machine learning.

### What is the role of dialogue-state tracker?
Maintain the current state of the dialogue, which includes the user's most recent dialogue act, plus the entire set of slot-filler constraints the user has expressed so far.

### What is the dialogue policy?
The dialogue policy decides what the system should do or say next. It can help a system decide when to answer the user's questions, when to instead ask the user a clarification question, when to make a suggestion, and so on.

### What does the natural language generation component do?
In Gus, the sentences that the generator produced were all from pre-written templates. A more sophisticated generation component can condition on the exact context to produce turns that seem much more natural.

### What are dialogue acts in dialogue-state systems?
Dialogue acts represent the interactive function of the turn or sentence, combining the idea of speech acts and grounding into a single representation. Different types of dialogue systems require labeling different kinds of acts, and so the tagset tends to be designed for particular tasks.

### What is the relationship between slot filling in dialogue-state systems and semantic parsing?
The task of slot-filling, and the simpler tasks of domain and intent classification, are special cases of the task of supervised semantic parsing. Semantic parsing is able to extract complex entities as slots also, for example. For the simple cases, we can pass the utterance through a contextual embedding model (e.g., BERT). A feed forward neural network on the CLS tag can be used for domain classification and intent classification and another sequence labeling network (e.g., another feedforward neural network with softmax or CRF outer layer).

### How are rule-based systems used for bootstrapping machine learning-based systems?
A rule-based system is first built for the domain, and a test set is carefully labeled. As new user utterances come in, they are paired with the labelling provided by the rule-based system to create training tuples. A classifier can then be trained on these tuples, using the test set to test the performance of the classifier against the rule-based system. Some heuristics can be used to eliminiate errorful training tuples, with the goal of increasing precision. As sufficient training samples become available, the resulting classifier can often outperform the original rule-based system. The rule-based systems may still have higher precision for complex cases like negation.

### What is the job the dialogue-state tracker?
Determine the current state of the frame (the fillers of each slot summarized from all the past user utterances), as well as the user's most recent dialogue act.

### How is a dialog-state tracker built?
The simplest dialogue state tracker might just take the output of a slot-filling sequence-model after each sentence. Alternatively, a more complex model can make use of the reading-comprehension architectures. For example, we can train a classifier for each slot to decide whether its value is being changed in the current sentence or should be carried over from the previous sentences. If the slot value is being changed, a span-prediction model is used to predict the start and end of the span with the slot filler.

### What is user correction act?
If a dialogue system misrecognizes or misunderstands an utterance, the user will generally correct the error by repeating or reformulating the utterances. The dialog acts corresponding to these are called user correction acts.

### What is hyperarticulation?
Hyperarticulation is when the utterance contains exaggerated energy, duration, or F0 contours. This is often seen when users are frustrated.

### What is the goal of dialog policy?
To decide what action the system should take next, that is, what dialogue act to generate.

### What is Ahat_i?
Ahat_i = argmax(P(A_i | A_0, U_0, ..., A_(i-1), U_(i-1)), A_i in A) = argmax(P(A_i | Frame_(i-1), A_(i-1), U_(i-1)), A_i in A)

### How can action probabilities be estimated?
We can use a neural classifier on the top of neural representations of the slot fillers (for example as spans) and the utterances (for examples as sentence embeddings computed over contextual embeddings). More sophisticated models train the policy via reinforcement learning. The system can get a large positive reward if the dialogue system terminates with the correct slot representation at the end, and a large negative reward if the slots are wrong, and a small negative reward for confirmation and elicitation questions to keep the system from reconfirming everything.

### What are two methods that modern dialog systems use to recover from mistakes?
1. confirming understandings with the user
1. rejecting utterances that the system is likely to have misunderstood

### What is explicit confirmation strategy?
A system asks the user a direct question to confirm the system's understanding. For example, the system can ask a yes/no confirmation question. Explicit confirmation makes it easier for users to correct the system's misrecognitions, but is awkward and increases the length of the conversation.

### What is implicit confirmation strategy?
A system demonstrates its understanding as a grounding strategy, for example repeating back the system's understanding as part of asking the next question. Implicit confirmation is conversationally natural.

### How does a dialog-state system handle rejection?
The system gives the user a prompt like *I'm sorry, I didn't understand that.* This is called rapid reprompting. Alternatively, it can follow the strategy of progressive prompting aka escalating detail. This gives the caller more guidance about how to formulate an utterance the system will understand. Users prefer rapid reprompting over progressive prompting as a first-level error prompt.

### What is the machine learning feature to know how likely the utterance is misrecognized?
Confidence is a metric that the speech recognizer can assign to its transcription of a sentence to indicate how confident it is in that transcription. Confidence is often computed from the acoustic log-likelihood of the utterance (greater probability means higher confidence), but prosodic features can also be used in confidence prediction. For example, utterances with large F0 excursions or longer durations, or those preceded by longer pauses, are likely to be misrecognized.

### When is the cost of making an error higher?
Cost of making an error is higher before a flight is actually booked, money in an account is moved, etc.

### How can we use a four-tiered level of confidence with three thresholds?
* < alpha = low confidence = reject
* >= alpha = above the threshold = confirm explicitly
* >= beta = high confidence = confirm implicitly
* >= gamma = very high confidence = don't confirm at all

### What are the two stages of NLG in dialogue-state architecture?
1. content planning (what to say)
1. sentence realization (how to say it)

### What is delexicalization?
Delexicalization is the process of replacing specific words in the training set that represent slot values with a generic placeholder token representing the slot.

### How can NLG map frames to delexicalized sentences?
Mapping from frames to delexicalized sentences is generally done by encoder decoder models trained on large hand-labeled corpora of task-oriented dialogue (e.g., MultiWoz). The input to the encoder is a sequence of tokens that represent the dialogue act and its arguments. The encoder reads all the input slot/value representations, and the decoder outputs a delexicalized English sentence. We can then use the input frame from the content planner to relexicalize.

### How does NLG work for clarification questions?
Targeted clarification questions can be created by rules (such as replacing “going to UNKNOWN WORD” with “going where”) or by building classifiers to guess which slots might have been misrecognized in the sentence.

### How can chatbots be evaluated by humans?
This can be the human who talked to the chatbot (participant evaluation) or a third party who reads a transcript of a human/chatbot conversation (observer evaluation).

### What are the eight dimensions capturing conversational quality?
1. avoiding repetition, 
1. interestingness, 
1. making sense, 
1. fluency, 
1. listening, 
1. inquisitiveness, 
1. humanness, and 
1. engagingness

### What is the acute-eval metric?
The acute-eval metric is an observer evaluation metric in which annotators look at two separate human-computer conversations (A and B) and choose the one in which the dialogue system participant performed better. They answer the following 4 questions (with these particular wordings shown to lead to high agreement):
1. Engagingness: Who would you prefer to talk to for a long conversation?
1. Interestingness: If you had to say one of these speakers is interesting and one is boring, who would you say is more interesting?
1. Humanness: Which speaker sounds more human?
1. Knowledgeable: If you had to say that one speaker is more knowledgeable and one is more ignorant, who is more knowledgeable?

### Why are automatic evaluations not used for chatbots?
That’s because computational measures of generation performance like BLEU or ROUGE or embedding dot products between a chatbot’s response and a human response correlate very poorly with human judgments. These methods perform poorly because there are so many possible responses to any given turn; simple word-overlap or semantic similarity metrics work best when the space of responses is small and lexically overlapping, which is true of generation tasks like machine translation or possibly summarization, but definitely not dialogue.

### What is adversarial evaluation?
The idea is to train a “Turing-like” evaluator classifier to distinguish between human-generated responses and machine-generated responses. The more successful a response generation system is at fooling this evaluator, the better the system.

### How do we evaluate task-based dialogue?
If the task is unambiguous, we can simply measure absolute task success (did the system book the right plane flight, or put the right event on the calendar). To get a more fine-grained idea of user happiness, we can compute a user satisfaction rating, having users interact with a dialogue system to perform a task and then having them complete a questionnaire:
1. TTS Performance: Was the system easy to understand ?
1. ASR Performance: Did the system understand what you said?
1. Task Ease: Was it easy to find the message/flight/train you wanted?
1. Interaction Pace: Was the pace of interaction with the system appropriate?
1. User Expertise: Did you know what you could say at each point?
1. System Response: How often was the system sluggish and slow to reply to you?
1. Expected Behavior: Did the system work the way you expected it to?
1. Future Use: Do you think you’d use the system in the future?

Responses are mapped into the range of 1 to 5, and then averaged over all questions to get a total user satisfaction rating.

### What is slot error rate or concept error rate?
Number of inserted/deleted/subsituted slots / Number of total reference slots for sentence

### What are efficiency costs?
Efficiency costs are measures of the system’s efficiency at helping users. This can be measured by the total elapsed time for the dialogue in seconds, the number of total turns or of system turns, or the total number of queries. Other metrics include the number of system non-responses and the “turn correction ratio”: the number of system or user turns that were used solely to correct errors divided by the total number of turns.

### What is quality cost?
Quality cost measures some aspects of the interactions that affect user’s perception of the system. One such measure is the number of times the ASR system failed to return any sentence, or the number of ASR rejection prompts. Similar metrics include the number of times the user had to barge in (interrupt the system), or the number of time-out prompts played when the user didn’t respond quickly enough. Other quality metrics focus on how well the system understood and responded to the user. The most important is the slot error rate, but other components include the inappropriateness (verbose or ambiguous) of the system’s questions, answers, and error messages or the correctness of each question, answer, or error message.

### What are the principles of voice user interface design?
The design of dialogue strategies, prompts, and error messages is often called voice user interface design, and generally follows user-centered design principles:
1. Study the user and task
1. Build simulation and prototypes
3. Iteratively test the design on users

### What are the ethical issues in dialogue system design?
1. safety of users: If users seek information from conversational agents in safety-critical situations like asking medical advice, or in emergency situations, or when indicating the intentions of self-harm, incorrect advice can be dangerous and even life-threatening
1. representational harm: Microsoft’s 2016 Tay chatbot, for example, was taken offline 16 hours after it went live, when it began posting messages with racial slurs, conspiracy theories, and personal attacks on its users. Simple changes like using the word ‘she’ instead of ‘he’ in a sentence caused neural dialogue systems in 2016 to respond more offensively and with more negative sentiment.
1. privacy: The ubiquity of in-home dialogue agents means they may often overhear private information.
1. gender equality: Current chatbots are overwhelmingly given female names, likely perpetuating the stereotype of a subservient female servant. And when users use sexually harassing language, most commercial chatbots evade or give positive responses rather than responding in clear negative ways.