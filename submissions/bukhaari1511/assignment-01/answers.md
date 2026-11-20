# Assignment 1 -- NLP Foundations

**Course:** Natural Language Processing\
**Assignment:** 1 -- NLP Foundations\
**Student:** \[bukhaari abdirisaaq abdile\]\
**Date:** \[18/11\2026]

------------------------------------------------------------------------

## 1. What Is Natural Language Processing?

Natural Language Processing (NLP) is a field of computer science and
artificial intelligence that develops methods for computers to process,
analyze, understand, and produce human language. Human language includes
written text and spoken language. NLP systems are used in applications
such as machine translation, chatbots, search engines, spam detection,
speech recognition, and text summarization (Jurafsky & Martin, 2026).

NLP is closely connected to Artificial Intelligence (AI) and Machine
Learning (ML), but the three terms do not mean the same thing:

-   **Artificial Intelligence (AI)** is the broad area of building
    computer systems that perform tasks associated with intelligent
    behavior, such as reasoning, planning, perception, and language use.
-   **Machine Learning (ML)** is a branch of AI in which a computer
    system learns patterns from data instead of relying only on rules
    written by a programmer.
-   **Natural Language Processing (NLP)** focuses specifically on
    computational work involving human language. NLP can use machine
    learning, deep learning, linguistic rules, dictionaries, or
    combinations of these methods.

Therefore, NLP is an area of AI, while machine learning is one of the
important approaches used to build NLP systems. Not every NLP system
must use machine learning; some systems use manually designed linguistic
rules. However, many modern NLP applications learn language patterns
from large collections of text.

### Real-life example: Email spam detection

Consider an email service that automatically moves unwanted messages
into a spam folder.

1.  **AI goal:** The system is designed to make an intelligent decision
    about whether an email is unwanted.
2.  **NLP processing:** The system processes the email's subject and
    message text. It may identify words, phrases, links, and other
    language-related patterns.
3.  **ML learning:** Developers train a machine-learning model using
    many previously labeled emails. The examples are marked as either
    *spam* or *not spam*.
4.  **Prediction:** When a new email arrives, the model examines its
    numerical representation and predicts a category.
5.  **Action:** The email service uses the prediction to decide whether
    the message should enter the inbox or spam folder.

This example shows how AI describes the broader intelligent application,
NLP handles the language data, and ML provides a way to learn
classification patterns from examples.

------------------------------------------------------------------------

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

These terms describe different levels and parts of a text dataset. To
explain them consistently, consider a small hypothetical email dataset
designed to train a spam detector.

  -----------------------------------------------------------------------
  Term                    Meaning                 Example in the email
                                                  dataset
  ----------------------- ----------------------- -----------------------
  **Corpus**              A collection of written A collection of 1,000
                          or spoken language data emails used to study
                          gathered for analysis   spam and normal
                          or research.            messages.

  **Document**            One individual text     One email with the
                          item inside a corpus.   subject "Meeting
                                                  tomorrow" and its
                                                  message body.

  **Token**               A unit produced when    The sentence "Win a
                          text is divided during  free phone!" could
                          tokenization. A token   become `Win`, `a`,
                          may be a word,          `free`, `phone`, `!`.
                          punctuation mark, or    
                          subword unit, depending 
                          on the tokenizer.       

  **Vocabulary**          The set of distinct     If the dataset contains
                          tokens recognized in a  the tokens `meeting`,
                          corpus or by a          `free`, `phone`, and
                          tokenizer.              `tomorrow`, these can
                                                  be members of its
                                                  vocabulary.

  **Feature**             A measurable property   The number of times the
                          or numerical value used word `free` occurs in
                          by a machine-learning   an email, or whether
                          model to make a         the email contains a
                          prediction.             suspicious link.

  **Label**               The target answer or    `spam` or `not_spam`
                          category assigned to an for each email.
                          example when a          
                          supervised model is     
                          trained.                
  -----------------------------------------------------------------------

### How the terms work together

Suppose the dataset contains this email:

> "Win a free phone today!"

The email is one **document** in the email **corpus**. A tokenizer
divides it into **tokens**. The unique tokens collected across all
emails form the **vocabulary**. A model can convert the email into
numerical **features**, such as token counts. Finally, the training
example has a **label**, for example `spam`.

It is important not to confuse a token with a vocabulary item. A token
is an occurrence in a particular text, while the vocabulary contains the
distinct token types known to the system. For example, the word `free`
may occur five times in one document, but it normally appears only once
as an entry in the vocabulary.

------------------------------------------------------------------------

## 3. Four Common NLP Tasks

NLP systems are designed for different tasks. The input and output
depend on the problem being solved.

  --------------------------------------------------------------------------------
  NLP task           Description    Input          Output          Example
  ------------------ -------------- -------------- --------------- ---------------
  **Text             Assigns one or A document,    A category or   Classifying a
  classification**   more           sentence,      set of          customer review
                     categories to  review, or     categories.     as positive,
                     an entire      email.                         negative, or
                     text.                                         neutral.

  **Sequence         Assigns a      A sentence     A sequence of   Identifying
  labeling**         label to each  divided into   labels          people,
                     token in a     tokens.        corresponding   locations, and
                     sequence.                     to the tokens.  organizations
                                                                   in a sentence.

  **Search and       Finds          A search query Ranked          Searching a
  similarity**       documents      and a          documents,      news archive
                     relevant to a  collection of  similarity      for articles
                     query or       documents, or  scores, or      about drought
                     measures how   a pair of      nearest         in Somalia.
                     similar two    texts.         matches.        
                     texts are.                                    

  **Text             Produces new   A prompt,      A generated     Producing a
  generation**       language based context, or    text sequence.  short summary
                     on an          partial                        or drafting a
                     instruction,   sentence.                      reply to a
                     prompt, or                                    message.
                     previous text.                                
  --------------------------------------------------------------------------------

### Further explanation

**Text classification:** A model considers the whole text and chooses a
category. For example, a message saying "The delivery arrived late" may
be classified as a complaint. The output is a document-level label
rather than a separate label for every word.

**Sequence labeling:** The model makes a prediction for each token. For
example, in "Amina visited Mogadishu," a named-entity recognition system
could label `Amina` as `PERSON`, `visited` as `O` (not a named entity),
and `Mogadishu` as `LOCATION`. The output has the same number of token
positions as the input sequence.

**Search and similarity:** A search system receives a user's query and
identifies relevant documents from a collection. A similarity system may
compare two sentences and estimate whether they express related
meanings. The result can be an ordered list of documents or a numerical
similarity score.

**Text generation:** A language model uses the prompt and learned
language patterns to produce a sequence of words or tokens. Examples
include machine translation, automatic replies, and summarization.
Generated text should still be checked because a system can produce
incorrect or unsupported information.

------------------------------------------------------------------------

## 4. Supervised and Unsupervised Learning in NLP

Machine-learning approaches differ in the type of information supplied
during training. Two important approaches are supervised learning and
unsupervised learning.

  -----------------------------------------------------------------------
  Aspect                  Supervised learning     Unsupervised learning
  ----------------------- ----------------------- -----------------------
  Training data           Examples include target Examples do not require
                          labels or answers.      target labels.

  Main purpose            Learn to predict a      Discover patterns or
                          known target.           structure in data.

  Example NLP task        Spam classification.    Grouping news articles
                                                  by topic.

  Expected result         A predicted label or    Clusters, patterns, or
                          value.                  representations.
  -----------------------------------------------------------------------

### Supervised learning example: Sentiment classification

Imagine a dataset of 5,000 customer reviews. Each review has already
been labeled as `positive`, `negative`, or `neutral`.

The model receives the review text and its correct label during
training. It learns which language patterns are associated with each
sentiment. After training, it can predict the sentiment of a new review.

**Why are target labels needed?**

The model must learn the relationship between the input text and the
correct answer. The labels provide the target against which its
predictions can be compared during training. Without them, the model
would not have the specified sentiment answers to learn to predict.

### Unsupervised learning example: News article clustering

Imagine collecting 10,000 news articles without assigning topic labels.
An unsupervised learning algorithm can examine patterns in the articles
and group similar documents together. One group might contain articles
about education, another about business, and another about sports.

**Why are target labels not required?**

The purpose is to discover structure in the data rather than predict
categories supplied in advance. The algorithm uses similarities and
patterns in the text to form groups. A researcher may inspect the groups
afterward and give them descriptive names, but those names were not
required as training labels.

In short, supervised learning learns from examples with known answers,
whereas unsupervised learning searches for patterns without being given
the desired answers.

------------------------------------------------------------------------

## 5. Why Text Must Be Converted into Numerical Representations

Most machine-learning algorithms operate on numbers. They perform
mathematical operations such as calculating distances, multiplying
values by model weights, and adjusting parameters during training.
Ordinary sentences are sequences of human-readable symbols, so they
cannot usually be supplied directly to a conventional numerical model.

Text must therefore be transformed into a numerical representation. This
process allows a model to work with information from words and documents
mathematically.

### One simple method: Bag of Words

The **Bag of Words (BoW)** method represents a document by counting how
often vocabulary words occur. It creates a vocabulary and assigns each
word a position in a numerical vector. Each document is then represented
by the counts at those positions.

Consider these two sentences:

-   Document 1: "Somali students learn NLP"
-   Document 2: "Students learn machine learning"

The combined vocabulary is:

`[Somali, students, learn, NLP, machine, learning]`

Their Bag of Words vectors are:

  Document       Somali   students   learn   NLP   machine   learning
  ------------ -------- ---------- ------- ----- --------- ----------
  Document 1          1          1       1     1         0          0
  Document 2          0          1       1     0         1          1

The numbers show how many times each vocabulary item appears in a
document. The vectors can now be used as input to a machine-learning
model.

### Main limitation of Bag of Words

Bag of Words does not preserve word order or fully represent meaning and
context. For example, "The student helps the teacher" and "The teacher
helps the student" contain the same words and can receive identical Bag
of Words vectors, even though their meanings differ.

It can also create very large, sparse vectors when the vocabulary is
large. In addition, words with similar meanings are treated as separate
vocabulary entries unless further processing is applied.

More advanced methods, such as word embeddings and contextual
representations, can capture relationships between words more
effectively. However, Bag of Words remains useful for understanding the
basic idea of numerical text representation (Jurafsky & Martin, 2026).

------------------------------------------------------------------------

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A low-resource language is a language for which relatively limited
digital language data, annotated datasets, linguistic tools, or other
resources are available for developing and evaluating NLP systems.
Somali is an important example because the amount of accessible,
carefully labeled data and language technology has historically been
much smaller than that available for widely researched languages such as
English.

The following challenges can affect the accuracy, reliability, and
usefulness of Somali NLP systems.

### 6.1 Limited digital datasets and annotated data

Many machine-learning systems require large amounts of representative
data. For supervised NLP, the data may also need human labels, such as
sentiment categories, named entities, or correct translations.

For Somali, researchers have reported that the lack of dedicated
electronic datasets has restricted the development of language-specific
NLP applications. For example, Badel et al. (2023) created a Somali
information retrieval corpus containing 2,335 documents because research
in Somali information retrieval had been limited and often depended on
translating queries into other languages.

**Effect on an NLP system:**

-   A model trained on a small dataset may fail to recognize language
    patterns that were not represented in its training examples.
-   It may perform poorly on new topics, dialects, or writing styles.
-   A small or unbalanced dataset can cause a model to favor the
    examples it has seen most often.
-   Evaluation results may be unreliable if the test data do not
    represent real-world Somali text.

Collecting more high-quality Somali text and creating carefully
annotated datasets can help address this problem.

### 6.2 Spelling variation and inconsistent text normalization

People may write the same word or expression in different ways,
especially in informal digital communication. Differences can come from
typing mistakes, shortened words, inconsistent spelling conventions,
repeated letters, punctuation, or the use of different forms of
transliteration.

Social media messages may also contain emojis, URLs, abbreviations, and
non-standard writing. A computer may treat two differently written forms
as separate tokens even when a human reader understands them as the same
word.

**Effect on an NLP system:**

-   The vocabulary becomes unnecessarily large.
-   A model may fail to match a word if it encounters a spelling form
    that was not common in its training data.
-   Search systems may miss relevant documents because a query and a
    document use different spellings.
-   Classification models may treat equivalent expressions as unrelated
    features.

Text normalization can help by applying consistent processing rules.
However, normalization must be designed carefully: changing a word
incorrectly can remove meaningful distinctions. Zupon, Crew, and Ritchie
(2021) examined text normalization and dataset quality across several
African low-resource languages, including Somali.

### 6.3 Morphological complexity and different word forms

Somali uses grammatical and derivational processes that can produce
related word forms from a root. A single root may appear in several
forms depending on grammatical function and word formation.

This creates a challenge for token-based systems. A model may encounter
a form that is rare or absent in its training vocabulary even though it
is related to a familiar root.

**Effect on an NLP system:**

-   The model may treat related forms as unrelated words.
-   The number of distinct vocabulary entries can grow.
-   Search may fail to retrieve a document when the query and document
    contain different forms of a related word.
-   Language models may need more examples to learn less frequent forms.

Lemmatization attempts to reduce inflected or derived forms to a common
base form where linguistically appropriate. Mohamed and Mohamed (2023)
developed a Somali lexicon and rule-based lemmatization approach,
demonstrating that Somali morphology requires language-specific
resources and processing.

### 6.4 Mixed-language text and code-switching

Somali speakers may mix Somali with English or Arabic in online messages
and everyday digital communication. A single sentence may contain words
from more than one language, as well as English technical terms.

**Effect on an NLP system:**

-   A Somali-only model may not recognize words from the other language.
-   The system may assign incorrect language-specific meanings or
    grammatical labels.
-   Sentiment analysis may miss the meaning of a sentence when positive
    or negative expressions appear in another language.
-   Search and translation systems may produce incomplete results.

A practical solution is to collect data that reflects real multilingual
communication and use language identification or multilingual models
where appropriate. Such systems should still be evaluated on actual
mixed-language Somali examples.

### Summary

These challenges are connected. Limited data makes it harder to learn
spelling variation, morphology, dialects, and code-switching. Improving
Somali NLP therefore requires not only more text, but also
representative data, careful annotation, suitable normalization, and
evaluation with native-language expertise.

------------------------------------------------------------------------

## 7. Case Study: Somali Information Retrieval Corpus

### Research paper

Badel, A., Zhong, T., Tai, W., and Zhou, F. (2023). **"Somali
Information Retrieval Corpus: Bridging the Gap between Query Translation
and Dedicated Language Resources."** Published in the *Proceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing
(EMNLP)*, pages 7463--7469.

### 7.1 Problem and task

The researchers focused on information retrieval for Somali. Information
retrieval means finding documents that are relevant to a user's
information need.

Before this work, Somali information retrieval research was limited and
often depended on translating a Somali search query into another
language. This approach can be problematic because translation may not
preserve every meaning or expression, and it does not replace a
dedicated Somali document collection.

The research aimed to help close this resource gap by building a corpus
specifically for Somali information retrieval and developing a retrieval
system using it.

### 7.2 Data

The researchers created an annotated corpus containing **2,335 Somali
documents**. The documents were collected from different online sources,
including:

-   Hiiraan Online
-   Dhacdo.net
-   Somali poetry books

The use of different sources helped provide a collection of
Somali-language material for retrieval research.

### 7.3 Method

The researchers worked with language experts and NLP researchers to
construct the corpus. They then developed a Somali information retrieval
system using **Pseudo-Relevance Feedback (PRF) query expansion**.

In simple terms, query expansion attempts to improve a search by adding
useful related terms to the original query. In pseudo-relevance
feedback, the system uses the top documents returned by an initial
search as if they were relevant, extracts useful terms from them, and
expands the query before performing another search.

For example, if a user searches for information about drought, the
system may identify related terms in initially retrieved documents and
use them to improve a second search. The purpose is to retrieve
documents that might not have matched the original query directly.

### 7.4 Findings and contribution

The study's main contribution was the creation of a dedicated Somali
information retrieval resource and a retrieval system based on that
resource. It addressed the shortage of electronically available Somali
datasets and provided a foundation for further research.

The paper's abstract describes the corpus and the PRF-based retrieval
system, but it does not provide a numerical retrieval-performance result
in the abstract. Therefore, it would not be accurate to claim a
particular percentage improvement based only on that summary.

The work is important because a dedicated corpus can support future
Somali search systems and other NLP research, including question
answering and text classification.

### 7.5 Which NLP task does it cover?

This case study covers **Search and Information Retrieval**, one of the
four NLP tasks discussed in Question 3.

-   **Input:** A user's search query and a collection of Somali
    documents.
-   **Processing:** The system searches the corpus and uses
    pseudo-relevance feedback to expand the query.
-   **Output:** A ranked set of documents that are expected to be
    relevant to the user's query.

This research demonstrates how creating language-specific datasets can
support practical NLP applications for a low-resource language.

------------------------------------------------------------------------

## Conclusion

Natural Language Processing enables computers to work with human
language and is an important area of Artificial Intelligence. Machine
Learning supports many NLP applications by learning patterns from text
data. Understanding the basic terms---corpus, document, token,
vocabulary, feature, and label---is necessary for working with NLP
datasets.

Common NLP tasks include text classification, sequence labeling, search
and similarity, and text generation. Supervised learning uses labeled
examples, while unsupervised learning discovers patterns without
requiring target labels. Because most machine-learning models operate on
numbers, text must be converted into numerical representations such as
Bag of Words vectors.

The Somali language illustrates why language-specific research is
necessary. Limited datasets, spelling variation, morphology, and
mixed-language communication can all affect system performance. The
Somali information retrieval study by Badel et al. (2023) provides a
practical example of researchers addressing these challenges through the
creation of a dedicated corpus and a search system.

------------------------------------------------------------------------

## References

Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information
retrieval corpus: Bridging the gap between query translation and
dedicated language resources. In *Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing* (pp. 7463--7469).
Association for Computational Linguistics.
https://doi.org/10.18653/v1/2023.emnlp-main.462

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing:
An introduction to natural language processing, computational
linguistics, and speech recognition with language models* (3rd ed.,
online manuscript, August 19, 2026). Stanford University.
https://web.stanford.edu/\~jurafsky/slp3/

Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word
lemmatization approach for Somali language. In *ICLR 2023 Workshops:
AfricaNLP*. https://mlanthology.org/iclrw/2023/mohamed2023iclrw-lexicon/

Zupon, A., Crew, E. E., & Ritchie, S. (2021). Text normalization for
low-resource languages of Africa. In *Proceedings of the AfricaNLP
Workshop*. Google Research.
https://research.google/pubs/text-normalization-for-low-resource-languages-of-africa/
