# Assignment 1 -- NLP Foundations


**Assignment:** 1 -- NLP Foundations\
**Topic:** Natural Language Processing

------------------------------------------------------------------------

## Question 1: What is Natural Language Processing (NLP), and how is it related to Artificial Intelligence (AI) and Machine Learning (ML)? Give one real-life example.

Natural Language Processing (NLP) is a field of computer science and
Artificial Intelligence (AI) that focuses on enabling computers to
process, analyze, understand, and generate human language, including
written text and spoken language.

NLP is closely related to AI and Machine Learning, but they are not
exactly the same:

-   **Artificial Intelligence (AI):** A broad field concerned with
    developing computer systems that perform tasks that normally require
    human intelligence, such as reasoning, problem-solving, and
    decision-making.
-   **Machine Learning (ML):** A branch of AI that allows computers to
    learn patterns from data and make predictions or decisions without
    being explicitly programmed for every situation.
-   **Natural Language Processing (NLP):** A field focused on working
    with human language. It uses different approaches, including
    linguistic rules, machine learning, and deep learning, to process
    language.

**Real-life example: Customer service chatbot**

A telecommunications company may use an NLP-powered chatbot to
communicate with customers.

For example, a customer sends the message:

> My internet is not working.

The NLP system processes the message and identifies that the customer is
reporting an internet problem. A machine learning model may classify the
message as a network complaint. The chatbot can then provide
troubleshooting instructions or direct the customer to a support agent.

This example demonstrates how NLP helps computers work with human
language, while machine learning helps them recognize patterns and
classify messages.

------------------------------------------------------------------------

## Question 2: Define corpus, document, token, vocabulary, feature, and label. Give one dataset example that shows how they are related.

These terms describe important elements of text data used in NLP and
machine learning.

1.  **Corpus:** A collection of text documents gathered for language
    analysis or NLP research.
2.  **Document:** One individual piece of text within a corpus, such as
    an email, article, or customer message.
3.  **Token:** A unit of text produced during tokenization. A token may
    be a word, part of a word, or punctuation mark, depending on the
    tokenizer.
4.  **Vocabulary:** The collection of unique tokens identified in a
    corpus or dataset.
5.  **Feature:** A measurable characteristic or numerical representation
    extracted from the text and used by a machine learning model.
6.  **Label:** The target category or answer assigned to an example that
    a supervised learning model is trained to predict.

### Example: A telecommunications customer message dataset

Imagine a company collecting customer messages to build a system that
automatically identifies complaints, requests, and questions.

  Document ID   Customer message                Label
  ------------- ------------------------------- -----------
  D1            My internet is slow             Complaint
  D2            I need a new SIM card           Request
  D3            When will my internet return?   Question
  D4            My network is not working       Complaint
  D5            I want to change my SIM card    Request

The relationship between the terms can be explained as follows:

-   **Corpus:** All five customer messages together form the corpus.
-   **Document:** Each individual message, such as D1, is one document.
-   **Token:** D1 can be tokenized into `My`, `internet`, `is`, and
    `slow`.
-   **Vocabulary:** The vocabulary contains unique tokens collected from
    all documents, such as `internet`, `slow`, `SIM`, `card`, and
    `network`.
-   **Feature:** The presence of the word `internet`, the presence of
    `slow`, or the number of words in a message can be used as features.
-   **Label:** Complaint, Request, and Question are the labels assigned
    to the messages.

In this example, the corpus provides the documents, tokenization breaks
the documents into smaller units, the vocabulary identifies unique
tokens, and features represent the text in a form that a machine
learning model can process. Labels provide the expected categories
during supervised training.

------------------------------------------------------------------------

## Question 3: Describe four common NLP tasks. For each task, explain its input and output.

NLP includes different tasks that allow computers to perform specific
operations on human language. Four common tasks are text classification,
sequence labeling, search and similarity, and text generation.

### 1. Text Classification

Text classification is the process of assigning a category or label to
an entire text or document based on its content.

-   **Input:** A text document, sentence, or message.
-   **Output:** A predicted category or label.

**Example:** An email filtering system receives an email and classifies
it as either spam or not spam. A telecommunications company could also
classify customer messages as complaints, requests, or questions.

### 2. Sequence Labeling

Sequence labeling assigns a label to each token or word in a sequence.
It is commonly used for tasks such as Named Entity Recognition (NER) and
Part-of-Speech (POS) tagging.

-   **Input:** A sequence of words or tokens.
-   **Output:** A label for each token.

**Example:** Consider the sentence:

> Ahmed works at the Ministry of Health in Somalia.

A Named Entity Recognition system could identify:

-   Ahmed --- PERSON
-   Ministry of Health --- ORGANIZATION
-   Somalia --- LOCATION

Unlike text classification, which assigns one label to an entire
document, sequence labeling assigns labels to individual tokens or
groups of tokens.

### 3. Search and Similarity (Information Retrieval)

Information retrieval is the process of finding and ranking documents
that are relevant to a user's search query. Text similarity measures how
closely two pieces of text are related.

-   **Input:** A search query and a collection of documents.
-   **Output:** A list of relevant documents, usually ranked by
    relevance, or similarity scores.

**Example:** A humanitarian worker searches for "drought preparedness in
Somalia." A search system retrieves reports and articles related to
drought preparedness, even if some documents use slightly different
wording.

### 4. Text Generation

Text generation is the process of producing new text based on an
instruction, prompt, or previous context.

-   **Input:** A prompt, instruction, or existing text.
-   **Output:** Newly generated text.

**Example:** A user asks an AI assistant to write a short paragraph
about drought preparedness. The system generates a paragraph describing
early warning systems, emergency supplies, and community preparedness.

### Summary

  -----------------------------------------------------------------------
  NLP task                Input                   Output
  ----------------------- ----------------------- -----------------------
  Text classification     A text or document      A category or label

  Sequence labeling       A sequence of tokens    A label for each token

  Search and similarity   Query and document      Relevant documents or
                          collection              similarity scores

  Text generation         Prompt or context       Newly generated text
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Question 4: Explain the difference between supervised and unsupervised learning. Give one NLP example of each and explain why labels are needed or not needed.

Machine learning approaches can be divided into different types
depending on the information available during training. Two common
approaches are supervised learning and unsupervised learning.

### Supervised Learning

Supervised learning is a machine learning approach in which a model
learns from labelled training data. Each training example contains an
input and its correct output or label. The model learns the relationship
between them and uses that knowledge to predict labels for new, unseen
data.

**NLP example: Somali sentiment analysis**

Imagine developing a system that identifies whether Somali customer
messages express positive or negative opinions.

The training dataset may contain:

  Customer message              Label
  ----------------------------- ----------
  The service is excellent      Positive
  The network is very poor      Negative
  I am happy with the service   Positive

The model learns from the labelled examples and later predicts whether
new messages express positive or negative sentiment.

**Why are labels needed?**

Labels are needed because they provide the correct answers during
training. The model uses them to learn what patterns are associated with
each category and to evaluate its predictions.

### Unsupervised Learning

Unsupervised learning is a machine learning approach in which a model
works with data that does not have predefined target labels. Instead of
learning from correct answers, it identifies patterns, similarities, or
structures within the data.

**NLP example: Grouping Somali customer messages**

Imagine a telecommunications company has collected thousands of Somali
customer messages but has not categorized them.

An unsupervised learning algorithm can group similar messages into
clusters. For example:

-   One cluster may contain messages about internet problems.
-   Another may contain messages about SIM cards.
-   A third may contain messages about mobile money.

After the algorithm creates the groups, a human can examine them and
assign descriptive names.

**Why are labels not needed?**

Labels are not required because the model is not learning to predict
predefined categories. Its purpose is to discover patterns and
similarities within the available data.

### Main Differences

  -----------------------------------------------------------------------
  Supervised learning                 Unsupervised learning
  ----------------------------------- -----------------------------------
  Uses labelled training data         Uses data without predefined target
                                      labels

  Learns from known input-output      Discovers patterns and structures
  examples                            

  Predicts predefined categories or   Groups or organizes similar data
  values                              

  Example: Sentiment classification   Example: Customer message
                                      clustering
  -----------------------------------------------------------------------

Both approaches can be useful in NLP. For example, researchers may first
use unsupervised learning to explore and group large collections of
Somali messages, then use human-labelled examples to train a supervised
classification model.

------------------------------------------------------------------------

## Question 5: Why do we convert text into numerical representations before using many machine learning algorithms? Explain one simple method and one limitation.

Human language consists of words, sentences, and symbols, but many
traditional machine learning algorithms require numerical input to
perform mathematical operations and learn patterns.

For this reason, text must be converted into numerical representations
before it can be used by many machine learning models. This process is
commonly called text vectorization or feature extraction.

### Simple Method: Bag of Words (BoW)

Bag of Words is a basic method of representing text numerically. It
creates a vocabulary of unique words from a collection of documents and
represents each document using the frequency or presence of those words.

Consider the following three documents:

-   D1: "Internet is working"
-   D2: "Internet is slow"
-   D3: "Network is working"

First, we create a vocabulary containing all unique words:

`[Internet, is, working, slow, Network]`

Next, each document is represented as a numerical vector. A value of 1
means the word appears in the document, while 0 means it does not.

  Document     Internet   is   working   slow   Network
  ---------- ---------- ---- --------- ------ ---------
  D1                  1    1         1      0         0
  D2                  1    1         0      1         0
  D3                  0    1         1      0         1

The resulting vectors allow machine learning algorithms to process the
documents as numerical data.

### One Limitation of Bag of Words

A major limitation of Bag of Words is that it ignores word order and
context. It counts words without understanding how their positions
affect meaning.

For example:

-   "The dog bites the man."
-   "The man bites the dog."

Both sentences contain the same words, so they can produce identical Bag
of Words representations. However, their meanings are different because
the word order changes who performs the action.

Bag of Words also creates sparse and potentially high-dimensional
vectors when the vocabulary is very large.

Therefore, although Bag of Words is simple and useful, more advanced
methods such as word embeddings and transformer-based representations
can capture additional information about meaning and context.

------------------------------------------------------------------------

## Question 6: What challenges arise when building NLP systems for low-resource languages such as Somali?

A low-resource language is a language that has limited digital resources
for developing natural language processing systems. These resources
include electronic text collections, labelled datasets, dictionaries,
language-processing tools, and evaluation benchmarks.

Somali faces several challenges in NLP compared with languages that have
much larger digital resources.

### 1. Limited Digital Data

There are fewer publicly available and carefully labelled Somali
datasets for training NLP models. For example, a sentiment-analysis
system needs many Somali sentences labelled as positive or negative.
Without enough examples, the system may not learn language patterns
accurately.

Research on Somali information retrieval has identified the lack of
dedicated Somali-language corpora as an important challenge.

### 2. Spelling Differences and Dialect Variation

Somali speakers may use different expressions, spellings, and dialect
forms. Informal online writing can also include abbreviations and
inconsistent spelling.

These differences can make it difficult for NLP systems to recognize
words or expressions with similar meanings. Research on Somali dialect
identification has demonstrated the importance of representing dialect
variation in datasets.

### 3. Morphological Complexity

Morphology is the study of how words are formed and how their parts
affect meaning. Somali has agglutinative and derivational word
formation, meaning that words can have different related forms.

An NLP system may treat related forms as separate words, making
searching and text analysis more difficult. Lemmatization can help by
reducing related word forms to a common root or base form.

Recent Somali research has developed a lemmatization resource containing
thousands of root words and derivative forms to help address this
challenge.

### 4. Code-Switching

Code-switching happens when people use more than one language in the
same conversation or text. For example, a Somali speaker might combine
Somali sentences with English technical terms.

An illustrative example is:

> Internet-kayga is not working.

A model trained only on formal, single-language Somali may struggle to
understand such mixed-language messages. It may not correctly identify
the language of each word or understand the full meaning.

### How Can Researchers Address These Challenges?

Researchers can address these problems by:

-   Collecting more Somali text from different sources, with appropriate
    permission.
-   Creating labelled datasets with help from native speakers and
    language experts.
-   Including different dialects and writing styles.
-   Developing spelling-normalization and lemmatization tools.
-   Using multilingual or subword-based models when appropriate.
-   Testing models on different types of Somali text, not only the data
    used during training.

**Conclusion:** Developing effective Somali NLP systems requires
sufficient language data, suitable linguistic tools, and collaboration
with Somali-speaking communities. More research and investment in Somali
language resources can help improve language technology for Somali
speakers.

------------------------------------------------------------------------

## Question 7: Describe one real-world case study where NLP was applied to solve a problem.

One example is the research paper by Badel et al. (2023), titled *Somali
Information Retrieval Corpus: Bridging the Gap between Query Translation
and Dedicated Language Resources*, published in the Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing
(EMNLP).

### 1. Problem

The researchers wanted to improve information retrieval for the Somali
language. Existing Somali search systems often depended on translating
search queries into other languages because dedicated Somali-language
resources were limited.

Translation could introduce errors or change the meaning of the original
query. The researchers aimed to develop a system that could search
Somali documents directly.

### 2. NLP Task

The study focused on **information retrieval**, which involves finding
and ranking relevant documents in response to a user's search query.

This is one of the search and similarity tasks discussed in Question 3.

### 3. Data

The researchers developed a Somali-language corpus containing:

-   2,335 documents.
-   16,450 sentences.
-   578,216 words.
-   16 search queries.

The documents were collected from Somali news websites and cultural and
history books. The researchers also worked with language experts and NLP
researchers to prepare and annotate the collection.

### 4. Method

The researchers developed an information retrieval system using two
important techniques:

-   **BM25:** A method for ranking documents according to how relevant
    their words are to a search query.
-   **Pseudo-Relevance Feedback (PRF):** A technique that first
    retrieves an initial set of documents, selects important words from
    the top results, and adds those words to the original query to
    improve the next search.

They compared their PRF approach with the traditional TF-IDF method.

### 5. Findings

The PRF approach performed better than TF-IDF in their experiments.

For the top 10 retrieved documents, PRF achieved an F-measure of
approximately 72%, compared with approximately 54% for TF-IDF.

For the top 20 documents, PRF achieved approximately 57%, compared with
approximately 44% for TF-IDF.

The study demonstrated the value of developing dedicated Somali-language
resources for information retrieval. However, the researchers also
identified spelling variation and the lack of a fully functional Somali
stemmer as limitations.

### Conclusion

This case study shows how NLP can help people find information in their
own language. It also demonstrates why building Somali-language datasets
and tools is important for improving language technology and making
digital information more accessible to Somali speakers.

------------------------------------------------------------------------

# References

1.  Jurafsky, D., & Martin, J. H. (2026). *Speech and Language
    Processing: An Introduction to Natural Language Processing,
    Computational Linguistics, and Speech Recognition with Language
    Models* (3rd ed., online manuscript). Stanford University.\
    https://web.stanford.edu/\~jurafsky/slp3/

2.  Bird, S., Klein, E., & Loper, E. (2009). *Natural Language
    Processing with Python: Analyzing Text with the Natural Language
    Toolkit*. O'Reilly Media.\
    https://www.nltk.org/book/

3.  Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to
    Information Retrieval*. Cambridge University Press.\
    https://nlp.stanford.edu/IR-book/

4.  Scikit-learn Developers. *Working with Text Data*. Scikit-learn User
    Guide.\
    https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

5.  Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., et al. (2026). *Somali
    dialect identification in low-resource settings using machine
    learning and deep learning*. Discover Data, 4, Article 10.\
    https://doi.org/10.1007/s44248-026-00109-y

6.  Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., et al. (2026).
    *Morphologically-informed Somali Lemmatization Corpus built with a
    Web-based Crowdsourcing Platform*. Proceedings of the 7th Workshop
    on African Natural Language Processing (AfricaNLP 2026).\
    https://doi.org/10.18653/v1/2026.africanlp-main.17

7.  Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). *Somali
    Information Retrieval Corpus: Bridging the Gap between Query
    Translation and Dedicated Language Resources*. Proceedings of the
    2023 Conference on Empirical Methods in Natural Language Processing,
    7463--7469.\
    https://doi.org/10.18653/v1/2023.emnlp-main.462

------------------------------------------------------------------------

**End of Assignment 1 -- NLP Foundations**
