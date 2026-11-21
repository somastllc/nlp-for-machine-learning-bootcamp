# Assignment 1 -- NLP Foundations

## Introduction

Natural Language Processing (NLP) is an important field of modern
computing that enables computers to process, analyze, interpret, and
generate human language. It combines ideas from computer science,
artificial intelligence, linguistics, statistics, and machine learning.

NLP is used in many everyday technologies, including search engines,
translation applications, chatbots, voice assistants, spam filters, and
automated customer-support systems.

This assignment explores the fundamental concepts behind NLP, explains
how text data is prepared for machine learning, examines the challenges
of processing low-resource languages such as Somali, and reviews a
research application involving African languages.

------------------------------------------------------------------------

## 1. Define Natural Language Processing (NLP). What is its relationship to Artificial Intelligence and Machine Learning?

### 1.1 Definition of NLP

Natural Language Processing (NLP) is a branch of computer science and
artificial intelligence concerned with enabling computers to work with
human languages, including written text and spoken language.

Human languages are complex because words can have different meanings
depending on their context, sentences can have different structures, and
people frequently use informal expressions, abbreviations, and language
variations.

NLP develops computational methods that help machines identify patterns
in language, understand textual information, extract meaning, and
produce understandable responses.

For example, when someone writes:

> Where can I find a hospital near me?

An NLP system can identify that the message is a question, that the user
is looking for a hospital, and that location is important to the
request. The system can then use this information to provide a relevant
response.

### 1.2 Relationship Between NLP, Artificial Intelligence, and Machine Learning

These three concepts are closely connected, but they have different
purposes.

-   **Artificial Intelligence (AI):** The broader field of developing
    computer systems that perform tasks associated with human
    intelligence, such as reasoning, problem-solving, decision-making,
    perception, and language understanding.
-   **Machine Learning (ML):** A subfield of AI that allows computers to
    learn patterns from data and improve their performance without
    requiring every decision to be explicitly programmed.
-   **Natural Language Processing (NLP):** A field concerned
    specifically with human language. It uses techniques from AI,
    machine learning, linguistics, and statistics to process and
    understand text and speech.

NLP is an AI research area, and machine learning provides many of the
methods used to build NLP systems. These fields overlap rather than
forming a strict sequence.

### 1.3 Real-Life Example: Email Spam Detection

An email spam detection system demonstrates how AI, ML, and NLP work
together.

Suppose an email contains the following message:

> Congratulations! You have won a free prize. Click here to claim your
> reward.

The system works through several stages:

1.  **NLP:** Processes the email text, identifies words and phrases, and
    extracts useful language information.
2.  **Machine Learning:** Uses patterns learned from previously
    classified emails to estimate whether the new message is spam.
3.  **Artificial Intelligence:** Provides the broader intelligent
    decision-making functionality that determines how the email should
    be handled.
4.  **Final result:** The system classifies the email as spam and moves
    it into the spam folder.

This example shows that NLP provides language-processing capabilities,
ML learns from examples, and AI brings these capabilities together to
perform an intelligent task.

------------------------------------------------------------------------

## 2. Explain the Terms Corpus, Document, Token, Vocabulary, Feature and Label

These terms describe the basic components of an NLP dataset. To explain
them, consider a hypothetical dataset containing customer reviews of an
online shopping application.

### 2.1 Definitions

**1. Corpus**

A corpus is a collection of written or spoken language samples gathered
for linguistic analysis or for developing and evaluating NLP systems. A
collection of 10,000 customer reviews about an online shopping
application is an example of a text corpus.

**2. Document**

A document is one individual piece of text within a corpus. It could be
an email, customer review, news article, social media post, or
paragraph.

Example:

> The application is easy to use and very helpful.

This single review represents one document.

**3. Token**

A token is an individual unit produced when text is divided into smaller
pieces during tokenization. Tokens are often words or punctuation marks,
although some NLP systems divide words into smaller subword units.

Example:

Sentence: `The application is very useful.`

Tokens: `["The", "application", "is", "very", "useful", "."]`

**4. Vocabulary**

Vocabulary refers to the set of unique tokens found in a corpus after
the chosen text-processing steps.

Consider these reviews:

-   The application is useful.
-   The application is fast.
-   The application is easy.

The vocabulary could be:

`["The", "application", "is", "useful", "fast", "easy"]`

Repeated words are included only once.

**5. Feature**

A feature is a measurable or representable piece of information that a
machine learning model uses to identify patterns and make predictions.

For text classification, features might include whether a particular
word appears, how frequently a word appears, or a numerical
representation of a sentence. The presence of words such as "excellent",
"helpful", and "fast" could provide useful information when predicting
whether a customer review is positive.

**6. Label**

A label is the correct category or target answer assigned to a data
example during supervised learning.

For customer reviews, possible labels include Positive, Negative, and
Neutral.

Example:

-   Review: "The application is excellent."
-   Label: Positive.

The model uses labeled examples to learn how to classify new reviews.

### 2.2 Example Dataset

  Review ID   Document                       Label
  ----------- ------------------------------ ----------
  1           The application is excellent   Positive
  2           The service is very poor       Negative
  3           The application works well     Positive
  4           The service is not good        Negative

The dataset can be described as follows:

  NLP Term     Application to the Dataset
  ------------ --------------------------------------------------
  Corpus       All four customer reviews together
  Document     One individual customer review
  Token        An individual word or punctuation unit
  Vocabulary   All unique tokens across the reviews
  Feature      Numerical information extracted from each review
  Label        Positive or negative sentiment

### 2.3 Why These Concepts Matter

A corpus provides the overall collection of language data. Documents
represent individual examples. Tokenization breaks documents into
smaller units. Vocabulary identifies the unique tokens. Features convert
useful information into a form suitable for computational models, and
labels provide the correct answers when training supervised systems.

Without properly organized data, an NLP model may struggle to identify
meaningful patterns or produce reliable predictions.

------------------------------------------------------------------------

## 3. Describe Four Common NLP Tasks

NLP includes different tasks depending on what a system is expected to
accomplish. Four important tasks are text classification, sequence
labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category to
a piece of text. The system examines the content and predicts which
category best describes it.

**Real-life example:** An email filtering system determines whether an
incoming email is spam or not spam.

**Input:** A complete email or text message.

**Output:** A predicted category or class.

Example:

-   Input: "Congratulations! You have won a free prize. Click here."
-   Output: Spam.

Other applications include sentiment analysis, news categorization,
topic detection, and offensive-language detection.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which a system assigns a label to
each token or element in a sequence. Unlike text classification, which
usually produces one category for an entire document, sequence labeling
produces labels for individual parts of the input.

**Real-life example: Named Entity Recognition (NER)**

Sentence:

> Ahmed works at Eurosom Cloud in Mogadishu.

An NER system might identify the entities as follows:

  Token       Label
  ----------- --------------
  Ahmed       PERSON
  works       O
  at          O
  Eurosom     ORGANIZATION
  Cloud       ORGANIZATION
  in          O
  Mogadishu   LOCATION

`O` means that the token is not part of one of the entity categories
being identified.

**Input:** A sentence or sequence of tokens.

**Output:** A corresponding sequence of labels.

Applications include identifying people, organizations, locations,
dates, and other important information in documents.

### 3.3 Search and Similarity

Search and similarity involve finding documents that match a user's
query or identifying texts that have similar meanings.

A search system can compare words directly or use more advanced
representations that capture semantic meaning.

**Real-life example:** A user searches, "How can I reset my password?" A
useful search system should also retrieve a document titled, "Steps to
recover your account credentials." Although the wording is different,
the two texts express related ideas.

**Input:** - A search query. - A collection of documents or previously
indexed text.

**Output:** - A ranked list of relevant documents. - Similarity scores
indicating how closely texts match.

Applications include search engines, digital libraries, customer-support
systems, and document retrieval.

### 3.4 Text Generation

Text generation is the process of automatically producing human-readable
text based on an instruction, question, or other input.

Modern language models use patterns learned from large collections of
text to generate appropriate sequences of words or tokens.

**Real-life example:** A user asks a chatbot, "What is Natural Language
Processing?" The system generates an explanation describing NLP and its
applications.

**Input:** A prompt, question, instruction, or existing text.

**Output:** Newly generated text, such as an answer, summary, email, or
translation.

Applications include chatbots, automatic summarization, machine
translation, writing assistants, and question-answering systems.

### 3.5 Comparison of the Four NLP Tasks

  -----------------------------------------------------------------------
  NLP Task          Input             Output            Example
  ----------------- ----------------- ----------------- -----------------
  Text              Complete text     One category      Spam detection
  classification                                        

  Sequence labeling Sequence of       Label for each    Identifying names
                    tokens            token             

  Search and        Query and         Ranked documents  Search engine
  similarity        documents         or scores         

  Text generation   Prompt or text    Newly generated   AI chatbot
                                      text              
  -----------------------------------------------------------------------

**Conclusion:** Classification assigns categories, sequence labeling
identifies information within text, search finds relevant information,
and generation produces new language. Together, these tasks support many
NLP applications used in everyday life.

------------------------------------------------------------------------

## 4. Compare Supervised and Unsupervised Learning in NLP

Machine learning systems can learn from data in different ways. Two
important approaches are supervised learning and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a machine learning approach in which a model is
trained using examples that contain both input data and correct target
labels.

The model examines the relationship between the input and the expected
output. It uses these examples to learn how to make predictions on
previously unseen data.

**NLP example: Sentiment Analysis**

Suppose we want to develop a system that identifies whether customer
reviews are positive or negative.

  Customer Review                      Label
  ------------------------------------ ----------
  The service is excellent             Positive
  The application is terrible          Negative
  I really like this product           Positive
  I am disappointed with the service   Negative

The model learns patterns associated with positive and negative reviews.
After training, it might classify "The application works perfectly" as
Positive.

**Why are target labels needed?**

The labels provide the correct answers during training. They allow the
model to compare its predictions against the expected results and adjust
its parameters to reduce prediction errors.

### 4.2 Unsupervised Learning

Unsupervised learning is a machine learning approach in which a model
receives data without predefined target labels.

Instead of learning to predict known answers, the model attempts to
identify structures, similarities, patterns, or groups within the data.

**NLP example: Customer Review Clustering**

Imagine collecting 5,000 customer reviews without assigning any
sentiment categories.

An unsupervised learning algorithm can analyze similarities between the
reviews and group related documents. It might discover groups
containing:

-   Reviews about application performance.
-   Reviews about customer service.
-   Reviews about payment problems.
-   Reviews about delivery experiences.

These groups are discovered from patterns in the data rather than from
manually supplied category labels.

**Why are target labels not needed?**

The objective is to discover patterns or relationships within the data,
not to predict an answer that has already been provided. The algorithm
organizes the examples based on their similarities.

### 4.3 Comparison

  -----------------------------------------------------------------------
  Comparison              Supervised Learning     Unsupervised Learning
  ----------------------- ----------------------- -----------------------
  Training data           Labeled data            Unlabeled data

  Target labels           Required                Not required

  Main objective          Learn to predict known  Discover hidden
                          targets                 patterns

  Example NLP task        Sentiment               Topic clustering
                          classification          

  Evaluation              Compare predictions     Evaluate cluster
                          with correct labels     quality and usefulness

  Human involvement       Often requires          Does not require
                          annotation              predefined target
                                                  labels
  -----------------------------------------------------------------------

### 4.4 Why Both Approaches Are Important

Supervised learning is useful when the desired output is known and
examples can be labeled. However, creating labeled datasets can be
expensive and time-consuming.

Unsupervised learning is useful when large amounts of text are available
but labeled examples are limited. It can reveal patterns that
researchers may not have identified in advance.

For languages such as Somali, unsupervised learning can be valuable
because manually labeled language datasets are often difficult to
obtain.

**Conclusion:** The main difference is that supervised learning learns
from examples with known answers, whereas unsupervised learning
discovers patterns from data without predefined answers.

------------------------------------------------------------------------

## 5. Why Must Text Be Converted into Numerical Representations Before Most Machine Learning Models Can Use It?

### 5.1 Understanding the Problem

Computers store and process information using numerical values. However,
human language consists of words, sentences, symbols, and grammatical
structures.

A machine learning algorithm generally cannot directly perform
mathematical operations on an ordinary sentence such as:

> The application is very useful.

Therefore, NLP systems transform text into numerical representations
that algorithms can process. This process is commonly called **text
vectorization** or **text representation**.

Numerical representations allow a machine learning model to calculate
similarities between documents, identify patterns in word usage, measure
the importance of words, learn relationships between textual examples,
and make predictions.

### 5.2 Simple Representation Method: Bag of Words (BoW)

Bag of Words is a traditional text representation technique that
converts documents into numerical vectors based on the occurrence or
frequency of words.

The method creates a vocabulary from the available documents and counts
how often each vocabulary word appears in each document.

Consider these three sentences:

-   Document 1: "I like machine learning."
-   Document 2: "I like natural language."
-   Document 3: "Machine learning is useful."

**Step 1: Create a vocabulary**

`["I", "like", "machine", "learning", "natural", "language", "is", "useful"]`

**Step 2: Convert documents into numerical vectors**

Count the occurrence of each vocabulary word in each document.

  Word         Document 1   Document 2   Document 3
  ---------- ------------ ------------ ------------
  I                     1            1            0
  like                  1            1            0
  machine               1            0            1
  learning              1            0            1
  natural               0            1            0
  language              0            1            0
  is                    0            0            1
  useful                0            0            1

The resulting numerical representations are:

``` text
Document 1: [1, 1, 1, 1, 0, 0, 0, 0]
Document 2: [1, 1, 0, 0, 1, 1, 0, 0]
Document 3: [0, 0, 1, 1, 0, 0, 1, 1]
```

Each number represents the frequency of a particular vocabulary word in
a document. The machine learning model can now use these vectors for
mathematical calculations and pattern recognition.

### 5.3 Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and
much of the context in which words appear.

For example:

-   "The student helped the teacher."
-   "The teacher helped the student."

These sentences contain the same words, but their meanings are
different. A basic Bag of Words representation produces identical
vectors for both sentences because it only considers word occurrence,
not word order.

Other limitations include:

1.  **Large vocabulary:** A large corpus can produce very
    high-dimensional vectors.
2.  **Limited semantic understanding:** Words with similar meanings are
    treated as different features.
3.  **Context loss:** The representation does not capture how word
    meanings depend on surrounding words.
4.  **Data sparsity:** Many documents contain only a small proportion of
    the words in the complete vocabulary.

### 5.4 Other Representation Methods

-   **TF-IDF:** Gives greater importance to words that are frequent in a
    particular document but less common across the entire collection.
-   **Word embeddings:** Represent words as dense numerical vectors that
    can capture semantic and linguistic relationships.
-   **Contextual embeddings:** Represent words according to their
    surrounding context, allowing the same word to have different
    representations in different sentences.

**Conclusion:** Text must generally be converted into numerical
representations because machine learning algorithms operate on numerical
data. Bag of Words provides a simple starting point, but more advanced
techniques are often needed to understand meaning, context, and
relationships between words.

------------------------------------------------------------------------

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

### 6.1 Understanding Low-Resource Languages

A low-resource language is a language for which relatively limited
digital language resources are available for developing and evaluating
NLP systems.

These resources can include digitized books and articles, annotated text
datasets, dictionaries and language lexicons, tokenizers and
morphological analyzers, machine translation datasets, speech recordings
and transcriptions, and pretrained language models.

Somali is an important example because, despite having millions of
speakers, its available NLP resources remain more limited than those
available for languages such as English.

Badel et al. (2023) describe how Somali information retrieval research
has historically depended heavily on translating queries into other
languages because of the lack of dedicated Somali-language datasets.

### 6.2 Challenge One: Limited Datasets and Language Resources

One major challenge is the limited availability of high-quality Somali
text datasets that can be used to train and evaluate machine learning
models.

For example, an English sentiment analysis system might have access to
millions of labeled reviews and social media posts. A Somali sentiment
analysis researcher may have to collect and manually annotate a much
smaller dataset.

This creates several problems:

1.  **Insufficient training data:** When the available Somali dataset is
    small, a model may fail to recognize different expressions and
    sentence structures.
2.  **Limited vocabulary coverage:** A dataset collected from formal
    news articles may not contain informal expressions commonly used in
    everyday conversations and social media.
3.  **Difficult model evaluation:** A small test dataset may not
    represent all Somali speakers, dialects, topics, and writing styles.
4.  **Expensive manual annotation:** Creating labeled datasets requires
    people who understand the language and can accurately identify the
    correct categories.

Badel et al. (2023) developed a dedicated Somali information retrieval
corpus containing 2,335 documents collected from different sources,
including online news websites and Somali poetry books. The research
demonstrates the importance of creating language-specific datasets
instead of depending entirely on translated resources.

### 6.3 Challenge Two: Spelling and Orthographic Variation

Another challenge is inconsistent spelling and writing conventions.

People may communicate using formal Somali spelling, informal writing,
shortened words, or inconsistent representations of the same expression.

For example, the hypothetical forms `internetka` and `internet-ka` may
be treated as separate vocabulary entries by a system that does not
normalize punctuation and spelling consistently, even though they refer
to the same concept.

**Effects on NLP systems:**

1.  A search engine may fail to retrieve relevant documents because the
    query uses a different spelling.
2.  A text classifier may fail to recognize words that were not common
    in its training data.
3.  A vocabulary may become unnecessarily large because different forms
    are stored separately.
4.  Similarity calculations may become less accurate.

**Possible solutions:**

-   Text normalization.
-   Consistent tokenization rules.
-   Language-specific dictionaries.
-   Spelling correction.
-   Training with different spelling variations.

### 6.4 Challenge Three: Morphological Complexity and Word Forms

Somali has morphological structures that allow words to change form
through grammatical and derivational processes. Changes in word forms
can communicate information about number, grammatical relationships, and
other linguistic features.

For example:

-   `macalin` -- teacher
-   `macalimiin` -- teachers

These words are related but have different surface forms. A system that
treats every form as an unrelated word may fail to recognize that they
share a common linguistic base.

**Effects on NLP systems:**

-   Larger vocabularies.
-   More difficulty matching related words.
-   Reduced search accuracy.
-   Problems with text classification.
-   Difficulty identifying grammatical structures.

**Possible solution: Lemmatization**

Lemmatization reduces different forms of a word to a common base or
dictionary form.

Mohamed and Mohamed (2023) investigated a lexicon-based and rule-based
Somali lemmatization approach. Their initial resources included 1,247
root words and 7,173 related derived forms. Their research demonstrates
how language-specific lexical resources and morphological rules can help
address Somali word-form variation.

### 6.5 Challenge Four: Dialect Differences and Mixed-Language Text

Somali has dialectal variation, and speakers may use different
expressions and vocabulary depending on their regional and linguistic
backgrounds.

A model trained mainly on one variety of Somali may not perform equally
well on another.

Additionally, people frequently mix Somali with English or Arabic when
communicating online.

For example, a hypothetical social media message could be:

> Maanta meeting ayaan leeyahay.

This sentence combines Somali with an English word. A system designed to
process only standard Somali may struggle to interpret the entire
message correctly.

**Effects on NLP systems:**

1.  Incorrect tokenization.
2.  Misclassification of mixed-language sentences.
3.  Reduced accuracy in sentiment analysis.
4.  Difficulty identifying dialect-specific vocabulary.
5.  Poor generalization to users whose writing differs from the training
    data.

**Possible solutions:**

-   Collect data from different Somali dialects.
-   Include naturally occurring mixed-language examples.
-   Develop language identification systems.
-   Use multilingual language models.
-   Involve native Somali speakers in dataset annotation.

### 6.6 Summary of Somali NLP Challenges

  -----------------------------------------------------------------------
  Challenge               Main Problem            Effect on NLP
  ----------------------- ----------------------- -----------------------
  Limited datasets        Insufficient annotated  Weak model training
                          text                    

  Spelling variation      Different forms of      Poor search and
                          similar words           classification

  Morphological           Multiple forms of       Vocabulary and matching
  complexity              related words           problems

  Dialect variation       Different regional      Reduced generalization
                          language forms          

  Mixed-language text     More than one language  Tokenization and
                          in a sentence           interpretation errors
  -----------------------------------------------------------------------

### 6.7 Recommendations for Improving Somali NLP

Based on these challenges, the following steps could help improve Somali
NLP research:

1.  Build larger, publicly accessible Somali text datasets.
2.  Collect text from different domains, including education, news,
    social media, and public services.
3.  Include different dialects and writing styles.
4.  Develop Somali-specific tokenization and normalization tools.
5.  Create dictionaries, morphological analyzers, and lemmatization
    resources.
6.  Involve native speakers in reviewing and annotating datasets.
7.  Evaluate NLP models using data from different sources rather than
    relying on one dataset.

**Conclusion:** Developing reliable NLP systems for Somali requires more
than applying algorithms designed for English. Researchers must consider
the language's available resources, spelling conventions, morphological
structure, dialects, and everyday communication patterns. Improving
these resources can make NLP technologies more accessible and useful for
Somali-speaking communities.

------------------------------------------------------------------------

## 7. Case Study: Somali Information Retrieval Corpus

### 7.1 Research Paper Information

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between
Query Translation and Dedicated Language Resources

**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou

**Publication:** Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing (EMNLP 2023), pages 7463--7469.

**DOI:** https://doi.org/10.18653/v1/2023.emnlp-main.462

**Original paper:** https://aclanthology.org/2023.emnlp-main.462/

### 7.2 Background and Problem

The research addresses the limited availability of dedicated
Somali-language resources for information retrieval.

Information retrieval is the process of finding relevant documents from
a collection based on a user's search query.

For example, someone might search for information about Somali
education, history, or culture.

The problem identified by the researchers was that Somali information
retrieval systems had often depended on translating Somali queries into
higher-resource languages, such as English.

However, translation-based searching can introduce errors and may fail
to preserve the original meaning and context of the user's query.

The researchers therefore investigated how a dedicated Somali-language
corpus could improve information retrieval.

### 7.3 Research Task

The main NLP task was **Information Retrieval (Search and Similarity)**.

The objective was to build a system that could retrieve relevant Somali
documents based on a user's query.

For example:

-   Input query: "Somali education"
-   Expected output: A ranked list of relevant Somali-language documents
    discussing education.

This directly corresponds to the search and similarity task described in
Question 3.

### 7.4 Dataset Used

The researchers created a Somali information retrieval corpus containing
2,335 documents.

The documents were collected from different sources, including:

-   Hiiraan Online.
-   Dhacdo.net.
-   Somali poetry books.
-   Other Somali-language content.

The researchers worked with language experts and NLP researchers to
construct the resource. They also developed a Somali stop-word list to
support document processing.

The corpus was designed to support Somali information retrieval research
and future NLP applications.

### 7.5 Methodology

The researchers developed an information retrieval system using a
technique called **Pseudo-Relevance Feedback (PRF)**.

The method can be understood through the following process:

1.  **User submits a query:** A user enters a Somali-language search
    query.
2.  **Initial document retrieval:** The system searches the Somali
    corpus and identifies documents that appear relevant to the query.
3.  **Pseudo-Relevance Feedback:** The system assumes that some of the
    highest-ranked documents are relevant and uses information from
    those documents to expand or improve the original query.
4.  **Query expansion:** Additional related terms are incorporated into
    the search process.
5.  **Improved retrieval:** The system performs another search and
    produces a revised ranking of documents.

The purpose of this approach is to improve the likelihood of retrieving
relevant documents.

### 7.6 Research Findings

The researchers compared their PRF approach with a traditional TF-IDF
retrieval baseline. The reported results showed improvements across the
evaluated retrieval metrics.

  Evaluation Metric          TF-IDF     PRF
  ------------------------ -------- -------
  Precision at 10 (P@10)      0.512   0.687
  Recall at 20 (R@20)         0.689   0.917
  F-measure at 10 (F@10)      0.537   0.721

These values are reported evaluation results from the research paper.
They should be checked against the original paper's experiment tables
before final academic submission.

The results suggest that the PRF technique improved document retrieval
performance compared with the TF-IDF baseline on the evaluated corpus.

In particular, the reported recall at 20 increased from 68.9% to 91.7%.
This means the PRF approach was more successful at retrieving relevant
documents within the first 20 results in that evaluation.

### 7.7 Importance of the Research

This research is important for several reasons.

**1. Supports Somali-language technology**

It provides a dedicated resource for developing search systems that
operate directly on Somali text.

**2. Reduces dependence on translation**

A native-language search system can reduce the need to translate every
query into English.

**3. Creates opportunities for future research**

The corpus can support further development of Somali search engines,
question-answering systems, text classification systems, and Somali
language processing tools.

**4. Supports language accessibility**

Improved information retrieval can help Somali speakers access relevant
digital information using their own language.

### 7.8 Limitations and Future Improvements

Although the research provides an important foundation, several
challenges remain.

The corpus contains a relatively limited number of documents compared
with large English-language information retrieval collections.

Somali morphological variation can also affect the matching of search
terms with documents. In addition, spelling differences can make it
difficult to retrieve documents containing alternative forms of a query
term.

Future improvements could include:

-   Expanding the corpus with more Somali documents.
-   Developing better Somali stemming and lemmatization tools.
-   Including additional dialects and writing styles.
-   Exploring semantic search using multilingual or Somali-specific
    language models.
-   Collecting more diverse evaluation queries.

### 7.9 Which NLP Task Does This Case Study Cover?

The case study primarily covers **Search and Similarity (Information
Retrieval)**.

It takes a search query as input and returns a ranked list of documents
as output. Its objective is to identify documents that are relevant to
the user's information needs.

**Case Study Conclusion:**

Badel et al. (2023) demonstrated how developing a dedicated
Somali-language corpus and applying pseudo-relevance feedback can
improve information retrieval. Their research is an important example of
how NLP can address practical problems in low-resource languages. It
also demonstrates that language-specific datasets and preprocessing
techniques are essential for developing useful NLP applications.

------------------------------------------------------------------------

## References

The following sources support the definitions, explanations, research
findings, and case study in this assignment. They include academic
books, peer-reviewed conference papers, and research publications.

1.  **Jurafsky, D., & Martin, J. H. (2026).** *Speech and Language
    Processing* (3rd ed., draft). Stanford University.
    https://web.stanford.edu/\~jurafsky/slp3/\
    Used for: NLP foundations, tokenization, vocabulary, language
    processing, and text representation.

2.  **Manning, C. D., Raghavan, P., & Schütze, H. (2008).**
    *Introduction to Information Retrieval.* Cambridge University Press.
    https://nlp.stanford.edu/IR-book/\
    Used for: Information retrieval, document representation, Bag of
    Words, TF-IDF, and search systems.

3.  **Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023).** Somali
    Information Retrieval Corpus: Bridging the Gap between Query
    Translation and Dedicated Language Resources. *Proceedings of the
    2023 Conference on Empirical Methods in Natural Language
    Processing*, 7463--7469.
    https://doi.org/10.18653/v1/2023.emnlp-main.462\
    Used for: Somali information retrieval, corpus development,
    pseudo-relevance feedback, and low-resource language challenges.

4.  **Mohamed, S. A., & Mohamed, M. A. (2023).** *Lexicon and Rule-based
    Word Lemmatization Approach for the Somali Language.* arXiv.
    https://arxiv.org/abs/2308.01785\
    Used for: Somali morphological complexity, lemmatization, and
    language-specific lexical resources.

5.  **Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A.,
    Hassan, F. M., & Assowe, H. A. (2026).** Morphologically-informed
    Somali Lemmatization Corpus built with a Web-based Crowdsourcing
    Platform. *Proceedings of the 7th Workshop on African Natural
    Language Processing (AfricaNLP 2026)*, 179--189.
    https://doi.org/10.18653/v1/2026.africanlp-main.17\
    Used for: Recent developments in Somali lemmatization resources and
    morphological processing.

6.  **Adelani, D. I., et al. (2023).** AfriSenti: A Twitter Sentiment
    Analysis Benchmark for African Languages. *Proceedings of the 2023
    Conference on Empirical Methods in Natural Language Processing.*
    https://aclanthology.org/2023.emnlp-main.862/\
    Used for: Annotated datasets and sentiment analysis research for
    African languages.

------------------------------------------------------------------------

## Overall Conclusion

Natural Language Processing is an important area of artificial
intelligence that allows computers to work with human language. Through
this assignment, I have learned that NLP involves several connected
concepts, including corpora, documents, tokens, vocabulary, features,
and labels.

I have also learned that different NLP tasks solve different problems.
Text classification assigns categories to documents, sequence labeling
identifies information within sentences, information retrieval finds
relevant documents, and text generation produces new human-readable
content.

Machine learning plays an important role in NLP because it allows
systems to learn patterns from numerical representations of language.
Supervised learning uses labeled examples, while unsupervised learning
discovers patterns without predefined target labels.

Another important lesson is that NLP development is not equally easy for
every language. Somali faces challenges involving limited digital
datasets, spelling variation, morphological complexity, dialect
differences, and mixed-language communication.

The Somali Information Retrieval Corpus research by Badel et al. (2023)
demonstrates that these challenges can be addressed through
language-specific datasets and appropriate NLP techniques.

In conclusion, developing successful NLP systems requires both technical
knowledge and an understanding of the language being processed. More
research, datasets, and language-specific tools are necessary to make
NLP technologies more accurate and accessible for Somali speakers and
other underrepresented language communities.
