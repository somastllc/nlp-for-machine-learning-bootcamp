# Natural Language Processing (NLP): Concepts, Machine Learning, Challenges, and Applications

## Introduction

Natural Language Processing (NLP) is an important area of computer
science that enables computers to work with human languages. It allows
machines to process, analyze, interpret, and generate written or spoken
communication.

NLP is used in everyday applications, including search engines,
translation systems, email spam detection, virtual assistants, and
customer-service chatbots.

This research examines seven important areas of NLP: its relationship to
Artificial Intelligence and Machine Learning, fundamental terminology,
common NLP tasks, learning approaches, numerical text representation,
challenges in low-resource languages, and a practical research case
study.

------------------------------------------------------------------------

## 1. Definition of Natural Language Processing (NLP) and Its Relationship to AI and ML

### 1.1 Definition of NLP

**Natural Language Processing (NLP)** is a field of computer science and
Artificial Intelligence (AI) concerned with developing computer systems
that can process, understand, interpret, and generate human language.

Human languages such as English, Somali, Arabic, and French contain
words, grammar, meaning, and context. Computers cannot automatically
understand these elements in the same way humans do. NLP provides
computational methods that enable computers to analyze language and
perform useful tasks.

For example, when someone writes:

> "I really enjoyed watching this movie."

An NLP system can analyze the sentence and identify that it expresses a
positive opinion.

NLP combines several techniques, including:

-   Linguistic rules and language analysis.
-   Statistical methods.
-   Machine Learning algorithms.
-   Deep Learning and neural networks.

### 1.2 Relationship between AI, ML, and NLP

Artificial Intelligence, Machine Learning, and Natural Language
Processing are related fields, but they have different responsibilities.

  -----------------------------------------------------------------------
  Concept                 Definition              Main purpose
  ----------------------- ----------------------- -----------------------
  Artificial Intelligence A broad field concerned Enables intelligent
  (AI)                    with developing         computer behavior.
                          machines that perform   
                          tasks associated with   
                          human intelligence.     

  Machine Learning (ML)   A branch of AI that     Learns patterns and
                          allows computers to     makes predictions.
                          learn patterns from     
                          data and improve their  
                          performance.            

  Natural Language        A field of AI that      Enables computers to
  Processing (NLP)        focuses on processing   work with text and
                          and understanding human speech.
                          language.               
  -----------------------------------------------------------------------

Their relationship can be understood as follows:

-   AI is the broader field.
-   ML is one of the approaches used to develop AI systems.
-   NLP is the language-focused field that uses AI techniques, including
    ML, to solve language-related problems.

Not every NLP system requires Machine Learning. Some NLP applications
use predefined linguistic rules, while modern applications frequently
combine rules, ML, and Deep Learning.

### 1.3 Real-life example: Email spam detection

Consider an email application that automatically identifies unwanted
messages.

Suppose a user receives the following email:

> "Congratulations! You have won a free smartphone. Click this link to
> claim your prize."

The system processes the message through several stages:

1.  **NLP:** Examines the email text and identifies relevant language
    patterns, such as suspicious phrases and promotional expressions.
2.  **Machine Learning:** Uses patterns learned from previously
    classified emails to estimate whether the new message is spam.
3.  **Artificial Intelligence:** Provides the intelligent application
    behavior of automatically deciding how to handle the email.
4.  **Final result:** The application classifies the email as spam and
    moves it into the spam folder.

This example demonstrates how NLP and ML can work together within an AI
application.

**Key point:** NLP provides language-processing capabilities, ML learns
patterns from examples, and AI describes the broader intelligent system
in which these capabilities operate.

------------------------------------------------------------------------

## 2. Fundamental NLP Terminology

Understanding the basic terminology of NLP is important because these
concepts describe how text data is collected, organized, processed, and
used in Machine Learning.

To explain these terms consistently, this section uses one original
example dataset: a collection of customer reviews about a fictional
online bookstore.

### 2.1 Definitions

  -----------------------------------------------------------------------
  Term                    Definition              Simple explanation
  ----------------------- ----------------------- -----------------------
  Corpus                  A structured collection The complete collection
                          of written or spoken    of texts.
                          language data used for  
                          linguistic analysis or  
                          NLP research.           

  Document                One individual text     A single review or
                          within a corpus.        message.

  Token                   A basic unit produced   A word or punctuation
                          by dividing text during mark, depending on the
                          tokenization.           tokenizer.

  Vocabulary              The collection of       All distinct words or
                          unique tokens           tokens.
                          identified in a corpus. 

  Feature                 A measurable numerical  Information the model
                          property extracted from uses to learn.
                          text and used by a      
                          Machine Learning model. 

  Label                   The target category or  The correct answer the
                          value assigned to an    model should learn to
                          example in supervised   predict.
                          learning.               
  -----------------------------------------------------------------------

### 2.2 Example dataset: Online bookstore reviews

Assume a bookstore collects the following three customer reviews:

  -----------------------------------------------------------------------
  Document ID             Customer review         Label
  ----------------------- ----------------------- -----------------------
  D1                      The book arrived        Positive
                          quickly and the quality 
                          was excellent.          

  D2                      The delivery was late   Negative
                          and the book was        
                          damaged.                

  D3                      The book was            Positive
                          interesting and the     
                          price was reasonable.   
  -----------------------------------------------------------------------

The six NLP terms can be understood using this dataset.

**1. Corpus**

The complete collection of the three reviews is the corpus. It contains
all the text documents collected for the research.

**2. Document**

Each individual customer review represents one document.

For example:

D1: "The book arrived quickly and the quality was excellent."

D1 is one document within the larger corpus.

**3. Token**

Tokens are the smaller units obtained when a document is divided into
separate elements.

For example, using whitespace-based word tokenization:

Original sentence:

"The book arrived quickly."

Tokens:

`["The", "book", "arrived", "quickly"]`

Each item is a token.

**4. Vocabulary**

Vocabulary contains the unique tokens appearing in the entire corpus.

For example, the vocabulary includes:

`{the, book, arrived, quickly, and, quality, was, excellent, delivery, late, damaged, interesting, price, reasonable}`

Repeated words are counted only once in the vocabulary.

*Note: The displayed vocabulary uses lowercase normalization and does
not include punctuation.*

**5. Feature**

Features represent numerical information extracted from documents.

For example, a system could use the following features:

-   Number of positive words.
-   Number of negative words.
-   Total number of words.
-   Frequency of particular words.

For D1:

  Feature                                           Value
  ----------------------------------------------- -------
  Number of positive words (quickly, excellent)         2
  Number of negative words                              0
  Total number of words                                 9

These numerical values can be used by a Machine Learning algorithm to
identify patterns in customer opinions.

**6. Label**

A label represents the expected output assigned to a document.

In this example:

-   D1 = Positive
-   D2 = Negative
-   D3 = Positive

The labels allow a supervised learning model to learn the relationship
between the text features and the correct sentiment category.

**Summary of the dataset structure:**

``` text
Corpus: Complete bookstore review collection
|
|-- Document D1
|   |-- Tokens
|   |-- Features
|   |-- Label: Positive
|
|-- Document D2
|   |-- Tokens
|   |-- Features
|   |-- Label: Negative
|
|-- Document D3
    |-- Tokens
    |-- Features
    |-- Label: Positive
```

The example illustrates how raw customer reviews can be transformed into
structured information suitable for NLP and Machine Learning.

------------------------------------------------------------------------

## 3. Four Common NLP Tasks

NLP systems perform different tasks depending on the problem they are
designed to solve. Four widely used tasks are text classification,
sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

**Definition:** Text classification is the process of assigning a
predefined category to a complete text document or message.

The system examines the text and predicts which category best describes
it.

**Example:** A university receives student feedback through an online
platform. The system automatically categorizes feedback as positive,
negative, or neutral.

-   **Input:** "The university library provides excellent services."
-   **Processing:** The system analyzes the words and their patterns.
-   **Output:** Positive.

  -----------------------------------------------------------------------
  Component                           Description
  ----------------------------------- -----------------------------------
  Input                               A text document, email, or message

  Processing                          Analysis of text features

  Output                              A predefined category

  Applications                        Spam detection, sentiment analysis,
                                      topic classification
  -----------------------------------------------------------------------

### 3.2 Sequence Labeling

**Definition:** Sequence labeling is an NLP task in which a system
assigns a label to each individual token in a sequence of text.

Unlike text classification, which normally produces one category for the
entire document, sequence labeling produces labels for individual words
or tokens.

**Example:** Consider the sentence:

"Ahmed studies Computer Science at Mogadishu University."

An NLP system performing Named Entity Recognition (NER) can identify the
following:

  Token        Predicted label
  ------------ -------------------
  Ahmed        PERSON
  studies      O
  Computer     EDUCATION_SUBJECT
  Science      EDUCATION_SUBJECT
  at           O
  Mogadishu    LOCATION
  University   ORGANIZATION

Here, `O` means that the token does not belong to one of the entity
categories being identified.

-   **Input:** A sentence or sequence of tokens.
-   **Output:** A sequence of corresponding labels.

Applications include:

-   Identifying people's names.
-   Recognizing locations and organizations.
-   Identifying dates and monetary amounts.
-   Part-of-speech tagging.

### 3.3 Search and Similarity

**Definition:** Search and similarity is an NLP task that identifies
documents or sentences related to a user's query based on their words,
content, or meaning.

Traditional search systems often rely on matching keywords. More
advanced NLP systems can also compare semantic meaning, allowing them to
retrieve relevant information even when the wording is different.

**Example:** A university student searches for:

> "How does a computer store information?"

The system retrieves a learning document titled:

> "Introduction to Computer Memory and Data Storage."

Although the exact words are different, the two texts are related in
meaning.

  -----------------------------------------------------------------------
  Component                           Description
  ----------------------------------- -----------------------------------
  Input                               A search query and a collection of
                                      documents

  Processing                          Keyword matching or semantic
                                      similarity comparison

  Output                              Ranked relevant documents

  Applications                        Search engines, digital libraries,
                                      academic document retrieval
  -----------------------------------------------------------------------

### 3.4 Text Generation

**Definition:** Text generation is an NLP task in which a computer
produces new human-readable text based on an input, instruction, or
previously supplied information.

Modern text generation systems commonly use neural networks and large
language models.

**Example:** A student asks an academic chatbot:

> "Explain the difference between hardware and software."

The chatbot generates the following response:

"Hardware refers to the physical components of a computer, such as the
keyboard, processor, and monitor. Software refers to the programs and
instructions that control computer operations."

The system creates a new response rather than simply selecting an
existing category.

  -----------------------------------------------------------------------
  Component                           Description
  ----------------------------------- -----------------------------------
  Input                               A prompt, question, or contextual
                                      information

  Processing                          Language modeling and prediction of
                                      subsequent tokens

  Output                              Newly generated text

  Applications                        Chatbots, report writing,
                                      summarization, question answering
  -----------------------------------------------------------------------

### 3.5 Comparison of the Four NLP Tasks

  -----------------------------------------------------------------------
  NLP task          Input             Output            Example
  ----------------- ----------------- ----------------- -----------------
  Text              Complete text     One category      Spam or not spam
  classification                                        

  Sequence labeling Sequence of       A label for each  Identifying
                    tokens            token             people's names

  Search and        Query and         Ranked relevant   Finding
  similarity        documents         documents         university notes

  Text generation   Prompt or context New text          Academic chatbot
                                                        response
  -----------------------------------------------------------------------

**Important distinction:** Text classification answers *which category
does this text belong to?* Sequence labeling answers *what role does
each word have?* Search answers *which documents are relevant?* Text
generation answers *what text should the system produce?*

------------------------------------------------------------------------

## 4. Supervised and Unsupervised Learning in NLP

Machine Learning provides different approaches for identifying patterns
in text. Two fundamental approaches are supervised learning and
unsupervised learning.

### 4.1 Supervised Learning

**Definition:** Supervised learning is a Machine Learning approach in
which a model learns from training examples that contain both input data
and their correct target labels.

The model examines the examples, learns the relationship between the
input and the expected output, and uses that knowledge to predict labels
for new data.

**Example: Student feedback classification**

Suppose a university collects 1,000 student feedback messages. Each
message is manually classified into one of three categories:

-   Positive
-   Negative
-   Neutral

Training dataset:

  Student feedback                             Label
  -------------------------------------------- ----------
  The lecturers explain lessons clearly.       Positive
  The internet connection is very poor.        Negative
  The examination starts at 9 AM.              Neutral
  The library has useful learning resources.   Positive

The model learns patterns associated with each category.

When a new message arrives:

> "The university provides excellent academic support."

The trained model may predict:

**Positive**

The target labels are necessary because they provide the correct answers
against which the model learns.

### 4.2 Unsupervised Learning

**Definition:** Unsupervised learning is a Machine Learning approach in
which a model analyzes data without being provided with predefined
target labels.

Instead of learning from known answers, the model identifies patterns,
similarities, relationships, or groups within the data.

**Example: Grouping university student feedback**

Imagine that a university collects 1,000 feedback messages but does not
classify them.

An unsupervised learning algorithm, such as K-Means clustering, can
analyze the text representations and group similar messages.

Possible results:

  -----------------------------------------------------------------------
  Group                   Example messages        Discovered topic
  ----------------------- ----------------------- -----------------------
  Cluster 1               "The library needs more Library resources
                          books." / "We need      
                          additional textbooks."  

  Cluster 2               "The internet is slow." Internet services
                          / "Wi-Fi disconnects    
                          frequently."            

  Cluster 3               "Lectures start late."  Class scheduling
                          / "Classes need better  
                          scheduling."            
  -----------------------------------------------------------------------

The system discovers these groups from patterns in the data. The topic
names are assigned by a human after examining the clusters; they are not
necessarily produced as predefined labels by the algorithm.

### 4.3 Comparison between Supervised and Unsupervised Learning

  -----------------------------------------------------------------------
  Comparison              Supervised learning     Unsupervised learning
  ----------------------- ----------------------- -----------------------
  Training data           Labeled data            Unlabeled data

  Target labels           Required                Not required

  Learning process        Learns from known       Discovers patterns and
                          input-output examples   structures

  Main purpose            Prediction and          Pattern discovery and
                          classification          grouping

  NLP example             Sentiment               Grouping similar
                          classification          documents

  Common algorithms       Naive Bayes, SVM,       K-Means, hierarchical
                          supervised neural       clustering, topic
                          networks                modeling
  -----------------------------------------------------------------------

### 4.4 Why Does One Require Labels and the Other Not?

The difference is related to the objective of each learning approach.

In supervised learning, the model must learn how to produce a specific
expected answer. Therefore, the training dataset needs target labels.

For example, when learning to classify feedback as positive or negative,
the model needs examples showing which messages belong to each category.

In unsupervised learning, the objective is different. The model is
expected to discover patterns or relationships independently. It does
not need predefined answers because its purpose is to organize or
explore the data rather than predict a known category.

**Conclusion:** Supervised learning learns from examples with known
answers, while unsupervised learning discovers hidden structures without
predefined answers.

------------------------------------------------------------------------

## 5. Why Text Must Be Converted into Numerical Representations

### 5.1 Why Numerical Representation Is Necessary

Human language consists of words, sentences, and symbols. However, most
traditional Machine Learning algorithms operate on numerical data rather
than directly processing ordinary written sentences.

For example, a computer program can display the sentence:

"The university provides quality education."

But a traditional ML algorithm cannot directly calculate mathematical
relationships between these words in their original textual form.

Therefore, NLP uses a process called **text vectorization**, which
converts textual information into numerical representations.

These representations allow algorithms to:

-   Calculate similarities between documents.
-   Identify word-frequency patterns.
-   Train classification models.
-   Discover relationships within text.
-   Make predictions using mathematical operations.

A numerical representation of text is commonly called a vector.

### 5.2 Bag-of-Words (BoW) Representation

One of the simplest methods of converting text into numerical values is
the **Bag-of-Words (BoW)** model.

Bag-of-Words represents each document using the frequency of words
appearing in it. It creates a vocabulary and assigns a numerical value
to each vocabulary word.

The method follows three main steps:

1.  Tokenize the documents into individual words.
2.  Create a vocabulary containing unique words.
3.  Count how many times each vocabulary word appears in each document.

### 5.3 Practical Example of Bag-of-Words

Consider these three original sentences:

-   D1: "Students study computer science."
-   D2: "Students study information technology."
-   D3: "Teachers teach computer science."

**Step 1: Create the vocabulary**

After tokenization and removal of duplicate vocabulary entries:

`[students, study, computer, science, information, technology, teachers, teach]`

**Step 2: Count word frequencies**

Create a numerical table showing the number of occurrences of each word.

  --------------------------------------------------------------------------------------------------
  Document     students    study   computer   science   information   technology   teachers    teach
  ---------- ---------- -------- ---------- --------- ------------- ------------ ---------- --------
  D1                  1        1          1         1             0            0          0        0

  D2                  1        1          0         0             1            1          0        0

  D3                  0        0          1         1             0            0          1        1
  --------------------------------------------------------------------------------------------------

**Step 3: Convert documents into vectors**

For example, document D1 becomes:

\[ D1=\[1,1,1,1,0,0,0,0\] \]

Document D2 becomes:

\[ D2=\[1,1,0,0,1,1,0,0\] \]

These vectors can now be processed by traditional Machine Learning
algorithms.

### 5.4 Main Limitation of Bag-of-Words

Although Bag-of-Words is simple and useful, it has an important
limitation: **it ignores word order and does not adequately capture
context or meaning.**

Consider the following two sentences:

1.  "The student helped the teacher."
2.  "The teacher helped the student."

Both sentences contain exactly the same words and therefore produce the
same Bag-of-Words representation.

However, their meanings are different because the subject and object
have changed.

This demonstrates why word frequency alone is insufficient for
understanding many language structures.

Other limitations include:

-   It creates large, sparse vectors when the vocabulary is extensive.
-   It does not naturally understand synonyms.
-   It cannot reliably identify context-dependent meanings.
-   It does not capture grammatical relationships.

More advanced methods, such as TF-IDF, Word2Vec, GloVe, and contextual
embeddings from transformer models, address some of these limitations in
different ways.

**Conclusion:** Numerical representation is necessary because it makes
language suitable for mathematical processing. Bag-of-Words is a simple
starting method, but more advanced representations are required when
meaning, context, and word order are important.

------------------------------------------------------------------------

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

### 6.1 Understanding Low-Resource Languages

A low-resource language is a language for which the digital and
computational resources needed to develop NLP applications are limited.

These resources include:

-   Large collections of electronic text.
-   Human-annotated training datasets.
-   Dictionaries and language-specific lexicons.
-   Tokenizers and morphological analyzers.
-   Language models and evaluation benchmarks.

Somali is an important example because, despite having millions of
speakers and a rich literary tradition, it has comparatively fewer
computational language resources than English and other widely
researched languages.

Research on Somali information retrieval identifies the lack of
dedicated electronic datasets as a major barrier to developing reliable
language-processing applications.

The following are four important challenges.

### 6.2 Challenge One: Limited Datasets and Training Resources

**Explanation**

Machine Learning models require suitable data to identify patterns in
language.

For languages such as English, researchers can access large collections
of books, articles, websites, and annotated datasets.

However, Somali has fewer publicly available, high-quality datasets for
tasks such as:

-   Sentiment analysis.
-   Named Entity Recognition.
-   Machine translation.
-   Information retrieval.
-   Speech recognition.

**How this affects NLP systems**

When a model is trained using a small dataset, it may fail to learn
sufficient language patterns. It can perform well on familiar examples
but produce inaccurate results when processing new or unfamiliar text.

For example, a Somali chatbot trained mainly on a small collection of
formal news articles may struggle to understand informal questions
written by university students.

**Possible solution**

Researchers can develop larger Somali text collections, collaborate with
native speakers, create manually annotated datasets, and reuse suitable
multilingual resources through transfer learning.

A relevant contribution is the Somali Information Retrieval Corpus
developed by Badel et al. (2023), which contains 2,335 documents.

### 6.3 Challenge Two: Spelling Variation and Inconsistent Writing

**Explanation**

People do not always write the same word in exactly the same way,
particularly in informal digital communication.

Social media messages may contain:

-   Misspellings.
-   Abbreviations.
-   Informal expressions.
-   Missing punctuation.
-   Inconsistent word spacing.

These differences can make it difficult for an NLP system to recognize
that two written forms refer to the same word or concept.

**How this affects NLP systems**

Consider a hypothetical Somali information retrieval system.

A student searches for a particular subject using one spelling, but the
relevant document contains a different spelling or an informal
variation.

The system may fail to retrieve the document because the written forms
do not match.

This reduces search accuracy and can also affect text classification and
chatbot responses.

**Possible solution**

Researchers can apply text normalization, develop Somali spelling
correction tools, and create standardized language datasets while
retaining useful original text.

### 6.4 Challenge Three: Rich Morphology and Different Word Forms

**Explanation**

Somali has relatively rich morphology, meaning that words can appear in
different forms through grammatical inflection and word formation.

A single root or base word may have multiple related forms.

This creates additional complexity for tokenization, word matching, and
language modeling.

**How this affects NLP systems**

A system that treats every word form as completely unrelated may fail to
recognize that several forms share the same underlying root.

For example, different forms of a word may appear in separate documents.
A simple keyword-based search might fail to retrieve all relevant
documents if it does not account for those variations.

**Possible solution**

Researchers can develop language-specific stemming and lemmatization
systems.

-   Stemming reduces words to an approximate stem.
-   Lemmatization attempts to identify the linguistically appropriate
    base or dictionary form.

A 2023 study by Mohamed and Mohamed developed a Somali lexicon
containing 1,247 root words and 7,173 derivationally related terms. A
later 2026 study by Gedi et al. expanded this research with a resource
containing more than 5,584 root words and 78,629 derivative forms.

### 6.5 Challenge Four: Code-Switching and Mixed-Language Text

**Explanation**

Code-switching occurs when speakers alternate between two or more
languages within the same conversation or written message.

For example, a Somali student might write:

> "Macallin, assignment-ka ma submit gareyn karaa maanta?"

This sentence combines Somali and English expressions.

**How this affects NLP systems**

A model trained only on standard Somali text may not correctly recognize
the English words or understand how they function within a Somali
sentence.

It may also misinterpret the grammatical relationship between words from
different languages.

This can affect:

-   Sentiment analysis.
-   Chatbot question understanding.
-   Text classification.
-   Search accuracy.

**Possible solution**

Researchers can collect real-world mixed-language datasets and train
multilingual models that recognize language changes within sentences.

### 6.6 Summary of Somali NLP Challenges

  -----------------------------------------------------------------------
  Challenge               Main problem            Effect on NLP
  ----------------------- ----------------------- -----------------------
  Limited datasets        Insufficient digital    Poor generalization and
                          and annotated text      unreliable predictions

  Spelling variation      Inconsistent written    Difficulty recognizing
                          forms                   equivalent words

  Rich morphology         Many forms derived from Reduced matching and
                          related roots           processing accuracy

  Code-switching          Mixing Somali and other Incorrect
                          languages               interpretation of
                                                  mixed-language
                                                  sentences
  -----------------------------------------------------------------------

**Overall discussion**

Developing effective NLP applications for Somali requires more than
simply applying algorithms originally designed for English. It requires
language-specific datasets, suitable preprocessing methods, linguistic
knowledge, and evaluation by native speakers.

The research by Gedi et al. (2026) demonstrates the importance of
building Somali-specific linguistic resources, while Badel et al. (2023)
demonstrates the value of developing dedicated Somali information
retrieval datasets.

------------------------------------------------------------------------

## 7. Research Case Study: Somali Information Retrieval Corpus

### 7.1 Background of the Study

A relevant research study is:

**"Somali Information Retrieval Corpus: Bridging the Gap between Query
Translation and Dedicated Language Resources."**

The paper was written by Abdisalam Badel, Ting Zhong, Wenxin Tai, and
Fan Zhou and published in the proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing (EMNLP).

The research addresses an important problem in Somali NLP: the limited
availability of dedicated Somali datasets for information retrieval.

Information retrieval is the process of finding relevant documents in
response to a user's search query.

The researchers aimed to create a Somali-language resource and
investigate an approach for retrieving relevant documents without
depending entirely on translating queries into another language.

### 7.2 Research Task

The main task was **Somali information retrieval**.

The researchers focused on developing a system that could retrieve
relevant Somali documents based on user search queries.

For example, a user might search for information about education in
Somalia.

The system should identify documents containing relevant information,
rather than requiring the user to search through every document
manually.

### 7.3 Dataset Used

The researchers constructed an annotated corpus containing:

**2,335 Somali text documents.**

The documents were collected from different sources, including:

-   Hiiraan Online.
-   Dhacdo.net.
-   Somali poetry books.

The research involved collaboration between language experts and NLP
researchers to construct the corpus.

The dataset provided a dedicated resource for investigating Somali
information retrieval and supporting future NLP research.

### 7.4 Method Used

The researchers developed a Somali information retrieval system using a
technique called **Pseudo-Relevance Feedback (PRF)**.

Pseudo-Relevance Feedback is a search improvement technique in which the
system initially retrieves documents for a query and then uses the
top-ranked results to identify additional useful search terms.

The process can be understood in four steps:

1.  **Initial query:** A user enters a search query.
2.  **Initial retrieval:** The system searches the Somali document
    collection and retrieves potentially relevant documents.
3.  **Query expansion:** The system examines highly ranked documents to
    identify additional terms that may improve the original query.
4.  **Improved retrieval:** The expanded query is used to retrieve a
    potentially more relevant set of documents.

**Simplified PRF information retrieval process:**

``` text
User Search Query
       |
       v
Initial Document Retrieval
       |
       v
Pseudo-Relevance Feedback
       |
       v
Expanded Query
       |
       v
Retrieved Documents
```

*Note: This diagram explains the general PRF workflow, not a
reconstruction of the paper's exact software architecture.*

### 7.5 Findings and Contributions

The study made several important contributions:

1.  **Created a dedicated Somali corpus:** The researchers introduced
    2,335 documents for Somali information retrieval research.
2.  **Addressed data scarcity:** The corpus helped address the lack of
    electronically available Somali datasets.
3.  **Developed a retrieval approach:** The researchers implemented an
    information retrieval system using pseudo-relevance feedback and
    query expansion.
4.  **Supported future research:** The dataset provides a foundation for
    further Somali NLP applications, including question answering and
    text classification.

An important research distinction is that the published abstract
establishes the corpus size, sources, and retrieval method. It does not
provide sufficient numerical evidence in the abstract to claim a
particular percentage improvement in retrieval accuracy. Therefore, no
unsupported performance figure is reported here.

### 7.6 Which NLP Task Does This Case Study Cover?

**Search and Similarity**

The case study primarily covers the **search and similarity** task
described in Question 3.

This is because its objective is to identify and retrieve documents
relevant to a user's query.

  -----------------------------------------------------------------------
  Question                            Case study answer
  ----------------------------------- -----------------------------------
  Research problem                    Limited Somali information
                                      retrieval resources

  NLP task                            Search and similarity

  Dataset                             2,335 Somali documents

  Data sources                        Hiiraan Online, Dhacdo.net, and
                                      Somali poetry books

  Method                              Pseudo-Relevance Feedback (PRF) and
                                      query expansion

  Main contribution                   A dedicated Somali information
                                      retrieval corpus and retrieval
                                      system

  Publication year                    2023
  -----------------------------------------------------------------------

### 7.7 Importance of the Case Study

This research demonstrates that creating language-specific resources is
an important step toward developing practical NLP systems for Somali.

Without suitable documents and datasets, it is difficult to build and
evaluate reliable information retrieval applications.

The study also demonstrates how linguistic expertise and Machine
Learning research can work together to address problems affecting
languages with limited digital resources.

It provides a useful foundation for developing Somali search engines,
digital libraries, academic information retrieval systems, and
question-answering applications.

------------------------------------------------------------------------

# References

The following references include scholarly conference papers, academic
books, and reliable technical publications. They provide the research
foundation for the explanations and examples used in this assignment.

1.  Aggarwal, C. C. (2022). *Machine Learning for Text* (2nd ed.).
    Springer. https://doi.org/10.1007/978-3-030-96623-2

2.  Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information
    Retrieval Corpus: Bridging the Gap between Query Translation and
    Dedicated Language Resources. In *Proceedings of the 2023 Conference
    on Empirical Methods in Natural Language Processing*
    (pp. 7463--7469). Association for Computational Linguistics.
    https://doi.org/10.18653/v1/2023.emnlp-main.462

3.  Chiang, D. (2021). *Natural Language Processing: Chapter 6 --
    Semantics*. University of Notre Dame.
    https://www3.nd.edu/\~dchiang/teaching/nlp/2021/notes/chapter6v2.pdf

4.  Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A.,
    Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed
    Somali Lemmatization Corpus built with a Web-based Crowdsourcing
    Platform. In *Proceedings of the 7th Workshop on African Natural
    Language Processing* (pp. 179--189). Association for Computational
    Linguistics. https://doi.org/10.18653/v1/2026.africanlp-main.17

5.  IBM. (2024, August 11). *What is Natural Language Processing (NLP)?*
    IBM Think.
    https://www.ibm.com/think/topics/natural-language-processing

6.  Kochmar, E. (2022). *Getting Started with Natural Language
    Processing*. Manning Publications.
    https://www.manning.com/books/getting-started-with-natural-language-processing

7.  Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-Based Word
    Lemmatization Approach for Somali Language. In *ICLR 2023 Workshops:
    AfricaNLP*.
    https://mlanthology.org/iclrw/2023/mohamed2023iclrw-lexicon/

8.  Nekoto, W., Marivate, V., Matsila, T., et al. (2020). Participatory
    Research for Low-resourced Machine Translation: A Case Study in
    African Languages. In *Findings of the Association for Computational
    Linguistics: EMNLP 2020*.
    https://doi.org/10.18653/v1/2020.findings-emnlp.195

9.  Jurafsky, D., & Martin, J. H. (2023). *Speech and Language
    Processing* (3rd ed., draft). Stanford University.
    https://web.stanford.edu/\~jurafsky/slp3/

------------------------------------------------------------------------

# Final Summary: Important Concepts to Remember

1.  **NLP:** Enables computers to process, understand, and generate
    human language.
2.  **AI, ML, and NLP:** AI is the broader field, ML learns patterns
    from data, and NLP focuses on human language.
3.  **Corpus:** A collection of text documents.
4.  **Document:** An individual text within a corpus.
5.  **Token:** A basic unit produced through text tokenization.
6.  **Vocabulary:** A collection of unique tokens.
7.  **Feature:** Numerical information used by a Machine Learning model.
8.  **Label:** The expected category or output in supervised learning.
9.  **Text classification:** Assigns categories to complete texts.
10. **Sequence labeling:** Assigns labels to individual tokens.
11. **Search and similarity:** Finds documents related to a query.
12. **Text generation:** Produces new human-readable text.
13. **Supervised learning:** Learns from labeled examples.
14. **Unsupervised learning:** Discovers patterns without predefined
    labels.
15. **Bag-of-Words:** Converts text into numerical word-frequency
    vectors.
16. **Low-resource NLP:** Faces challenges such as insufficient
    datasets, spelling variation, morphology, and code-switching.
17. **Somali case study:** Badel et al. (2023) developed a
    2,335-document Somali information retrieval corpus.

**Overall conclusion:** NLP connects human language with computer
intelligence. Its effectiveness depends not only on Machine Learning
algorithms but also on appropriate numerical representations, reliable
datasets, linguistic knowledge, and careful evaluation. For Somali and
other low-resource languages, developing language-specific datasets and
tools is essential for creating accurate and useful NLP applications.
