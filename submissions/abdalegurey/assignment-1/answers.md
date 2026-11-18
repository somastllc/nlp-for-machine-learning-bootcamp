# Assignment 1 – NLP Foundations

**Subject:** Natural Language Processing  
**Topic:** Foundations of Natural Language Processing  
**Assignment:** 1

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a branch of computer science and Artificial Intelligence (AI) that enables computers to understand, process, analyze, and generate human languages such as English, Somali, and Arabic.

NLP allows computers to perform language-related activities such as translating sentences, detecting spam messages, analyzing customer reviews, recognizing names, searching documents, and generating answers.

For example, consider an SMS message:

"Congratulations! You have won a free prize. Click here now!"

A human can recognize that this message might be spam. An NLP system can also analyze the message and predict whether it is spam or a normal message.

### Relationship between AI, Machine Learning, and NLP

These three concepts are connected, but they have different meanings.

- **Artificial Intelligence (AI):** A broad field of technology focused on creating computer systems that perform tasks requiring human-like intelligence.
- **Machine Learning (ML):** A branch of AI that allows computers to learn patterns from data and make predictions without programming every decision manually.
- **Natural Language Processing (NLP):** A field that focuses on enabling computers to work with human language. NLP can use Machine Learning, Deep Learning, or rule-based techniques.

### Real-life example: Email spam detection

Imagine an email application that automatically identifies unwanted messages.

1. **AI:** Provides the overall intelligent system that makes decisions about incoming emails.
2. **NLP:** Processes the email text and identifies relevant language patterns.
3. **Machine Learning:** Learns from previously labeled emails to predict whether a new email is spam or legitimate.

For example, if a model has learned from thousands of spam and normal emails, it can analyze a new email and predict its category.

Therefore, AI is the broader field, ML provides learning techniques, and NLP focuses on human language processing.

## 2. Core NLP Terms

To understand NLP, it is important to understand the basic terms used when working with text datasets.

For this question, I will use an SMS spam detection dataset containing messages classified as spam or ham.

| Term | Definition | Example |
|---|---|---|
| Corpus | A collection of text documents used for NLP research or model training. | A dataset containing 10,000 SMS messages. |
| Document | One individual piece of text in a corpus. | "Are you coming today?" |
| Token | A unit of text produced by a tokenizer. It can be a word, part of a word, or punctuation. | `Python`, `is`, `useful`, `.` |
| Vocabulary | The collection of unique tokens found or recognized in a dataset. | `['free', 'prize', 'meeting']` |
| Feature | A numerical value representing information from text that a Machine Learning model uses as input. | The number of times the word `prize` appears. |
| Label | The correct category assigned to a document during supervised learning. | `spam` or `ham`. |

### Practical example

Consider these four SMS messages:

| Document | SMS Message | Label |
|---|---|---|
| 1 | Are you coming today? | ham |
| 2 | Claim your cash prize! | spam |
| 3 | Call me when you arrive. | ham |
| 4 | Win money now! | spam |

In this example:

- **Corpus:** All four SMS messages together.
- **Document:** Each individual SMS message.
- **Token:** Words such as `Claim`, `cash`, `prize`, and `money`.
- **Vocabulary:** The unique tokens collected from the messages.
- **Feature:** Numerical information such as whether the word `money` appears in a message.
- **Label:** The category assigned to each message, either spam or ham.

An important difference is that a token can appear several times, while vocabulary contains unique tokens only.

For example, the sentence:

`Python Python is useful.`

Can produce five token occurrences:

`['Python', 'Python', 'is', 'useful', '.']`

However, it contains only four unique vocabulary entries.

## 3. Four Common NLP Tasks

NLP systems perform different tasks depending on the problem they are designed to solve.

### 3.1 Text Classification

Text classification is an NLP task that assigns a category or class to an entire text document.

It is commonly used in spam detection, sentiment analysis, news categorization, and customer feedback analysis.

**Example:** A system analyzes an SMS and determines whether it is spam or a normal message.

- **Input:** A complete text document or SMS message.
- **Output:** A predicted category or label, such as `spam` or `ham`.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task that assigns labels to individual words or tokens within a sentence.

It is useful for Named Entity Recognition (NER), identifying people's names, locations, organizations, and other important information.

**Example:**

Sentence: "Amina lives in Muqdisho."

| Word | Label |
|---|---|
| Amina | PERSON |
| lives | OTHER |
| in | OTHER |
| Muqdisho | LOCATION |

- **Input:** A sentence divided into tokens.
- **Output:** A corresponding label for each token.

### 3.3 Search and Similarity

Search and similarity is an NLP task that identifies documents related to a user's query or finds texts with similar meanings or content.

For example, a customer searching for "battery problems" may receive product reviews discussing batteries that stop working quickly.

This task can use numerical text representations to compare documents and rank relevant results.

- **Input:** A search query and a collection of documents.
- **Output:** A ranked list of relevant or similar documents.

### 3.4 Text Generation

Text generation is an NLP task in which a computer produces new human-readable text based on an input, instruction, or previous context.

It is used in chatbots, automatic translation, summarization, and AI writing assistants.

**Example:** A user asks an AI assistant, "Explain Machine Learning in simple English."

The system generates an explanation based on the request.

- **Input:** A prompt, question, or existing text.
- **Output:** Newly generated text, such as an answer, summary, or translation.

### Summary of NLP Tasks

| NLP Task | Input | Output | Example |
|---|---|---|---|
| Text Classification | Complete text | Category or class | Spam detection |
| Sequence Labeling | Sentence or tokens | Label for each token | Name recognition |
| Search and Similarity | Query and documents | Ranked relevant documents | Document search |
| Text Generation | Prompt or existing text | New text | Chatbot response |

## 4. Supervised vs. Unsupervised Learning in NLP

Machine Learning approaches in NLP can be divided into supervised and unsupervised learning.

### Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from training data containing both input examples and their correct target labels.

The model examines the relationship between the text and its labels to learn patterns that can be used to predict the labels of new documents.

**Example: SMS Spam Classification**

Consider a dataset containing thousands of SMS messages. Each message is manually classified as either spam or ham.

| Message | Label |
|---|---|
| Win a free prize now! | spam |
| Are you coming home? | ham |
| Claim your cash reward! | spam |

The model learns from these examples and predicts whether new messages are spam or ham.

**Why are target labels needed?**

Labels provide the correct answers during training. They help the model understand which category should be predicted for each example.

Without these target labels, the model cannot learn the intended spam-versus-ham classification directly through supervised training.

### Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model works with data that does not contain predefined target labels for the task.

Instead of learning to predict known categories, the system discovers patterns, relationships, or groups within the data.

**Example: Customer Review Grouping**

Imagine collecting 5,000 customer reviews without assigning positive or negative labels.

An unsupervised algorithm such as K-Means can group reviews according to similarities in their numerical representations.

For example, the resulting groups might contain:

- Reviews discussing product quality.
- Reviews discussing delivery services.
- Reviews discussing product prices.

These groups are discovered from patterns in the data rather than supplied target labels.

**Why are target labels not needed?**

The objective is to discover patterns or structures in the data rather than predict a predefined correct answer.

### Comparison

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labeled training data. | Uses data without target labels for the task. |
| Learns from input and correct answers. | Discovers patterns and relationships. |
| Predicts predefined categories or values. | Finds groups, structures, or similarities. |
| Example: Spam classification. | Example: Customer review clustering. |
| Evaluation can compare predictions with known labels. | Evaluation may use clustering or other structure-based measures. |

The main difference is that supervised learning requires target labels because it learns to predict known answers, while unsupervised learning does not require them because it discovers patterns independently.

## 5. Why Must Text Be Converted into Numerical Representations?

Most traditional Machine Learning algorithms cannot directly process raw human-readable text as mathematical input.

These algorithms work with numerical values, such as integers and floating-point numbers. Therefore, text must first be converted into numerical representations.

This process is called **text vectorization**.

For example, a computer cannot directly use the sentence "I love Machine Learning" as a numerical feature matrix. A vectorizer converts the text into numbers that an algorithm can process.

### Bag of Words (BoW)

Bag of Words is a simple text representation technique that converts documents into numerical vectors based on the words they contain.

It creates a vocabulary from the text and assigns a numerical position to each vocabulary word.

Consider this vocabulary:

`['free', 'meeting', 'prize']`

We can represent two messages using binary values:

| Message | free | meeting | prize | Vector |
|---|---:|---:|---:|---|
| free prize | 1 | 0 | 1 | [1, 0, 1] |
| free meeting | 1 | 1 | 0 | [1, 1, 0] |

Here:

- `1` means the word appears in the document.
- `0` means the word does not appear in the document.

The resulting vectors can now be used as input for a Machine Learning model.

### Main Limitation of Bag of Words

The main limitation of Bag of Words is that it does not understand the meaning or context of words.

It also ignores word order.

For example:

- "The dog bites the man."
- "The man bites the dog."

These sentences contain the same words, so a basic Bag-of-Words representation may produce identical vectors even though their meanings are different.

Additionally, Bag of Words can create large and sparse vectors when a dataset contains many unique words.

Therefore, although Bag of Words is useful for simple NLP applications, it cannot capture all the meaning and relationships within human language.

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have relatively limited digital text collections, annotated datasets, linguistic tools, and computational resources available for developing NLP applications.

Somali is an example of a language that faces several challenges in NLP research and development.

Although Somali is widely spoken, the availability of high-quality digital resources for NLP remains limited compared with languages such as English.

### 6.1 Limited Datasets and Annotated Data

One of the biggest challenges is the limited availability of high-quality Somali text datasets, especially datasets that have been manually labeled for specific NLP tasks.

For example, developing a Somali spam detection system requires many SMS messages labeled as spam or ham.

If researchers cannot obtain enough examples, the model may struggle to learn reliable language patterns.

**Effects on NLP systems:**

- Models may perform poorly on new messages.
- Training data may not represent different types of Somali writing.
- Evaluation becomes more difficult.
- Researchers may need to collect and manually annotate their own datasets.

A study by Zupon, Crew, and Ritchie (2021) examined text normalization and dataset quality for several African low-resource languages, including Somali. It highlights why data quality is important when developing language models.

### 6.2 Spelling Variation

Somali users may write the same word in different ways, particularly in informal digital communication.

For example:

- `Soomaaliya`
- `Somalia`

A basic word-level tokenizer may recognize these as two different tokens, even though they refer to the same country.

This creates difficulties when building a vocabulary and training a Machine Learning model.

**Effects on NLP systems:**

- The vocabulary becomes unnecessarily large.
- Similar words may be treated as unrelated.
- Search systems may fail to retrieve relevant documents.
- Classification models may have difficulty recognizing familiar words written differently.

Text normalization can help reduce these problems by standardizing spelling where appropriate.

### 6.3 Word Forms and Morphological Complexity

Somali uses different word forms to express grammatical information.

For example:

- `nin` – man
- `ninka` – the man

Although these words are related, a simple word-level representation may treat them as separate vocabulary entries.

This can make it harder for NLP models to recognize relationships between words.

**Effects on NLP systems:**

- Similar words may be represented differently.
- The vocabulary may contain many related forms.
- Search and information retrieval may miss relevant documents.
- Models may require more training examples to learn different forms.

Techniques such as stemming and lemmatization can help reduce different word forms to a common base or root.

### 6.4 Mixed-Language Text

Many Somali speakers combine Somali and English in everyday digital communication.

For example:

"Meeting-ka maanta ayaan leenahay."

This sentence combines the English word `Meeting` with the Somali suffix `-ka`.

A tokenizer trained mainly on English may not process this combination correctly.

**Effects on NLP systems:**

- Incorrect tokenization.
- Poor vocabulary coverage.
- Difficulty identifying the language of individual words.
- Reduced accuracy in classification and translation.

### Conclusion

Developing NLP applications for Somali requires more than choosing a powerful Machine Learning algorithm.

Researchers must also consider data availability, spelling variation, word forms, language varieties, and mixed-language communication.

Creating high-quality Somali datasets and developing language-specific preprocessing tools can improve the accuracy and usefulness of future NLP systems.

## 7. Case Study: Somali Information Retrieval Corpus

### Research Paper

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Published:** 2023, Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

### 7.1 Research Problem

The researchers identified a problem in Somali NLP: there was a lack of dedicated Somali information retrieval datasets.

As a result, many existing approaches relied on translating search queries into other languages rather than searching directly using Somali language resources.

The researchers wanted to address this problem by developing a dedicated Somali information retrieval corpus and a system for retrieving relevant documents.

### 7.2 Data Used

The researchers created a corpus containing **2,335 Somali documents**.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The corpus was developed through collaboration between language experts and NLP researchers.

This dataset provided a resource for developing and evaluating Somali information retrieval approaches.

### 7.3 Method Used

The researchers developed a Somali information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF) query expansion**.

The general idea of query expansion is to improve an initial search query by adding useful related terms.

For example, if a user searches for a particular Somali topic, the system can use relevant information from initially retrieved documents to expand the query and improve subsequent retrieval.

The purpose is to help the search system identify more relevant documents.

### 7.4 Findings

The research produced two important contributions:

1. It introduced a dedicated Somali information retrieval corpus containing 2,335 documents.
2. It developed an information retrieval system using pseudo-relevance feedback and query expansion.

The study also demonstrated the importance of creating language-specific datasets for low-resource languages such as Somali.

The published abstract does not provide numerical retrieval-performance results, so it would not be appropriate to claim a particular accuracy or improvement percentage.

### 7.5 Which NLP Task Does This Case Study Cover?

This research primarily covers **Search and Similarity**.

The reason is that the system receives a user's search query and attempts to retrieve relevant documents from a collection of Somali texts.

It is an example of how NLP can solve a real information-access problem for Somali-speaking users.

### 7.6 Importance of the Research

This study is important because it contributes to the development of Somali language technology.

A dedicated Somali information retrieval corpus can support future applications such as:

- Somali search engines.
- Digital libraries.
- Question-answering systems.
- Somali document retrieval.
- Other Machine Learning-based NLP applications.

It also demonstrates that building high-quality language datasets is an important step toward developing useful NLP systems for low-resource languages.

## Overall Conclusion

Natural Language Processing is an important field of Artificial Intelligence that allows computers to process and generate human language.

Understanding fundamental concepts such as corpora, documents, tokens, vocabulary, features, and labels is necessary before developing NLP applications.

Common NLP tasks include text classification, sequence labeling, search and similarity, and text generation. These tasks can use different Machine Learning approaches depending on whether target labels are available.

Text must also be converted into numerical representations before most traditional Machine Learning models can process it.

Finally, languages such as Somali face challenges involving limited datasets, spelling variation, morphology, and mixed-language communication. Research and development of Somali-specific datasets and tools can help overcome these challenges.



