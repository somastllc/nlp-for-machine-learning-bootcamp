# Assignment 1 – NLP Foundations

**Student:** Atika Isse Ali  
**GitHub:** [atikamTech](https://github.com/atikamTech)  
**Course:** NLP for Machine Learning Bootcamp

---

## 1. Definition of Natural Language Processing (NLP)

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that focuses on enabling computers to work with human language. It allows computer systems to process, analyze, understand and generate text or speech in languages that people use to communicate.

Human language is complex because the same word can have different meanings depending on its context, and people express similar ideas in many different ways. NLP provides methods that help computers handle these challenges and perform language-related tasks.

### Relationship between AI, Machine Learning and NLP

Artificial Intelligence, Machine Learning and NLP are closely connected, but they do not mean the same thing.

- **Artificial Intelligence (AI):** A broad field of computer science concerned with building systems that can perform tasks that normally require human intelligence, such as reasoning, problem-solving and understanding language.
- **Machine Learning (ML):** A branch of AI that enables computers to learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed for every situation.
- **Natural Language Processing (NLP):** A field of AI concerned with processing and working with human language. It often uses machine learning techniques to analyze text and speech.

NLP can use machine learning to learn language patterns from examples. However, not every NLP system requires machine learning; some systems also use dictionaries, linguistic rules and other approaches.

### Real-life example: Email spam detection

An email spam filter demonstrates how these fields can work together.

Imagine receiving hundreds of emails, including advertisements, unwanted messages and important personal communication.

1. **NLP:** The system processes the email text and identifies useful information, such as words, phrases and message content.
2. **Machine Learning:** The system learns patterns from previously identified spam and legitimate emails.
3. **Artificial Intelligence:** The complete system uses the learned patterns to make decisions and automatically classify incoming messages.

For example, an email containing suspicious promotional language and misleading links may be classified as spam, while a normal message from a colleague may be classified as legitimate.

This example shows how NLP helps a computer work with language, machine learning helps it recognize patterns, and AI provides the broader intelligent system.

---

## 2. Core NLP Terminology

Before building an NLP system, it is important to understand how text data is organized and represented. The terms corpus, document, token, vocabulary, feature and label describe different parts of the data used in NLP.

For this question, I will use a small example dataset containing customer reviews of an online shopping service.

### Example dataset

| ID | Customer review | Label |
|---|---|---|
| 1 | The delivery was fast and excellent. | Positive |
| 2 | The product quality was poor. | Negative |
| 3 | I received my order on time. | Positive |
| 4 | The delivery was late and disappointing. | Negative |

### Explanation of the terms

**1. Corpus**

A corpus is a collection of text documents gathered for language analysis or NLP research. It provides the data that researchers use to study language or train and evaluate NLP systems.

In this example, all four customer reviews together form the corpus.

**2. Document**

A document is one individual unit of text within a corpus. Depending on the dataset, a document can be a sentence, customer review, article, email or complete book.

In the example, each customer review represents one document. Therefore, the dataset contains four documents.

**3. Token**

A token is an individual unit produced when text is divided into smaller pieces through a process called tokenization. Tokens can be words, punctuation marks or smaller subword units, depending on the tokenization method.

For example, the sentence:

"The delivery was fast."

Can be divided into these tokens:

`["The", "delivery", "was", "fast", "."]`

This example contains five tokens, including the punctuation mark.

**4. Vocabulary**

Vocabulary refers to the collection of unique tokens found in a corpus after applying a chosen tokenization and normalization method.

For example, if we consider the reviews in the sample dataset and convert words to lowercase, the vocabulary contains unique words such as:

`{the, delivery, was, fast, and, excellent, product, quality, poor, i, received, my, order, on, time, late, disappointing}`

Repeated words are included only once in the vocabulary. Punctuation has been excluded from this illustrative vocabulary.

**5. Feature**

A feature is a measurable or numerical representation of information that a machine learning model uses to identify patterns and make predictions.

For example, the number of times a word appears in a customer review can be used as a feature.

| Review | fast | excellent | poor | late |
|---|---:|---:|---:|---:|
| Review 1 | 1 | 1 | 0 | 0 |
| Review 2 | 0 | 0 | 1 | 0 |
| Review 3 | 0 | 0 | 0 | 0 |
| Review 4 | 0 | 0 | 0 | 1 |

These numerical values represent word-count features.

**6. Label**

A label is the expected category or correct answer associated with a data example. In supervised learning, labels help a model learn the relationship between input data and the desired output.

In this dataset, the labels are Positive and Negative. They indicate the sentiment expressed in each customer review.

### Summary of the terminology

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of documents | All four reviews |
| Document | One individual text | Review 1 |
| Token | A unit of text | "delivery" |
| Vocabulary | Unique tokens in the corpus | Unique review words |
| Feature | Numerical information used by a model | Count of "excellent" |
| Label | Expected category or output | Positive |

These concepts are connected. A corpus contains documents, documents are divided into tokens, tokens contribute to a vocabulary, and features and labels help prepare the data for machine learning.

---

## 3. Four Common NLP Tasks

NLP supports many applications, and each task has a different purpose. Some tasks identify categories in text, while others recognize individual words, retrieve information or generate new sentences.

The following are four common NLP tasks.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category to a piece of text based on its content.

It is commonly used in spam detection, sentiment analysis, news categorization and customer feedback analysis.

- **Input:** A text document, sentence or message.
- **Output:** A category or class assigned to the input.

**Example:**

Input: "I really enjoyed using this application."

Output: Positive sentiment.

A sentiment analysis
