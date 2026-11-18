
# Assignment 1 – NLP Foundations
## Natural Language Processing

---

## 1. What Is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that focuses on enabling computers to process, analyze, understand, translate, and generate human languages such as English, Somali, Arabic, and others.

NLP helps AI systems work with human language. Some common applications include **Sentiment Analysis**, **Machine Translation**, **Text Generation**, and **Speech Recognition**.

### Relationship Between AI, Machine Learning, and NLP

> **Artificial Intelligence (AI)** is the general field of Computer Science that focuses on developing systems capable of performing tasks that normally require human intelligence, such as understanding, decision-making, and problem-solving.

> **Machine Learning (ML)** is a branch of AI that enables computers to learn from data and identify patterns so they can make predictions or decisions instead of being explicitly programmed with rules for every task.

> **Natural Language Processing (NLP)** is a field of AI that focuses on human language. It uses techniques such as Machine Learning to help computers process and understand human text and speech.

---

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

### Example: Student Feedback Analysis Dataset

For this example, I am creating a **Student Feedback Analysis Dataset** to analyze students' comments about their education and identify whether their feedback is positive, negative, or neutral.

| Document ID | Student Feedback | Label |
|---|---|---|
| D-1 | The teacher explains lessons clearly and helps students. | Positive |
| D-2 | The lessons are difficult and the teaching is poor. | Negative |
| D-3 | The classroom is good and the lessons are normal. | Neutral |

### 2.1 Corpus

A **Corpus** is a collection of texts or documents gathered for research or Natural Language Processing tasks.

In this example, the three student feedback comments together form one corpus. A corpus can contain three documents, millions of documents, or even more, depending on the amount of collected data.

### 2.2 Document

A **Document** is a single text or piece of writing within a corpus.

For example:

D-1: "The teacher explains lessons clearly and helps students."

This is one document in our dataset.

### 2.3 Token

A **Token** is a piece of text produced when a sentence or document is divided into individual words or smaller units.

Example:

```text
The classroom is good
```

Tokens:

```python
["The", "classroom", "is", "good"]
```

Each individual word is considered a token.

### 2.4 Vocabulary

**Vocabulary** is the collection of all unique tokens found in a corpus. If a word appears multiple times, it is included only once in the vocabulary.

For example, the vocabulary of the three feedback comments includes:

```python
["the", "teacher", "explains", "lessons", "clearly",
 "and", "helps", "students", "are", "difficult",
 "teaching", "is", "poor", "classroom", "good", "normal"]
```

**Note:** In this example, all words have been converted to lowercase so that "The" and "the" are treated as the same token.

### 2.5 Feature

A **Feature** is a characteristic or piece of information extracted from text that a Machine Learning model uses to learn patterns or make predictions.

Examples of features include:
- Whether the word `good` appears in a comment.
- How many times the word `lessons` appears.
- The total number of words in a comment.

### 2.6 Label

A **Label** is the correct category or answer assigned to a document. It tells the model what it is expected to predict during supervised learning.

The labels in our dataset are:
- `Positive`
- `Negative`
- `Neutral`

---

## 3. Four Common NLP Tasks

NLP involves different tasks. The following are four common examples:

| Task | What it does | Input | Output | Example |
|---|---|---|---|---|
| **Text Classification** | Assigns a category to an entire text. | An article, email, or message. | One label, such as `Technology`. | News article classification. |
| **Sequence Labeling** | Assigns a label to each token in a sentence. | A sentence divided into tokens. | One label for each token, such as `PERSON` or `LOCATION`. | Named Entity Recognition (NER). |
| **Search and Similarity** | Finds and ranks documents according to their relevance to a query. | A search query and a collection of documents. | A ranked list of relevant documents. | Searching for relevant research papers. |
| **Text Generation** | Produces new text based on a prompt or input text. | A prompt or source text. | Newly generated text. | Machine translation or text summarization. |

---

## 4. Supervised and Unsupervised Learning in NLP

**Supervised Learning** is a Machine Learning approach in which a model is trained using labeled data. The model learns from examples that contain inputs and their correct answers.

**Unsupervised Learning** is a Machine Learning approach in which a model is given data without labels. The model searches for hidden patterns and can group similar data together through clustering.

### Comparison

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labeled training data. | Uses data without labels. |
| Learns from examples with known answers. | Discovers patterns and groups in data. |
| Example: Spam detection. | Example: Grouping similar customer reviews. |

### Examples

**Supervised Learning – Spam Detection**

The model is given emails that have already been labeled as `Spam` or `Ham` (not spam). It learns the differences between these categories and then predicts whether a new email is Spam or Ham.

**Unsupervised Learning – Customer Segmentation**

The model is given customer data without labels. It groups customers with similar characteristics, such as purchasing behavior or spending habits.

This helps identify different customer groups (*customer segments*) without providing the model with predefined group labels.

---

## 5. Why Does Text Need Numerical Representation Before Most ML Models?

Computers and most Machine Learning models cannot directly process ordinary human-readable text (*raw text*) in the same way they process numerical data.

They need numerical values to perform calculations, identify patterns, and learn from data.

Therefore, text must be converted into numbers or numerical vectors before it can be provided to most Machine Learning models.

### One Simple Representation Method: Bag of Words (BoW)

**Bag of Words (BoW)** is a method of converting text into numerical vectors. First, a vocabulary containing all unique words is created. Then, the method counts how many times each word appears in each document or sentence.

For example, consider these two sentences:

```text
1. I love learning AI.
2. I love learning Python.
```

The vocabulary of these two sentences is:

```python
["I", "love", "learning", "AI", "Python"]
```

### Bag of Words Table

| Sentence | I | love | learning | AI | Python |
|---|---:|---:|---:|---:|---:|
| I love learning AI. | 1 | 1 | 1 | 1 | 0 |
| I love learning Python. | 1 | 1 | 1 | 0 | 1 |

- `1` = The word appears in the sentence.
- `0` = The word does not appear in the sentence.

Therefore:

- Sentence 1 → `[1, 1, 1, 1, 0]`
- Sentence 2 → `[1, 1, 1, 0, 1]`

### One Limitation of Bag of Words

Bag of Words does not consider the order of words (*word order*) or the overall meaning of a sentence (*context*). It mainly focuses on which words appear and how frequently they occur.

For example:

```text
I love learning AI.
AI love I learning.
```

These sentences can produce the same Bag of Words vector, even though their word order is different and their meanings are not the same.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A **Low-Resource Language** is a language that has limited digital data and linguistic resources available for developing and training Natural Language Processing (NLP) systems.

Somali is considered a low-resource language because there is limited availability of high-quality digital datasets and linguistic resources for NLP research compared with high-resource languages such as English.

There are several challenges affecting the development of AI and NLP technologies for Somali. This section focuses on two major challenges: **Limited Datasets** and **Code-Switching**.

### 6.1 Limited Datasets

One of the major challenges is the lack of large, high-quality, and human-annotated datasets for training NLP models.

For example, Badel et al. (2023) explained that Somali information retrieval research lacked a dedicated annotated corpus. To address this problem, they developed a Somali information retrieval corpus containing **2,335 documents**.

When training data is limited, NLP models may struggle to learn language patterns accurately. This can reduce their performance in tasks such as Text Classification, Question Answering, and Information Retrieval.

### 6.2 Code-Switching

**Code-Switching** occurs when people mix two or more languages within the same sentence or conversation.

For example, Somali speakers sometimes combine Somali and English words when communicating online or searching for information on the internet.

This creates difficulties for NLP models because they must recognize and understand words from different languages within the same sentence.

As a result, a model may misunderstand the meaning or context of the text and produce incorrect predictions.

### Summary of the Two Challenges

| Challenge | Explanation | Effect on NLP |
|---|---|---|
| **Limited Datasets** | There are not enough high-quality, annotated Somali text datasets for training models. | Models may struggle to learn language patterns and produce accurate results. |
| **Code-Switching** | Somali speakers sometimes mix Somali and English in the same sentence. | Models may misunderstand the context and make incorrect predictions. |

---

## 7. Case Study: Somali Sentiment Analysis

### Research Paper

**Title:** *Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification*

**Authors:** Kadar Bahar and Nehad T. A. Ramaha (2023).

### 7.1 Task

The main objective of this research was to develop an approach for analyzing sentiment in Somali text (*Sentiment Analysis*).

The researchers aimed to identify people's opinions and emotions expressed in Somali text and classify them into different sentiment categories:

- **Positive:** Expresses a favorable opinion, positive feeling, or praise.
- **Negative:** Expresses an unfavorable opinion, negative feeling, or criticism.
- **Neutral:** Expresses information without a clearly positive or negative sentiment.

### 7.2 Data

The researchers collected Somali text from different sources, including social media platforms, news articles, online forums, and blogs.

The corpus covered different topics, such as politics, sports, entertainment, and everyday conversations. It included both formal and informal Somali, regional dialects, and mixed-language text commonly used in digital communication.

They used a **Crowdsourcing** approach, which means assigning different people the task of labeling text samples according to their sentiment.

The labels included Positive, Negative, and Neutral. The researchers also used annotation guidelines and multiple rounds of review to improve labeling consistency.

### 7.3 Method

The researchers proposed a resource-light approach that combines lexicon-based methods and feature engineering techniques.

Their methodology included the following techniques:

- **Tokenization:** Dividing text into individual words or smaller units called tokens.
- **Stemming:** Reducing words to their base or root forms.
- **Stopword Removal:** Removing common words that do not contribute significantly to sentiment analysis.
- **Negation Handling:** Identifying negative expressions that can change the meaning or sentiment of a sentence.
- **Feature Extraction:** Extracting useful information from text using techniques such as n-grams and language-specific features.

The researchers also explored machine learning classifiers such as Support Vector Machines (SVM), Naive Bayes, and Random Forest.

They used supervised learning, meaning that the models were trained using text that had already been labeled according to its sentiment.

### 7.4 Findings

The study demonstrated that Somali sentiment analysis is possible even with limited datasets and linguistic resources.

The researchers reported that their resource-light approach achieved competitive performance compared with existing approaches for sentiment analysis in other under-resourced languages.

The study also highlighted the importance of developing Somali-specific datasets and language-processing tools.

It identified several challenges that require further research, including:
- Limited annotated datasets.
- Differences between regional dialects.
- Code-switching between Somali and other languages.
- Informal writing and expressions used on social media.

The researchers recommended expanding Somali sentiment datasets and exploring techniques such as transfer learning and cross-lingual approaches to improve future NLP systems.

### 7.5 Connection to Question 3

This research is mainly related to **Text Classification** because its purpose is to classify complete Somali text into sentiment categories such as Positive, Negative, and Neutral.

For example, a Somali sentence expressing happiness can be classified as Positive, while a sentence expressing dissatisfaction can be classified as Negative.

### Reference

Bahar, K., & Ramaha, N. T. A. (2023). *Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification.* International Conference on Applied Engineering and Natural Sciences, 1(1), 620–628.

https://doi.org/10.59287/icaens.1069

---

## References

1. Kowsari, K., et al. (2022). A survey on text classification algorithms: From text to predictions. *Information, 13*(2), 83. https://doi.org/10.3390/info13020083

2. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. *Proceedings of CoNLL-2003*, 142–147. https://aclanthology.org/W03-0419/

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Paaß, G., & Giesselbach, S. (2023). *Foundation Models for Text Generation*. Springer. https://doi.org/10.1007/978-3-031-23190-2

5. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information retrieval corpus: Bridging the gap between query translation and dedicated language resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462

6. Bahar, K., & Ramaha, N. T. A. (2023). Exploring Somali sentiment analysis: A resource-light approach for small-scale text classification. *International Conference on Applied Engineering and Natural Sciences, 1*(1), 620–628. https://doi.org/10.59287/icaens.1069
