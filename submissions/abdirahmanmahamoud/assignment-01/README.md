# Assignment 1 – NLP Foundations

**Subject:** Natural Language Processing (NLP)  
**Assignment:** 01 – NLP Foundations

---

## 1. Introduction to Natural Language Processing (NLP)

### 1.1 Definition of NLP

Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) that focuses on enabling computers to understand, interpret, analyze, and generate human language. Human languages include English, Somali, Arabic, and many others.

NLP combines ideas from computer science, linguistics, statistics, and machine learning to process both written and spoken language.

NLP is used in many everyday applications, including search engines, machine translation, chatbots, voice assistants, spam detection, and automatic text summarization.

For example, when a person writes a question in English or Somali to an AI chatbot, NLP techniques help the system process the words, understand the user's intention, and generate an appropriate response.

### 1.2 Relationship Between AI, Machine Learning, and NLP

Artificial Intelligence, Machine Learning, and Natural Language Processing are closely related, but they have different roles.

- **Artificial Intelligence (AI):** A broad field of computer science concerned with creating systems that perform tasks associated with human intelligence, such as reasoning, learning, and problem-solving.
- **Machine Learning (ML):** A subfield of AI that enables computers to learn patterns from data and make predictions or decisions without requiring every rule to be manually programmed.
- **Natural Language Processing (NLP):** A field of AI concerned with processing and understanding human language. It often uses machine learning methods to perform language-related tasks.

Machine Learning is one of the important approaches used to develop modern NLP systems. However, not every NLP system requires machine learning; some systems also use manually designed linguistic rules.

### 1.3 Real-Life Example: Google Translate

Google Translate demonstrates how AI, ML, and NLP work together.

When a user enters a sentence in Somali and requests an English translation:

1. NLP techniques process the input text and identify linguistic patterns.
2. Machine Learning models use patterns learned from large collections of translated texts to predict an appropriate translation.
3. AI provides the broader intelligent system that combines language processing and other computational techniques to deliver the translation.

Therefore, NLP provides the language-processing capabilities, ML provides learning methods, and AI represents the broader field in which the application operates.

---

## 2. Fundamental NLP Terminology

To understand how NLP systems work, it is important to understand six fundamental terms: corpus, document, token, vocabulary, feature, and label.

For this section, consider an example dataset containing customer reviews of a mobile application.

### 2.1 Corpus

A corpus is a structured collection of text or speech data gathered for linguistic analysis, research, or machine learning.

For example, a dataset containing 5,000 customer reviews of a mobile application can be considered a text corpus.

### 2.2 Document

A document is an individual unit of text within a corpus. Depending on the dataset, a document can be a sentence, paragraph, customer review, email, or news article.

For example:

"I really enjoy using this application."

This individual customer review represents one document in the corpus.

### 2.3 Token

A token is an individual unit produced when text is divided into smaller pieces through a process called tokenization. Tokens may be complete words, punctuation marks, or subword units.

For example, the sentence:

"I really enjoy this application."

Can be tokenized into:

`["I", "really", "enjoy", "this", "application", "."]`

Each item represents a token.

### 2.4 Vocabulary

Vocabulary refers to the collection of unique tokens identified in a corpus after applying the chosen text-processing rules.

For example, consider these two reviews:

- "I love this application."
- "I love this service."

Their vocabulary, ignoring punctuation and capitalization, is:

`["I", "love", "this", "application", "service"]`

Repeated words are included only once in the vocabulary.

### 2.5 Feature

A feature is a measurable property or numerical representation of text that a machine learning model uses to identify patterns and make predictions.

For example, a feature could indicate whether the word "excellent" appears in a customer review.

Other features may include word frequency, sentence length, TF-IDF values, and numerical embeddings.

### 2.6 Label

A label is the target category or expected answer assigned to a data example during supervised machine learning.

For example, customer reviews can be assigned sentiment labels:

- Positive
- Negative
- Neutral

### 2.7 Summary Table

| Term       | Example from the customer review dataset          |
| ---------- | ------------------------------------------------- |
| Corpus     | Collection of 5,000 customer reviews              |
| Document   | One individual customer review                    |
| Token      | An individual word or punctuation mark            |
| Vocabulary | All unique tokens in the dataset                  |
| Feature    | Word frequency or the presence of a specific word |
| Label      | Positive, Negative, or Neutral                    |

These concepts are important because they explain how raw text is organized, processed, and prepared for machine learning.

---

## 3. Four Common NLP Tasks

NLP systems perform different tasks depending on the problem they are designed to solve. Four common NLP tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is an NLP task that assigns a predefined category to an entire text document based on its content.

**Example:** An email filtering system determines whether an incoming email is spam or not spam.

- **Input:** An email or text document.
- **Output:** A category such as Spam or Not Spam.

Other applications include sentiment analysis, news categorization, and topic classification.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which each token in a sequence receives a corresponding label. Unlike text classification, which assigns a category to an entire document, sequence labeling identifies the role or category of individual words.

**Example:** Named Entity Recognition (NER) identifies people's names, organizations, and geographical locations in a sentence.

Input:

"Abdirahman lives in Somalia."

Possible output:

| Token      | Label    |
| ---------- | -------- |
| Abdirahman | PERSON   |
| lives      | O        |
| in         | O        |
| Somalia    | LOCATION |

The label O means that the token does not belong to one of the named entity categories being identified.

- **Input:** A sentence or sequence of tokens.
- **Output:** A sequence of labels corresponding to the input tokens.

Other applications include Part-of-Speech tagging and identifying important information in documents.

### 3.3 Search and Similarity

Search and similarity are NLP-related tasks that help systems retrieve relevant documents or identify texts with similar meanings.

Traditional search systems often use matching words, while semantic search systems can also use contextual representations to identify related meanings.

**Example:** A user searches for "how to learn programming," and a search engine retrieves documents about software development, coding tutorials, and programming courses.

- **Input:** A search query and a collection of documents.
- **Output:** A ranked list of relevant documents or similarity scores.

Applications include search engines, document retrieval, recommendation systems, and question-answering systems.

### 3.4 Text Generation

Text generation is an NLP task in which a computer produces new human-readable text based on an input, instruction, or previously processed information.

**Example:** A chatbot receives the question:

"What is Artificial Intelligence?"

It generates a response explaining the concept.

- **Input:** A prompt, instruction, or existing text.
- **Output:** Newly generated text, such as an answer, summary, translation, or paragraph.

Modern conversational AI systems use advanced language models to perform text generation.

### 3.5 Comparison of NLP Tasks

| NLP Task              | Input                         | Output                                |
| --------------------- | ----------------------------- | ------------------------------------- |
| Text Classification   | A text document               | One or more predefined categories     |
| Sequence Labeling     | A sequence of tokens          | A label for each token                |
| Search and Similarity | Query and document collection | Ranked documents or similarity scores |
| Text Generation       | Prompt or input text          | Newly generated text                  |

Although these tasks have different objectives, they all involve processing human language to solve practical problems.

---

## 4. Supervised and Unsupervised Learning in NLP

Machine learning approaches can be divided into different learning types. Two important approaches are supervised learning and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using labeled data. Each training example contains an input and its corresponding correct output.

The model learns the relationship between the input features and their target labels. After training, it can use the learned patterns to predict labels for new, unseen examples.

**Example: Sentiment Classification**

Consider the following dataset:

| Customer Review               | Label    |
| ----------------------------- | -------- |
| I love this application       | Positive |
| This application is terrible  | Negative |
| The service is excellent      | Positive |
| I hate using this application | Negative |

The model uses the labeled examples to learn patterns associated with positive and negative opinions.

When it receives a new review, it predicts the appropriate sentiment category.

**Why are labels necessary?**

Labels are necessary because they provide the expected answers during training. The model uses these answers to measure prediction errors and adjust its parameters to improve its performance.

### 4.2 Unsupervised Learning

Unsupervised learning is a machine learning approach in which a model analyzes data without predefined target labels.

Instead of learning from known answers, the model attempts to discover patterns, structures, or relationships within the data.

**Example: Customer Review Clustering**

Imagine having 10,000 customer reviews without sentiment labels.

An unsupervised learning algorithm, such as K-Means clustering, can group similar reviews based on their numerical representations.

Possible groups might include:

- Reviews discussing application performance.
- Reviews discussing customer support.
- Reviews discussing payment problems.

These groups are discovered from patterns in the data rather than predefined labels.

**Why are labels not required?**

Labels are not required because the purpose is to discover hidden structures in the dataset rather than predict a known target category.

### 4.3 Comparison

| Aspect            | Supervised Learning              | Unsupervised Learning            |
| ----------------- | -------------------------------- | -------------------------------- |
| Training data     | Labeled data                     | Unlabeled data                   |
| Target labels     | Required                         | Not required                     |
| Main objective    | Learn to predict known outputs   | Discover hidden patterns         |
| Example           | Sentiment classification         | Customer review clustering       |
| Common algorithms | Naive Bayes, Logistic Regression | K-Means, Hierarchical Clustering |

In conclusion, supervised learning is appropriate when examples have known target answers, while unsupervised learning is useful when the objective is to explore and discover patterns in data without predefined answers.

---

## 5. Numerical Representations of Text

### 5.1 Why Must Text Be Converted into Numbers?

Most traditional machine learning algorithms operate on numerical data rather than raw human-readable text.

Words and sentences are symbolic representations of language. A computer can store them as characters, but machine learning algorithms generally need numerical features to perform mathematical operations such as calculating distances, probabilities, and model parameters.

Therefore, text must be transformed into numerical representations before it can be processed by most machine learning models.

This transformation is commonly called text vectorization or text representation.

### 5.2 Bag of Words (BoW)

Bag of Words is a simple and widely known method of representing text numerically.

It creates a vocabulary from a collection of documents and represents each document using the frequency of vocabulary words.

The method focuses on which words appear and how often they occur, rather than their grammatical order.

**Example:**

Consider two sentences:

- Document 1: "I love learning NLP."
- Document 2: "I love learning Python."

First, create a vocabulary:

`["I", "love", "learning", "NLP", "Python"]`

Then represent each document as a numerical vector.

| Document   |   I | love | learning | NLP | Python |
| ---------- | --: | ---: | -------: | --: | -----: |
| Document 1 |   1 |    1 |        1 |   1 |      0 |
| Document 2 |   1 |    1 |        1 |   0 |      1 |

The numerical vectors allow a machine learning algorithm to process the documents mathematically.

### 5.3 Main Limitations of Bag of Words

Despite its simplicity, Bag of Words has several limitations:

1. **Loss of word order:** It does not preserve the order in which words appear.
2. **Limited understanding of meaning:** It cannot independently understand the context or semantic meaning of words.
3. **Large vocabulary size:** A large corpus may produce thousands or millions of features, resulting in high-dimensional vectors.
4. **Sparse representations:** Most documents contain only a small portion of the entire vocabulary, so many vector values are zero.

For example:

"The dog bites the man."

"The man bites the dog."

A basic Bag of Words representation gives both sentences the same word-frequency vector, even though their meanings are different.

More advanced approaches, including TF-IDF, Word2Vec, and contextual embeddings, address some of these limitations.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

### 6.1 Introduction

Low-resource languages are languages that have limited digital linguistic resources available for developing and evaluating language technologies.

These resources may include large text collections, annotated datasets, dictionaries, speech recordings, translation datasets, and pretrained language models.

Somali is an important example of a language that faces several challenges in computational language processing.

Although Somali is widely spoken in the Horn of Africa, the availability of high-quality digital resources for NLP has historically been limited compared with languages such as English.

The following are some major challenges.

### 6.2 Limited Datasets and Annotated Resources

One major challenge is the shortage of large, high-quality, and properly annotated Somali datasets.

Modern machine learning models often require substantial amounts of training data to learn language patterns effectively.

For example, a sentiment analysis system requires Somali sentences labeled as positive, negative, or neutral.

Without enough correctly labeled examples, the model may fail to recognize different expressions, opinions, and contextual meanings.

This can lead to:

- Lower prediction accuracy.
- Poor understanding of less common words.
- Weak generalization to new text.
- Difficulty evaluating model performance reliably.

Research by Badel et al. (2023) highlights the shortage of dedicated Somali information retrieval resources and introduces a corpus designed to address this problem.

### 6.3 Spelling Variation and Text Normalization

Another challenge is variation in how people write words.

In informal digital communication, users may write the same word differently because of typing habits, regional usage, abbreviations, or inconsistent spelling.

For example, a dataset may contain different written forms of the same intended word.

When these forms are not normalized, an NLP system may treat them as separate vocabulary items.

This creates several problems:

- The vocabulary becomes unnecessarily large.
- Word-frequency statistics become less reliable.
- Similar expressions may not be recognized as related.
- Search systems may fail to retrieve relevant documents.

Text normalization and language-specific preprocessing can help reduce this problem.

### 6.4 Morphological Complexity and Word Forms

Somali has agglutinative and derivational morphological characteristics. This means that words can have different forms through the addition of grammatical or derivational elements.

Consequently, a single root may be related to multiple surface forms.

For example, different forms of a word may express grammatical information or related meanings.

A basic NLP system that treats every surface form as an unrelated word may struggle to identify these relationships.

This can affect:

- Search and information retrieval.
- Vocabulary management.
- Text classification.
- Machine translation.
- Language modeling.

Lemmatization, stemming, and morphological analysis can help systems recognize relationships between different word forms.

### 6.5 Code-Switching and Mixed-Language Text

Code-switching occurs when speakers use two or more languages within the same conversation or sentence.

For example:

"Waxaan rabaa inaan learn programming."

This sentence combines Somali and English.

A language-processing system designed only for standard Somali may struggle to identify the correct language of each word and understand the sentence accurately.

This can affect tokenization, language identification, sentiment analysis, and machine translation.

Developing multilingual and code-switching-aware systems can help address this challenge.

### 6.6 Summary of Challenges

| Challenge                | Effect on NLP systems                                | Possible solution                               |
| ------------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| Limited datasets         | Weak model training and evaluation                   | Build high-quality annotated corpora            |
| Spelling variation       | Duplicate vocabulary entries and poor matching       | Text normalization                              |
| Morphological complexity | Difficulty connecting related word forms             | Lemmatization and morphological analysis        |
| Code-switching           | Incorrect language identification and interpretation | Multilingual models and language identification |

### 6.7 Conclusion

Developing reliable NLP systems for Somali requires more than applying algorithms originally designed for English.

Researchers need to develop language-specific datasets, preprocessing tools, tokenizers, morphological resources, and evaluation benchmarks.

Community participation is also important because native speakers can contribute linguistic knowledge and help verify the quality of collected data.

Improving these resources will make it easier to develop useful Somali applications, including search engines, translation systems, chatbots, and educational technologies.

---

## 7. Case Study: Somali Information Retrieval Corpus

### 7.1 Research Paper

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).

**DOI:** https://doi.org/10.18653/v1/2023.emnlp-main.462

### 7.2 Research Problem and Objective

The research addresses the limited availability of dedicated information retrieval resources for the Somali language.

Information retrieval systems help users find relevant documents based on their search queries.

Before this research, Somali information retrieval research often relied on translating Somali queries into other languages because dedicated Somali datasets were limited.

The researchers aimed to create an annotated Somali information retrieval corpus and investigate methods for retrieving relevant Somali documents.

### 7.3 Data Used

The researchers developed a corpus containing 2,335 Somali documents.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.
- Other online textual resources.

The research involved collaboration between language experts and NLP researchers to construct a dedicated evaluation resource for Somali information retrieval.

### 7.4 Method Used

The researchers developed an information retrieval system and investigated a technique called Pseudo-Relevance Feedback (PRF).

Pseudo-Relevance Feedback is a query expansion technique that uses initially retrieved documents to identify additional useful terms for improving a search query.

The research also compared the PRF approach with a traditional TF-IDF ranking method.

The purpose was to determine whether expanding search queries using relevant terms could improve document retrieval performance.

### 7.5 Research Findings

The researchers reported that the PRF approach performed better than the traditional TF-IDF method on their evaluation corpus.

One important result was recall at 20 documents (R@20):

| Method                          | Recall at 20 (R@20) |
| ------------------------------- | ------------------: |
| TF-IDF                          |                 69% |
| Pseudo-Relevance Feedback (PRF) |                 92% |

This indicates that PRF retrieved a larger proportion of relevant documents within the first 20 results in their evaluation.

The research also introduced a publicly available Somali information retrieval evaluation corpus, providing an important resource for future Somali NLP research.

These results should be understood within the scope of the study's dataset and evaluation setup rather than as a guarantee that PRF will outperform TF-IDF in every search application.

### 7.6 Which NLP Task Does This Research Cover?

This research primarily covers **Search and Similarity**, specifically Information Retrieval.

The system receives a search query and a collection of documents as input. It then ranks documents according to their relevance to the query.

The output is an ordered list of documents that are likely to contain useful information for the user.

### 7.7 Importance of the Research

This study is important because it demonstrates that practical NLP systems can be developed for Somali when researchers create dedicated language resources.

It also provides a foundation for future applications, such as:

- Somali search engines.
- Somali question-answering systems.
- Educational information retrieval.
- Digital libraries.
- Somali document discovery systems.

The study demonstrates the importance of building language-specific datasets rather than depending entirely on translation-based approaches.

---

## 8. Overall Conclusion

Natural Language Processing is an important field of Artificial Intelligence that enables computers to process, analyze, and generate human language.

This assignment has explored the foundational concepts of NLP, including the relationship between AI, Machine Learning, and NLP, fundamental text terminology, common NLP tasks, learning approaches, and numerical text representations.

It has also examined the challenges of developing NLP technologies for low-resource languages such as Somali.

The Somali Information Retrieval Corpus case study demonstrates how dedicated datasets and appropriate techniques can improve language-specific applications.

Overall, successful NLP development requires a combination of linguistic understanding, high-quality data, appropriate computational methods, and reliable evaluation.

For Somali, continued research, dataset development, and collaboration with native speakers are essential for building accurate and useful language technologies.

---

## 9. References

1. Bird, S., Klein, E., & Loper, E. (2009). _Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit_. O'Reilly Media.  
   https://www.nltk.org/book/

2. Jurafsky, D., & Martin, J. H. (2025). _Speech and Language Processing_ (3rd edition draft). Stanford University.  
   https://web.stanford.edu/~jurafsky/slp3/

3. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, 7463–7469.  
   https://doi.org/10.18653/v1/2023.emnlp-main.462

4. Hussen, K. Y., Sewunetie, W. T., Ayele, A. A., Imam, S. H., Muhammad, S. H., & Yimam, S. M. (2025). The State of Large Language Models for African Languages: Progress and Challenges.  
   https://arxiv.org/abs/2506.02280

5. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-Based Word Lemmatization Approach for Somali Language. _International Conference on Learning Representations Workshop (ICLRW 2023)._  
   https://mlanthology.org/iclrw/2023/mohamed2023iclrw-lexicon/

6. Bird, S., Klein, E., & Loper, E. (2009). Learning to Classify Text. In _Natural Language Processing with Python_.  
   https://www.nltk.org/book/ch06.html

7. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. _Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)._  
   https://aclanthology.org/2026.africanlp-main.17/
