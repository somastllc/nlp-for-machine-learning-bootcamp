# Assignment 1 – NLP Foundations
**Student:** Hibo Mohamed  
**GitHub Username:** Hibo-moha


## Introduction

Natural Language Processing (NLP) is an important area of computer science that enables computers to process, analyze, and generate human language. It combines ideas from artificial intelligence, machine learning, and linguistics to help computers work with written and spoken communication.

NLP is used in many everyday applications, including search engines, chatbots, translation systems, email filtering, and voice assistants. This assignment explores the fundamental concepts of NLP, its common tasks, learning approaches, text representation, and the challenges of developing NLP systems for low-resource languages such as Somali.

---

## 1. Definition of Natural Language Processing (NLP) and Its Relationship to AI and ML

Natural Language Processing (NLP) is a branch of computer science and artificial intelligence that focuses on enabling computers to understand, interpret, analyze, and generate human language.

NLP helps computers work with natural languages such as English, Somali, and Arabic rather than only programming languages.

### Relationship between NLP, Artificial Intelligence, and Machine Learning

These three fields are closely connected:

- **Artificial Intelligence (AI):** A broad field concerned with developing computer systems that perform tasks requiring human-like intelligence, such as reasoning, problem-solving, and understanding language.
- **Machine Learning (ML):** A subfield of AI that enables computers to learn patterns from data and improve their performance without requiring every instruction to be manually programmed.
- **Natural Language Processing (NLP):** An area of AI that focuses specifically on processing human language. Machine learning is commonly used to build NLP systems.

In simple terms, AI is the broader field, machine learning provides techniques for learning from data, and NLP applies these techniques to human language.

### Real-life example: A customer service chatbot

Consider a chatbot used by an online shopping company.

1. A customer writes: "Where is my order?"
2. The NLP component processes the sentence and identifies that the customer is asking about delivery.
3. The machine learning model uses patterns learned from previous customer messages to recognize the request.
4. The AI system connects the request to the appropriate service or database.
5. The chatbot generates a response such as: "Your order is currently being delivered."

This example demonstrates how AI, ML, and NLP work together to provide an automated customer service experience.

---

## 2. Explanation of Corpus, Document, Token, Vocabulary, Feature, and Label

These terms are important in NLP because they describe how text data is collected, organized, and prepared for machine learning.

For this explanation, consider a small dataset containing customer reviews about an online shopping service.

| Term | Definition | Example |
|---|---|---|
| **Corpus** | A collection of text documents gathered for language analysis or NLP research. | All customer reviews collected by the shopping company. |
| **Document** | One individual unit of text within a corpus. | "The delivery was very fast." |
| **Token** | A unit of text produced by dividing text into smaller pieces, usually words or subwords. | The tokens are: The, delivery, was, very, fast. |
| **Vocabulary** | The collection of unique tokens found in a corpus after tokenization. | delivery, fast, slow, product, excellent. |
| **Feature** | A measurable property or numerical representation used by a machine learning model. | The frequency of the word "fast" in a review. |
| **Label** | The target category or answer assigned to a document in supervised learning. | Positive, negative, or neutral. |

### Example dataset

Suppose we have three customer reviews:

| Document ID | Customer Review | Label |
|---|---|---|
| D1 | The delivery was fast | Positive |
| D2 | The product was damaged | Negative |
| D3 | The service was excellent | Positive |

In this example:

- The three reviews together form a corpus.
- Each individual review is a document.
- Words such as "delivery," "fast," and "product" are tokens.
- All unique words form the vocabulary.
- Word frequencies can be used as numerical features.
- Positive and negative are the labels.

This dataset could be used to train a sentiment classification model that learns to identify whether a customer review expresses a positive or negative opinion.

---

## 3. Four Common NLP Tasks

NLP includes several tasks that allow computers to work with human language. Four important tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category to a complete text document based on its content.

**Input:** A sentence, message, email, or document.

**Output:** A predicted category or class.

**Example:** An email filtering system receives an email and classifies it as either spam or not spam.

Other applications include sentiment analysis, news categorization, and identifying customer complaints.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which a label is assigned to each individual token in a sequence. The system examines the words and their surrounding context to determine the correct label.

**Input:** A sentence containing several words.

**Output:** A sequence of labels corresponding to the individual words.

**Example:**

Sentence: "Hodan studies at Hargeisa University."

| Word | Label |
|---|---|
| Hodan | PERSON |
| studies | VERB |
| at | PREPOSITION |
| Hargeisa | LOCATION |
| University | ORGANIZATION |

This task is useful for Part-of-Speech tagging and Named Entity Recognition (NER).

### 3.3 Search and Similarity

Search and similarity involve finding documents or sentences that are relevant to a user's query or similar to a given text.

**Input:** A search query or a text document together with a collection of documents.

**Output:** A ranked list of relevant or similar documents.

**Example:** A student searches for "Introduction to Data Analysis" on a search engine. The system retrieves documents containing relevant information, even when their wording is not exactly the same.

This task is commonly used in search engines, recommendation systems, and information retrieval applications.

### 3.4 Text Generation

Text generation is the process of producing new human-readable text based on an input, instruction, or previous conversation.

**Input:** A question, instruction, prompt, or previous text.

**Output:** A newly generated sentence, paragraph, or document.

**Example:** A user asks a chatbot, "Explain what machine learning means." The system generates a meaningful explanation.

Text generation is used in chatbots, automatic summarization, translation, and writing assistants.

### Summary of the Four Tasks

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | A text document | A category or label |
| Sequence Labeling | A sequence of tokens | A label for each token |
| Search and Similarity | A query or document | Ranked relevant documents |
| Text Generation | A prompt or previous text | Newly generated text |

---

## 4. Comparison Between Supervised and Unsupervised Learning in NLP

Machine learning approaches in NLP can be divided into different categories. Two important approaches are supervised learning and unsupervised learning.

### Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using labeled data. Each training example contains an input and its correct expected output.

The model learns the relationship between the input text and its corresponding label. After training, it can predict labels for new, unseen text.

**Example: Email spam detection**

A dataset contains emails labeled as either "Spam" or "Not Spam."

The model studies these examples and learns patterns associated with spam messages. When a new email arrives, it predicts the appropriate category.

### Unsupervised Learning

Unsupervised learning is a machine learning approach in which a model works with data that does not contain predefined target labels.

Instead of learning from known answers, the model identifies hidden patterns, structures, or similarities within the data.

**Example: Customer review grouping**

A company collects thousands of customer reviews without assigning categories to them.

An unsupervised learning algorithm can group similar reviews together. For example, reviews about delivery problems may form one group, while reviews about product quality may form another.

### Comparison Table

| Feature | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Target labels | Required | Not required |
| Main purpose | Predict known categories or values | Discover hidden patterns |
| Example | Spam detection | Customer review clustering |
| Learning process | Learns from correct examples | Finds structures and similarities |

### Why Does One Require Labels and the Other Does Not?

Supervised learning requires target labels because the model needs correct answers during training to understand what it should predict.

Unsupervised learning does not require target labels because its purpose is to explore the data and discover patterns independently.

Therefore, the main difference is that supervised learning learns from examples with known answers, while unsupervised learning discovers relationships in data without predefined answers.

---

## 5. Why Text Must Be Converted into Numerical Representations

Most machine learning algorithms cannot directly process raw human language in the same way that people read and understand sentences.

Computers and machine learning models generally require numerical data arranged in a structured format. Therefore, text must first be converted into numbers before it can be used for mathematical operations and model training.

This conversion process is called **text vectorization**.

### One Simple Representation Method: Bag of Words (BoW)

Bag of Words is a basic text representation technique that converts documents into numerical vectors by counting how frequently each word appears.

It creates a vocabulary from the available documents and represents each document using word counts.

### Example

Consider the following two sentences:

- Document 1: "I like machine learning."
- Document 2: "I like data analysis."

First, create a vocabulary:

[I, like, machine, learning, data, analysis]

Then count the occurrences of each word.

| Document | I | like | machine | learning | data | analysis |
|---|---:|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| Document 2 | 1 | 1 | 0 | 0 | 1 | 1 |

The sentences have now been transformed into numerical vectors that machine learning algorithms can process.

### Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and the relationship between words.

For example:

- "The student teaches the teacher."
- "The teacher teaches the student."

These sentences contain almost the same words, but their meanings are different. Bag of Words may represent them with identical word counts.

It also struggles to represent context and can produce large, sparse numerical matrices when the vocabulary becomes very large.

Despite these limitations, Bag of Words is useful for simple text classification and basic NLP applications.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited digital text collections, annotated datasets, language-processing tools, and other resources required to develop effective NLP systems.

Somali is an important example because it has a large speaking community, but its available language technology resources are still developing compared with widely supported languages such as English.

Research on African NLP highlights the unequal representation of African languages in modern language technologies (Alabi et al., 2025).

Several challenges affect the development of NLP systems for Somali.

### 6.1 Limited Datasets and Language Resources

One major challenge is the shortage of large, high-quality, human-annotated Somali datasets.

Machine learning models need sufficient training examples to learn language patterns. For example, developing a Somali sentiment analysis system requires many Somali sentences labeled as positive, negative, or neutral.

When only a small amount of labeled data is available:

- The model may fail to recognize different expressions.
- Its predictions may be less accurate.
- It may perform poorly on text from different sources.
- It becomes difficult to evaluate whether the model works reliably.

Gedi et al. (2026) identify the lack of human-annotated lemmatization datasets as an important problem for Somali NLP research.

### 6.2 Spelling Variation and Non-Standard Writing

Another challenge is inconsistent spelling and writing styles.

People may write Somali words differently in social media messages, informal conversations, and other digital platforms.

For example, a person may write the same expression using different spellings, shortened forms, or repeated letters to express emotion.

This creates difficulties because an NLP model may treat two forms of the same word as different tokens.

As a result, the system may fail to recognize related expressions, reducing its ability to classify text, retrieve information, or understand messages accurately.

### 6.3 Morphological Complexity and Word Forms

Somali has morphological structures in which prefixes, suffixes, and other grammatical elements can affect word forms.

A single root can produce multiple related forms. This can increase the size of the vocabulary and make it difficult for NLP systems to recognize that different forms are related.

For example, a model that encounters an unfamiliar word form may fail to connect it with a related word it has already learned.

This problem affects tokenization, stemming, lemmatization, search, and machine learning performance.

The 2026 Somali lemmatization study by Gedi and colleagues demonstrates why language-specific lexical resources and morphological analysis are important for Somali NLP.

### 6.4 Code-Switching and Mixed-Language Text

Code-switching occurs when speakers combine two or more languages within the same conversation or sentence.

For example:

"Maanta I have an assignment to finish."

This sentence combines Somali and English.

An NLP system trained mainly on standard Somali may struggle to process such sentences because it encounters words and grammatical structures from different languages.

Consequently, it may incorrectly identify tokens, misunderstand the meaning, or produce inaccurate predictions.

### Conclusion of Question 6

These challenges show that developing effective Somali NLP systems requires more than simply applying machine learning algorithms. Researchers also need reliable datasets, language-specific tools, standardized text processing, and collaboration with Somali language experts.

---

## 7. Case Study: Developing a Somali Lemmatization Corpus

### Research Paper

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., and Assowe, H. A. (2026).

**Title:** 
*Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform.*

Published in the Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026), pages 179–189.

### A. Research Problem and Task

The research addresses the shortage of reliable Somali language resources for NLP applications.

In particular, the researchers focus on lemmatization, which is the process of reducing different forms of a word to a common dictionary or base form.

Lemmatization is important because it helps NLP systems recognize related word forms and improves applications such as information retrieval, text indexing, and language processing.

### B. Data Used

The researchers developed a Somali lemmatization lexicon containing:

- 5,584 linguistically verified root words.
- 78,629 derived word forms.
- Part-of-Speech (POS) annotations for the derivative forms.

They also evaluated their system using a collection of 294 documents covering news articles, social media posts, and short messages.

The research focused on the MAXAA TIRI written dialect of Somali.

### C. Method Used

The researchers developed a web-based crowdsourcing platform to support the collection, annotation, and expansion of Somali language resources.

They combined three main techniques:

1. **Lexicon-based approach:** A dictionary containing Somali root words and their related forms was used to identify word relationships.
2. **POS information:** Part-of-Speech annotations helped distinguish grammatical categories and improve word analysis.
3. **Rule-based processing:** Linguistic rules were used to handle some words that were not found in the dictionary.

The researchers then evaluated the system on different types of Somali text.

### D. Research Findings

The evaluation produced the following lemmatization accuracy results:

| Type of Text | Accuracy |
|---|---:|
| Full articles | 51.27% |
| Excerpts | 44.14% |
| Short texts such as tweets | 59.51% |

The results showed that the system performed better on shorter texts than on excerpts and full articles.

The study demonstrates that combining linguistic knowledge, lexical resources, POS information, and rule-based techniques can help address morphological challenges in Somali.

However, the accuracy results also show that further improvements are needed before the system can reliably process all Somali text.

### E. Which NLP Task Does This Case Study Cover?

The main task is **lemmatization**, which is a morphological text-processing task.

Among the four tasks discussed in Question 3, its closest relationship is with **sequence labeling**, because the system analyzes individual words and identifies their corresponding normalized forms.

However, lemmatization is technically a separate NLP task rather than exactly the same as sequence labeling.

It can support other NLP applications, including text classification, information retrieval, and search and similarity.

### F. Importance of the Research

This research is important because it contributes a reusable Somali language resource that can support future NLP development.

It also demonstrates how researchers can combine linguistic knowledge and technology to develop language tools for communities whose languages receive less attention in global AI development.

The study is particularly relevant to Somali speakers because better language resources can support future applications such as Somali search engines, text analysis systems, educational technologies, and language models.

---

## Overall Conclusion

Natural Language Processing plays an important role in connecting human communication with computer technology. Through NLP, computers can classify text, identify important information, retrieve relevant documents, and generate meaningful responses.

This assignment has explored the relationship between NLP, AI, and machine learning, together with important text-processing concepts and common NLP tasks.

It has also explained how text is converted into numerical representations and compared supervised and unsupervised learning approaches.

Finally, the discussion of Somali NLP challenges and the Somali lemmatization case study demonstrates why developing language-specific datasets and tools is important.

In conclusion, improving NLP for low-resource languages such as Somali requires continuous research, high-quality language data, appropriate computational methods, and collaboration between researchers and native speakers.

---

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., online manuscript). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

2. Scikit-learn Developers. (n.d.). *Feature extraction: Text feature extraction and Bag of Words representation*. Scikit-learn Documentation. https://scikit-learn.org/stable/modules/feature_extraction.html

3. Alabi, J. O., Hedderich, M. A., Adelani, D. I., & Klakow, D. (2025). *Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead*. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 27807–27841. https://doi.org/10.18653/v1/2025.emnlp-main.1414

4. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). *Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform*. Proceedings of the 7th Workshop on African Natural Language Processing, 179–189. https://doi.org/10.18653/v1/2026.africanlp-main.17

5. Course Material. (n.d.). *Chapter 6: Natural Language Processing*. Course lecture PDF.
