# Assignment 1 – NLP Foundations

## Introduction

Natural Language Processing (NLP) is an important area of computing that enables machines to process and work with human language. It supports applications such as machine translation, sentiment analysis, information retrieval, chatbots, and automatic text classification.

NLP combines concepts from computer science, Artificial Intelligence (AI), Machine Learning (ML), and linguistics to develop systems that can analyze and understand human communication.

This assignment examines the fundamental concepts behind NLP, explains how machine learning is applied to language data, and discusses the challenges of developing NLP systems for low-resource languages, particularly Somali. It also presents a research case study demonstrating how NLP can address a real-world problem.

---

## 1. Definition of NLP and Its Relationship to AI and ML

### Definition of Natural Language Processing

**Natural Language Processing (NLP)** is a field of computer science and Artificial Intelligence concerned with developing computer systems that can process, analyze, interpret, and generate human language.

Human language includes written text and spoken communication. Unlike programming languages, natural languages contain complex grammatical structures, different meanings, context, and cultural expressions.

NLP helps computers perform language-related activities such as translating sentences, identifying opinions, understanding questions, and generating responses.

### Relationship between AI, ML, and NLP

Artificial Intelligence, Machine Learning, and Natural Language Processing are closely related, but they have different purposes.

- **Artificial Intelligence (AI):** A broad field concerned with developing computer systems capable of performing tasks associated with human intelligence, such as reasoning, decision-making, and problem-solving.
- **Machine Learning (ML):** A branch of AI that enables computer systems to learn patterns from data and use those patterns to make predictions or decisions.
- **Natural Language Processing (NLP):** A field focused on enabling computers to process and work with human language. NLP can use machine learning, deep learning, linguistic rules, or combinations of these approaches.

Machine learning is one of the approaches used to solve NLP problems. However, not every NLP system must use machine learning because some systems rely on manually designed linguistic rules.

### Real-life example: Email spam detection

Consider an email service that automatically identifies unwanted messages.

1. **AI:** The overall intelligent system is designed to identify and filter unwanted emails.
2. **NLP:** The system processes the language in the email by examining words, phrases, and textual patterns.
3. **ML:** A machine-learning model learns from previously labelled examples of spam and legitimate emails.
4. **Prediction:** The trained model analyzes new emails and predicts whether they are spam or legitimate messages.

For example, an email containing suspicious promotional language such as "claim your free prize" may be classified as spam based on patterns learned during training.

Therefore, AI provides the broader goal, NLP provides language-processing techniques, and ML provides a way for the system to learn from examples and make predictions.

---

## 2. Core NLP Terms

To understand how NLP systems work, it is important to understand the basic terminology used when preparing and analyzing text data.

For this explanation, consider a small fictional dataset containing customer reviews of an online shop.

| Review | Label |
|---|---|
| "The delivery was fast and excellent." | Positive |
| "The product was damaged." | Negative |
| "The service was helpful." | Positive |
| "My order arrived late." | Negative |

The following concepts apply to this dataset.

### 2.1 Corpus

A **corpus** is an organized collection of text documents used for linguistic analysis or for developing and evaluating NLP systems.

In this example, the complete collection of four customer reviews forms the corpus.

A corpus can contain thousands or millions of documents, depending on the research problem.

### 2.2 Document

A **document** is one individual text item within a corpus.

For example:

"The product was damaged."

This single customer review is one document. Therefore, the example corpus contains four documents.

A document does not have to be a complete book or formal report. It can also be a sentence, message, social media post, or customer review.

### 2.3 Token

A **token** is an individual unit produced when text is divided through a process called tokenization.

Depending on the tokenizer, tokens may represent complete words, parts of words, characters, or punctuation marks.

For example, the sentence:

"The delivery was fast."

Can be divided into these word tokens:

`["The", "delivery", "was", "fast", "."]`

The punctuation mark is also a token in this example.

### 2.4 Vocabulary

A **vocabulary** is the collection of distinct tokens identified in a particular text corpus after applying a chosen tokenization and normalization method.

For example, suppose the processed reviews contain these words:

`delivery, fast, excellent, product, damaged, service, helpful, order, arrived, late`

These unique terms form the vocabulary in this simplified example.

If a word appears several times, it is still counted only once in the vocabulary.

### 2.5 Feature

A **feature** is a measurable property or numerical representation that a machine-learning model uses to identify patterns and make predictions.

For example, a feature could indicate whether the word "excellent" appears in a review.

A simplified feature table might look like this:

| Review | Contains "excellent" | Contains "damaged" |
|---|---:|---:|
| Delivery was fast and excellent | 1 | 0 |
| Product was damaged | 0 | 1 |
| Service was helpful | 0 | 0 |
| Order arrived late | 0 | 0 |

Here, 1 means that the word is present, while 0 means it is absent.

### 2.6 Label

A **label** is the expected category or answer associated with a training example in a supervised machine-learning task.

In the customer-review dataset, the labels are Positive and Negative.

For example:

- "The delivery was fast and excellent." → Positive
- "The product was damaged." → Negative

The model learns from these labelled examples and attempts to predict the correct sentiment of new reviews.

### Summary of the terms

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of text documents | All four customer reviews |
| Document | One text item | One customer review |
| Token | A unit of divided text | "delivery" |
| Vocabulary | Unique tokens in a corpus | delivery, fast, product |
| Feature | Numerical information used by a model | Presence of "excellent" |
| Label | Target category | Positive or Negative |

---

## 3. Four Common NLP Tasks

NLP supports many applications. Four important tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category to a complete text document.

**Input:** A text document, such as an email, review, or news article.

**Output:** A predicted category or label.

Example:

- Input: "I am very satisfied with this service."
- Output: Positive sentiment.

Other applications include spam detection, topic classification, and identifying harmful content.

### 3.2 Sequence Labeling

Sequence labeling assigns a label to each token or relevant unit in a sequence of text.

Unlike ordinary text classification, which assigns one category to an entire document, sequence labeling identifies the role or category of individual words.

**Input:** A sentence divided into tokens.

**Output:** A label associated with each token.

Example:

Sentence: "Amina studies at Plasma University."

| Token | Label |
|---|---|
| Amina | PERSON |
| studies | O |
| at | O |
| Plasma | ORGANIZATION |
| University | ORGANIZATION |

This example illustrates Named Entity Recognition (NER), a common sequence-labeling task. The label O means that the token is not part of one of the entity categories being identified.

Sequence labeling is also used for part-of-speech tagging and identifying dates, locations, and organizations.

### 3.3 Search and Similarity

Search and similarity involve finding documents that are relevant or similar to a user's query or another document.

**Input:** A search query or reference document, together with a collection of documents.

**Output:** A ranked list of relevant or similar documents.

Example:

A student searches for:

"Introduction to machine learning."

A search system examines available educational documents and returns resources related to machine learning.

Similarity systems can also compare two sentences and estimate how closely their meanings are related.

Applications include search engines, document retrieval, recommendation systems, and duplicate-content detection.

### 3.4 Text Generation

Text generation is the process of producing new text based on learned language patterns and a given instruction or context.

**Input:** A prompt, instruction, or previous text.

**Output:** Newly generated text.

Example:

- Input: "Write a short explanation of artificial intelligence."
- Output: "Artificial intelligence is a field of computing that develops systems capable of performing tasks associated with human intelligence."

Applications include chatbots, writing assistants, automatic summarization, and machine translation.

### Comparison of the four tasks

| NLP Task | Input | Output |
|---|---|---|
| Text classification | Complete text | Category or label |
| Sequence labeling | Tokenized sentence | Label for each token |
| Search and similarity | Query or reference text and documents | Ranked relevant documents |
| Text generation | Prompt or context | New text |

---

## 4. Supervised and Unsupervised Learning in NLP

Machine learning methods used in NLP can be divided into different learning approaches. Two important approaches are supervised and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a machine-learning approach in which a model learns from examples that contain both input data and correct target labels.

The model compares its predictions with the expected answers during training and adjusts its parameters to reduce prediction errors.

**Example: Email spam classification**

A training dataset may contain:

| Email | Label |
|---|---|
| "Congratulations! Claim your prize." | Spam |
| "The meeting begins at 10 AM." | Not Spam |
| "You have won a free gift." | Spam |

The model learns relationships between the language patterns and their corresponding labels.

When it receives a new email, it predicts whether the message is spam or not spam.

**Why are target labels necessary?**

Labels provide the correct answers that guide the model during training. Without them, the model cannot directly learn the intended spam or non-spam categories through supervised learning.

### 4.2 Unsupervised Learning

Unsupervised learning is an approach in which a model works with data that does not have predefined target labels. It attempts to discover structures, similarities, or patterns within the data.

**Example: Customer review clustering**

Imagine collecting 5,000 customer reviews without assigning sentiment categories.

A clustering algorithm can examine similarities between reviews and group related documents together.

Possible groups might contain:

- Reviews discussing delivery.
- Reviews discussing product quality.
- Reviews discussing customer service.

The algorithm discovers groups based on patterns in the data. A human may examine the groups afterward and assign descriptive names.

**Why are target labels not required?**

The objective is to discover patterns or structures rather than predict a known answer supplied for each example. Therefore, the algorithm does not require predefined target labels.

### Comparison

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labelled data | Uses data without target labels |
| Learns from known answers | Discovers patterns in data |
| Common example: spam classification | Common example: document clustering |
| Predicts predefined targets | Identifies hidden structures |

The main difference is the availability of target labels and the objective of the learning process.

---

## 5. Why Text Must Be Converted into Numerical Representations

Most traditional machine-learning algorithms cannot directly process raw sentences as meaningful mathematical input. They operate on numerical values.

For this reason, text must first be transformed into numerical representations. This process allows an algorithm to examine patterns, calculate mathematical relationships, and learn from language data.

For example, a computer cannot directly calculate the distance between two ordinary sentences in the same way it calculates the distance between two numerical vectors.

A vectorizer converts text into a structured numerical representation that a machine-learning model can use.

### Bag-of-Words Representation

One simple text-representation technique is **Bag-of-Words (BoW)**.

Bag-of-Words represents a document according to the words it contains and how frequently those words appear. It creates a vocabulary and assigns each word a position in a numerical vector.

Consider two documents:

- Document 1: "AI improves education."
- Document 2: "AI supports education."

The vocabulary is:

`["AI", "improves", "education", "supports"]`

The documents can be represented as follows:

| Document | AI | improves | education | supports |
|---|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 0 |
| Document 2 | 1 | 0 | 1 | 1 |

The numbers indicate whether each vocabulary word appears in the document.

A machine-learning algorithm can now use these numerical vectors as input.

### Main limitation of Bag-of-Words

The main limitation is that Bag-of-Words does not adequately capture word order, grammar, or deeper meaning.

For example:

- "The teacher helps the student."
- "The student helps the teacher."

These sentences contain almost the same words, but their meanings are different. A basic Bag-of-Words representation produces identical vectors for both sentences.

Therefore, although Bag-of-Words is simple and useful for basic text-classification problems, it does not fully represent the meaning and context of natural language.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited computational resources available for developing and evaluating language technologies. These resources may include large text collections, annotated datasets, dictionaries, language-processing tools, and pretrained models.

Somali is an example of a language that faces several resource-related challenges in NLP research.

### 6.1 Limited Datasets and Linguistic Resources

One major challenge is the limited availability of large, high-quality Somali datasets that can be used to train and evaluate NLP models.

Many machine-learning systems require substantial amounts of representative data. For supervised learning, they also need correctly labelled examples.

When a Somali dataset is small or does not represent different topics, writing styles, and communities, a model may fail to learn patterns that apply to new texts.

For example, a Somali sentiment-analysis model trained on a small collection of product reviews may perform poorly when analyzing political discussions or informal social-media posts.

Limited linguistic resources also make it harder to build tools such as tokenizers, lemmatizers, and part-of-speech taggers.

**Effect on NLP systems:** Insufficient training data can reduce model accuracy and make it difficult for a system to generalize to unfamiliar language examples.

### 6.2 Morphological Complexity and Word Forms

Somali has rich word-formation and morphological patterns. A word can appear in different forms depending on grammatical information and its role in a sentence.

This creates difficulties for NLP systems because related word forms may be treated as separate tokens.

For example, a system may encounter different forms of a word that share the same basic meaning or root. If it cannot recognize their relationship, it may treat them as unrelated vocabulary items.

This can increase vocabulary size, reduce the number of examples available for each form, and make search or text classification less reliable.

A 2026 study by Gedi et al. demonstrates the importance of this issue. The researchers developed a Somali lemmatization lexicon containing more than 5,584 root words and 78,629 derivative forms. Each form was annotated with part-of-speech information.

**Effect on NLP systems:** Models may struggle to recognize related words, retrieve relevant documents, and understand grammatical relationships.

Lemmatization and stemming can help normalize related forms, but these processes require language-specific knowledge and carefully developed tools.

### 6.3 Spelling Variation and Non-Standard Writing

Somali text found online may contain inconsistent spelling, informal abbreviations, and variations in how words are written.

For example, users may write the same expression differently across social-media platforms or informal messages.

A computer system may interpret these variations as different words even when the writer intends the same meaning.

This can reduce the effectiveness of text matching, increase the number of unique tokens, and cause classification errors.

**Effect on NLP systems:** Search systems may fail to retrieve relevant information, while classification models may fail to recognize expressions they have previously encountered in another written form.

Text normalization can reduce some of these differences, but normalization must be designed carefully so that meaningful distinctions are not removed.

### 6.4 Mixed-Language Text and Code-Switching

Code-switching occurs when speakers combine two or more languages within the same sentence or conversation.

Somali speakers may combine Somali and English in online communication, education, and technology discussions.

For example:

"Waxaan maanta leeyahay online meeting."

This sentence combines Somali with English terminology.

A language-processing system designed only for standard Somali may not recognize all the English terms or understand how they are used within the Somali sentence.

**Effect on NLP systems:**

- Incorrect language identification.
- Tokenization difficulties.
- Reduced translation accuracy.
- Incorrect sentiment classification.
- Difficulty understanding mixed-language expressions.

Researchers can address this problem by developing multilingual datasets and training models with examples of real-world mixed-language communication.

### Overall impact

| Challenge | Main problem | Effect on NLP |
|---|---|---|
| Limited datasets | Insufficient training and annotated data | Poor accuracy and generalization |
| Morphological complexity | Multiple forms of related words | Difficulty recognizing word relationships |
| Spelling variation | Inconsistent written forms | Incorrect text matching and classification |
| Code-switching | Combining different languages | Incorrect language identification and interpretation |

These challenges can reduce the accuracy and reliability of NLP applications. Improving Somali NLP requires more high-quality datasets, language-specific processing tools, consistent annotation practices, and research that considers real-world language use.

---

## 7. Case Study: Somali Sentiment Analysis

### Research title

*Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification*

**Authors:** Kadar Bahar and Nehad T. A. Ramaha

**Publication year:** 2023

**Publication:** International Conference on Applied Engineering and Natural Sciences, Volume 1, Issue 1, pages 620–628.

**DOI:** https://doi.org/10.59287/icaens.1069

### 7.1 Research Problem

Sentiment analysis is an NLP application used to identify opinions and emotions expressed in written text.

It is widely researched for languages with large digital datasets. However, Somali has received less attention because of its limited linguistic and computational resources.

The researchers addressed this problem by investigating whether a resource-light approach could support sentiment analysis in Somali.

### 7.2 Task

The main task was **sentiment classification**.

The objective was to analyze Somali text and determine the sentiment expressed by the writer.

For example, a system could classify a customer review as positive or negative.

This case study therefore corresponds to text classification, one of the four NLP tasks discussed in Question 3.

### 7.3 Data

The researchers created a sentiment-annotated Somali dataset through crowdsourcing.

Crowdsourcing involves collecting contributions or annotations from multiple people. In this study, it was used to obtain text examples with sentiment labels.

The dataset was developed specifically to support the training and evaluation of a Somali sentiment-classification model.

The published abstract identifies the dataset as small-scale but does not provide its exact size or a detailed breakdown of the class distribution. Therefore, those figures should not be assumed.

### 7.4 Method

The researchers proposed a resource-light methodology that combined:

1. **Lexicon-based methods:** Using sentiment-related vocabulary to help identify positive or negative expressions.
2. **Feature engineering:** Extracting useful characteristics from text so that a classification model can process them.
3. **Text-processing techniques:** The study identifies tokenization, stemming, stopword removal, and negation handling among its relevant methodological concepts.
4. **Sentiment classification:** Training and evaluating a model using the annotated Somali dataset.

Negation handling is particularly important because a negative word can change the meaning of an otherwise positive expression.

The approach was designed to work with the limited linguistic resources available for Somali.

### 7.5 Findings

The researchers reported that their approach achieved competitive performance compared with existing techniques for under-resourced languages.

Their findings suggested that Somali sentiment analysis is possible even when only a small dataset and limited linguistic resources are available.

The study also demonstrated the importance of developing language-specific approaches rather than assuming that methods designed for high-resource languages will work equally well for Somali.

However, the published abstract does not provide exact numerical performance scores. Consequently, it is more accurate to report the authors' qualitative findings than to invent accuracy or F1-score values.

### 7.6 Importance of the Study

This research is valuable because it demonstrates that Somali can be used in practical NLP applications despite limited resources.

It also highlights the need to create annotated datasets and develop processing methods that account for the characteristics of the language.

Future research can build on this work by collecting larger datasets, improving annotation quality, and evaluating different machine-learning approaches.

### 7.7 Which NLP Task Does the Case Study Cover?

The case study primarily covers **Text Classification**, specifically sentiment analysis.

The input is a Somali sentence or text message.

The output is a predicted sentiment category.

For example:

| Input text | Expected classification |
|---|---|
| A positive customer review | Positive |
| A negative customer review | Negative |
| A neutral statement | Neutral |

The model analyzes the language and predicts the most appropriate sentiment category.

Therefore, the research is a practical example of applying text classification to a low-resource African language.

---

## Conclusion

Natural Language Processing enables computers to work with human language and supports applications such as sentiment analysis, text classification, information retrieval, and text generation.

Understanding concepts such as corpora, documents, tokens, vocabulary, features, and labels is essential when preparing text data for machine learning.

Numerical representation is also an important part of NLP because most machine-learning models require structured numerical input rather than raw text.

However, developing NLP systems for low-resource languages such as Somali presents challenges, including limited datasets, morphological complexity, spelling variation, and mixed-language communication.

The Somali sentiment-analysis case study demonstrates that research and practical NLP applications are possible even with limited resources. Continued development of Somali datasets and language-processing tools can help make language technologies more accessible to Somali-speaking communities.

---

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

2. Bahar, K., & Ramaha, N. T. A. (2023). Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification. *International Conference on Applied Engineering and Natural Sciences, 1*(1), 620–628. https://doi.org/10.59287/icaens.1069

3. Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and Rule-based Word Lemmatization Approach for the Somali Language*. arXiv:2308.01785. https://doi.org/10.48550/arXiv.2308.01785

4. Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 13968–13981. Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.862

5. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. *Proceedings of the 7th Workshop on African Natural Language Processing*, 179–189. https://doi.org/10.18653/v1/2026.africanlp-main.17

6. Abdirahman, A. A., Hashi, A. O., Dahir, U. M., Elmi, M. A., & Rodriguez, O. E. R. (2023). Enhancing Natural Language Processing in Somali Text Classification: A Comprehensive Framework for Stop Word Removal. *International Journal of Engineering Trends and Technology, 71*(12), 40–49. https://doi.org/10.14445/22315381/IJETT-V71I12P205