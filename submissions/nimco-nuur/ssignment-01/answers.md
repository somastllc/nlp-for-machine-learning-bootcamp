# Assignment 1 – NLP Foundations

**Student Name:** [Nimco Nuur Gesey]  
**Assignment:** 01 – NLP Foundations  
**Date:** 17 November 2026

---

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that focuses on developing computer systems capable of processing, analyzing, understanding, and generating human language. It combines ideas from computer science, linguistics, statistics, and machine learning to enable computers to work with languages such as English, Somali, and Arabic (Jurafsky & Martin, 2009).

NLP is used in many real-world applications, including machine translation, chatbots, sentiment analysis, speech recognition, information retrieval, and automatic text classification.

### Relationship Between AI, ML, and NLP

Artificial Intelligence, Machine Learning, and Natural Language Processing are closely related, but they have different roles.

- **Artificial Intelligence (AI):** A broad field concerned with developing computer systems that perform tasks associated with human intelligence, such as reasoning, problem-solving, and language processing.
- **Machine Learning (ML):** A branch of AI that enables computers to learn patterns from data and make predictions without requiring programmers to write every decision rule manually.
- **Natural Language Processing (NLP):** An area of AI that focuses specifically on processing and working with human language.

NLP can use Machine Learning to improve its ability to process language. However, not every NLP system requires Machine Learning. Some systems use predefined linguistic rules, while others learn patterns from examples.

### Real-Life Example: Email Spam Detection

Consider an email service that automatically separates unwanted messages from normal emails.

The system works through the following stages:

1. **AI:** Provides the overall intelligent system that identifies and filters unwanted emails.
2. **NLP:** Processes the language in each email, examining words, phrases, and other text patterns.
3. **Machine Learning:** Learns from previously labeled emails to identify patterns associated with spam and legitimate messages.
4. **Prediction:** When a new email arrives, the trained model predicts whether it belongs to the spam or legitimate category.

For example, an email containing suspicious advertising language may be classified as spam, while an ordinary personal email may be classified as legitimate.

This example demonstrates how NLP processes language, ML learns from examples, and AI brings these capabilities together in a practical application.

---

## 2. Explain the Core NLP Terms

To understand how NLP systems process text, it is important to understand the basic terms used to describe text data.

For this explanation, consider a small dataset containing customer reviews of a mobile application.

### Example Dataset

| Document ID | Text | Label |
|---|---|---|
| D1 | The application is easy to use | Positive |
| D2 | The application crashes frequently | Negative |
| D3 | I enjoy using this application | Positive |

### 2.1 Corpus

A corpus is an organized collection of text documents used for language analysis or computational research.

In this example, all three customer reviews together form one corpus.

Therefore, the corpus contains three documents.

### 2.2 Document

A document is an individual unit of text within a corpus. Its size depends on the task being performed.

A document could be a sentence, customer review, email, news article, or an entire book.

In the example dataset, D1, D2, and D3 are individual documents.

### 2.3 Token

A token is a unit of text produced when a text-processing system divides a document into smaller parts.

Tokens can represent complete words, subwords, characters, or punctuation marks, depending on the tokenizer being used.

For example, the sentence:

`The application is easy to use`

A simple word-level tokenizer produces:

`["The", "application", "is", "easy", "to", "use"]`

This sentence contains six token occurrences.

### 2.4 Vocabulary

Vocabulary refers to the collection of distinct tokens identified across a corpus.

For example, consider these two sentences:

- The application is useful.
- The application is simple.

Their combined vocabulary is:

`["The", "application", "is", "useful", "simple"]`

Repeated words are included only once in the vocabulary.

### 2.5 Feature

A feature is a measurable numerical property extracted from data and provided as input to a Machine Learning model.

For text classification, features may represent word frequency, word presence, or numerical representations of words.

For example, the number of times the word "crashes" appears in a review can be used as a feature.

| Document | crashes count |
|---|---:|
| D1 | 0 |
| D2 | 1 |
| D3 | 0 |

Here, the frequency of the word "crashes" is a numerical feature.

### 2.6 Label

A label is the target category assigned to a particular document in a supervised learning dataset.

In the example dataset:

- D1 has the label Positive.
- D2 has the label Negative.
- D3 has the label Positive.

The labels represent the correct categories that a supervised classification model should learn to predict.

### Summary of Core Terms

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of text documents | Three customer reviews |
| Document | One individual text example | D1 |
| Token | A unit produced by a tokenizer | application |
| Vocabulary | Collection of unique tokens | application, easy, useful |
| Feature | Numerical information used by a model | Frequency of "crashes" |
| Label | Correct target category | Positive or Negative |

These concepts are connected. A corpus contains documents, documents are divided into tokens, tokens contribute to a vocabulary, features represent the text numerically, and labels provide target answers for supervised learning.

---

## 3. Four Common NLP Tasks

NLP includes several tasks designed to solve different language-processing problems. The input and output depend on the particular task.

### 3.1 Text Classification

Text classification is the process of assigning one or more predefined categories to a complete text document.

It is commonly used in email filtering, sentiment analysis, news categorization, and customer feedback analysis.

**Input:** A text document, such as a customer review.

**Output:** A predicted category or class.

Example:

Input: "The new application works perfectly."

Output: Positive sentiment.

Another example is classifying news articles into categories such as politics, sports, business, and technology.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which individual elements of a text sequence receive labels.

Unlike text classification, which assigns a category to an entire document, sequence labeling identifies specific words or parts of a sentence.

One common application is Named Entity Recognition (NER).

Example:

Input: "Amina studies at Mogadishu University."

| Word | Predicted Label |
|---|---|
| Amina | Person |
| studies | Other |
| at | Other |
| Mogadishu University | Organization |

**Input:** A sentence or sequence of tokens.

**Output:** A label assigned to each relevant token or sequence of tokens.

This task is useful for extracting people's names, locations, organizations, dates, and other important information from documents.

### 3.3 Search and Similarity

Search and similarity techniques help systems identify documents that are relevant to a user's query or similar to another document.

For example, an online learning platform may allow students to search for materials related to database management.

Input: "Database security"

The search system may retrieve documents discussing database protection, access control, and data privacy.

**Input:** A search query or document.

**Output:** A ranked list of relevant or similar documents.

Some systems use keyword matching, while others use numerical text representations and semantic similarity to identify related content.

### 3.4 Text Generation

Text generation involves producing new human-readable language based on an input, instruction, or previously learned patterns.

Modern language models use this task in applications such as chatbots, automatic summarization, and language translation.

Example:

Input: "Explain the importance of artificial intelligence."

Output: A newly generated explanation describing how AI supports different industries.

**Input:** A prompt, question, or existing text.

**Output:** Newly generated text, such as an answer, summary, or translated sentence.

### Comparison of the Four NLP Tasks

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | Complete text | Category or class |
| Sequence Labeling | Sentence or tokens | Labels for individual tokens |
| Search and Similarity | Query or document | Ranked relevant documents |
| Text Generation | Prompt or existing text | Newly generated text |

Although these tasks use language data, their objectives and expected outputs are different.

---

## 4. Supervised and Unsupervised Learning in NLP

Machine Learning approaches in NLP can be divided into different learning types. Two important approaches are supervised learning and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from training examples that contain both input data and corresponding target labels.

The model compares its predictions with the correct answers during training and adjusts its parameters to reduce prediction errors.

**Example: Customer Review Classification**

Suppose a company collects customer reviews and assigns each review a sentiment label.

| Review | Label |
|---|---|
| Excellent customer service | Positive |
| The delivery was very slow | Negative |
| I am satisfied with the product | Positive |

The model uses these labeled examples to learn patterns associated with positive and negative reviews.

After training, it can predict the sentiment of a new review that it has not previously encountered.

**Why are target labels needed?**

Target labels are necessary because they provide the correct answers that guide the model during supervised training. Without them, the model cannot directly learn the intended mapping between a particular input and its expected category.

### 4.2 Unsupervised Learning

Unsupervised learning is an approach in which a model analyzes data without receiving predefined target labels for the task.

Instead of learning from provided answers, the system attempts to discover patterns, similarities, or structures within the available data.

**Example: Grouping News Articles**

Imagine collecting 5,000 news articles without assigning categories to them.

An unsupervised learning algorithm can analyze similarities between the articles and group related documents together.

For example, the algorithm may discover groups containing:

- Sports-related articles.
- Business-related articles.
- Political articles.

These groups are discovered from patterns in the data rather than from predefined category labels.

**Why are target labels not required?**

The objective of unsupervised learning is to discover relationships or structures that already exist within the data. Since the system is not being trained to reproduce predefined target answers, it does not require manually assigned labels for that task.

### Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Target labels | Required | Not required |
| Main objective | Learn to predict known targets | Discover patterns and structures |
| Example | Sentiment classification | Grouping similar articles |
| Common methods | Naive Bayes, SVM | K-means clustering |

Both approaches are useful in NLP. The appropriate choice depends on the problem and the availability of labeled data.

---

## 5. Why Must Text Be Converted into Numerical Representations?

Most traditional Machine Learning algorithms operate on numerical data rather than directly processing ordinary sentences.

Human language consists of words, phrases, and sentences, which are not automatically suitable as numerical model inputs.

Therefore, text must first be transformed into a numerical representation that a Machine Learning algorithm can process.

This transformation allows the system to identify patterns, compare documents, calculate relationships, and make predictions.

### Bag of Words (BoW)

Bag of Words is a simple and widely used method for converting text into numerical features.

It creates a vocabulary from a collection of documents and represents each document according to the frequency or presence of vocabulary terms.

For example, consider these two sentences:

- Document 1: "Technology improves education."
- Document 2: "Technology supports education."

First, we create a vocabulary:

`["Technology", "improves", "education", "supports"]`

Next, we represent each document using the vocabulary.

| Document | Technology | improves | education | supports |
|---|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 0 |
| D2 | 1 | 0 | 1 | 1 |

The resulting numerical vectors are:

- D1 = [1, 1, 1, 0]
- D2 = [1, 0, 1, 1]

The values represent whether each vocabulary word appears in the document.

A frequency-based Bag of Words representation could also use the actual number of occurrences instead of only 0 and 1.

### Main Limitation of Bag of Words

One major limitation of Bag of Words is that it does not preserve word order or fully represent the relationships between words.

For example:

- "The student helps the teacher."
- "The teacher helps the student."

These sentences contain the same words, but their meanings are different.

A basic Bag of Words representation produces identical vectors for both sentences because it only considers the words and their frequencies.

Consequently, Bag of Words may lose important grammatical and contextual information.

Despite this limitation, it remains useful for simple text-classification tasks because it is easy to implement and computationally efficient.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A low-resource language is a language that has limited computational resources available for developing and evaluating language technologies.

These resources may include large text collections, labeled datasets, dictionaries, tokenizers, morphological tools, and pretrained language models.

Somali is an important example of a language that has historically received less computational NLP research and resource development than widely studied languages such as English.

Several challenges affect the development of reliable Somali NLP systems.

### 6.1 Limited Datasets and Language Resources

One major challenge is the limited availability of large, diverse, and properly labeled Somali text datasets.

Machine Learning models generally require sufficient examples to learn useful language patterns.

When only a small amount of training data is available, an NLP model may:

- Learn incomplete language patterns.
- Perform poorly on unfamiliar sentences.
- Become overly dependent on a limited collection of examples.
- Fail to generalize to different writing styles and sources.

For example, a sentiment classification model trained using a small collection of formal Somali articles may struggle when processing informal social media comments.

Research by Badel et al. (2023) addressed this problem by developing a dedicated Somali Information Retrieval Corpus containing 2,335 documents.

Their work demonstrates the importance of creating language-specific datasets to support Somali NLP development.

### 6.2 Spelling Variation and Non-Standard Writing

Another challenge is the variation in how people write the same language.

Online communication often contains informal spelling, abbreviations, shortened words, and inconsistent writing styles.

For example, the same place name might appear as:

- Soomaaliya
- Somalia

A basic word-based NLP system may treat these forms as different tokens, even when they refer to the same place.

This can divide useful information across separate vocabulary entries.

As a result, a model may fail to connect documents containing different spellings of the same term.

This problem becomes more significant when processing social media messages, where writing conventions are not always consistent.

### 6.3 Morphological Complexity and Word Forms

Somali has morphological characteristics that can create difficulties for text-processing systems.

Words may appear in different grammatical forms depending on their function and context.

For example, different forms of a word may express grammatical information that a simple word-level representation does not recognize as related.

If an NLP system treats every form as a completely separate token, its vocabulary may become larger and important relationships between words may be missed.

Lemmatization can help by identifying a word's base or dictionary form.

Mohamed and Mohamed (2023) investigated a lexicon-based and rule-based approach to Somali word lemmatization. Their work illustrates the need for language-specific preprocessing tools.

### 6.4 Mixed-Language Text and Dialect Variation

Somali speakers may combine Somali and English within the same message. This is commonly known as code-switching.

For example, a person might write:

"I have an assignment cusub today."

A system trained only on standard English or standard Somali may struggle to process such sentences correctly.

Dialect variation also creates difficulties. Somali includes varieties such as Maxaa Tiri and Maay, and an NLP model trained mostly on one variety may not perform equally well on another.

Gedi et al. (2026) investigated automatic Somali dialect identification and discussed challenges related to shared vocabulary, mixed dialect usage, and inconsistent social media spelling.

### Summary of Low-Resource NLP Challenges

| Challenge | Effect on NLP Systems |
|---|---|
| Limited datasets | Reduces the amount and diversity of training information |
| Spelling variation | Causes similar words to be treated as different tokens |
| Morphological complexity | Makes it difficult to recognize related word forms |
| Code-switching | Creates difficulties for language identification and text processing |
| Dialect variation | Can reduce performance on varieties not represented in training data |

These challenges show that developing reliable Somali NLP systems requires more than selecting a powerful algorithm. It also requires high-quality datasets, appropriate preprocessing techniques, and consideration of the language's linguistic characteristics.

---

## 7. Case Study: Automatic Identification of Somali Dialects Using Machine Learning and Deep Learning

### Research Paper

Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., et al. (2026). *Somali dialect identification in low-resource settings using machine learning and deep learning*. Discover Data, 4, Article 10.

Research link: https://doi.org/10.1007/s44248-026-00109-y

### 7.1 Background and Problem

Somali contains different dialects, and automatically identifying these varieties is an important challenge for developing language technologies.

An NLP system trained using one dialect may have difficulty processing another dialect because of differences in vocabulary, spelling, and language usage.

The researchers investigated how Machine Learning and Deep Learning methods could identify two major Somali varieties: Maxaa Tiri and Maay.

The main purpose was to develop and evaluate automatic methods for distinguishing between these varieties.

### 7.2 Research Objective

The main objective of the study was to develop and evaluate computational models capable of automatically identifying Somali dialects.

Specifically, the researchers aimed to:

- Distinguish between Maxaa Tiri and Maay dialects.
- Evaluate traditional Machine Learning and Deep Learning algorithms.
- Investigate the effectiveness of numerical text representations for dialect classification.
- Contribute to the development of NLP resources for the Somali language.

### 7.3 NLP Task

The primary task was **Text Classification**, specifically dialect classification.

The system receives a Somali text sample and predicts which dialect category it belongs to.

The expected output is one of two classes:

- Maxaa Tiri
- Maay

This is a binary text-classification problem.

### 7.4 Dataset

The researchers constructed a manually annotated dataset containing 8,947 Somali text samples.

The dataset contained:

- 5,505 Maxaa Tiri samples.
- 3,442 Maay samples.

The samples were collected from different sources, including social media, news websites, blogs, and other written documents.

The dataset was designed to support research on Somali dialect identification.

### 7.5 Methods Used

The researchers evaluated traditional Machine Learning and Deep Learning approaches.

**Traditional Machine Learning methods included:**

1. Naive Bayes.
2. Support Vector Machine (SVM).
3. Random Forest.
4. Logistic Regression.

The study used TF-IDF (Term Frequency–Inverse Document Frequency) to represent text numerically for traditional classification models.

**Deep Learning methods included:**

1. Convolutional Neural Networks (CNN).
2. Recurrent Neural Networks (RNN).
3. Long Short-Term Memory (LSTM).
4. Bidirectional LSTM (BiLSTM).

These models were evaluated to compare their ability to recognize patterns and distinguish between the two dialects.

### 7.6 Findings and Results

The study reported strong classification performance from both traditional Machine Learning and Deep Learning methods.

Among the traditional classifiers, Naive Bayes achieved 99.05% accuracy in the reported model comparison. SVM also achieved high performance, although its reported accuracy figures are inconsistent in different sections of the paper.

Among the Deep Learning models, BiLSTM and CNN achieved approximately 98% accuracy.

These findings suggest that traditional Machine Learning methods can remain competitive when appropriate numerical text representations and language-specific features are available.

However, high accuracy alone does not guarantee that a model will perform equally well on all naturally occurring Somali dialect data.

### 7.7 Limitations of the Study

Despite its promising results, the research has some limitations:

1. **Limited Natural Maay Data:** Naturally occurring written Maay resources were limited. Some Maay samples were created through expert-assisted translation from Maxaa Tiri.

2. **Short Text Samples:** Short sentences may not contain enough linguistic information for accurate dialect identification.

3. **Shared Vocabulary:** Maxaa Tiri and Maay share some vocabulary, which can make it difficult for models to distinguish between them.

4. **Spelling Variation:** Inconsistent spelling and informal writing, particularly on social media, can affect classification performance.

These limitations show that more naturally collected, diverse, and representative Somali dialect datasets are needed for future research.

### 7.8 Importance of the Research

This research is important because it addresses the limited availability of computational resources for the Somali language.

Identifying Somali dialects can support the development of language technologies such as:

- Somali information retrieval systems.
- Machine translation.
- Dialect-aware language models.
- Text-processing applications.
- Other Somali NLP tools.

The study also demonstrates how Machine Learning and Deep Learning can be applied to improve language technology for low-resource languages.

### 7.9 Connection to NLP Concepts and Tasks

This research connects several concepts discussed in this assignment:

- **NLP:** The study uses computational techniques to process Somali language data.
- **Corpus:** The 8,947 text samples form a dataset used for the research.
- **Features:** TF-IDF converts text into numerical representations for traditional Machine Learning models.
- **Labels:** Maxaa Tiri and Maay are the target categories assigned to the text samples.
- **Supervised Learning:** The models learn from labeled dialect examples to predict the dialect of new text.
- **Text Classification:** The system assigns each Somali text sample to one of the two dialect categories.

Therefore, this case study provides a practical example of how NLP concepts and Machine Learning techniques work together to solve a real language-processing problem.

### 7.10 Case Study Conclusion

The study demonstrates that Machine Learning and Deep Learning techniques can be useful for automatically identifying Somali dialects.

The results show that computational models can achieve strong classification performance when trained using appropriate datasets and text representations.

However, the limited availability of naturally written Maay data and other linguistic challenges indicate that further research is necessary.

Developing larger and more representative Somali datasets will help researchers build more reliable NLP systems that support both Maxaa Tiri and Maay.

Overall, this research represents an important contribution to Somali language technology and low-resource NLP research.

## Conclusion

Natural Language Processing provides methods for enabling computers to work with human language. It is an important area of Artificial Intelligence and can use Machine Learning techniques to process text, discover patterns, classify documents, identify information, and generate language.

Understanding fundamental concepts such as corpus, document, token, vocabulary, feature, and label is essential before developing NLP applications.

Different NLP tasks require different approaches and outputs. Supervised learning uses labeled examples to learn prediction tasks, while unsupervised learning discovers patterns without predefined target labels.

Traditional Machine Learning models require numerical representations of text. Techniques such as Bag of Words provide a simple solution, although they have limitations in representing word order and context.

Research into Somali NLP also demonstrates that language-specific challenges must be considered. Limited datasets, spelling variation, morphological complexity, and dialect differences can affect the reliability of NLP systems.

The reviewed Somali dialect identification study demonstrates that Machine Learning and Deep Learning can be applied to practical language problems. However, continued dataset development and careful evaluation remain important for building reliable language technologies.

Overall, developing effective NLP applications requires a combination of linguistic knowledge, appropriate data, suitable computational methods, and careful evaluation.

---

## References

1. Jurafsky, D., & Martin, J. H. (2009). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (2nd ed.). Prentice Hall.

   https://library.au.int/speech-and-language-processing

2. Stanford Institute for Human-Centered Artificial Intelligence. (n.d.). *What is Natural Language Processing (NLP)?*

   https://hai.stanford.edu/ai-definitions/what-is-nlp

3. Stanford Institute for Human-Centered Artificial Intelligence. (2022). *Brief Definitions of Key Terms in AI.*

   https://hai.stanford.edu/policy/brief-definitions-of-key-terms-in-ai

4. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469.

   https://doi.org/10.18653/v1/2023.emnlp-main.462

5. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-Based Word Lemmatization Approach for Somali Language. *ICLRW 2023.*

   https://arxiv.org/abs/2308.01785

6. Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., et al. (2026). Somali dialect identification in low-resource settings using machine learning and deep learning. *Discover Data, 4*, Article 10.

   https://doi.org/10.1007/s44248-026-00109-y

7. Bag-of-Words Technique in Natural Language Processing: A Primer for Radiologists. (2021). *Journal of the American College of Radiology.*

   https://pmc.ncbi.nlm.nih.gov/articles/PMC8415041/
