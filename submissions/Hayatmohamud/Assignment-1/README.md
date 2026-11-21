# Assignment 1 – NLP Foundations

**Name:** Hayat Mohamud Hassan  


## Table of Contents

1. [Definition of NLP](#1-definition-of-natural-language-processing-nlp)
2. [Fundamental NLP Terms](#2-fundamental-nlp-terms)
3. [Four Common NLP Tasks](#3-four-common-nlp-tasks)
4. [Supervised and Unsupervised Learning](#4-supervised-and-unsupervised-learning-in-nlp)
5. [Numerical Representation of Text](#5-numerical-representation-of-text)
6. [Challenges of NLP for Low-Resource Languages](#6-challenges-of-nlp-for-low-resource-languages-such-as-somali)
7. [Case Study: Somali Information Retrieval](#7-case-study-somali-information-retrieval-corpus)
8. [References](#references)

---

## 1. Definition of Natural Language Processing (NLP)

### 1.1 What is NLP?

Natural Language Processing (NLP) is a field of computer science and artificial intelligence that focuses on developing systems capable of processing, analyzing, interpreting, and generating human language.

Human languages, such as English, Somali, and Arabic, are complex because they contain grammar, different word forms, meanings, context, and cultural expressions. NLP provides computational methods that allow computers to work with this complexity.

For example, when someone sends a message saying:

> I am not happy with this service.

An NLP system can analyze the sentence, recognize its words, and classify it as expressing negative sentiment.

NLP is used in many real-world applications, including:

- Chatbots and virtual assistants.
- Machine translation.
- Spam detection.
- Sentiment analysis.
- Search engines.
- Speech recognition.
- Automatic text summarization.

### 1.2 Relationship between NLP, Artificial Intelligence, and Machine Learning

These three concepts are related, but they do not mean the same thing.

| Concept | Explanation |
|---|---|
| Artificial Intelligence (AI) | A broad field concerned with building machines that perform tasks requiring human-like intelligence. |
| Machine Learning (ML) | A branch of AI that enables computers to learn patterns from data without explicitly programming every rule. |
| Natural Language Processing (NLP) | A field concerned with enabling computers to process and work with human language. |

NLP is an area of AI, while machine learning is one of the main approaches used to build modern NLP systems. The fields overlap: not every NLP method uses machine learning, and machine learning is also used outside NLP.

### 1.3 Real-Life Example: Email Spam Detection

Consider an email service that automatically separates spam messages from legitimate emails.

1. **AI:** The overall intelligent application makes decisions about incoming emails.
2. **NLP:** The system processes the email's subject and message, examining words and language patterns.
3. **ML:** A model learns from previously labeled examples of spam and legitimate emails and uses those patterns to classify new messages.

The system produces an output such as:

- Spam
- Not Spam

This example shows how NLP processes language, machine learning learns patterns from examples, and AI brings these capabilities together in an application.

---

## 2. Fundamental NLP Terms

To understand how NLP systems work, it is important to understand the basic terminology used when preparing and processing text data.

For this question, I will use a small fictional dataset containing customer reviews of an online shopping service.

### 2.1 Definitions

**1. Corpus**

A corpus is an organized collection of written or spoken language data gathered for linguistic analysis or computational processing. A collection of customer reviews can be used as a corpus for sentiment analysis.

**2. Document**

A document is one individual unit of text within a corpus. Depending on the application, it could be an email, paragraph, article, customer review, or social media post.

**3. Token**

A token is an individual unit produced when text is divided into smaller pieces during tokenization. Tokens can represent words, punctuation marks, subwords, or characters.

Example:

Original sentence:

`The service is excellent!`

Word-level tokens:

`["The", "service", "is", "excellent", "!"]`

This sentence contains five tokens under this tokenization approach.

**4. Vocabulary**

Vocabulary refers to the collection of unique tokens identified in a corpus after applying a specified preprocessing and tokenization method.

For example, if the same word appears twenty times, it normally appears only once in the vocabulary.

**5. Feature**

A feature is a measurable property or numerical representation extracted from text that a machine learning algorithm can use to identify patterns and make predictions. Examples include word frequencies, document length, and word embeddings.

**6. Label**

A label is the target category or expected answer assigned to a training example in supervised machine learning. For example, a customer review might have a label of Positive or Negative.

### 2.2 Example Dataset

Consider the following fictional dataset:

| Document ID | Customer review | Label |
|---|---|---|
| D1 | The service is excellent | Positive |
| D2 | The delivery was very slow | Negative |
| D3 | I love this service | Positive |
| D4 | The product is terrible | Negative |

Each term applies to the dataset as follows:

| Term | Application in the dataset |
|---|---|
| Corpus | All four customer reviews together. |
| Document | Each individual review, such as D1. |
| Token | Individual units such as "service", "excellent", and "slow". |
| Vocabulary | Unique tokens extracted from all four reviews. |
| Feature | Numerical word-frequency values representing each review. |
| Label | Positive or Negative, indicating the review's sentiment. |

For example, consider D1:

- **Document:** The service is excellent.
- **Tokens:** The, service, is, excellent.
- **Vocabulary entries:** service, excellent, among others.
- **Features:** Numerical counts representing the words.
- **Label:** Positive.

This demonstrates how raw text is organized into data that can be processed by an NLP model.

---

## 3. Four Common NLP Tasks

NLP includes different tasks depending on what a system needs to accomplish. Four important examples are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category or label to an entire text document.

**Example:** A system that identifies whether a customer review expresses positive or negative sentiment.

- **Input:** `The product quality is excellent.`
- **Output:** `Positive`

Other applications include spam detection, news categorization, and emotion detection.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which the system assigns a label to each token or element in a sequence rather than assigning one label to the entire document.

One common application is Named Entity Recognition (NER), which identifies names of people, locations, organizations, and other entities.

**Example:**

Input:

`Hodan lives in Mogadishu.`

Output:

| Token | Predicted label |
|---|---|
| Hodan | PERSON |
| lives | OTHER |
| in | OTHER |
| Mogadishu | LOCATION |

This task is useful in information extraction, document analysis, and identifying important entities in news articles.

### 3.3 Search and Similarity

Search and similarity involve finding documents that are relevant or similar to a user's query.

An NLP-based search system can represent text numerically and compare the representations to identify relevant results.

**Example:**

A user searches:

`How can I learn Python programming?`

The system may return documents containing:

- Beginner Python tutorials.
- Introduction to Python programming.
- Learning Python from scratch.

Even if the documents do not contain exactly the same words, they may have similar meanings.

- **Input:** A search query and a collection of documents.
- **Output:** A ranked list of relevant or similar documents.

Applications include search engines, recommendation systems, and document retrieval.

### 3.4 Text Generation

Text generation is the process of producing new human-readable text based on an input, instruction, or previously supplied context.

Modern language models use learned patterns to generate sequences of words or tokens.

**Example:**

Input:

`Write a short explanation of artificial intelligence.`

Possible output:

> Artificial intelligence is a field of computer science that develops systems capable of performing tasks that normally require human intelligence.

Applications include chatbots, automated writing assistants, translation systems, and summarization tools.

### 3.5 Comparison of the Four Tasks

| NLP task | Input | Output |
|---|---|---|
| Text classification | A text document | One category or label |
| Sequence labeling | A sequence of tokens | A label for each token |
| Search and similarity | Query and document collection | Ranked relevant documents |
| Text generation | Prompt or contextual text | Newly generated text |

---

## 4. Supervised and Unsupervised Learning in NLP

Machine learning approaches can be divided into different learning types depending on whether the training data contains predefined target labels.

### 4.1 Supervised Learning

Supervised learning is a machine learning approach in which a model learns from examples containing both input data and corresponding correct outputs (labels).

During training, the algorithm compares its predictions with the actual labels and adjusts its parameters to reduce prediction errors.

**Example: Somali Sentiment Classification**

Suppose we want to develop a system that identifies whether Somali social media comments express positive or negative opinions.

A training dataset might contain:

| Somali text | Label |
|---|---|
| Adeeggan aad ayaan u jeclahay | Positive |
| Adeeggu aad buu u xun yahay | Negative |
| Waa adeeg aad u wanaagsan | Positive |
| Kuma qanacsani adeeggan | Negative |

The model learns relationships between the language used in the comments and their assigned sentiment categories. After training, it can predict the sentiment of a new comment.

**Why are labels necessary?**

Labels provide the correct answers during training. They allow the model to learn which patterns correspond to particular categories.

### 4.2 Unsupervised Learning

Unsupervised learning is a machine learning approach in which an algorithm analyzes data without predefined target labels.

Instead of learning to predict known answers, the algorithm identifies patterns, structures, or similarities within the data.

**Example: Grouping Somali Social Media Comments**

Imagine collecting 5,000 Somali social media comments without assigning categories.

A clustering algorithm could analyze their numerical representations and group similar comments together.

The resulting groups might contain:

- Comments discussing education.
- Comments discussing politics.
- Comments discussing sports.
- Comments discussing business.

These categories are possible interpretations of the discovered groups; the algorithm does not necessarily understand or name their topics automatically.

**Why are labels unnecessary?**

The objective is to discover relationships and patterns within the available data rather than predict a predefined answer. Therefore, the algorithm can organize the data using similarities without manually assigned target labels.

### 4.3 Comparison

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Main objective | Predict known categories or values | Discover patterns and structure |
| Target labels | Required | Not required |
| Example NLP task | Sentiment classification | Document clustering |
| Evaluation | Often compared with known correct labels | Often evaluated using clustering measures and human interpretation |

**Conclusion:** The main difference between the two approaches is the availability of target labels. Supervised learning uses known answers to learn predictions, whereas unsupervised learning discovers structures and similarities without predefined answers.

---

## 5. Numerical Representation of Text

### 5.1 Why Must Text Be Converted into Numbers?

Most traditional machine learning algorithms cannot directly process human-readable sentences as mathematical input. They require numerical data that can be represented as vectors, matrices, or other mathematical structures.

For example, a computer can display the sentence:

`I enjoy learning machine learning.`

However, a traditional ML algorithm cannot directly calculate mathematical relationships between the words in that sentence.

To make the text usable, an NLP system first converts it into numerical representations. This process is called **text vectorization** or **text representation**.

Numerical representations allow machine learning algorithms to:

- Calculate similarities between documents.
- Identify patterns in word usage.
- Measure relationships between text features.
- Train classification models.
- Make predictions about new text.

### 5.2 Bag of Words (BoW)

Bag of Words is a basic text representation technique that converts documents into numerical vectors based on the frequency of words.

It creates a vocabulary from a collection of documents and counts how often each vocabulary word appears in each document.

Consider three sentences:

- D1: I love machine learning.
- D2: I love artificial intelligence.
- D3: Machine learning is interesting.

First, create a vocabulary:

`[I, love, machine, learning, artificial, intelligence, is, interesting]`

Next, count the occurrences of each word in each document.

| Document | I | love | machine | learning | artificial | intelligence | is | interesting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| D2 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| D3 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |

For example, the vector for D1 is:

`[1, 1, 1, 1, 0, 0, 0, 0]`

The model can now use these numbers as input.

### 5.3 Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and much of the contextual meaning of language.

For example:

- Sentence A: The teacher helped the student.
- Sentence B: The student helped the teacher.

Both sentences contain the same words and therefore produce identical Bag of Words vectors, even though their meanings are different.

Another limitation is that it does not naturally capture synonyms. For example, the words "excellent" and "great" are treated as different vocabulary entries even though they have similar meanings.

Consequently, Bag of Words is useful as a simple starting point, but more advanced representations, such as word embeddings and contextual embeddings, are often needed for tasks requiring semantic understanding.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

### 6.1 What Is a Low-Resource Language?

A low-resource language is a language for which relatively limited digital linguistic resources are available for developing and evaluating NLP systems.

These resources include:

- Digitized text collections.
- Annotated training datasets.
- Dictionaries and lexical databases.
- Language processing tools.
- Speech datasets.
- Standardized evaluation benchmarks.

Somali is an important example of a language facing these challenges. Although millions of people communicate in Somali, its availability in computationally prepared datasets and language-processing resources is considerably more limited than that of languages such as English.

### 6.2 Challenge 1: Limited Datasets and Linguistic Resources

One major challenge is the lack of sufficiently large, diverse, and accurately labeled Somali datasets.

Many modern NLP models require substantial amounts of training data to recognize language patterns and perform tasks reliably.

For example, developing a Somali sentiment analysis model requires a collection of Somali comments that have been reviewed and assigned sentiment labels by knowledgeable annotators.

Without sufficient examples, a model may:

- Fail to recognize less common Somali expressions.
- Perform poorly on unfamiliar topics.
- Memorize limited training examples instead of learning general patterns.
- Produce unreliable predictions on new data.

**Research evidence:**

Badel et al. (2023), in their paper *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*, address the shortage of dedicated Somali information retrieval resources.

The researchers developed a corpus containing 2,335 documents collected from online Somali sources and poetry books.

### 6.3 Challenge 2: Spelling Variation and Inconsistent Writing

Somali users may write the same word or expression differently, particularly in informal digital communication.

Differences may involve:

- Missing or additional vowels.
- Informal abbreviations.
- Typographical errors.
- Inconsistent spelling.
- Different ways of writing borrowed words.

This creates problems because an NLP model may treat different spellings as unrelated vocabulary entries.

**Effect on NLP systems:**

Suppose a sentiment analysis model was trained using standardized Somali text. If users frequently write informal or inconsistent versions of those expressions, the model may fail to recognize their intended meanings.

This can reduce classification accuracy and increase the number of unknown or unfamiliar words.

### 6.4 Challenge 3: Morphological Complexity and Different Word Forms

Somali has a rich morphological structure. Words can change their forms through grammatical inflections and derivational processes.

A single root may have multiple related forms depending on grammatical or linguistic context.

For example, the Somali word *macalin* (teacher) has the plural form *macalimiin* (teachers).

A system that does not understand these relationships may fail to recognize that the two forms are related.

**Effect on NLP systems:**

- Vocabulary size may increase unnecessarily.
- Similar words may be treated as unrelated.
- Search systems may fail to retrieve documents containing different forms of a search term.
- Text classification may become less accurate.

**Research evidence:**

Mohamed and Mohamed (2023) investigated Somali lemmatization using a lexicon and rule-based approach.

Their initial system included 1,247 root words and 7,173 derivationally related forms. The study demonstrates the importance of understanding Somali morphology when building language-processing tools.

### 6.5 Challenge 4: Mixed-Language Text and Code-Switching

Code-switching occurs when speakers alternate between two or more languages within the same conversation or sentence.

Somali speakers may combine Somali and English in social media posts, messages, and everyday communication.

For example:

`Maanta assignment ayaan submit gareeyay.`

This sentence combines Somali and English vocabulary.

Such text can be difficult for NLP systems trained exclusively on standard Somali or English.

**Effect on NLP systems:**

- Language identification becomes more difficult.
- A tokenizer may process unfamiliar combinations incorrectly.
- A model may misunderstand expressions that depend on both languages.
- Sentiment analysis and other classification tasks may produce incorrect results.

### 6.6 Summary of the Challenges

| Challenge | Main problem | Effect on NLP |
|---|---|---|
| Limited datasets | Insufficient training and evaluation data | Poor generalization and unreliable predictions |
| Spelling variation | Different spellings of similar expressions | Vocabulary fragmentation and recognition errors |
| Morphological complexity | Multiple forms of related words | Difficulty identifying related words |
| Code-switching | Mixing Somali and other languages | Incorrect language processing and interpretation |

### 6.7 Possible Solutions

To improve NLP development for Somali, researchers and developers can:

1. Build larger, publicly available Somali text datasets.
2. Collaborate with native Somali speakers to annotate and verify data.
3. Develop Somali tokenizers, stemmers, lemmatizers, and part-of-speech taggers.
4. Normalize spelling carefully without removing meaningful linguistic differences.
5. Collect real-world mixed-language text for training and evaluation.
6. Explore multilingual pretrained models and transfer learning.

The development of language-specific resources is particularly important because simply applying models designed for high-resource languages does not guarantee reliable performance in Somali.

---

## 7. Case Study: Somali Information Retrieval Corpus

### 7.1 Research Paper

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Year:** 2023.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

**Research paper:** https://doi.org/10.18653/v1/2023.emnlp-main.462

### 7.2 Background and Problem

Information retrieval is the process of finding relevant documents from a collection based on a user's search query.

For languages with extensive digital resources, search systems can be developed using large collections of native-language documents.

However, Somali has historically had limited dedicated information retrieval resources.

As a result, some approaches have depended on translating Somali search queries into other languages, particularly English, before retrieving information.

This creates a problem because translation may fail to preserve the original meaning or linguistic expressions.

The researchers investigated how dedicated Somali-language resources could help address this limitation.

### 7.3 Task

The main task was developing a Somali information retrieval resource and exploring a search system that could retrieve relevant Somali documents.

The research focused on creating a dedicated corpus and using query expansion to improve information retrieval.

### 7.4 Dataset

The researchers developed a corpus containing **2,335 Somali documents**.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The dataset provided a dedicated collection of Somali-language material that could be used for developing and studying information retrieval systems.

### 7.5 Methodology

The researchers worked with language experts and NLP researchers to construct and annotate the corpus.

They then developed a Somali information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF)**.

Pseudo-Relevance Feedback is a query expansion technique that uses initially retrieved documents to identify additional terms that may improve a search query.

The general process is:

1. A user submits a search query.
2. The system retrieves an initial set of potentially relevant documents.
3. The system analyzes the initially retrieved results to identify useful additional terms.
4. The query is expanded using those terms.
5. The system performs an improved search using the expanded query.

This approach attempts to improve retrieval by considering related terms rather than relying only on the original query.

### 7.6 Findings and Contributions

The study made several important contributions:

- It introduced a dedicated Somali information retrieval corpus containing 2,335 documents.
- It provided a resource for developing Somali-language search systems.
- It demonstrated an approach to Somali information retrieval using pseudo-relevance feedback.
- It highlighted the importance of creating language-specific datasets for low-resource NLP research.

The main contribution of this research is the development of the corpus and retrieval approach. Numerical performance claims should be checked against the full experimental results before being reported.

### 7.7 Which NLP Task Does This Case Study Cover?

This case study primarily covers **Search and Similarity**, which was discussed in Question 3.

The system receives a user's search query and attempts to retrieve documents relevant to that query.

For example:

- **Input:** A Somali search query about education.
- **Processing:** The system searches its Somali document collection and uses query expansion.
- **Output:** A ranked collection of relevant Somali documents.

### 7.8 Why Is This Research Important?

This research is important because it addresses a practical problem affecting Somali speakers who want to search for information in their own language.

It also demonstrates that developing NLP systems for low-resource languages requires more than simply choosing a machine learning algorithm.

Researchers must first build reliable datasets, understand the language, and develop appropriate processing methods.

The study provides a foundation for future Somali applications, including document search, digital libraries, question-answering systems, and other information access technologies.

---

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., online manuscript). Stanford University.  
   https://web.stanford.edu/~jurafsky/slp3/

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit.* O'Reilly Media.  
   https://www.nltk.org/book/

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.  
   https://nlp.stanford.edu/IR-book/

4. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469.  
   https://doi.org/10.18653/v1/2023.emnlp-main.462

5. Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and Rule-based Word Lemmatization Approach for the Somali Language.*  
   https://arxiv.org/abs/2308.01785

6. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). *Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform.* Proceedings of the 7th Workshop on African Natural Language Processing.  
   https://doi.org/10.18653/v1/2026.africanlp-main.17
