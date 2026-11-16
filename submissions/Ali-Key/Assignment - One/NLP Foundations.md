# Assignment 1: NLP Foundations

**Author:** Ali Omar Abdi
**Subject:** Natural Language Processing (NLP)  


## Table of Contents

1. [Introduction](#introduction)
2. [What Is Natural Language Processing?](#1-what-is-natural-language-processing-nlp)
3. [Basic Terms Used in NLP](#2-basic-terms-used-in-nlp)
4. [Four Common NLP Tasks](#3-four-common-nlp-tasks)
5. [Supervised and Unsupervised Learning](#4-supervised-and-unsupervised-learning-in-nlp)
6. [Why Text Must Be Converted into Numbers](#5-why-text-must-be-converted-into-numerical-representations)
7. [Challenges of NLP for Somali](#6-challenges-of-nlp-for-low-resource-languages-such-as-somali)
8. [Case Study: Somali Information Retrieval](#7-case-study-somali-information-retrieval-corpus)
9. [References](#references)

---

## Introduction

People use language to communicate, ask questions, share information, and express ideas. Computers do not understand human language in the same way people do. They need methods that allow them to process, analyse, and generate language.

**Natural Language Processing (NLP)** is a field of computer science and Artificial Intelligence (AI) that develops these methods. It allows computers to work with written and spoken languages, including English, Arabic, and Somali.

This assignment introduces the main concepts of NLP, common NLP tasks, Machine Learning approaches, text representation, and challenges involved in developing language technologies for low-resource languages such as Somali. Everyday examples are included to make the concepts easier to understand.

## 1. What Is Natural Language Processing (NLP)?

Natural Language Processing is a branch of Artificial Intelligence that enables computers to work with human language. It includes analysing, understanding, interpreting, and generating written or spoken language.

For example, when a customer sends a message in Somali, an NLP system can analyse the words and identify whether the customer is asking a question, making a complaint, or requesting information.

NLP is used in:

- Chatbots that answer questions.
- Translation systems that convert text from one language to another.
- Email filters that identify unwanted messages.
- Voice assistants that respond to spoken instructions.
- Search engines that find information from written queries.

### Relationship between AI, Machine Learning, and NLP

These concepts are connected, but they have different meanings.

| Concept | Meaning |
|---|---|
| **Artificial Intelligence (AI)** | A broad field concerned with developing computer systems that perform tasks associated with human intelligence. |
| **Machine Learning (ML)** | A branch of AI that allows computers to learn patterns from data and make predictions. |
| **Natural Language Processing (NLP)** | A field concerned with enabling computers to process and work with human language. |

NLP is an area of AI. Machine Learning provides many of the methods used to develop modern NLP applications. However, NLP can also use linguistic rules and other methods that do not require Machine Learning.

### Everyday example: A Somali customer-support chatbot

Imagine a cargo company that receives customer messages through WhatsApp.

A customer writes:

> "Xaggee ayay maraysaa alaabtaydu?"

Meaning: "Where is my package?"

The three fields can work together:

1. **AI** provides the overall intelligent system that responds to customer questions.
2. **NLP** processes the Somali sentence and identifies that the customer is asking about a shipment.
3. **Machine Learning** can help the system learn from previously labelled messages and recognise similar questions.

The chatbot can then search the shipment database and return the latest tracking information.

This example shows how AI, NLP, and Machine Learning can work together to solve a practical communication problem.

## 2. Basic Terms Used in NLP

Before developing an NLP system, it is useful to understand how text data is organised.

Consider a small dataset containing messages sent by customers to a Somali cargo company.

### Example dataset

| Document ID | Customer message | Label |
|---|---|---|
| D1 | alaabtaydu way timid | Delivery |
| D2 | Waxaan rabaa inaan alaab diro | Shipping |
| D3 | Xagee marayaa baakadaydu? | Tracking |
| D4 | Qiimaha raristu waa imisa? | Price |

The dataset contains four customer messages. Each message has a different purpose.

### Important terms

| Term | Definition | Example from the dataset |
|---|---|---|
| **Corpus** | A collection of text documents gathered for research or NLP development. | All four customer messages together. |
| **Document** | One individual piece of text within a corpus. | D1: "alaabtaydu way timid." |
| **Token** | A word or another unit produced when text is divided into smaller parts. | "alaabtaydu", "way", and "timid" are three word tokens. |
| **Vocabulary** | The collection of unique tokens found in a corpus after tokenisation. | The distinct words appearing across the four messages. |
| **Feature** | A measurable property of text that a Machine Learning model uses to identify patterns or make predictions. | Word frequency, message length, or the presence of a particular word. |
| **Label** | The category or correct answer assigned to a document during supervised learning. | Delivery, Shipping, Tracking, or Price. |

### How the terms fit together

Suppose the company collects 1,000 customer messages:

- The complete collection is the **corpus**.
- Each individual message is a **document**.
- Dividing a message into words produces **tokens**.
- Collecting all different tokens produces the **vocabulary**.
- Information such as word frequency and message length can become **features**.
- The category assigned to each message is its **label**.

For example, a message containing the words "qiimaha" and "imisa" may help a model recognise that a customer is asking about price. The model can learn this pattern from previously labelled examples.

## 3. Four Common NLP Tasks

NLP systems perform different tasks depending on the problem they are designed to solve. Some identify categories, some label individual words, some retrieve information, and others generate new text.

### 3.1 Text Classification

Text classification is the process of assigning a category to a complete text document or message.

**Example**

A customer writes:

> "Waxaan helay adeeg ka wanaagsan sidii aan filayay."

Meaning: "The service you provided was very good."

The system may classify this message as positive feedback.

| Component | Explanation |
|---|---|
| **Input** | A written customer message. |
| **Process** | The model analyses words and patterns associated with different categories. |
| **Output** | A category, such as Positive sentiment. |

Other categories may include negative, neutral, spam, complaint, or customer enquiry.

### 3.2 Sequence Labeling

Sequence labeling assigns a label to each token or word in a sentence. It differs from text classification because it focuses on individual parts of the sentence rather than assigning one category to the whole document.

**Example**

Sentence:

> "Ali wuxuu joogaa Muqdisho."

| Word | Example label |
|---|---|
| Ali | PERSON |
| wuxuu | OTHER |
| joogaa | OTHER |
| Muqdisho | LOCATION |

| Component | Explanation |
|---|---|
| **Input** | A sentence or sequence of tokens. |
| **Process** | The model identifies the role or category of each token. |
| **Output** | Tokens with their assigned labels. |

This task is useful for identifying people's names, locations, dates, and other information. The labels above are illustrative; a real annotation project would define a consistent labelling scheme.

### 3.3 Search and Similarity

Search and similarity help users find documents related to their questions or interests. A search system compares a user's query with information stored in a collection of documents and returns relevant results.

**Example**

A customer searches:

> "Sidee loo ogaadaa halka alaabtaydu marayso?"

The system searches cargo-related documents and retrieves information about shipment tracking.

| Component | Explanation |
|---|---|
| **Input** | A search query and a collection of documents. |
| **Process** | The system compares the query with available documents using keywords or similarity measures. |
| **Output** | A ranked list of relevant documents or results. |

This task is used in search engines, digital libraries, and customer-support systems.

### 3.4 Text Generation

Text generation is the process of producing new, readable text from an instruction, prompt, or other input.

**Example**

A customer asks:

> "Goorma ayay alaabtaydu imaanaysaa?"

A chatbot might respond:

> "alaabtaadu waxay ku jirtaa safarka, waxaana la filayaa inay dhowaan timaaddo."

| Component | Explanation |
|---|---|
| **Input** | A question, instruction, or text prompt. |
| **Process** | The model uses patterns learned from language data to produce a suitable response. |
| **Output** | A newly generated sentence or passage. |

Text generation is used in chatbots, writing assistants, translation systems, and summarisation tools.

### Comparison of the four tasks

| NLP task | Main purpose | Input | Output |
|---|---|---|---|
| **Text classification** | Categorise text. | A document or message. | A category or label. |
| **Sequence labeling** | Label individual tokens. | A sentence or token sequence. | Tokens with assigned labels. |
| **Search and similarity** | Find relevant information. | A query and document collection. | Ranked relevant documents. |
| **Text generation** | Produce new text. | A prompt or instruction. | Generated text. |

## 4. Supervised and Unsupervised Learning in NLP

Machine Learning allows computers to learn patterns from data. Two common approaches are supervised learning and unsupervised learning. Their main difference is whether the training data contains target labels.

### 4.1 Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from examples that already have correct answers, called **labels**.

During training, the model examines each text example together with its label. It learns patterns that can help it predict labels for new messages.

**Example: Classifying Somali customer messages**

| Customer message | Correct label |
|---|---|
| alaabtaydu way timid | Delivery |
| Waxaan rabaa inaan alaab diro | Shipping |
| Xagee marayaa baakadaydu? | Tracking |
| Qiimaha raristu waa imisa? | Price |

After training, the model receives a new message:

> "Waxaan rabaa inaan ogaado qiimaha rarista."

It may predict the category **Price**.

The correct labels are needed because they show the model what it is expected to predict.

### 4.2 Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model works with data that does not have predefined target labels.

The model examines the data and identifies patterns, similarities, or groups without being given the correct category for each example.

**Example: Grouping Somali customer messages**

Suppose a cargo company collects 500 customer messages without assigning categories. An algorithm may group messages that use similar words or discuss similar subjects.

| Possible group | Example messages |
|---|---|
| Group 1 | alaabtaydu way timid; Alaabtii waa la keenay. |
| Group 2 | Qiimaha raristu waa imisa?; Immisa ayaa laga qaadaa halkii kiilo? |
| Group 3 | Xagee marayaa alaabtaydu?; Goorma ayay imaanaysaa? |

The algorithm discovers the groups. A researcher can inspect them afterwards and give them meaningful names.

### Main differences

| Comparison | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Labelled data. | Unlabelled data. |
| Target labels | Required during training. | Not required. |
| Main purpose | Predict known categories or values. | Discover patterns and groups. |
| Example | Classifying messages as complaints or enquiries. | Grouping similar customer messages. |
| Common techniques | Naive Bayes, Decision Trees, Support Vector Machines. | K-means clustering and topic modelling. |

### Why does one need labels and the other does not?

Supervised learning needs target labels because the model learns by comparing its predictions with the correct answers provided in the training data.

Unsupervised learning does not need target labels because its purpose is to discover patterns in the data. It can group similar documents without being told what each group means.

Both approaches are useful. The best choice depends on the problem, the available data, and the expected result.

## 5. Why Text Must Be Converted into Numerical Representations

Most traditional Machine Learning algorithms cannot directly use ordinary sentences as their mathematical input. They perform calculations using numbers.

Text must therefore be converted into numerical representations. This process is commonly called **text vectorisation** or **feature extraction**.

### Example: Bag of Words

One simple method is called **Bag of Words (BoW)**. It represents a document by counting how often each vocabulary word appears.

Consider these three messages:

- D1: "Alaabtu way timid."
- D2: "Alaabtu way baxday."
- D3: "Waxaan rabaa alaab."

First, create a vocabulary:

`[Alaabtu, way, timid, baxday, Waxaan, rabaa, alaab]`

Next, count the occurrence of each word in each document.

### Bag-of-Words table

| Document | Alaabtu | way | timid | baxday | Waxaan | rabaa | alaab |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| D2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| D3 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

The text has now been converted into numerical vectors.

For example:

`D1 = [1, 1, 1, 0, 0, 0, 0]`

A Machine Learning model can use these numbers to compare documents and learn patterns.

### Main limitation of Bag of Words

Bag of Words is simple, but it does not preserve word order or fully represent sentence meaning.

Consider:

1. "The customer received the package."
2. "The package received the customer."

The two sentences contain the same words, but their meanings are different. A basic Bag-of-Words representation produces the same word counts for both sentences. It therefore cannot identify the difference in meaning from word order alone.

It may also treat related word forms as separate vocabulary items unless additional text processing is applied.

Other methods, such as TF-IDF and word embeddings, represent text in different ways and address some of these limitations.

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A **low-resource language** is a language with limited digital resources available for developing and training language technologies. These resources may include electronic text collections, dictionaries, annotated datasets, and speech recordings.

Somali is an example of a language for which researchers have had fewer digital NLP resources than are available for widely studied languages such as English.

Several challenges affect the development of accurate Somali NLP systems.

### 6.1 Limited Datasets and Language Resources

One major challenge is the shortage of high-quality Somali datasets that researchers can use to train and evaluate NLP models.

For example, an English sentiment-analysis project may have access to thousands or millions of labelled customer reviews. A Somali researcher may find it difficult to obtain a similarly large collection of correctly labelled Somali messages.

This can cause several problems:

- Models have fewer examples from which to learn.
- Some Somali words and expressions may be missing from the training data.
- Models may perform poorly on unfamiliar messages.
- Researchers may have difficulty evaluating system performance.

**Everyday example**

Suppose a company wants to develop a chatbot that understands Somali customer questions. If it has collected only 100 training messages, the chatbot may struggle when customers use expressions that do not appear in those examples.

The system may misunderstand a question or provide an incorrect response.

### 6.2 Spelling Variation and Informal Writing

People do not always write messages in the same way, particularly on social media and messaging applications. They may use informal spelling, shorten words, or make typing mistakes.

For example, a customer might write:


- "Xaggee maraysaa alaabtaydu?"
- "laabtaydu heerkee ayay maraysaa?"

Both messages are intended to ask about the location of a package, although their written forms differ.

If the training data contains only one form, a model may struggle to recognise the other. This can reduce the accuracy of classification, search, and chatbot responses.

### 6.3 Word Forms and Morphological Complexity

Words can appear in different grammatical forms depending on how they are used in a sentence. Related forms may therefore look different in written text.

A system that recognises only one form may fail to identify other forms that share the same root.

**Lemmatization** is a text-processing technique that attempts to convert related word forms into a common dictionary form.

Mohamed and Mohamed (2023) studied a lexicon- and rule-based approach to Somali word lemmatization. Their work illustrates how language-specific word resources and rules can support Somali text processing.

Lemmatization can help an NLP system recognise related word forms, although the quality of the result depends on the language resources and rules available.

### 6.4 Mixed-Language Text and Dialect Differences

Somali speakers sometimes combine Somali with English or Arabic, especially in informal digital communication.

For example:

> "Waxaan rabaa inaan check gareeyo tracking number-kayga."

This sentence combines Somali and English words.

A model trained only on formal Somali sentences may have difficulty processing this type of message.

Dialect differences can also affect performance. A model trained mainly on one variety of Somali may not work equally well with another variety. Shared vocabulary, different expressions, and non-standard online spelling can make the task more difficult.

### Summary of the challenges

| Challenge | Main problem | Possible effect on an NLP system |
|---|---|---|
| **Limited datasets** | Not enough high-quality training examples. | Poor predictions and limited coverage. |
| **Spelling variation** | The same expression appears in different written forms. | Difficulty recognising similar messages. |
| **Word forms** | Words have different grammatical forms. | Incorrect word identification or analysis. |
| **Mixed-language text and dialects** | Users combine languages or use different dialects. | Misinterpretation and reduced accuracy. |

Developing Somali NLP systems therefore requires language-specific datasets, suitable text-processing methods, and contributions from people who understand the language and its varieties.

## 7. Case Study: Somali Information Retrieval Corpus

### Research paper

**Title:** *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*  
**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou  
**Year:** 2023  
**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)  
**DOI:** [10.18653/v1/2023.emnlp-main.462](https://doi.org/10.18653/v1/2023.emnlp-main.462)

### 7.1 Background and Research Problem

The researchers investigated the problem of finding information written in Somali.

Some search systems have relied on translating Somali queries into other languages because dedicated Somali information-retrieval resources are limited. However, translation may fail to preserve the original meaning of a user's query.

The researchers aimed to develop a dedicated Somali information-retrieval resource that could support searching with Somali text directly.

### 7.2 Data Used in the Study

The researchers created an annotated Somali information-retrieval corpus containing **2,335 documents**.

The documents came from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

These sources provided Somali written content for developing and testing information-retrieval methods.

### 7.3 Method Used

The researchers developed an information-retrieval approach using a technique called **Pseudo-Relevance Feedback (PRF)**.

In simple terms, this technique uses documents retrieved in an initial search to help improve the original query.

For example, a user searches for information about education in Somalia. The system retrieves documents that appear relevant. It can then use useful terms from those documents to expand or improve the search and identify other potentially relevant documents.

### 7.4 Findings and Contribution

The study produced a dedicated Somali information-retrieval corpus containing 2,335 documents and presented a retrieval approach based on pseudo-relevance feedback.

The work highlights the value of developing language-specific resources rather than relying mainly on translating Somali queries into other languages.

The corpus can also support further research on Somali information retrieval and related language technologies. The numerical corpus size is a reported resource contribution; it should not be confused with a search-accuracy score.

### 7.5 Which NLP Task Does the Study Cover?

The study mainly covers **Search and Similarity (Information Retrieval)**, the third task described in Section 3.

The system receives a search query and uses a collection of Somali documents to identify relevant information.

| Component | Application in the study |
|---|---|
| **Input** | A Somali search query and a collection of documents. |
| **Data** | 2,335 Somali documents. |
| **Method** | Information retrieval using pseudo-relevance feedback. |
| **Output** | Documents retrieved as relevant to the search. |
| **NLP task** | Search and Similarity. |

### 7.6 Why This Research Matters

This study addresses a practical problem in Somali language technology: the limited availability of dedicated digital resources.

A Somali information-retrieval system can help people find information using their own language. It may also provide a foundation for future work on Somali question-answering systems, digital libraries, and other language-processing applications.

The study illustrates why building suitable datasets is an important step in developing practical NLP systems.

## References

1. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. https://doi.org/10.18653/v1/2023.emnlp-main.462

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/

3. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed., online manuscript). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

5. Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and Rule-based Word Lemmatization Approach for the Somali Language*. arXiv. https://doi.org/10.48550/arXiv.2308.01785

6. Scikit-learn Developers. (n.d.). *Feature Extraction: Working with Text Data*. Scikit-learn Documentation. https://scikit-learn.org/stable/modules/feature_extraction.html

---
