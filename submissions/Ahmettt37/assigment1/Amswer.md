# Assignment 1 – NLP Foundations

**Subject:** Natural Language Processing  
**Assignment:** 01 – NLP Foundations

---

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of Artificial Intelligence that helps computers understand, process, analyze, and generate human language. It allows computers to work with languages that people use in everyday communication, such as English and Somali.

### Relationship Between AI, Machine Learning, and NLP

**Artificial Intelligence (AI):** AI is a field of computer science that focuses on creating computer systems that can perform tasks that normally require human intelligence.

**Machine Learning (ML):** Machine Learning is a part of AI that allows computers to learn patterns from data and make predictions without being programmed for every situation.

**Natural Language Processing (NLP):** NLP is a field of AI that focuses on helping computers work with human language, including understanding text, translating languages, and answering questions.

### Real-Life Example: Email Spam Detection

A good example of how AI, Machine Learning, and NLP work together is email spam detection.

When someone receives an email, NLP helps the system process and analyze the words and sentences in the message. Machine Learning uses patterns learned from previously labeled emails to predict whether a new email is spam or not. Finally, the AI system uses this prediction to help organize the email into the correct folder.

This example shows how NLP and Machine Learning can work together as part of an intelligent AI system.

---

## 2. Explain the Terms: Corpus, Document, Token, Vocabulary, Feature, and Label

In NLP, text data is organized into different parts before it can be used to train a Machine Learning model.

To understand these terms, consider a small dataset containing Somali customer reviews about a mobile phone shop.

### Example Dataset

| Document ID | Customer Review | Label |
|---|---|---|
| D1 | good work | Positive |
| D2 | the mopile is expensive | Negative |
| D3 | workers are a good | Positive |
| D4 | work is a bad | Negative |

### 2.1 Corpus

A corpus is a collection of written or spoken language data used for NLP research and analysis.

In this example, all four customer reviews together form one corpus.

### 2.2 Document

A document is one individual piece of text within a corpus. It can be a sentence, customer review, email, or article.

For example:

`Adeeg fiican ayaan helay`

This customer review is one document.

### 2.3 Token

A token is a small unit of text produced when a sentence or document is divided into smaller parts. Tokens can be words, punctuation marks, or subword units.

For example:

Original sentence:

`Adeeg fiican ayaan helay`

Tokens:

`["Adeeg", "fiican", "ayaan", "helay"]`

This sentence contains four word tokens.

### 2.4 Vocabulary

Vocabulary is the collection of unique tokens found in a corpus. Repeated words are counted only once.

For example, some words from our dataset are:

`["Adeeg", "fiican", "ayaan", "helay", "Telefoonka", "waa", "qaali"]`

The complete vocabulary contains all unique words from the four reviews.

### 2.5 Feature

A feature is an input variable or characteristic that a Machine Learning model uses to learn patterns and make predictions.

For example, the frequency of the word `fiican` can be used as a feature when analyzing customer reviews.

Other possible features include the number of words in a review and the frequency of positive or negative words.

### 2.6 Label

A label is the correct answer or target value that a supervised Machine Learning model tries to predict.

In our dataset:

- Positive means the customer is satisfied.
- Negative means the customer is dissatisfied.

For example, the review `Adeeg fiican ayaan helay` has the label `Positive`.

### Summary Table

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of texts | All four reviews |
| Document | One individual text | Review D1 |
| Token | A unit of text | Adeeg |
| Vocabulary | All unique tokens | fiican, qaali |
| Feature | Input used by a model | Word frequency |
| Label | Expected prediction | Positive |

---

## 3. Describe Four Common NLP Tasks

NLP systems perform different tasks depending on the problem they are designed to solve. Four common NLP tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is the process of assigning a category or label to an entire text document.

It is commonly used in spam detection, sentiment analysis, and topic classification.

**Example:**

Input: `The service was excellent.`

Output: `Positive`

- Input: A sentence, review, email, or document.
- Output: A predicted category or label.

The model learns from previously labeled examples and uses the learned patterns to classify new text.

### 3.2 Sequence Labeling

Sequence labeling is a task in which a model assigns a label to each token in a sentence instead of assigning one label to the entire document.

One example is Named Entity Recognition (NER), which identifies names of people, places, and organizations.

**Example:**

Sentence: `Ahmed lives in Mogadishu.`

| Token | Label |
|---|---|
| Ahmed | PERSON |
| lives | O |
| in | O |
| Mogadishu | LOCATION |

Here, O means the word does not belong to one of the selected named-entity categories.

- Input: A sequence of words or tokens.
- Output: A corresponding label for each token.

### 3.3 Search and Similarity

Search and similarity involve finding documents that are relevant or similar to a user's query.

A search system analyzes a query and retrieves documents that contain useful information.

**Example:**

Input: `How can I buy a smartphone?`

Output: A list of documents about purchasing smartphones.

- Input: A search query and a collection of documents.
- Output: A ranked list of relevant documents.

This task is used in search engines, information retrieval systems, and recommendation applications.

### 3.4 Text Generation

Text generation is the process of producing new human-readable text based on an input, instruction, or previous text.

It is used in chatbots, translation systems, and writing assistants.

**Example:**

Input: `What is Artificial Intelligence?`

Output: `Artificial Intelligence is a field of computer science that develops intelligent computer systems.`

- Input: A prompt, question, or previous text.
- Output: Newly generated text.

Modern language models use neural networks, especially Transformer architectures, to learn language patterns and generate responses.

### Comparison of NLP Tasks

| NLP Task | Input | Output |
|---|---|---|
| Text classification | Complete text | One category |
| Sequence labeling | Sequence of tokens | Label for each token |
| Search and similarity | Query and documents | Ranked relevant documents |
| Text generation | Prompt or context | New text |

---

## 4. Compare Supervised and Unsupervised Learning in NLP

Machine Learning is an important part of NLP. Two common learning approaches are supervised learning and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from training data that contains input features and correct target labels.

The model studies the relationship between the inputs and their correct answers. After training, it can predict labels for new examples.

**Example: Somali Sentiment Analysis**

Imagine we have 1,000 Somali customer reviews. Each review has a label that shows whether the customer's opinion is positive or negative.

| Customer Review | Label |
|---|---|
| Adeeg fiican | Positive |
| Adeeg xun | Negative |
| Aad baan ugu faraxsanahay | Positive |
| Kuma qanacsani | Negative |

The model learns from these examples and can predict the sentiment of a new customer review.

**Why are labels needed?**

Labels are needed because the model must learn which answers are correct during training. They help the model understand the relationship between text and its intended category.

### 4.2 Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model receives data without predefined target labels.

Instead of learning from known answers, the model tries to discover hidden patterns, similarities, or groups in the data.

**Example: Grouping Somali Customer Reviews**

Imagine collecting 1,000 Somali customer reviews without assigning positive or negative labels.

An unsupervised clustering algorithm, such as K-Means, can group similar reviews based on their numerical representations.

Possible groups might include:

- Group 1: Reviews discussing customer service.
- Group 2: Reviews discussing product prices.
- Group 3: Reviews discussing delivery problems.

These groups are discovered from similarities in the data rather than predefined answers.

**Why are target labels not required?**

Target labels are not required because the purpose of unsupervised learning is to discover patterns and relationships within the data. The algorithm works with the input data without being given correct target answers.

### Comparison Table

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labeled data | Uses data without target labels |
| Learns from known answers | Discovers hidden patterns |
| Predicts predefined categories or values | Finds groups or structures |
| Example: Sentiment classification | Example: Customer review clustering |

The main difference is that supervised learning learns from examples with known answers, while unsupervised learning discovers patterns without those answers.

---

## 5. Why Must Text Be Converted into Numerical Representations?

Most traditional Machine Learning algorithms cannot directly process ordinary human-readable text as numerical input.

For example, a computer can perform mathematical operations on numbers such as 5, 10, and 20. However, words such as `good`, `bad`, and `excellent` cannot be directly used as numerical input by most traditional Machine Learning models.

Therefore, text must be converted into numerical representations before it can be processed by these models.

This process is called text vectorization or text representation.

### Bag of Words (BoW)

Bag of Words is a simple text representation method that converts text into numerical vectors based on the occurrence or frequency of words.

It creates a vocabulary from a collection of documents and counts how many times each vocabulary word appears in each document.

**Example:**

Consider these three sentences:

- D1: I like machine learning.
- D2: I like NLP.
- D3: I study machine learning.

First, create a vocabulary:

`["I", "like", "machine", "learning", "NLP", "study"]`

Next, count the words in each document.

| Document | I | like | machine | learning | NLP | study |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 1 | 0 | 0 |
| D2 | 1 | 1 | 0 | 0 | 1 | 0 |
| D3 | 1 | 0 | 1 | 1 | 0 | 1 |

The result is a numerical matrix that a Machine Learning model can process.

### Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and much of the meaning and context of language.

For example:

- Sentence 1: The dog chased the cat.
- Sentence 2: The cat chased the dog.

Both sentences contain the same words, so they can produce the same basic Bag of Words representation, even though their meanings are different.

Another limitation is that large vocabularies can create high-dimensional and sparse numerical matrices.

More advanced methods, such as TF-IDF, Word2Vec, and Transformer-based embeddings, can provide more informative text representations.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A low-resource language is a language that has relatively limited digital text data, annotated datasets, linguistic tools, and other resources available for developing NLP systems.

Somali is an example of a language that has faced these challenges. Many NLP technologies have received more research and development attention for languages such as English than for Somali.

Several challenges can affect the development and performance of Somali NLP systems.

### 6.1 Limited Datasets and Linguistic Resources

One major challenge is the limited availability of large, high-quality, and properly annotated Somali text datasets.

Many Machine Learning models require sufficient training data to learn reliable language patterns.

For example, a sentiment analysis system needs Somali customer reviews that have been correctly labeled as positive, negative, or neutral.

When only a small dataset is available:

- The model may not learn enough language patterns.
- Predictions may become less reliable.
- The model may perform poorly on new examples.
- It may overfit the limited training data.

A lack of linguistic resources, such as dictionaries, tokenizers, and annotated corpora, also makes text processing more difficult.

### 6.2 Spelling Variation and Writing Differences

Somali text can contain spelling differences, informal writing, typing errors, and inconsistent word forms.

For example, people may write the same expression differently in social media messages and informal communication.

A computer may treat these different written forms as separate vocabulary items, even when they express the same meaning.

This can create several problems:

- The vocabulary becomes unnecessarily large.
- Similar words may be treated as unrelated.
- Important language patterns may be missed.
- Search and text classification accuracy may decrease.

Text normalization can help reduce these problems by standardizing writing where appropriate.

### 6.3 Morphological Complexity and Word Forms

Somali has morphological processes that create different forms of words through grammatical changes and word formation.

This means that a word can appear in different related forms depending on its grammatical use.

If an NLP system cannot recognize these relationships, it may treat related word forms as completely different words.

This can affect:

- Tokenization.
- Vocabulary size.
- Search results.
- Text classification.
- Machine translation.

Lemmatization and morphological analysis can help identify relationships between different forms of words.

### 6.4 Mixed-Language Text

Somali speakers sometimes combine Somali and English in the same message, especially in social media and online communication. This is known as code-switching.

For example:

`Maanta waxaan haystaa new phone.`

This sentence contains both Somali and English.

A system trained only on formal Somali may struggle to process such messages because it may not recognize all the words and language structures.

This can reduce the accuracy of sentiment analysis, language identification, and other NLP applications.

### Summary of Challenges

| Challenge | Effect on NLP Systems | Possible Solution |
|---|---|---|
| Limited datasets | Poor model performance | Collect and annotate more data |
| Spelling variation | Duplicate vocabulary and missed patterns | Text normalization |
| Morphological complexity | Difficulty recognizing related words | Lemmatization and morphological analysis |
| Mixed-language text | Incorrect processing and classification | Multilingual and code-switching-aware models |

These challenges show why developing NLP technology for Somali requires language-specific research, high-quality datasets, and suitable preprocessing techniques.

---

## 7. Case Study: Somali Information Retrieval System

A useful research example is the paper titled:

**Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources**

The paper was written by Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou in 2023. It was published in the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

### 7.1 Research Problem

The researchers identified a problem: Somali information retrieval research was limited because there were not enough dedicated Somali-language datasets.

Many existing systems relied on translating Somali search queries into other languages instead of searching directly using Somali resources.

The researchers wanted to create a dedicated Somali information retrieval resource to help solve this problem.

### 7.2 Data Used

The researchers created an annotated Somali information retrieval corpus containing 2,335 documents.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo Net.
- Somali poetry books.

Language experts and NLP researchers collaborated in constructing the dataset.

### 7.3 Method Used

The researchers developed an information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF)**.

The general process works as follows:

1. A user submits a search query.
2. The system retrieves documents that appear relevant.
3. The system examines the initially retrieved documents.
4. It identifies additional useful terms from those documents.
5. It expands the original query and performs another search.

This method attempts to improve search results by adding related terms to the original query.

### 7.4 Findings

The research contributed a dedicated Somali information retrieval corpus and demonstrated an approach for developing Somali-language search systems.

The study highlighted the importance of creating language-specific datasets because limited digital resources are a major challenge for Somali NLP development.

It also provided a useful resource for future research in information retrieval, question answering, and text classification.

### 7.5 Which NLP Task Does It Cover?

This case study covers **Search and Similarity (Information Retrieval)**, which is the third NLP task discussed in Question 3.

The system receives a user's search query and retrieves relevant documents from a Somali text collection.

For example:

**Input:** A Somali search query about education.

**Output:** A ranked collection of relevant Somali documents.

This research demonstrates how NLP can help people search for and access information written in their own language.

---

---

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media.  
   https://www.nltk.org/book/

2. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed., online draft). Stanford University.  
   https://web.stanford.edu/~jurafsky/slp3/

3. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.  
   https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing/

4. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469.  
   https://doi.org/10.18653/v1/2023.emnlp-main.462

5. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-based Word Lemmatization Approach for the Somali Language.  
   https://arxiv.org/abs/2308.01785

6. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. *Proceedings of the 7th Workshop on African Natural Language Processing*.  
   https://doi.org/10.18653/v1/2026.africanlp-main.17
