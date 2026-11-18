# Assignment 1 – NLP Foundations

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of computing concerned with enabling computers to work with human language. It includes tasks such as analyzing text, finding relevant documents, identifying names and entities, classifying messages, translating languages, and generating language.

NLP is related to Artificial Intelligence (AI) and Machine Learning (ML), but the three terms are not identical. **AI** is the broad field concerned with building systems that perform tasks associated with intelligence. **Machine Learning** is one approach within AI in which a model learns patterns from data rather than relying only on manually written rules. **NLP** focuses specifically on human language. Therefore, an NLP application may use ML as its learning method, although not every NLP system has to use ML.

### Real-life example

Consider an email filtering system:

1. **NLP** is used to process and represent the words and language in an email.
2. **Machine Learning** can learn patterns from previously labeled emails.
3. **AI** is the broader system that uses the learned model to make an automatic decision.

For example, after learning from many examples, the system can classify a new email as either *spam* or *not spam*.

---

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

These terms describe different parts of an NLP dataset. To show their relationship, consider a small movie-review dataset:

| ID | Review | Label |
|---|---|---|
| 1 | "The movie was excellent" | Positive |
| 2 | "The acting was poor" | Negative |
| 3 | "I enjoyed the movie" | Positive |

### Corpus

A **corpus** is a collection of text documents used for analysis or model development.

In this example, the three movie reviews together form the corpus.

### Document

A **document** is one individual piece of text in the corpus.

For example:

> "The acting was poor"

is one document.

### Token

A **token** is a unit produced when text is divided by a tokenizer. Depending on the tokenizer, a token can be a word, part of a word, character, or punctuation mark.

For example, a simple word-level tokenization of:

> "The movie was excellent"

could produce:

`["The", "movie", "was", "excellent"]`

### Vocabulary

A **vocabulary** is the collection of unique token types represented by a tokenizer or vectorizer.

For the three reviews, a simplified vocabulary could contain:

`the, movie, was, excellent, acting, poor, i, enjoyed`

Repeated words such as **the**, **movie**, and **was** are represented once in the vocabulary.

### Feature

A **feature** is a numerical value that represents some property of an example and can be used as input to a machine-learning model.

For example, with a Bag-of-Words representation, the feature corresponding to the word **movie** could be:

- Review 1: 1
- Review 2: 0
- Review 3: 1

Thus, the word "movie" becomes a numerical feature.

### Label

A **label** is the target answer assigned to an example in supervised learning.

In this dataset:

- Review 1 → **Positive**
- Review 2 → **Negative**
- Review 3 → **Positive**

The model learns from the relationship between the text features and these labels.

---

## 3. Four Common NLP Tasks

| NLP task | Input | Output | Example |
|---|---|---|---|
| Text classification | A complete text | One or more categories | Review → Positive/Negative |
| Sequence labeling | A sequence of tokens | A label for each token or part | "Amina visited Muqdisho" → Amina=Person, Muqdisho=Place |
| Search / similarity | Query and/or documents | Relevant or similar documents | "Somali agriculture" → related documents |
| Text generation | A prompt or input text | Newly generated text | Question → generated answer |

### 3.1 Text Classification

Text classification assigns a category to an entire text. The categories may be binary, such as positive/negative, or multiclass, such as sports/politics/business.

**Input:** a movie review  
**Output:** `Positive`

### 3.2 Sequence Labeling

Sequence labeling assigns labels to individual tokens or spans within a text. Named Entity Recognition (NER) is a common example.

**Input:**  
`Amina visited Muqdisho.`

**Output:**  
`Amina → Person`  
`Muqdisho → Place`

### 3.3 Search and Similarity

Search and similarity systems identify documents or texts that are relevant or semantically related to a query.

**Input:**  
`Somali agriculture`

**Output:**  
A ranked list of documents about Somali agriculture.

### 3.4 Text Generation

Text generation produces new language from an input prompt or context. Examples include generating an answer, summary, translation, or other text.

**Input:**  
`Explain climate change in simple language.`

**Output:**  
A newly generated explanation.

---

## 4. Supervised vs. Unsupervised Learning in NLP

### Supervised learning

In supervised learning, each training example has a target answer, called a **label**. The model learns a relationship between the input features and the known target labels.

**NLP example:** sentiment classification.

Training data might contain:

| Text | Label |
|---|---|
| "The service was excellent." | Positive |
| "The service was terrible." | Negative |

The labels are necessary because the model needs to learn what the desired output is for each training example.

### Unsupervised learning

In unsupervised learning, the training data does not contain a target label for each example. The algorithm attempts to discover useful patterns, structures, or relationships in the data.

**NLP example:** grouping news articles according to similarity without giving the algorithm predefined topic labels.

For example, a system may represent articles as vectors and discover groups of articles that are similar to each other.

### Comparison

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Target labels | Required during training | Not required |
| Main goal | Learn to predict a known target | Discover patterns or structure |
| NLP example | Sentiment classification | Document clustering |
| Evaluation | Can compare predictions with known targets | Often evaluates discovered structure or usefulness |

---

## 5. Why Does Text Need to Become Numerical?

Most traditional machine-learning algorithms cannot directly process raw text as a variable-length sequence of human-readable symbols. They generally require numerical feature vectors with a consistent structure.

A common solution is **Bag-of-Words (BoW)**. BoW creates a vocabulary and represents each document using the number or presence of vocabulary terms.

For example, suppose the vocabulary is:

`[apple, book, school]`

The sentence:

> `apple school`

could be represented as:

`[1, 0, 1]`

The values are numerical features that a machine-learning algorithm can process.

### Limitation

Bag-of-Words mainly represents token occurrence and does not naturally capture word order or deeper context. For example:

> "The dog chased the cat."

and

> "The cat chased the dog."

can have very similar word-count representations even though their meanings are different.

Therefore, numerical representation is necessary for many ML models, but the choice of representation affects what information the model can learn.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Somali is a low-resource language for NLP compared with languages such as English. Research has identified several challenges.

### 6.1 Limited datasets and annotated resources

A major challenge is the limited availability of large, high-quality, human-annotated datasets and other digital language resources.

Badel et al. (2023) note that Somali information retrieval research had often relied on query translation because of the lack of a dedicated Somali information-retrieval corpus. They developed an annotated corpus containing **2,335 documents** to address this gap.

The effect is important: when fewer labeled examples are available, it becomes harder to train and evaluate reliable NLP models for tasks such as classification, question answering, and information retrieval.

### 6.2 Spelling variation

Somali text can contain different spellings for the same word. Earlier corpus research documented spelling variation such as **madaxweyne** and **madaxwayne**.

This can cause a system to treat two forms of the same underlying word as different tokens. As a result, the vocabulary becomes unnecessarily larger and a model may have fewer examples for each form.

### 6.3 Morphological and word-form variation

Somali has morphological processes that produce many related word forms. This creates a challenge for tokenization, normalization, lemmatization, and other NLP components.

Recent Somali lemmatization research describes Somali as having agglutinative and derivational morphology and reports a large lexicon containing thousands of roots and many derivative forms.

If related forms are not handled properly, a model may fail to connect words that are linguistically related. This can reduce the usefulness of text representations and downstream NLP systems.

### 6.4 Mixed-language and informal text

Somali users may also communicate in informal digital environments where Somali is mixed with English or other languages. Such text can make tokenization, vocabulary construction, language identification, and downstream modeling more difficult.

### Overall effect

These challenges can increase vocabulary sparsity, produce out-of-vocabulary words, reduce the amount of useful training data per form, and make it harder to build robust NLP models that generalize across different sources and writing styles.

---

## 7. Case Study: Somali Information Retrieval Corpus

### Research paper

**Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). _Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources._ Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 7463–7469.**

### Problem / Task

The paper addresses **information retrieval for Somali**. The researchers explain that Somali information-retrieval research had been limited and had often depended on translating queries into other languages because dedicated Somali resources were lacking.

### Data

The researchers created an annotated Somali information-retrieval corpus containing **2,335 documents**. The documents were collected from several online sources, including Somali news websites and Somali poetry books.

### Method

The researchers developed a Somali information-retrieval system and used **pseudo-relevance feedback (PRF) query expansion**. The basic idea is to use information from initially retrieved documents to expand or improve the query so that the retrieval system can find more relevant Somali documents.

### Findings / Contribution

The main contribution was the creation of a dedicated Somali information-retrieval corpus and a retrieval system built around it. The work demonstrates how creating language-specific resources can help address the lack of digital datasets for Somali NLP.

The corpus can also support future research and development of Somali NLP applications, including information retrieval, question answering, and text classification.

### Connection to Question 3

The case study belongs primarily to the **search / similarity** NLP task from Question 3 because the system is designed to retrieve documents that are relevant to a user's information need.

---

## Conclusion

NLP connects human language with computational methods. Core concepts such as corpus, document, token, vocabulary, feature, and label describe how text data is organized and prepared for machine learning. Common NLP tasks include classification, sequence labeling, search/similarity, and generation.

For low-resource languages such as Somali, limited datasets, spelling variation, morphological variation, and informal or mixed-language text create additional difficulties. Research such as the Somali Information Retrieval Corpus shows that building dedicated Somali resources is an important step toward developing stronger NLP systems for Somali.

---

## References

1. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 7463–7469.

2. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). *Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform*. Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026), 179–189.

3. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed., August 2026 draft). Stanford University.

4. Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and Rule-Based Word Lemmatization Approach for Somali Language*. ICLRW 2023.

5. Mohamed, S. A., & others. (2014). *Building and Evaluating Somali Language Corpora*. Proceedings of the Workshop on Language Technology for African Languages.

6. scikit-learn developers. (2026). *Text feature extraction*. scikit-learn documentation.
