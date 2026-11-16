# Assignment 1: NLP Foundations

# 1. Natural Language Processing and Its Relationship to AI and Machine Learning

**Natural Language Processing (NLP)** is the field of computing concerned with processing, analyzing, understanding, and generating human language. NLP can be used for applications such as machine translation, text classification, information retrieval, question answering, and text generation (Jurafsky & Martin, 2024; Eisenstein, 2019).

NLP is closely connected to **Artificial Intelligence (AI)** because understanding or generating human language is considered an intelligent task. AI is the broader field, while NLP focuses specifically on human language.

**Machine Learning (ML)** is one of the main approaches used to build NLP systems. Instead of manually writing rules for every possible situation, a machine-learning model can learn patterns from examples (Google for Developers, 2026).

A simple way to understand the relationship is:

**Artificial Intelligence → Machine Learning → many NLP applications**

However, not every NLP system has to use machine learning. A rule-based system can also perform an NLP task without learning from data (Eisenstein, 2019).

### Real-life example: Customer support ticket routing

Consider an online shopping company that receives thousands of customer support messages every day. The company wants to automatically send each message to the correct department.

For example:

> "My package arrived damaged and I would like a replacement."

The system could classify this as a **Product/Delivery Complaint**.

The three areas work together in the following way:

1. **AI:** The overall goal is to automate an intelligent decision that would normally require a human support agent.
2. **NLP:** The system processes the customer's natural-language message and identifies what the message is about.
3. **ML:** A classification model can learn from previously categorized support messages and use those patterns to classify new messages.
4. **Prediction:** When a new message arrives, the system converts its text into numerical features and predicts the appropriate department.

Therefore, AI provides the broader intelligent objective, NLP defines the language problem, and ML provides a way for the system to learn patterns from examples.

---

# 2. Corpus, Document, Token, Vocabulary, Feature, and Label

To explain these concepts using one dataset, consider a small **online shopping review dataset**. These concepts are fundamental to how NLP datasets are represented and processed (Jurafsky & Martin, 2024; Eisenstein, 2019).

| Document ID | Text | Label |
|---|---|---|
| D1 | "The delivery was very fast." | Positive |
| D2 | "The package arrived damaged." | Negative |
| D3 | "The quality is excellent." | Positive |
| D4 | "The product stopped working." | Negative |

## Corpus

A **corpus** is a collection of text documents created or collected for a particular purpose (Jurafsky & Martin, 2024).

In this example, the four shopping reviews together form the corpus.

**Corpus = {D1, D2, D3, D4}**

A real NLP corpus can contain thousands or millions of documents.

## Document

A **document** is one individual piece of text within a corpus.

For example:

> "The delivery was very fast."

is one document, D1.

Depending on the application, a document could be a review, email, social-media post, article, or another unit of text.

## Token

A **token** is a unit of text produced by a tokenizer.

For example, D1 could be divided into:

> ["The", "delivery", "was", "very", "fast"]

Each item is a token under this word-level tokenization approach.

A token does not always have to be a complete word. NLP systems can also use subword or character-based tokens (Jurafsky & Martin, 2024).

## Vocabulary

A **vocabulary** is the collection of unique token types recognized from the corpus after the chosen preprocessing and tokenization steps.

For example, a simplified vocabulary from the dataset could include:

> {the, delivery, was, very, fast, package, arrived, damaged, quality, is, excellent, product, stopped, working}

If the word **the** appears several times, it still appears only once in the vocabulary because the vocabulary contains unique entries.

## Feature

A **feature** is a numerical value that represents some property of a text and can be used as input to a machine-learning model (Eisenstein, 2019).

For example, with a Bag-of-Words representation, a feature could represent whether the word **damaged** occurs in a document.

For D2:

> `damaged = 1`

The value is numerical, allowing a machine-learning model to use it as an input.

## Label

A **label** is the target value assigned to an example in a supervised-learning problem.

In this dataset:

- D1 → Positive
- D2 → Negative
- D3 → Positive
- D4 → Negative

Here, **Positive** and **Negative** are the possible classes, while the label is the specific class assigned to each document.

### Summary

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of text | All four reviews |
| Document | One text item | D1 |
| Token | Unit produced by tokenization | `delivery` |
| Vocabulary | Unique token types | `{delivery, package, quality, ...}` |
| Feature | Numerical input for a model | `damaged = 1` |
| Label | Target assigned to an example | `Negative` |

The distinction between these terms is important because a dataset can contain many documents while having only a small number of possible classes (Jurafsky & Martin, 2024).

---

# 3. Four Common NLP Tasks

NLP includes many different tasks. Four important examples are text classification, sequence labeling, search and similarity, and text generation (Jurafsky & Martin, 2024; Eisenstein, 2019).

## 3.1 Text Classification

**Text classification** assigns one or more predefined categories to an entire piece of text.

### Input

A document, message, review, or sentence.

### Output

A class label.

For example:

> **Input:** "I cannot log into my account."

> **Output:** `Account Access`

Other examples include:

- Sentiment classification
- Spam detection
- Topic classification
- Customer-intent classification

The output represents the category assigned to the whole text.

---

## 3.2 Sequence Labeling

**Sequence labeling** assigns labels to individual parts of a text sequence, usually tokens.

For example, consider:

> "Hodan works at Hormuud."

A named-entity recognition system could produce:

| Token | Label |
|---|---|
| Hodan | PERSON |
| works | O |
| at | O |
| Hormuud | ORGANIZATION |

Here, the system does not give one label to the entire sentence. Instead, it identifies which tokens belong to particular categories.

One common sequence-labeling task is **Named Entity Recognition (NER)**, which identifies entities such as people, organizations, locations, dates, and other predefined types (Jurafsky & Martin, 2024).

---

## 3.3 Search and Similarity

Search and similarity systems are used to find information that is relevant or similar to a user's request.

### Input

A user query and a collection of documents.

For example:

> **Query:** "computer science scholarships"

### Output

A ranked list of documents that are considered relevant to the query.

A similarity system can also compare two pieces of text and produce a similarity score. Information retrieval and text similarity are important applications of NLP (Eisenstein, 2019).

For example, a university website could use text similarity to find articles that are related to a student's search query.

---

## 3.4 Text Generation

**Text generation** produces new natural-language text based on an input, prompt, or source information.

### Input

A prompt or source text.

For example:

> "Translate this paragraph from Somali into English."

### Output

A newly generated text in the target language.

Applications include:

- Machine translation
- Text summarization
- Question answering
- Dialogue systems
- Writing assistance

Unlike classification, where the output normally comes from a predefined set of labels, text generation produces a sequence of tokens (Jurafsky & Martin, 2024).

### Comparison

| NLP Task | Typical Input | Typical Output |
|---|---|---|
| Text classification | A document | Class label |
| Sequence labeling | A sequence of tokens | Label for each token |
| Search and similarity | Query + document collection | Ranked documents or similarity score |
| Text generation | Prompt or source text | Newly generated text |

---

# 4. Supervised vs. Unsupervised Learning in NLP

The main difference between supervised and unsupervised learning is whether the training data contains target answers for the task (Google for Developers, 2026; Jurafsky & Martin, 2024).

## Supervised Learning

In **supervised learning**, each training example has an input and a corresponding target label.

For example:

| Customer Message | Label |
|---|---|
| "I want to return this item." | Return |
| "Where is my order?" | Delivery |
| "I need to change my password." | Account |

The model learns relationships between the text and the target categories.

After training, it may receive a new message:

> "I need to send this product back."

and predict:

> **Return**

The labels are necessary because the model needs examples of the correct answers it is expected to learn.

## Unsupervised Learning

In **unsupervised learning**, the training data does not contain predefined target labels for the task.

For example, suppose a company has thousands of customer messages but does not know what topics they contain. An unsupervised clustering method could group messages according to similarities in their language.

The resulting groups might represent topics such as:

- Payments
- Delivery
- Technical problems
- Returns

The algorithm discovers patterns in the data rather than being told beforehand which group each message belongs to (Eisenstein, 2019).

### Why does supervised learning need labels?

A supervised learning problem has a target to predict:

**Input → Target**

For example:

**Customer message → Department**

The target labels tell the model what it should learn to predict.

In contrast, an unsupervised task may ask the system to discover structure in the data without specifying the correct answer in advance.

Therefore, supervised and unsupervised learning are not simply two versions of the same task. They address different learning objectives.

---

# 5. Why Text Must Be Converted into Numerical Representations

Most traditional machine-learning algorithms expect their inputs to be numerical feature vectors. Raw text, however, consists of words, symbols, and sequences rather than numerical measurements.

For example:

> "The laptop battery lasts long."

is meaningful to a human, but a conventional machine-learning model cannot directly use the sentence as a numerical feature vector.

Therefore, NLP systems often convert text into a numerical representation before giving it to a machine-learning model (Jurafsky & Martin, 2024; Eisenstein, 2019).

## Bag-of-Words

One simple representation is **Bag-of-Words (BoW)**. It represents a document using the words that occur in it, usually as counts or binary indicators (scikit-learn developers, 2026).

Suppose the vocabulary is:

> [laptop, battery, expensive]

For the sentence:

> "laptop battery"

a binary Bag-of-Words representation would be:

> **[1, 1, 0]**

The positions correspond to the vocabulary:

- `laptop` → 1
- `battery` → 1
- `expensive` → 0

For:

> "expensive laptop"

the representation would be:

> **[1, 0, 1]**

This gives the machine-learning model a fixed numerical representation for each document.

## Main limitation

A major limitation of basic Bag-of-Words is that it does not preserve word order.

For example:

> "The teacher praised the student."

and:

> "The student praised the teacher."

contain the same words, so a basic unigram Bag-of-Words representation can produce the same word counts even though the meanings are different (Eisenstein, 2019).

Bag-of-Words therefore represents word occurrence well, but it does not fully represent syntax, context, or relationships between words.

Another limitation is that large vocabularies can produce high-dimensional and sparse feature matrices (scikit-learn developers, 2026).

These limitations are among the reasons NLP has developed richer representations such as word embeddings, contextual representations, and subword representations (Jurafsky & Martin, 2024).

---

# 6. Challenges of NLP for Low-Resource Languages Such as Somali

A **low-resource language** is a language for which computational resources such as large corpora, annotated datasets, linguistic tools, pretrained models, and evaluation benchmarks are relatively limited compared with high-resource languages (Eisenstein, 2019).

Somali is considered an under-resourced language for NLP, despite being widely spoken. Research on Somali has identified several challenges that affect the development and performance of NLP systems.

## 6.1 Limited Annotated Data

One major challenge is the limited availability of high-quality annotated datasets.

Supervised NLP models require labeled examples for training and evaluation. If only a small amount of reliable labeled data exists, models have fewer examples from which to learn linguistic patterns.

This can lead to:

- Lower model performance
- Poor generalization to new text
- Difficulty evaluating systems reliably
- Greater dependence on transfer learning and multilingual resources

Recent Somali research on lemmatization specifically identifies the lack of human-annotated datasets and reliable ground truth as an important research challenge. The researchers developed a Somali lemmatization resource containing more than **5,584 root words and 78,629 derivative forms** (Gedi et al., 2026).

This shows that the problem is not simply a lack of written Somali. The bigger issue is the lack of **well-structured and annotated resources** that can be directly used for NLP research.

## 6.2 Spelling Variation

Another challenge is variation in how Somali words are written.

A word-based NLP system may treat two different spellings as two separate vocabulary entries, even when they refer to the same word.

For example, the Somali word for Prime Minister can appear in different forms such as:

> `ra'iisulwasaare`

and

> `raysulwasaare`

Research on Somali information retrieval identified spelling variation as one of the difficulties affecting retrieval.

### Effect on NLP systems

Spelling variation can cause:

- More out-of-vocabulary words
- Missed search results
- Fragmented word statistics
- Lower classification or retrieval performance

This is particularly difficult for low-resource languages because there may not be enough data for a model to learn every common variation.

## 6.3 Morphological Complexity

Somali also presents challenges related to morphology and word formation. Different grammatical or derived forms of words can create many surface forms.

If a basic word-based system treats each form as a completely separate token, related words may not share enough information during learning.

This can increase vocabulary size and make individual forms less frequent.

Recent Somali research describes the language's agglutinative and derivational morphology and shows why lemmatization resources are useful for reducing this variation (Gedi et al., 2026).

### Effect on NLP systems

Morphological variation can result in:

- Larger vocabularies
- More rare or unseen words
- Difficulty identifying related word forms
- Reduced performance on new text

Techniques such as lemmatization, stemming, subword representations, and morphology-aware processing can help address these problems.

## 6.4 Mixed-Language Text

Somali speakers may also combine Somali with English or other languages, especially in informal digital communication.

For example:

> "Waxaan update-gareeyay application-ka."

A system trained only on standard Somali or only on English may not process this type of sentence correctly.

Mixed-language text can create problems with:

- Tokenization
- Language identification
- Vocabulary coverage
- Morphological analysis
- Model generalization

This shows why NLP systems should be evaluated on realistic language use rather than only carefully edited text.

---

# 7. Case Study: Somali-English Machine Translation

## Study

A relevant Somali NLP case study is:

**Duh, K., McNamee, P., Post, M., & Thompson, B. (2020). "Benchmarking Neural and Statistical Machine Translation on Low-Resource African Languages." Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020).**

The paper studies machine translation for **Somali and Swahili**, focusing on how statistical and neural translation systems perform when only limited training data is available (Duh et al., 2020).

### NLP Task

This study covers **text generation**, specifically **machine translation**.

The input is a sentence in Somali, and the output is a newly generated English translation.

---

## Problem

Machine translation systems usually perform better when they have access to large amounts of parallel training data. However, many African languages do not have the millions of sentence pairs available for languages such as English.

The researchers therefore investigated how **Statistical Machine Translation (SMT)** and **Neural Machine Translation (NMT)** perform under low-resource conditions, focusing on Somali-English and Swahili-English translation (Duh et al., 2020).

---

## Data

For Somali-English, the baseline training data contained approximately:

- **24.8K sentence pairs**
- **758K Somali-English words**

The validation set contained about 2.6K sentences, while the first test set contained about 4.0K sentences. The researchers also used a second test set containing speech transcripts to test performance on a different type of data (Duh et al., 2020).

The paper describes the data as coming from the IARPA MATERIAL program and covering different genres.

---

## Method

The researchers compared two main approaches:

- **Statistical Machine Translation (SMT)**
- **Neural Machine Translation (NMT)**

The SMT system used a phrase-based approach, while the NMT system used a Transformer architecture.

The researchers also tested different NMT hyperparameters, including model architecture, embedding size, attention heads, learning rate, and Byte Pair Encoding settings. This was important because neural models can be sensitive to these choices when training data is limited (Duh et al., 2020).

They also investigated whether additional resources could improve translation. These included:

- Web-mined parallel text
- Dictionaries
- Additional existing parallel text

---

## Findings

With the baseline low-resource training data, the final Somali-English systems achieved similar but slightly different BLEU scores:

| System | Somali-English BLEU |
|---|---:|
| SMT | 15.1 |
| NMT | 14.4 |

The researchers found that NMT required careful hyperparameter tuning to reach competitive performance in this low-resource setting.

More importantly, adding additional resources substantially improved NMT performance. When dictionary resources, additional parallel text, and web-mined parallel data were combined, Somali-English NMT reached a BLEU score of **25.0**, compared with **14.4** for the baseline NMT system (Duh et al., 2020).

This represents a substantial improvement and demonstrates how strongly low-resource NLP systems can benefit from additional language data.

---

## Why This Case Study Matters

This study is relevant to Somali NLP because it shows that the main challenge is not simply choosing a more advanced model. **Data availability is also a major factor.**

The results show that neural models can work with limited Somali-English data, but they require careful tuning and benefit significantly from additional resources (Duh et al., 2020).

It also connects directly to the challenges discussed in Question 6. Limited parallel data makes Somali machine translation difficult, while collecting and combining additional language resources can improve performance.

---

# Conclusion

NLP enables computers to work with human language and is closely connected to both Artificial Intelligence and Machine Learning. Understanding its foundations requires knowing how text is organized into documents and tokens, how vocabularies and features are created, and how labels are used in supervised learning (Jurafsky & Martin, 2024; Eisenstein, 2019).

Different NLP tasks have different inputs and outputs. Text classification produces a category for a document, sequence labeling assigns labels to parts of a sequence, search systems retrieve relevant information, and text generation produces new language.

Because most traditional machine-learning models cannot directly process raw text, NLP systems need numerical representations such as Bag-of-Words. However, simple representations have limitations, including their inability to preserve word order and contextual relationships (Eisenstein, 2019).

These challenges become more significant for low-resource languages such as Somali. Limited annotated data, spelling variation, morphological complexity, and mixed-language text can all affect NLP performance (Gedi et al., 2026).

The Somali-English machine translation case study demonstrates the importance of these issues in a real NLP application. It shows that low-resource translation can be challenging, but additional language resources and careful model development can substantially improve results (Duh et al., 2020).

---

# References

1. Duh, K., McNamee, P., Post, M., & Thompson, B. (2020). *Benchmarking Neural and Statistical Machine Translation on Low-Resource African Languages*. Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), 2667–2675. Association for Computational Linguistics.

2. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). *Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform*. Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026). Association for Computational Linguistics.

3. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., draft). Stanford University.

4. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.

5. scikit-learn developers. (2026). *Feature extraction: Text feature extraction and the Bag-of-Words representation*. scikit-learn documentation.

6. Google for Developers. (2026). *Machine Learning Crash Course: What is Machine Learning?* Google.