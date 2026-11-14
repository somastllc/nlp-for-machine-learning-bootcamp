# NLP Foundations

## Goal

Understand the **foundations of NLP**: what NLP is, the basic terms for text data (corpus, document, token, vocabulary, feature, label), the common NLP tasks, supervised and unsupervised learning with text, and why language must become numbers before a model can use it.

## 1. What is Natural Language Processing (NLP)?

**Natural Language Processing (NLP)** is a field of computing that helps computers work with human language, such as Somali, English, and Arabic. NLP systems can analyze text, search for similar documents, identify names, classify messages, translate sentences, and generate new text.

Imagine you receive two SMS messages. The first says, "Are we meeting tomorrow?" The second says, "Congratulations! Claim your free prize now!" A person may recognize the second message as suspicious. But how can a computer make a similar decision? We can use NLP techniques to represent the messages and train a classifier to predict whether each is **spam** (unwanted) or **ham** (normal).

NLP is not only about modern chatbots or Large Language Models (LLMs). Simple keyword rules, search tools, and traditional Machine Learning classifiers are also part of NLP.

### Where NLP fits within AI and Machine Learning

**Artificial Intelligence (AI)** is the broad field of building systems that perform tasks associated with intelligence. **Machine Learning (ML)** is one approach: a model learns patterns from data instead of relying only on rules written by a programmer. **NLP** is the area focused on human language.

Some NLP systems use Machine Learning, and some do not. A program that looks for a particular word is a rule-based NLP system. A spam detector trained on labeled messages is a Machine Learning-based NLP system. In this bootcamp, we will mainly study ML-based NLP.

## 2. Core terms you must know

| Term | Simple meaning | Example |
|---|---|---|
| **NLP** | Making computers work with human language | Classifying an SMS as spam or ham |
| **Corpus** | A collection of text documents | All 10,000 SMS messages in a dataset |
| **Document** | One piece of text we treat as an example | One SMS, review, email, or article |
| **Label** | The answer assigned to a particular example | The message's label is `spam` |
| **Class** | One possible category the model can predict | `spam` is one class; `ham` is another |
| **Token** | One unit of text produced by a tokenizer | `Python`, `ing`, or `!` |
| **Vocabulary** | The collection of unique token types recognized by a representation | `['I', 'love', 'Python']` |
| **Vocabulary size** | The number of entries in that vocabulary | The vocabulary above has size **3** |
| **Feature** | A numerical value used as model input | The count of the word `prize` |
| **Vectorizer** | A method that turns text into numerical features | Bag of Words or TF-IDF |
| **Training** | Learning patterns from examples | Fitting a spam classifier on labeled messages |
| **Inference** | Using a trained model on new input | Predicting whether a new SMS is spam |

### Corpus, documents, classes, and labels: one example

Suppose we collect **10,000 SMS messages**. The entire collection is our **corpus**, and each individual SMS is one **document**. For a supervised spam classification task, every training message needs a **label**.

| Document (text) | Label |
|---|---|
| "Are you coming today?" | `ham` |
| "Claim your cash prize!" | `spam` |
| "Call me when you arrive." | `ham` |
| "Win money now!" | `spam` |

There are **four labeled documents** in this small table, but only **two classes**: `spam` and `ham`. A class is a possible category; a label is the category assigned to a specific example. When the trained model predicts a new message's category, we call that its **predicted label**.

### Token vs. vocabulary: why we say “token” instead of “word”

A **token** is simply a piece of text that our tokenizer works with. If we choose a *word-level tokenizer*, a token can be a complete word. We use the word *token* because other tokenizers may use **parts of words, characters, or punctuation**. So a token does not always mean a whole word.

Take this sentence:

> Python Python is useful.

A simple tokenizer that separates words and punctuation could produce:

`['Python', 'Python', 'is', 'useful', '.']`

That is **5 token occurrences**. Notice that `Python` appears twice, so it counts twice in the token count. But the unique entries are `['Python', 'is', 'useful', '.']`: the **vocabulary size is 4** for this toy example.

**Vocabulary** means *which unique tokens are available*. **Vocabulary size** means *how many unique entries there are*. For example, a tokenizer with vocabulary size **48K** has about **48,000 distinct entries**; that does **not** mean a document can contain only 48,000 tokens. The same entries can be used over and over.

## 3. What does text data look like?

In traditional ML, you may work with **structured data**, such as a table with columns for age, income, and price. Text is often **unstructured**, such as a free-form review or message. Text can also be **semi-structured**, such as an email containing both a message body and metadata like the sender and time.

For a supervised task, each example usually needs at least the **text** and its **label**. Extra information such as time, sender, or source is called **metadata**. Metadata can be useful, but we should not automatically use every metadata field as a feature. For example, a sender ID may help a model memorize people seen during training without learning how to recognize suspicious text.

## 4. Common NLP tasks

**Text classification** predicts a category for a whole text. For example, a spam detector predicts `spam` or `ham`, and a sentiment classifier predicts whether a review is positive or negative. With two possible classes it is **binary** classification; with several, such as `sports`, `business` and `politics`, it is **multiclass**.

**Sequence labeling** identifies information inside a text. For example, in "Amina lives in Muqdisho," a named-entity recognition system may label `Amina` as a person and `Muqdisho` as a place. Here, the labels apply to parts of the sentence, not to the entire document.

**Search and similarity** finds relevant or related text. For example, when someone searches "battery problems," a review-search system may return reviews that discuss batteries failing quickly.

**Text generation** produces new language. Examples include writing a summary, answering a question, or translating a sentence. These tasks involve different outputs, but all require techniques for processing language.

## 5. Supervised vs. unsupervised NLP

In **supervised learning**, every training example has a target answer. Suppose we collect messages and manually mark each one as `spam` or `ham`. The model sees the message and its correct label during training, then learns patterns that help it predict labels for unseen messages. We can evaluate it by comparing predictions against correct labels in a held-out test set.

In **unsupervised learning**, we do not provide target labels for the task. Imagine we have thousands of customer reviews but no positive/negative labels. We can still represent the reviews as vectors and find ones that are similar to a user's search query. The system returns a ranked list of related reviews instead of predicting a predefined sentiment label.

An important idea is that **the same text representation can be useful in both setups**. TF-IDF vectors can feed a supervised classifier or support document search. Word2Vec can learn word representations from surrounding text without spam labels; those representations can later help a supervised task.

## 6. What changes from tabular Machine Learning?

In tabular ML, a dataset might already contain the columns `age`, `salary`, and `experience`. A model can receive the same set of features for every row. With text, one message may be "Call me," while another may contain fifty words. Their lengths and vocabulary differ, so we need a consistent way to describe them numerically.

A **vectorizer** solves this for many classical ML models. It learns or uses a fixed set of feature positions and converts each message into numbers using that same layout. A **Bag-of-Words** vectorizer, for example, can represent how often vocabulary words appear. **TF-IDF** gives different weights to words depending on how common or distinctive they are across documents.

### A tiny numeric example

Suppose our vocabulary is:

`['free', 'meeting', 'prize']`

We create three feature positions in that exact order. A binary Bag-of-Words representation puts **1** where a word appears and **0** where it does not.

| Message | `free` | `meeting` | `prize` | Vector |
|---|---:|---:|---:|---|
| "free prize" | 1 | 0 | 1 | `[1, 0, 1]` |
| "free meeting" | 1 | 1 | 0 | `[1, 1, 0]` |

Both messages become vectors with **three features**, even though their words are different. A classifier can now use the numerical features to learn patterns from labeled examples. These vectors do **not** capture all meaning; they are just a simple representation of which words appear.

The feature positions must stay consistent. If position 2 means `meeting` during training, it must still mean `meeting` during inference. That is why we reuse the **same fitted vectorizer** when making predictions.

## 7. Language is part of the problem: Somali examples

Text-processing choices that work for English may not work equally well for Somali. The language in your corpus affects which spellings, words, and combinations the model will see.

**Spelling variation:** Somali's official Latin orthography was adopted in 1972. Standard spelling uses doubled letters for long vowels, as in `Soomaaliya`, but informal messages may write `Somalia`. A basic word-based vectorizer may treat those as two different vocabulary entries.

**Word forms:** Somali can express grammatical differences through word endings. For example, `nin` means "man" and `ninka` means "the man." A simple word-level system treats them as different tokens even though they are related.

**Mixed-language messages:** Someone might write `meeting-ka`, combining an English word with a Somali ending. A tokenizer trained only on English might divide or represent such text poorly.

**Language varieties:** Written language can vary across standard Somali, Maay, and other varieties. If a dataset covers only one variety, a model may perform worse on another. We should inspect actual examples instead of assuming one set of rules works equally well for everyone.

**Limited labeled data:** For some Somali NLP tasks, there may be fewer ready-made labeled datasets than for English. We may need to collect text carefully and create reliable labels ourselves.

## 8. From raw text to an output

Most NLP systems, classical or modern, follow the same general path:

**Raw text → text preparation and tokenization → numerical representation → model → output**

The output depends on the task: a class for classification, labels on words for sequence labeling, a ranked list for search, or new text for generation.

During **training**, the system learns from example text. We fit the vectorizer on training text, convert the text into numerical features, and train the model, with target labels for a supervised task or without them for an unsupervised one. We then evaluate the system on data that was not used to train it. We must not fit the vocabulary on the test data, because that would leak information from the test set into the training process.

During **inference**, new text arrives. The system applies the same preparation steps, uses the **already fitted vectorizer**, and passes the resulting features to the trained model, which returns its output.

**Training means learning from examples. Inference means applying what was learned.** A model does not automatically retrain itself simply because it reads new text.

## 9. What can a model *not* know from text alone?

A model can learn that some phrases are often associated with spam, but a prediction does not prove the sender's actual intention. The message "You won a prize" could be a real notification, a scam, or a joke. Without more context, the model cannot reliably know which it is just from those words.

Text alone also may not tell a model whether a claim is true, whether the writer is being sarcastic, or what happened earlier in a conversation. A model may make useful predictions while still missing slang, dialects, unusual spellings, or hidden context. **A high score is not the same thing as complete understanding.**

## 10. Why data quality matters

Even a strong algorithm can fail if the examples are poor. A text dataset might contain duplicated messages, incorrect labels, outdated scams, or text from only one region or language variety. If we evaluate a model only on messages similar to its training set, it may look successful but perform badly on messages from a different platform. This change between training and real-world data is called **distribution shift**.

We must also check **class balance**. Suppose 90 out of 100 SMS messages are `ham` and only 10 are `spam`. A model that predicts `ham` for every message achieves **90% accuracy** but detects **zero spam messages**. That is why we also look at which mistakes the model makes, and use metrics such as **precision**, **recall**, and **F1-score**, rather than trusting accuracy alone.

An **annotator** is the person or process assigning labels to examples. If annotators disagree or use unclear rules, the dataset can contain inconsistent labels. Good NLP work includes checking those labels, not just training a model.

## 11. Define the problem before choosing a model

Before we start coding, we should answer five questions about our application:

1. **What is the input?** One SMS, one sentence, one review, or an entire article?
2. **What should the output be?** A class label, several labels, or a ranked list of similar documents?
3. **Who will use the result?** An ordinary user, a support agent, a business, or a researcher?
4. **Which mistakes matter most?** Missing spam, blocking an important message, or returning irrelevant reviews?
5. **What information is available at prediction time?** Only the text, or also reliable metadata?

For example, blocking a genuine bank message can be more harmful than leaving an unwanted advertisement in an inbox. Understanding the cost of mistakes helps us decide how to evaluate and use the model.

## 12. Mini thought experiment

If a training corpus mostly uses `Soomaaliya` but a new message uses `Somalia`, what will our tokenizer do? Will it treat them as different pieces? Will our vectorizer still recognize the message? These are exactly the kinds of questions we will examine during the practical lessons.

## Lesson recap

- **NLP** is about working with human language using computers; Machine Learning is one way to build NLP systems.
- A **corpus** contains **documents**. A tokenizer produces **tokens** from the text.
- A **vocabulary** is the collection of unique token entries; **vocabulary size** counts those entries. **Token count** is how many token occurrences a given text produces.
- A **class** is a possible category; a **label** is the category assigned to a particular example.
- Classical ML models generally need numeric **features**. Vectorizers such as Bag of Words and TF-IDF can create them from text.
- **Supervised** tasks use target labels during training; **unsupervised** methods do not need target labels for their task.
- **Training** learns from examples; **inference** makes predictions with the fitted system.
- Text alone may not reveal real intent, truth, sarcasm, or missing context.
- Data quality, Somali language variation, and the cost of mistakes matter when evaluating a model.
