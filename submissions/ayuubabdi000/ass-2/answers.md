# NLP Assignment 2 – Text Preprocessing

## 1. Text Preprocessing

Text processing is crucial step in NLP that involves raw, unstructured data to be converted to clean structured format that machine can understand.

### Why can more cleaning sometimes make results worse?

Before training we remove common words thats called stopword removal, its purpose is to reduce noise.

**Example:**

- This product is good → product good
- This product is not good → product good

---

## 2. Text Normalization

Text normalization is the process of converting textual data into a clean and consistent format before processing it in Natural Language Processing.

- **Lowercasing:** Converts uppercase to lowercase.
- **Whitespace:** Removes extra spaces.
- **Unicode normalization:** Removes punctuation marks.

---

## 3. Tokenization

“Let us learn tokenization.”

### Word-Based Tokenization

A word-based tokenization algorithm will break the sentence into words. The most common one is splitting based on space.

`["Let", "us", "learn", "tokenization."]`

### Subword-Based Tokenization

A subword-based tokenization algorithm will break the sentence into subwords.

`["Let", "us", "learn", "token", "ization."]`

### Character-Based Tokenization

A character-based tokenization algorithm will break the sentence into characters.

`["L", "e", "t", "u", "s", "l", "e", "a", "r", "n", "t", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n", "."]`

### Vocabulary Size

Vocabulary size refers to the total number of unique tokens (words, subwords, or characters) that an AI model can recognize.

- **Word-level:** Vocabulary size grows directly with unique words in the training corpus (including inflections, typos, and rare terms), quickly becoming huge and hard to bound.
- **Character-level:** Vocabulary size is tiny and roughly fixed by the alphabet/script and symbols used, which keeps embedding tables small at the cost of much longer sequences.
- **Subword-level:** Vocabulary size is typically a set hyperparameter chosen before training. Algorithms like Byte-Pair Encoding (BPE) start from base characters and merge frequent adjacent pairs until reaching the target vocabulary size—letting you tune the balance between smaller tables and shorter sequences.

---

## 4. Stopwords

Stop words are frequently occurring words such as “is”, “the”, “and”, and “of” that may contribute little to certain text analysis tasks.

Removing them may reduce number of feature but removing stop words can remove the meaning.

**Example:**

- I am happy → happy
- I am not happy → happy

The right stopword list depends on the task because some common words, like not, carry important meaning, and on the language because different languages have different grammar and common words.

---

## 5. Stemming and Lemmatization

Stemming removes prefix and suffix using simple rules while lemmatizing reduces words from to their dictionary base from lemma.

### Porter Stemmer

The porter stemmer removes common suffixes thought series of rule based system.

### WordNet Lemmatizer

Word net lemmatizer uses dictionary wordnet and part-of-speech information to find a valid base word.

### When to Prefer Each

- **Stemming:** Prefer when speed is important.
- **Lemmatization:** Prefer when meaningful grammatically important.

---

## 6. Regular Expressions (Regex)

Regular expressions (regex) are used in text cleaning to find and remove or replace unwanted characters or patterns.

### Example 1: `\d+`

Matches one or more digits. It can be used to remove numbers from text.

### Example 2: `[!@#$%^&*]`

Matches special characters such as !, @, and #. It can be used to remove unnecessary symbols.

An overly broad pattern may remove useful information, such as deleting numbers that are part of dates, phone numbers, or product names.

---

## 7. Case Study

### A Comparative Evaluation of Pre-processing Techniques for Twitter Sentiment Analysis

**Task:**

The researchers aimed to classify Twitter posts based on their sentiment using machine learning.

**Preprocessing Options Compared:**

They tested techniques such as lowercasing, stopword removal, stemming, lemmatization, punctuation removal, and number removal using different machine learning algorithms.

**Results:**

The study found that lemmatization, number removal, and replacing contractions improved accuracy. However, removing punctuation did not provide significant improvement. The researchers also found that combining preprocessing techniques could affect performance differently.

**Conclusion:**

Preprocessing techniques should be selected based on the NLP task because some methods improve accuracy while others have little or no effect.

**Source:**

Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). *A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis.*
