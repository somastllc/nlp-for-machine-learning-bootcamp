# Assignment 2 – Text Preprocessing

## Part A – Research

### 1. Definition and Importance of Text Preprocessing

Text preprocessing is the process of preparing raw textual data before it is used by a Natural Language Processing (NLP) system. It involves transforming unstructured text into a more consistent and usable format through operations such as lowercasing, tokenization, whitespace normalization, stopword removal, stemming, and lemmatization.

NLP systems require preprocessing because raw text often contains inconsistencies, unnecessary formatting, spelling variations, punctuation, and other elements that can make language processing difficult. Preprocessing helps reduce data complexity, improve consistency, and prepare text for machine learning algorithms.

However, excessive cleaning can negatively affect model performance because some information that appears unnecessary may actually carry important meaning.

**Real-life example:**

Consider the customer review:

"I don't like this product!"

If preprocessing removes the word "don't" and all punctuation, the remaining text could become:

"like this product"

The original sentence expresses a negative opinion, but the cleaned version may appear positive. This demonstrates why preprocessing must depend on the purpose of the NLP application.

### 2. Text Normalization

Text normalization is the process of converting different representations of text into a consistent format. It helps NLP systems recognize that different written forms may represent the same underlying information.

Common normalization techniques include:

- **Lowercasing:** Converting uppercase letters into lowercase letters.
- **Whitespace normalization:** Removing unnecessary spaces, tabs, and repeated line breaks.
- **Unicode normalization:** Converting characters with equivalent representations into a standardized form.

Unicode normalization is particularly important because visually identical characters can sometimes have different underlying Unicode representations. NFC and NFKC are two commonly used normalization forms.

**Example where normalization helps:**

The words "HELLO", "Hello", and "hello" can be converted into "hello". This reduces unnecessary differences when building a text classification system.

**Example where normalization removes useful information:**

In a spam detection system, converting all text to lowercase removes capitalization patterns. A message containing "URGENT!!!" may be more suspicious than a normally written message. Removing capitalization and repeated punctuation can eliminate potentially useful spam indicators.

Therefore, normalization should preserve information that is relevant to the target task.

### 3. Word-Level, Subword, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. A token may represent a complete word, part of a word, or an individual character.

| Tokenization type | Description                                                 | Advantages                                            | Disadvantages                                           |
| ----------------- | ----------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| Word-level        | Divides text into complete words.                           | Easy to understand and interpret.                     | Large vocabulary and difficulties with unknown words.   |
| Subword-level     | Divides words into smaller meaningful or statistical units. | Handles unfamiliar words and reduces vocabulary size. | Some words require multiple tokens.                     |
| Character-level   | Divides text into individual characters.                    | Handles spelling variations and unknown words.        | Produces longer sequences and requires more processing. |

**Examples:**

Original text: "unbelievable"

- Word-level: `["unbelievable"]`
- Subword-level: `["un", "believ", "able"]` (illustrative segmentation)
- Character-level: `["u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "e"]`

**Vocabulary size** refers to the number of unique tokens that a tokenizer can recognize or use.

Word-level tokenization generally requires a larger vocabulary because every distinct word form may become a separate token. Subword tokenization reduces this problem by combining smaller units. Character-level tokenization normally uses a relatively small vocabulary consisting of letters, digits, punctuation, and other characters.

The tokenizer choice affects vocabulary size, sequence length, memory requirements, and the ability to process unfamiliar words.

### 4. Stopwords

Stopwords are common words that are sometimes considered less informative for particular NLP tasks. Examples in English include "the", "is", "a", "and", and "of".

Removing stopwords can reduce the number of tokens and simplify text analysis. It can be useful in information retrieval and some document classification tasks.

However, stopword removal can also damage meaning.

For example:

- Original: "I do not recommend this product."
- After removing stopwords: "recommend product."

The second sentence no longer communicates the original negative opinion.

The appropriate stopword list depends on the task and language. A sentiment analysis system should preserve negation words such as "not", "never", and "no". A search engine may use a different list depending on its indexing requirements.

Stopword lists must also be language-specific. English stopwords cannot simply be applied to Somali, Arabic, or other languages because their grammatical structures and frequently occurring words are different.

### 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce different word forms to a common representation.

**Stemming**

Stemming removes prefixes or suffixes using predefined linguistic or rule-based procedures. It does not necessarily produce a valid dictionary word.

A well-known example is the Porter Stemmer, introduced by Martin Porter in 1980. It applies a sequence of suffix-removal rules to reduce related English words to similar stems.

For example:

- connected → connect
- connecting → connect
- organization → organ (depending on the algorithm and implementation)

The Lancaster Stemmer is another algorithm. It generally applies more aggressive reduction rules and can produce shorter stems.

**Lemmatization**

Lemmatization converts a word into its dictionary base form, called a lemma. Unlike basic stemming, it considers vocabulary and may use grammatical information such as part of speech.

Examples:

- running → run
- wolves → wolf
- better → good (when interpreted as an adjective and supported by the lemmatizer)

WordNetLemmatizer in NLTK is a commonly used dictionary-based approach.

**Comparison**

| Feature  | Stemming                          | Lemmatization                             |
| -------- | --------------------------------- | ----------------------------------------- |
| Method   | Removes word affixes using rules. | Uses lexical and grammatical information. |
| Output   | May not be a real word.           | Usually produces a valid dictionary form. |
| Speed    | Generally faster.                 | Usually more computationally expensive.   |
| Accuracy | Less linguistically precise.      | More linguistically informed.             |
| Example  | studies → studi                   | studies → study                           |

Stemming may be preferred when processing large document collections where speed and reducing vocabulary size are important. Lemmatization is more appropriate when preserving meaningful linguistic forms is important, such as in language analysis and sentiment classification.

### 6. Regular Expressions in Text Cleaning

Regular expressions, commonly called regex, are patterns used to identify and manipulate specific sequences of characters in text.

In Python, the `re` module provides functions such as `re.sub()`, `re.findall()`, and `re.search()`.

Two examples are:

**Example 1:**

Pattern:

`r"\d+"`

This matches one or more consecutive digits.

For example, in:

"Order number 12345"

The pattern matches:

"12345"

**Example 2:**

Pattern:

`r"\s+"`

This matches one or more consecutive whitespace characters, including spaces, tabs, and line breaks.

It can be used to replace repeated whitespace with a single space.

**Risk of overly broad patterns:**

Consider:

`r"[^a-z\s]"`

This pattern matches every character that is not a lowercase English letter or whitespace.

It removes numbers, punctuation, emojis, currency symbols, and uppercase characters if applied to text that has not already been lowercased.

For example, it may remove the £ symbol and the digits in a financial transaction message. In a spam detection application, these elements may be important evidence.

Therefore, regex rules should be designed according to the information that the application needs to preserve.

### 7. Case Study: Evaluating the Effectiveness of Text Pre-Processing in Sentiment Analysis

A relevant research study is:

**Palomino, M. A., and Aider, F. (2022). "Evaluating the Effectiveness of Text Pre-Processing in Sentiment Analysis." Applied Sciences, 12(17), 8765.**

The researchers investigated how different preprocessing techniques affect Twitter sentiment analysis. Their work focused on preparing informal social media messages for sentiment classification.

**Task:**

The study evaluated preprocessing methods for identifying positive and negative sentiment in Twitter messages.

**Preprocessing options compared:**

The researchers examined several techniques, including:

- Lowercasing.
- Removing URLs and unnecessary formatting.
- Removing unnecessary whitespace.
- Punctuation handling.
- Stopword removal.
- Emoji and emoticon processing.
- Lemmatization.
- Different combinations and orders of preprocessing operations.

**Results:**

The researchers found that preprocessing choices and their order can influence classification performance. Some combinations improved Naive Bayes classification results. Lemmatization was useful for improving the text index, although it did not produce a significant improvement in sentiment analysis quality in their experiments.

An important finding was that preprocessing should not be treated as a fixed sequence of cleaning operations. Different combinations can produce different results depending on the classifier and dataset.

**My interpretation:**

This research demonstrates that removing more text does not automatically improve an NLP model. For sentiment analysis, retaining negation, emojis, and useful punctuation may be more important than removing every unusual character.

## References

1. Bird, S., Klein, E., & Loper, E. (2009). _Natural Language Processing with Python_. O'Reilly Media. https://www.nltk.org/book/

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_. Cambridge University Press. https://nlp.stanford.edu/IR-book/

3. Porter, M. F. (1980). An algorithm for suffix stripping. _Program, 14_(3), 130–137.

4. Porter, M. F. (2001). Snowball: A language for stemming algorithms. https://snowballstem.org/texts/introduction.html

5. Chai, C. P. (2023). Comparison of text preprocessing methods. _Natural Language Engineering, 29_(3), 509–553. https://doi.org/10.1017/S1351324922000213

6. Palomino, M. A., & Aider, F. (2022). Evaluating the effectiveness of text pre-processing in sentiment analysis. _Applied Sciences, 12_(17), 8765. https://doi.org/10.3390/app12178765

7. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. _Expert Systems with Applications_. https://doi.org/10.1016/j.eswa.2018.06.022
