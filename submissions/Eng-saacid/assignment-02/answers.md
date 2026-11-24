# Assignment 2 – Preprocessing

## – My Research 

### 1. What is Text Preprocessing?

Text preprocessing is the process of preparing raw text before it is used in Natural Language Processing (NLP) applications. It involves transforming unstructured text into a cleaner and more consistent format that computers can analyze.

NLP systems need preprocessing because human language contains spelling variations, punctuation, unnecessary spaces, different word forms, and other inconsistencies. These differences can make it difficult for machine learning models to identify useful patterns.

However, excessive cleaning can remove important information and negatively affect model performance.

**Real-life example:**

Consider a spam detection system that receives this message:

`URGENT!!! You have WON a £1,000 gift card!`

If the system removes all capital letters, punctuation, and numbers, it may lose useful spam indicators. Words such as "URGENT" and expressions such as "!!!" can help identify suspicious messages.

Therefore, preprocessing should preserve information that is relevant to the task.

### 2. Text Normalization

Text normalization is the process of converting text into a consistent representation while maintaining its original meaning.

Common normalization techniques include:

- **Lowercasing:** Converting uppercase letters into lowercase.
- **Whitespace normalization:** Removing unnecessary spaces and standardizing spacing.
- **Unicode normalization:** Converting different Unicode representations of the same character into a consistent form.

**Example:**

Original:

`HELLO     WORLD`

Normalized:

`hello world`

**When normalization helps:**

Lowercasing helps a search engine recognize that "Computer", "COMPUTER", and "computer" usually refer to the same word. This reduces unnecessary differences in the text.

**When normalization removes useful information:**

In sentiment analysis, converting all text to lowercase may remove emphasis. For example, "This is GREAT!" expresses stronger emotion than "This is great."

Therefore, normalization should depend on the information required by the NLP task.

### 3. Word-Level, Subword and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. A tokenizer determines how sentences are separated into words, subwords, or characters.

| Tokenization | Description | Example: "unhappiness" |
|---|---|---|
| Word-level | Treats complete words as tokens | `["unhappiness"]` |
| Subword | Divides words into smaller meaningful or frequently occurring units | `["un", "happiness"]` |
| Character-level | Divides text into individual characters | `["u","n","h","a","p","p","i","n","e","s","s"]` |

**What is vocabulary size?**

Vocabulary size refers to the total number of unique tokens that a tokenizer can recognize or represent.

The choice of tokenizer affects vocabulary size:

- Word-level tokenization often produces a large vocabulary because different word forms may be treated separately.
- Subword tokenization generally provides a balance between vocabulary size and the ability to represent unfamiliar words.
- Character-level tokenization uses a relatively small vocabulary because it works with individual characters.

For example, subword tokenization is useful when processing languages with many word variations or handling words that were not previously encountered.

### 4. Stopwords

Stopwords are common words that some NLP systems remove because they may contribute little to a particular task.

Examples in English include:

- the
- is
- a
- an
- and
- of

**Benefits of removing stopwords:**

Removing unnecessary common words can reduce the amount of text that a model processes. It can also help certain information-retrieval and text-classification systems focus on more informative terms.

**Risks of removing stopwords:**

Some common words carry important meaning. For example:

- "I am happy."
- "I am not happy."

If the word "not" is removed, the second sentence loses its negative meaning.

The appropriate stopword list depends on the NLP task and language. Sentiment analysis must preserve negation words, while some search applications may remove certain frequent words. Different languages also have different grammatical structures and common words.

Therefore, stopword removal should not be applied automatically to every NLP application.

### 5. Stemming vs. Lemmatization

Stemming and lemmatization are techniques used to reduce words to a common base form. However, they work differently.

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Removes word endings using rules | Uses vocabulary and linguistic information |
| Output | May produce a non-dictionary word | Produces a valid dictionary form |
| Speed | Usually faster | Usually slower |
| Accuracy | Less linguistically accurate | Generally more linguistically accurate |
| Example | `studies → studi` | `studies → study` |

**Stemming algorithm: Porter Stemmer**

The Porter Stemmer is a well-known stemming algorithm introduced by Martin Porter in 1980. It applies a sequence of suffix-removal rules to reduce related words to a common stem.

For example:

- connected → connect
- connecting → connect
- connection → connect

It is useful when processing large collections of text where speed and grouping related word forms are important.

**Lemmatization approach: WordNet Lemmatizer**

The WordNet Lemmatizer uses WordNet, a lexical database of English words, to identify dictionary base forms.

Unlike stemming, lemmatization can use part-of-speech information to determine the correct base form.

For example:

- running → run (verb)
- better → good (adjective)

Lemmatization is preferred when meaningful dictionary words and linguistic accuracy are important.

### 6. Regular Expressions in Text Cleaning

Regular expressions (regex) are patterns used to search, identify, extract, or replace specific sequences of characters in text.

They are useful for removing unwanted symbols, identifying phone numbers, replacing dates, and cleaning structured text.

**Example 1: Removing special characters**

```python
import re

text = "Hello!!! Welcome @2026"
result = re.sub(r"[^a-zA-Z0-9\s]", "", text)

print(result)
```

The pattern identifies characters that are not letters, numbers, or whitespace and removes them.

**Example 2: Identifying phone numbers**

```python
import re

text = "Call me at 0712345678"

result = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", text)

print(result)
```

The pattern identifies a sequence of exactly ten digits surrounded by word boundaries and replaces it with `PHONE_NUMBER`.

**Risk of overly broad patterns:**

An overly broad regex may remove useful information. For example, removing every non-letter character could delete currency symbols, numbers, punctuation, and emojis.

This can damage spam detection and sentiment analysis because these elements may contain important information.

### 7. Case Study: Preprocessing Choices in Text Classification

A relevant research study is Uysal and Gunal (2014), titled *"The Impact of Preprocessing on Text Classification."*

The researchers investigated how different preprocessing techniques affect the performance of text classification systems. They examined four preprocessing techniques: tokenization, stopword removal, lowercase conversion, and stemming.

The study used email and news datasets in two languages, English and Turkish, to evaluate how different combinations of preprocessing techniques influence classification accuracy.

**Main Findings and Implications:**

The study demonstrated that preprocessing choices can significantly affect text classification performance. The researchers found that different combinations of preprocessing techniques may produce different results depending on the language and text domain.

For example, stopword removal and stemming may be useful in some classification tasks, but applying every preprocessing technique does not always improve accuracy.

The main lesson is that preprocessing techniques should be selected and evaluated according to the dataset and classification objective rather than applying every available cleaning technique automatically.

---

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

2. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Jurafsky, D., & Martin, J. H. *Speech and Language Processing*. https://web.stanford.edu/~jurafsky/slp3/

5. Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006