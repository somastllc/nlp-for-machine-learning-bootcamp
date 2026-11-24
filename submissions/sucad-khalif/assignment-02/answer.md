## Part A – Research

### 1. What is Text Preprocessing?

Text preprocessing is the process of preparing raw text before using it in a Natural Language Processing (NLP) system. It includes operations such as lowercasing, tokenization, removing unnecessary spaces, and reducing word variations.

NLP systems need preprocessing because raw text often contains inconsistent formatting and unnecessary variations. Preparing text helps algorithms process information more consistently.

### 2. What is Text Normalization?

Text normalization is the process of converting text into a consistent format while preserving its intended meaning.

Common techniques include:

- **Lowercasing:** Converting uppercase letters into lowercase.
- **Whitespace normalization:** Removing unnecessary spaces.
- **Unicode normalization:** Converting equivalent Unicode representations into a consistent form, such as NFC.

### 3. Word-Level, Subword and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens.

| Tokenization type | Description | Example |
|---|---|---|
| Word-level | Divides text into complete words. | `playing` |
| Subword | Divides words into smaller meaningful or frequent units. | `play` + `ing` |
| Character-level | Divides words into individual characters. | `p`, `l`, `a`, `y` |

Vocabulary size refers to the total number of unique tokens available to a tokenizer. Word-level tokenization can produce a large vocabulary because different word forms are treated separately. Subword tokenization usually controls vocabulary size by combining frequently occurring character sequences. Character-level tokenization generally uses a smaller vocabulary but produces longer token sequences.

### 4. What are Stopwords?

Stopwords are common words that may contribute relatively little information to certain NLP tasks. Examples in English include "the", "is", "and" and "of".

Removing stopwords can reduce text size and help information retrieval or topic analysis focus on more distinctive terms.

### 5. Stemming vs. Lemmatization

Stemming and lemmatization are techniques used to reduce different forms of words.

| Stemming | Lemmatization |
|---|---|
| Removes word endings using predefined rules. | Converts words into their dictionary base forms. |
| May produce incomplete words. | Usually produces valid dictionary words. |
| Generally faster. | Requires more linguistic information. |
| Example: `connected` → `connect`. | Example: `better` → `good` (with appropriate analysis). |

### 6. How are Regular Expressions Used in Text Cleaning?

Regular expressions (Regex) are patterns used to identify, extract, replace or remove specific text sequences.

Examples:

1. `\d+` – Matches one or more digits, such as `12345`. It can be used to identify numbers in messages.
2. `\b[A-Z][a-z]+\b` – Matches capitalized words, such as `London` or `Somalia`.

Regex can simplify text cleaning, but overly broad patterns can remove useful information. For example, removing every non-alphabetic character may delete emojis, currency symbols and punctuation that help identify emotions or suspicious messages.

### 7. Case Study: Preprocessing for Twitter Sentiment Analysis

Symeonidis, Effrosynidis and Arampatzis (2018) conducted a comparative study of preprocessing techniques for Twitter sentiment analysis.

The research compared 16 preprocessing techniques using two Twitter datasets and four machine learning algorithms, including Logistic Regression, Linear SVC, Bernoulli Naive Bayes and Convolutional Neural Networks.

The researchers found that preprocessing choices affected classification accuracy. In particular, lemmatization, number removal and contraction replacement improved accuracy in their experiments. The study also examined how different preprocessing techniques interacted when combined.

This case study demonstrates that preprocessing techniques do not have the same effect on every NLP application. Their usefulness depends on the dataset, algorithm and task.

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

2. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.

3. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

4. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.

5. Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode Normalization Forms*. https://www.unicode.org/reports/tr15/

6. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022