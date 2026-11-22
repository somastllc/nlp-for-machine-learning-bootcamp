# Assignment 2 – Text Preprocessing

## Part A – Research

### 1. What Is Text Preprocessing?

Text preprocessing is the process of preparing raw text before it is used by a Natural Language Processing (NLP) system. It involves transforming unstructured text into a more consistent and suitable format so that computers can analyze and process it effectively.

Common preprocessing techniques include lowercasing, removing unnecessary spaces, tokenization, punctuation handling, stopword removal, stemming, and lemmatization.

**Why do NLP systems need text preprocessing?**

Raw text often contains inconsistencies, unnecessary characters, spelling variations, and different forms of the same word. These differences can increase the complexity of NLP tasks and make it harder for machine learning models to identify useful patterns.

Preprocessing helps to:
- Reduce unnecessary variations in text.
- Improve consistency across documents.
- Reduce the number of features in text classification.
- Make text easier to analyze.
- Improve processing efficiency and, in some cases, model performance.

**Why can excessive cleaning make results worse?**

Removing too much information can destroy important meaning. Some words, punctuation marks, numbers, and symbols provide useful information depending on the NLP task.

For example, consider the sentence:

"I do not like this product."

If preprocessing removes the word "not" as a stopword, the remaining sentence may become "I like this product." This changes the original negative opinion into a positive one.

Another example is spam detection. A message containing "WIN!!! £1,000 NOW!" may use capital letters, repeated exclamation marks, and money amounts as indicators of suspicious activity. Removing all these features could make the message harder to identify as spam.

Therefore, preprocessing should be selected according to the purpose of the NLP application rather than removing every unusual element.

### 2. Text Normalization

Text normalization is the process of converting different representations of text into a consistent format while attempting to preserve its intended meaning.

It helps NLP systems recognize that different written forms may represent the same information.

Common normalization techniques include:

**a. Lowercasing**

Converting uppercase letters into lowercase letters.

Example:
- Original: "MACHINE Learning Is Powerful"
- Normalized: "machine learning is powerful"

This helps a system treat words such as "Apple" and "apple" as the same token when capitalization is not important.

**b. Whitespace normalization**

Removing unnecessary spaces and replacing repeated spaces, tabs, or line breaks with consistent spacing.

Example:
- Original: "Natural    Language     Processing"
- Normalized: "Natural Language Processing"

This prevents identical text from being represented differently because of formatting.

**c. Unicode normalization**

Unicode allows some characters to have different internal representations even when they look identical. Unicode normalization converts equivalent representations into a consistent form.

For example, the character "é" may be represented as one combined character or as the letter "e" followed by a combining accent. NFC normalization makes canonically equivalent representations consistent.

**When normalization is useful**

Lowercasing is useful in document classification when the system should recognize "COMPUTER", "Computer", and "computer" as the same word. It reduces unnecessary vocabulary differences.

**When normalization can remove useful information**

Lowercasing may be harmful in Named Entity Recognition (NER). For example, "Apple" refers to a company in a sentence such as "Apple released a new product." Changing it to lowercase removes a capitalization signal that may help identify proper names.

Similarly, converting all punctuation into nothing can remove useful information from messages where repeated exclamation marks indicate urgency or emotion.

The Unicode Consortium explains that NFC and NFD handle canonical equivalence, while NFKC and NFKD remove certain compatibility distinctions and may lose information.

Therefore, normalization should preserve features that are important to the task.

### 3. Word-Level, Subword-Level, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. A token can be a complete word, part of a word, or an individual character.

There are three common approaches:

**a. Word-level tokenization**

This method divides text into complete words.

Example:

Text: "Natural language processing is interesting."

Tokens: `["Natural", "language", "processing", "is", "interesting"]`

Advantages:
- Easy to understand and interpret.
- Preserves complete word meanings.
- Suitable for traditional text classification systems.

Disadvantages:
- Requires a large vocabulary.
- Unseen words may become unknown tokens.
- Different word forms may be treated as separate vocabulary entries.

**b. Subword-level tokenization**

Subword tokenization divides words into smaller meaningful units. Common approaches include Byte Pair Encoding (BPE) and WordPiece.

Example (illustrative segmentation):

Word: "unhappiness"

Tokens: `["un", "happi", "ness"]`

Advantages:
- Handles rare and previously unseen words.
- Reduces vocabulary size compared with a complete word vocabulary.
- Can represent prefixes, suffixes, and word structures.

Disadvantages:
- A word may be divided into several tokens.
- Token sequences can be harder to interpret.

Sennrich, Haddow, and Birch (2016) demonstrated that subword units help neural machine translation systems process rare and unknown words without relying entirely on fixed word vocabularies.

**c. Character-level tokenization**

This method divides text into individual characters.

Example:

Word: "cat"

Tokens: `["c", "a", "t"]`

Advantages:
- Can represent almost any word.
- Does not require a large word vocabulary.
- Can help process spelling variations and unusual words.

Disadvantages:
- Produces longer sequences.
- Requires models to learn word structures from individual characters.
- May increase computational requirements.

**Comparison Table**

| Feature | Word-level | Subword-level | Character-level |
|---|---|---|---|
| Token unit | Complete words | Word parts | Individual characters |
| Vocabulary size | Usually large | Medium | Small |
| Handles rare words | Poorly | Well | Very well |
| Sequence length | Shorter | Medium | Longer |
| Example for "playing" | playing | play + ing | p, l, a, y, i, n, g |

**What is vocabulary size?**

Vocabulary size is the total number of unique tokens that a tokenizer recognizes in its vocabulary.

For example, if a dataset contains the tokens:

`["cat", "dog", "cat", "bird"]`

Its vocabulary size is 3 because there are only three unique tokens.

Word-level tokenization usually requires more vocabulary entries because different complete words are stored separately. Subword tokenization can reduce this requirement by combining smaller units. Character-level tokenization generally uses a small set of characters.

The choice of tokenizer affects the vocabulary size, sequence length, and the ability of an NLP model to process unfamiliar words.

### 4. Stopwords

Stopwords are frequently occurring words in a language that are sometimes considered less useful for particular NLP tasks. In English, examples include "the", "is", "and", "of", and "to".

These words are common in ordinary sentences and may contribute little to some information retrieval or document classification tasks.

**Benefits of removing stopwords**

1. Reduces the number of tokens that a system needs to process.
2. Reduces the size of the feature space in traditional text classification.
3. Helps emphasize important topic-related words.
4. Can reduce computational requirements.

**Risks of removing stopwords**

Not every frequently occurring word is unimportant. Removing certain words may change the meaning of a sentence.

For example:

"I am not satisfied with the service."

If the word "not" is removed, the remaining sentence may suggest a positive opinion instead of a negative one.

Stopword removal can also damage grammatical relationships and information needed for question answering, machine translation, and sentiment analysis.

**Why do stopword lists depend on tasks and languages?**

Different NLP applications require different information. A search engine may remove common words to focus on meaningful search terms, while a sentiment analysis system should preserve negation words.

Stopword lists are also language-dependent because frequently occurring words differ between English, Somali, Arabic, and other languages. Even within one language, a word that is unimportant for one task may be essential for another.

Therefore, stopword removal should use a carefully selected list that matches both the language and the purpose of the application.

### 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce different forms of words to a common base representation. However, they use different methods.

**a. Stemming**

Stemming reduces words by removing or modifying prefixes and suffixes according to a set of rules. It does not necessarily produce a valid dictionary word.

Example:

- connected → connect
- connecting → connect
- organization → organ

The final result depends on the stemming algorithm.

**Porter Stemming Algorithm**

Martin Porter introduced the Porter stemming algorithm in 1980. It uses a sequence of suffix-removal rules to reduce related English words to common stems.

For example:

- connected → connect
- connecting → connect
- connection → connect

The algorithm is useful in information retrieval and search systems because different forms of related words can be grouped under a common stem.

**Lancaster Stemming Algorithm**

The Lancaster algorithm, also known as the Paice/Husk stemmer, uses a collection of rules that can repeatedly remove or replace word endings. It is generally more aggressive than Porter stemming.

This can reduce words significantly, but it may sometimes produce stems that are difficult to recognize as English words.

**b. Lemmatization**

Lemmatization converts words into their dictionary base forms, known as lemmas. Unlike stemming, it uses linguistic information and may consider the word's part of speech.

Examples:

- running → run
- better → good
- wolves → wolf
- was → be

WordNet is a lexical database that organizes English nouns, verbs, adjectives, and adverbs into synonym sets and connects them through semantic relationships.

The WordNetLemmatizer in NLTK uses WordNet to find a valid base form. Providing the correct part-of-speech information can improve its results.

**Comparison Table**

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix reduction | Dictionary and linguistic analysis |
| Output | May not be a real word | Usually a valid dictionary lemma |
| Context/POS | Usually does not require POS | Can use POS information |
| Speed | Generally faster | Generally slower |
| Accuracy | Less linguistically precise | More linguistically precise |
| Example | wolves → wolv (depending on stemmer) | wolves → wolf |

**When should each be preferred?**

Stemming is useful when processing speed and reducing vocabulary size are important, such as in some information retrieval systems.

Lemmatization is preferable when preserving the correct meaning and grammatical form is important, such as in sentiment analysis, language understanding, and linguistic research.

### 6. Regular Expressions in Text Cleaning

Regular expressions (Regex) are patterns used to search, identify, extract, and replace specific sequences of characters in text.

In NLP, regular expressions help identify unwanted formatting, URLs, phone numbers, dates, email addresses, and other text patterns.

Python provides the `re` module for working with regular expressions.

**Example 1: Matching phone numbers**

Pattern:

`r"\b\d{10}\b"`

Explanation:
- `\b` identifies a word boundary.
- `\d` matches a digit from 0 to 9.
- `{10}` requires exactly ten digits.
- The final `\b` marks the end of the number.

This pattern can identify a ten-digit phone number such as `0712345678`.

**Example 2: Matching email addresses**

Pattern:

`r"[\w.-]+@[\w.-]+\.\w+"`

Explanation:
- `[\w.-]+` matches one or more letters, digits, underscores, dots, or hyphens.
- `@` matches the at symbol.
- `\.` matches a literal dot.
- `\w+` matches the final domain component.

This pattern can match an address such as `student@example.com`. It is a simplified pattern and does not validate every possible email address.

**How can an overly broad regex damage text?**

A regular expression that removes everything except lowercase letters and spaces may delete numbers, currency symbols, punctuation, and emojis.

For example, cleaning:

`"WIN £1,000 NOW!!!"`

with the pattern `[^a-z\s]` results in:

`"win  now"`

The money amount and repeated exclamation marks disappear. These elements may be important indicators for a spam detection system.

Therefore, regular expressions should be designed carefully, and replacing meaningful information with labels is sometimes better than deleting it.

### 7. Case Study: Comparing Preprocessing Techniques for Twitter Sentiment Analysis

A useful research study is:

**Symeonidis, S., Effrosynidis, D., and Arampatzis, A. (2018). A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis.**

**Research task**

The researchers investigated how different text preprocessing techniques affect sentiment classification on Twitter data. The goal was to determine which preprocessing methods improve the ability of machine learning models to classify messages according to their sentiment.

**Preprocessing techniques compared**

The study experimentally compared sixteen preprocessing techniques using two Twitter datasets and four machine learning algorithms:

- Linear Support Vector Classification.
- Bernoulli Naive Bayes.
- Logistic Regression.
- Convolutional Neural Networks.

The researchers evaluated methods including:
- Lowercasing.
- Lemmatization.
- Number removal.
- Punctuation handling.
- Replacing contractions.
- Other common text cleaning operations.

They also investigated how different preprocessing methods interact when combined into a single pipeline.

**Research findings**

The results showed that preprocessing techniques do not have equal effects on model performance.

The researchers found that:
- Lemmatization improved classification accuracy.
- Removing numbers improved accuracy in their experiments.
- Replacing contractions also improved accuracy.
- Removing punctuation did not consistently improve results.
- Replacing repeated punctuation could be useful when combined with other preprocessing methods.

**Conclusion**

This study demonstrates that preprocessing decisions should be based on experimental evidence rather than applying every cleaning technique automatically.

For example, removing punctuation may seem useful because it reduces noise, but it can also remove information that helps identify sentiment or message characteristics.

The findings support the idea that an effective NLP pipeline should use carefully selected preprocessing methods according to the task and dataset.

---

## References

1. Camacho-Collados, J., & Pilehvar, M. T. (2018). On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 40–46. https://doi.org/10.18653/v1/W18-5406

2. Cambridge University Press. (2023). Comparison of Text Preprocessing Methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

3. Miller, G. A. (1995). WordNet: A Lexical Database for English. *Communications of the ACM, 38*(11), 39–41.

4. Paice, C. D. (1990). Another Stemmer. *ACM SIGIR Forum, 24*(3), 56–61.

5. Porter, M. F. (1980). An Algorithm for Suffix Stripping. *Program, 14*(3), 130–137.

6. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725. https://doi.org/10.18653/v1/P16-1162

7. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2018.06.022

8. Unicode Consortium. (2026). Unicode Standard Annex #15: Unicode Normalization Forms. https://www.unicode.org/reports/tr15/

9. Uysal, A. K., & Gunal, S. (2014). The Impact of Preprocessing on Text Classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006

10. Course lecture slides. (n.d.). *Chapter 6: Natural Language Processing*, especially pages 21–27. Course PDF.
