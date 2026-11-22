# ASSIGNMENT 02
# NLP TEXT PREPROCESSING

**Course:** Natural Language Processing (NLP)  
**Assignment:** 02 – Text Preprocessing  
**Sections:** Part A – Research | Part B – Practical

---

# PART A – RESEARCH

## 1. Introduction to Text Preprocessing

Text preprocessing is the process of transforming raw text into a more organized and consistent format before using it in Natural Language Processing (NLP) and machine learning systems. Raw text collected from websites, social media, customer reviews, and other sources often contains inconsistent capitalization, unnecessary spaces, punctuation, spelling variations, and special characters.

Common preprocessing techniques include text normalization, tokenization, stopword removal, stemming, lemmatization, and regular expressions. These methods help prepare text for computational analysis by reducing unnecessary variations and organizing information into a format that NLP systems can process (Chai, 2023).

NLP systems need preprocessing because human language is complex and often inconsistent. For example, the words "RUN," "Run," and "running" have related meanings but different written forms. Without suitable preprocessing, a system may treat some variations as different tokens.

However, more cleaning does not always produce better results. Excessive preprocessing can remove important information, change the meaning of a sentence, or negatively affect model performance.

A real-life example is a spam detection system. A message such as "URGENT!!! You won £1,000. Call 0712345678 now!" contains capitalization, repeated punctuation, money amounts, and a phone number. Removing all numbers and punctuation may make the text cleaner, but it can also remove useful indicators of spam. Therefore, preprocessing should depend on the purpose of the NLP application rather than removing every unusual character.

## 2. Text Normalization

Text normalization is the process of converting text into a consistent and standardized representation before further processing. Common normalization techniques include lowercasing, whitespace normalization, and Unicode normalization (Chai, 2023).

### 2.1 Lowercasing

Lowercasing converts uppercase letters into lowercase letters.

For example:

- Original: `HELLO`, `Hello`, `hello`
- Normalized: `hello`

This helps reduce unnecessary differences between words and can be useful in text classification and information retrieval.

However, lowercasing can remove useful information. For example, `US` may refer to the United States, while `us` is a pronoun. Converting both into lowercase removes this distinction.

### 2.2 Whitespace Normalization

Whitespace normalization removes unnecessary spaces, tabs, and repeated whitespace characters.

For example:

- Original: `The   delivery    was   fast.`
- Normalized: `The delivery was fast.`

This produces cleaner text and prevents unnecessary differences caused by spacing.

### 2.3 Unicode Normalization

Unicode normalization ensures that characters represented in different Unicode forms can be treated consistently.

For example, some accented characters can be represented either as a single character or as a combination of a base character and a combining accent. Unicode normalization can convert equivalent representations into a consistent form.

Unicode normalization is useful when processing multilingual text, but it should be applied carefully because removing accents or changing special characters may damage information in some languages.

### 2.4 When Normalization Helps and When It Can Cause Problems

Normalization helps when a system needs to treat different capitalization or spacing patterns as equivalent. For example, a search system can match `Python`, `PYTHON`, and `python` more consistently after lowercasing.

However, normalization may remove useful information when capitalization has a special meaning, such as distinguishing `US` from `us`. Therefore, normalization should be selected according to the NLP task (Chai, 2023).

## 3. Tokenization

Tokenization is the process of dividing text into smaller units called tokens. Depending on the tokenizer, these units may be complete words, subword units, or individual characters.

Tokenization is important because it determines how an NLP system represents and processes text (Chai, 2023).

### 3.1 Word-Level Tokenization

Word-level tokenization divides text into individual words or word-like units.

Example:

`Natural Language Processing is useful.`

Tokens:

`["Natural", "Language", "Processing", "is", "useful"]`

It is easy to understand and interpret. However, it can create a large vocabulary and may struggle with unfamiliar words.

### 3.2 Subword Tokenization

Subword tokenization divides words into smaller units. A word that is unfamiliar to a model may be represented using smaller units that the model already knows.

For example, a word such as `unhappiness` may be divided into smaller components such as `un`, `happi`, and `ness`, depending on the tokenizer.

Subword tokenization is useful for handling rare and unfamiliar words without requiring every complete word to exist in the vocabulary.

### 3.3 Character-Level Tokenization

Character-level tokenization divides text into individual characters.

Example:

`CAT`

Tokens:

`["C", "A", "T"]`

It uses a relatively small basic vocabulary and can handle unfamiliar words. However, it creates longer sequences and may require more processing.

### 3.4 Vocabulary Size

Vocabulary size refers to the number of unique tokens that a tokenizer can recognize or represent.

| Tokenization Type | Basic Unit | Vocabulary Characteristics |
|---|---|---|
| Word-level | Complete words | Usually larger vocabulary |
| Subword | Parts of words | Usually medium-sized vocabulary |
| Character-level | Individual characters | Usually smaller vocabulary |

The choice of tokenizer affects vocabulary size, sequence length, computational requirements, and the ability to handle unfamiliar words. Word-level tokenization may require many unique entries, while subword and character-level methods can represent words using smaller units.

## 4. Stopword Removal

Stopword removal is a preprocessing technique that removes frequently occurring words that may contribute little information to a particular NLP task.

Common English stopwords include:

- the
- is
- a
- an
- in
- of
- and

Removing these words can reduce the number of tokens and help some NLP systems focus on more informative words (Chai, 2023).

### 4.1 Benefits

- Reduces unnecessary tokens.
- Can reduce the size of text representations.
- May improve efficiency in some information retrieval tasks.
- Can help certain models focus on informative words.

### 4.2 Risks

Stopword removal can damage text meaning when important words are removed.

For example:

Original sentence:

`I do not like this movie.`

After removing `not`:

`I like this movie.`

The meaning has changed from negative to positive. This can cause problems in sentiment analysis and other text classification tasks.

### 4.3 Why Stopword Lists Depend on the Task and Language

A word that is unimportant in one application may be essential in another. In sentiment analysis, words such as `not`, `never`, and `no` can be important because they express negative meaning.

Stopword lists also depend on the language. English stopword lists cannot be directly applied to Somali text because the languages have different vocabulary, grammar, and linguistic structures.

Therefore, stopword removal should be customized according to the language, dataset, and purpose of the NLP application.

## 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce different forms of a word to a common representation. They help reduce unnecessary word variations in text processing (Chai, 2023).

### 5.1 Stemming

Stemming is a rule-based process that removes prefixes or suffixes from words to produce a stem.

For example:

| Original Word | Possible Stem |
|---|---|
| connected | connect |
| connecting | connect |
| studies | studi |

The resulting stem is not always a valid dictionary word.

#### Porter Stemmer

The Porter Stemmer is a well-known English stemming algorithm developed by Martin Porter. It applies a sequence of rules to remove common English suffixes.

It is useful in information retrieval and search applications where grouping related word forms is more important than producing grammatically correct dictionary words.

#### Lancaster Stemmer

The Lancaster Stemmer is another rule-based algorithm. It is generally more aggressive than Porter and may produce shorter stems.

For example, in the practical results, Lancaster converted `organization` into `org`, while Porter produced `organ`.

### 5.2 Lemmatization

Lemmatization converts words into their dictionary base forms, known as lemmas. Unlike stemming, it uses vocabulary and grammatical information to determine a valid base form.

Examples:

| Original Word | Lemma |
|---|---|
| studies | study |
| wolves | wolf |
| ran | run |
| better | good |

The WordNet Lemmatizer in NLTK uses the WordNet lexical database. It can use part-of-speech information to improve the result.

For example, `ran` may remain unchanged with the default noun setting, but it becomes `run` when the verb hint is supplied.

### 5.3 Comparison

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Removes prefixes or suffixes using rules | Uses vocabulary and linguistic information |
| Output | May not be a real word | Usually a dictionary word |
| Speed | Generally faster | Often slower |
| Accuracy | Less linguistically accurate | Usually more linguistically accurate |
| Example | studies → studi | studies → study |
| Suitable Use | Search and information retrieval | Applications requiring meaningful word forms |

Stemming may be preferred when speed and grouping related word forms are important. Lemmatization may be preferred when preserving meaningful dictionary forms and grammatical information is more important.

## 6. Regular Expressions (Regex)

Regular expressions are patterns used to search for, identify, extract, or replace specific character sequences in text.

In NLP preprocessing, Regex is useful for cleaning text, identifying numbers, extracting information, and replacing unwanted patterns.

### 6.1 Example One: Matching Numbers

Pattern:

```python
\d+
```

This pattern matches one or more consecutive digits.

Example input:

`Bille is 21 years old and has 2 brothers.`

Matched values:

`21`, `2`

### 6.2 Example Two: Matching English Alphabetic Words

Pattern:

```python
[A-Za-z]+
```

This pattern matches one or more consecutive uppercase or lowercase English alphabetic characters.

Example input:

`Bille2026 is learning Python3`

Matched values:

`Bille`, `is`, `learning`, `Python`

### 6.3 Risks of Overly Broad Regex Patterns

An overly broad regular expression may remove important information instead of only removing unwanted characters.

For example, the pattern:

```python
[^a-z\s]
```

removes every character that is not a lowercase English letter or whitespace. This includes numbers, currency symbols, punctuation, and emojis.

In a spam detection system, removing phone numbers and money amounts could remove useful spam indicators. Therefore, Regex patterns should be designed carefully, and important information may be replaced with labels rather than deleted.

## 7. Case Study: Preprocessing Techniques in Twitter Sentiment Analysis

Symeonidis, Effrosynidis, and Arampatzis (2018) conducted a study titled *A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis*.

### 7.1 Research Objective

The study investigated how different preprocessing techniques affect the performance of sentiment classification systems using Twitter data.

### 7.2 Methods Compared

The researchers evaluated 16 preprocessing techniques using two Twitter datasets and four machine learning algorithms:

- Linear SVC
- Bernoulli Naive Bayes
- Logistic Regression
- Convolutional Neural Networks (CNNs)

The preprocessing methods included lemmatization, number removal, punctuation removal, and contraction replacement.

### 7.3 Findings

The results showed that some preprocessing techniques improved classification accuracy, including lemmatization, number removal, and contraction replacement.

However, removing punctuation did not consistently improve performance. The researchers also examined how different preprocessing techniques interact when combined in a pipeline.

Another study by Palomino and Aider (2022) investigated preprocessing methods for Twitter sentiment analysis. Their findings showed that the order in which preprocessing components are applied can affect classifier performance. They reported improvements in Naive Bayes performance from preprocessing order, while lemmatization did not significantly improve sentiment analysis accuracy in their experiments.

### 7.4 Discussion

These studies demonstrate that preprocessing techniques do not always improve every NLP task. The effectiveness of a technique depends on the dataset, classification task, and machine learning model.

For example, removing punctuation may be acceptable in one application but harmful in another where punctuation carries useful information. Therefore, preprocessing techniques should be tested and evaluated rather than applied automatically (Symeonidis et al., 2018; Palomino & Aider, 2022).

## 8. Conclusion

Text preprocessing is an important stage in Natural Language Processing because it prepares raw text for computational analysis.

Techniques such as normalization, tokenization, stopword removal, stemming, lemmatization, and regular expressions can improve text consistency, reduce unnecessary variation, and help machine learning models process text more effectively.

However, preprocessing does not always improve model performance. Removing important words, punctuation, or other information may change the meaning of a sentence and lead to inaccurate predictions.

Therefore, preprocessing should be carefully selected, tested, and evaluated rather than applied blindly. The main goal is not to remove as much text as possible, but to prepare text in a way that preserves useful information and supports the intended NLP task.

## 9. References

1. Chai, C. P. (2023). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

2. Palomino, M. A., & Aider, F. (2022). Evaluating the effectiveness of text pre-processing in sentiment analysis. *Applied Sciences, 12*(17), 8765. https://doi.org/10.3390/app12178765

3. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022

---

# PART B – PRACTICAL

## Task 1: Lowercasing

**Question: What information could a spam detector lose?**

Lowercasing converts uppercase letters into lowercase letters, making text more consistent. However, a spam detector may lose capitalization patterns that indicate urgency or emphasis, such as the word `URGENT` and the repeated uppercase letters in promotional messages. Therefore, capitalization may be useful information for spam detection.

## Task 2: Whitespace Normalization

The original text contained unnecessary spaces between words. After whitespace normalization, repeated spaces were replaced with single spaces, producing a cleaner sentence. The `repr()` function was used to make the whitespace differences visible.

## Task 3: Regex Cleaning

**Question: What was deleted, and why can this be a problem?**

The pattern `[^a-z\s]` removed punctuation marks such as `!!!`, `.`, and `!`, the currency symbol `£`, the digits and comma in `1,000`, and the phone number `0712345678`. These removals can be harmful because money amounts, phone numbers, and repeated punctuation may be useful indicators of spam. Custom replacements such as `MONEY` and `PHONE_NUMBER` preserve their meaning instead of deleting them.

## Task 4: Tokenization

The Python `split()` method produced 14 tokens, while NLTK's `word_tokenize()` produced 16 tokens because it separated the contraction and punctuation differently. For T4, NLTK produced 11 tokens and 10 unique tokens. This demonstrates that tokenization methods can affect token counts and vocabulary size.

## Task 5: Stopwords and Negation

**Question: How did the meaning change between the two results?**

The default stopword list removed `Not` but kept `n't` in the practical output. The custom stopword list preserved both forms of negation, making the negative meaning of the customer's complaint clearer. This is important because removing negation words can change the meaning of a review and negatively affect sentiment classification.

## Task 6: Stemming Comparison

### Stemming Results

| Original Word | Porter Stemmer | Lancaster Stemmer | Snowball Stemmer |
|---|---|---|---|
| connection | connect | connect | connect |
| connected | connect | connect | connect |
| connecting | connect | connect | connect |
| organization | organ | org | organ |
| organized | organ | org | organ |
| university | univers | univers | univers |
| universal | univers | univers | univers |
| flies | fli | fli | fli |
| wolves | wolv | wolv | wolv |
| better | better | bet | better |
| ran | ran | ran | ran |

**Question: Find a pair of words merged into the same stem and one stem that is not a real word.**

The words `connection`, `connected`, and `connecting` were all converted into `connect` by the three stemmers. This demonstrates how stemming groups different forms of a word. An example of a stem that is not a complete dictionary word is `organ`, produced from `organization` by Porter and Snowball.

## Task 7: Lemmatization

**Question: Which words changed only when you gave the POS hint?**

The default WordNet lemmatizer did not convert every word into its correct base form because it used the default noun setting. Providing POS hints changed `connected` and `connecting` into `connect`, `organized` into `organize`, `better` into `good`, and `ran` into `run`. This shows that grammatical information can improve lemmatization accuracy.

## Task 8: Sentence Splitting

**Question: How many sentences did you get? Did Mr. and a.m. break the sentences incorrectly?**

The `sent_tokenize()` function divided T3 into two sentences. The abbreviations `Mr.` and `a.m.` did not cause incorrect sentence splitting in this example. NLTK correctly identified the two sentence boundaries.

## Task 9: POS Tagging and Named Entity Recognition

### Entity Recognition Results

| Entity | NLTK Output | Expected Label | Correct? |
|---|---|---|---|
| Mr. | PERSON | Title, not a separate person | No |
| Smith | PERSON | PERSON | Partly |
| London | GPE | Location | Yes |
| Friday | Not recognized | DATE / Day | No |
| Apple | Not recognized | ORGANIZATION | No |

**Discussion:**

NLTK identified `Mr.` and `Smith` as separate PERSON entities instead of recognizing the complete name `Mr. Smith` as one entity. It correctly identified London as a geographical location. However, it did not recognize Friday as a day or Apple as an organization, even though Apple refers to a company in the sentence. This demonstrates that named entity recognition systems can miss entities or assign incomplete labels depending on their training and context.

## Task 10: Full Preprocessing Pipeline

**Question: Which version keeps the customer's complaint?**

The default pipeline removed the word `not` but retained `n't`, while the custom pipeline preserved both negation forms. The custom version therefore retained more of the negative meaning of the customer's complaint. For sentiment-related classification, preserving negation is important because it helps the model distinguish positive and negative opinions.

## Spam Detector: Preprocessing Recommendations

For a spam detector, I would use lowercasing, whitespace normalization, tokenization, and carefully designed regular expressions. The practical results showed that the pattern `[^a-z\s]` removed the money amount and phone number, which could be useful spam indicators. Instead of deleting them, I would replace them with labels such as `MONEY` and `PHONE_NUMBER`. I would also avoid blindly removing negation words and all punctuation because these elements can contain useful information. The custom stopword results showed that preserving negation helps retain the meaning of a message. Therefore, preprocessing should clean irrelevant text while preserving features that help identify spam.

---

# END OF ASSIGNMENT 02