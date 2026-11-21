# Part A – Research

## Introduction

Text preprocessing prepares raw text for NLP tasks by making it more consistent and suitable for analysis. The correct approach depends on the language, dataset, and task.

## 1. What is text preprocessing, and why is it needed?

Text preprocessing is the process of preparing raw text for computer analysis. It can include cleaning, normalization, tokenization, and removing or transforming selected words.

It helps reduce irrelevant variation and makes text easier for NLP models to process. However, excessive cleaning can remove important meaning. For example, removing the word **“not”** from “I do not like this product” changes the sentiment and may make the review appear positive.

## 2. What is text normalization?

Text normalization converts different forms of text into a consistent representation.

Common methods include:
- **Lowercasing:** changes `HELLO` to `hello`.
- **Whitespace normalization:** changes repeated spaces into a single space.
- **Unicode normalization:** converts equivalent character representations into a consistent form, such as NFC.

Normalization helps when matching search terms written with different capitalization or spacing. However, it can remove useful information: lowercasing may erase the distinction between `Apple` (a company) and `apple` (a fruit).

## 3. How do word, subword, and character tokenization differ?
Tokenization divides text into smaller units called tokens.

| Type | How it works | Main advantage | Main limitation |
|---|---|---|---|
| Word-level | Splits text into complete words | Easy to interpret | Large vocabulary; rare words can be unknown |
| Subword-level | Splits words into smaller units | Handles rare and unfamiliar words | Word pieces may be harder to interpret |
| Character-level | Splits text into individual characters | Can represent almost any spelling | Creates long sequences |

**Vocabulary size** is the number of distinct tokens a model can use. Word tokenizers often need larger vocabularies, subword tokenizers use a controlled set of word pieces, and character tokenizers usually use a small character set.

## 4. What are stopwords, and should they be removed?
Stopwords are common words such as `the`, `is`, `and`, and `of` that may contribute little to some tasks.

Removing them can reduce the number of features and help certain document-classification or search systems. But removal can also destroy meaning. For example, deleting `not` makes “I like it” and “I do not like it” much more similar.

Stopword lists must be selected according to the task and language. A list designed for English may not work for Somali, and sentiment analysis or question answering may need common grammatical words to be preserved.

## 5. What is the difference between stemming and lemmatization?
Both methods reduce related word forms, but they work differently.

- **Stemming** removes word endings using rules. The Porter Stemmer, introduced by Martin Porter in 1980, may reduce `studies` to `studi`. It is fast and useful in some information-retrieval systems, but its output may not be a dictionary word.
- **Lemmatization** identifies a word's dictionary base form (lemma), often using vocabulary and grammatical information. For example, `children` becomes `child` and `better` may become `good`. WordNet is one established English lexical resource used for lemmatization.

Stemming is often preferred when speed and simple matching are priorities. Lemmatization is preferable when linguistically meaningful base forms are important.

## 6. How are regular expressions used in text cleaning?
Regular expressions (RegEx) are patterns used to find, extract, or replace text.

**Example 1 — repeated whitespace**

```python
r"\s+"
```

This matches one or more whitespace characters. It can be used to replace repeated spaces or line breaks with a single space.

**Example 2 — a simple email pattern**

```python
r"\b[\w.-]+@[\w.-]+\.\w+\b"
```

This matches many common email-address formats, although it is not a complete email validator.

An overly broad pattern can delete useful information. For example, removing every character that is not an English letter can erase spaces, numbers, punctuation, and letters from other writing systems.

## 7. What does research show about preprocessing choices?
HaCohen-Kerner, Miller, and Yigal (2020) studied how preprocessing affects text classification using a Bag-of-Words representation. They tested four datasets and compared methods including lowercasing, spelling correction, HTML and punctuation removal, stopword removal, and repeated-character reduction.

Their findings showed that preprocessing effects varied by dataset. Stopword removal improved results for three datasets in the reported baseline comparisons, while lowercasing was the only individual technique that significantly improved the fourth. Combinations of techniques also produced improvements.

The key lesson is that no single cleaning method is best for every NLP task. Preprocessing choices should be tested on the actual dataset and evaluated using an appropriate metric.


# Part B – Practical Report

### Task 1: Lowercasing
Lowercasing `T1` converts all uppercase letters (`URGENT`, `WON`, `NOW`) to lowercase (`urgent`, `won`, `now`). A spam detector could lose important stylistic signals, because excessive capitalization (ALL CAPS) is a strong indicator of urgency, aggressive marketing, and spam messages.

### Task 3: Regex Cleaning
The pattern `[^a-z\s]` removed all non-lowercase-letter and non-whitespace characters, specifically deleting `!!!`, `£`, `1`, `,`, `0`, `.`, `0712345678`, and `!`. Replacing sensitive or structural entities with placeholders (`PHONE_NUMBER`, `MONEY`, `!`) is better than deleting them because it preserves the context that the message contains financial bait and a call-to-action phone number.

### Task 5: Stopwords and Negation
When using NLTK's default stopword list, the negation words (`n't` and `not`) were removed, leaving positive-sounding keywords (`think`, `jacket`, `worth`, `price`, `happy`). Keeping negation words (`n't`, `not`) in the custom stopword list preserved the true negative sentiment of the customer's complaint.

### Task 6: Stemming Comparison

| Word | Porter | Lancaster | Snowball |
| :--- | :--- | :--- | :--- |
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

Porter and Snowball merged `organization` and `organized` into the same stem (`organ`), and all three stemmers merged `university` and `universal` into `univers`. Several resulting stems—such as `univers`, `fli`, and `wolv`—are not real English words because stemmers aggressively strip suffixes using rules rather than a dictionary.

### Task 7: Lemmatization
Without a POS hint, `WordNetLemmatizer` assumes words are nouns by default, so verbs and adjectives like `connected`, `connecting`, `organized`, `better`, and `ran` remained unchanged. They only changed to their true dictionary base forms (`connect`, `organize`, `good`, `run`) when the correct POS hints (`pos="v"` or `pos="a"`) were provided.

### Task 8: Sentence Splitting
`sent_tokenize` split  into **2 sentences**: `"Mr. Smith arrived in London on Friday."` and `"He met the Apple team at 9 a.m."`. The abbreviations `Mr.` and `a.m.` did not break the sentences in the wrong places because NLTK's Punkt tokenizer is trained to recognize common titles and time abbreviations.

### Task 9: POS Tagging and Named Entity Recognition (NER)

| Entity | Predicted Label | Correct? | Notes |
| :--- | :--- | :--- | :--- |
| Smith | PERSON | Yes | Correctly identified as a person's name. |
| London | GPE | Yes | Correctly identified as a location (city). |
| Apple | PERSON | No | Misclassified as a person; in this context, it is an `ORGANIZATION`. |

The model missed **Friday** (a `DATE` entity) and did not include **Mr.** as part of the person's title/name, while also misclassifying **Apple** as a `PERSON` instead of an `ORGANIZATION`.

### Task 10: Full Pipeline
The second version (using the custom stopword list that keeps negation words) kept the customer's complaint because it preserved `not` alongside `happy` (`['dont', 'think', 'jacket', 'worth', 'price', 'not', 'happy']`). The first version removed `not`, leaving behind `happy` and reversing the meaning of the customer's feedback.

### Preprocessing Strategy for a Spam Detector
For a spam detector, I would use **regex entity replacement** and **tokenization**, while avoiding **lowercasing**, **default stopword removal**, and **aggressive stemming**. As shown in Task 3, replacing specific numbers and currency with placeholders like `MONEY` and `PHONE_NUMBER` captures spam patterns without overfitting to unique digits. Conversely, Task 1 demonstrated that lowercasing erases `ALL CAPS` shouting (`URGENT!!!`, `WON`, `NOW`), which is a strong indicator of spam, while Task 5 showed that default stopword removal can erase critical negation or urgency cues. Finally, avoiding aggressive stemming (Task 6) prevents distinct words like `university` and `universal` from colliding into distorted roots (`univers`), preserving exact vocabulary nuances that separate legitimate messages from spam.
