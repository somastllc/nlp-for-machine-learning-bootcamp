# Assignment 2: Text Preprocessing

## Part A – Research

### 1. What is Text Preprocessing?

Text preprocessing is the process of cleaning, transforming, and preparing raw text before using it in Natural Language Processing (NLP) or Machine Learning models. Raw text may contain unnecessary spaces, punctuation, inconsistent capitalization, spelling variations, or other elements that make it difficult for a computer to process.

Text preprocessing is important because it helps make text more consistent and suitable for analysis. However, excessive preprocessing can remove information that is important for understanding the meaning of a sentence.

**Example:**

Original text:

`URGENT!!! You have WON a £1,000  cargiftd. Call NOW!`

After lowercasing and removing unnecessary punctuation:

`urgent you have won a 1000 gift card call now`

This text is easier to process, but some information, such as the urgency expressed through capital letters and repeated punctuation, has been removed.

In a spam detection system, removing too much information may reduce accuracy because words such as "urgent," "won," and "gift card" may be useful for identifying spam messages.

Therefore, preprocessing should be based on the purpose of the NLP task rather than removing everything that looks unnecessary.

---

### 2. Text Normalization

Text normalization is the process of transforming text into a consistent format. It reduces unnecessary variations in text and helps NLP systems process similar words or expressions consistently.

Common normalization techniques include:

| Technique                 | Description                                    | Example                                            |
| ------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| Lowercasing               | Converts uppercase letters to lowercase        | `HELLO` → `hello`                                  |
| Whitespace normalization  | Removes unnecessary spaces                     | `Hello   world` → `Hello world`                    |
| Unicode normalization     | Standardizes different Unicode representations | Similar characters may be represented consistently |
| Punctuation normalization | Removes or standardizes selected punctuation   | `hello!!!` → `hello!`                              |

**Example:**

Original:

`  HELLO    WORLD!!!  `

After normalization:

`hello world!!!`

Normalization can improve consistency, especially when text contains different forms of the same word.

However, normalization may also remove useful information. For example, in sentiment analysis, repeated punctuation and capitalization can express emotion or emphasis.

`I LOVE THIS!!!`

If the text is converted to:

`i love this`

The basic meaning remains, but the emotional intensity is reduced.

Unicode normalization is also useful when text contains visually similar characters represented differently. For example, some accented characters may have different underlying Unicode representations. Normalization can make these representations consistent.

Normalization should therefore be applied carefully, depending on the language and the task.

---

### 3. Tokenization

Tokenization is the process of dividing text into smaller units called tokens. Tokens may be words, subwords, characters, or sentences.

Tokenization allows NLP systems to process text as separate units instead of treating an entire sentence as one string.

There are three common tokenization approaches:

| Tokenization type | Description                                           | Example                           |
| ----------------- | ----------------------------------------------------- | --------------------------------- |
| Word-level        | Splits text into individual words                     | `I love NLP` → `I`, `love`, `NLP` |
| Subword-level     | Splits words into smaller meaningful or learned units | `unhappiness` → `un`, `happiness` |
| Character-level   | Splits text into individual characters                | `cat` → `c`, `a`, `t`             |

**Word-level tokenization**

Word-level tokenization is useful when the task depends on individual words, such as traditional sentiment analysis or text classification.

However, it may create a large vocabulary because different word forms are treated as separate tokens.

For example:

`connect`, `connected`, `connecting`, `connection`

These may be treated as four different vocabulary items.

**Subword-level tokenization**

Subword tokenization divides words into smaller units. It is commonly used in modern language models because it can handle unfamiliar words and reduce the number of completely unknown words.

For example, a word such as `unbelievable` may be divided into smaller units rather than requiring the entire word to exist in the vocabulary.

**Character-level tokenization**

Character-level tokenization processes text character by character. It can help with spelling variations, misspellings, and languages where word boundaries are difficult to identify.

However, character-level sequences can become longer and may require more computational resources.

**Vocabulary size**

Word-level tokenization may produce a larger vocabulary because different word forms are stored separately. Subword tokenization can reduce vocabulary size by reusing smaller units across different words. Character-level tokenization usually requires a relatively small set of characters, but the resulting sequences are longer.

The choice of tokenizer affects the vocabulary, sequence length, information representation, and performance of an NLP model.

---

### 4. Stopwords

Stopwords are common words that occur frequently in a language, such as `the`, `is`, `and`, `a`, and `of`. Some NLP tasks remove stopwords to reduce the number of words that the model processes.

**Example:**

Original sentence:

`The student is studying the subject.`

After removing common stopwords:

`student studying subject`

Stopword removal may reduce unnecessary text and simplify some text analysis tasks.

However, stopwords are not always unimportant. Their importance depends on the task.

For example:

`I do not like this product.`

If the stopwords `I`, `do`, and `not` are removed, the sentence may become:

`like product`

This changes the meaning because the original sentence expresses a negative opinion.

Stopword removal can also be harmful in question answering, translation, text generation, and other tasks where grammatical structure and meaning are important.

Stopwords should not automatically be removed from every NLP dataset. The decision should depend on the task, the language, and the meaning of the words.

---

### 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce words to a simpler or more normalized form.

They can help reduce vocabulary variations and allow a model to recognize related words more consistently.

#### Stemming

Stemming removes or modifies word endings using predefined rules. It does not always produce a valid dictionary word.

For example:

| Word         | Porter Stemmer example |
| ------------ | ---------------------- |
| connecting   | connect                |
| connected    | connect                |
| connection   | connect                |
| organization | organ                  |
| organized    | organ                  |

Stemming is generally faster and less linguistically complex because it mainly uses rules for removing or modifying word endings.

However, the resulting stems may not be real English words.

Different stemmers may produce different results because they use different rules.

**Porter Stemmer**

Porter Stemmer applies a series of rules to remove common word suffixes. It is commonly used in English text processing.

**Lancaster Stemmer**

Lancaster Stemmer uses more aggressive rules. It can reduce words to shorter forms than Porter, which may sometimes result in less readable stems.

**Snowball Stemmer**

Snowball Stemmer is an improved and language-oriented stemming algorithm. It provides stemming rules for multiple languages and is commonly used in NLP.

#### Lemmatization

Lemmatization converts a word into its dictionary form, called a lemma. It considers linguistic information and may use the word's part of speech.

Examples:

| Word    | Lemma |
| ------- | ----- |
| running | run   |
| better  | good  |
| wolves  | wolf  |
| flies   | fly   |

Unlike stemming, lemmatization aims to produce meaningful dictionary words.

In NLTK, WordNetLemmatizer can be used to lemmatize words. The result may depend on the supplied part-of-speech information.

For example:

- `running` may become `running` if treated as a noun.
- `running` may become `run` if treated as a verb.
- `better` may become `good` when treated as an adjective.

**When to use stemming or lemmatization**

Stemming may be suitable when speed and reducing word variations are important, and exact dictionary forms are not required.

Lemmatization may be more suitable when meaningful words and linguistic accuracy are important, such as in more meaning-sensitive text analysis.

---

### 6. Regular Expressions (Regex) in Text Cleaning

Regular expressions (Regex) are patterns used to search for, match, replace, or remove specific text elements.

They are useful in preprocessing because they can identify unwanted characters, numbers, email addresses, phone numbers, punctuation, and other text patterns.

**Example 1: Removing non-letter characters**

Pattern:

`[^a-zA-Z\s]`

This pattern matches characters that are not English letters or whitespace.

Example:

Original:

`Hello!!! I paid $100.`

After applying the pattern:

`Hello I paid `

This pattern removes punctuation and numbers, but it may also remove important information such as prices, phone numbers, and symbols.

**Example 2: Matching a phone number**

Pattern:

`\b\d{10}\b`

This pattern matches a sequence of exactly ten digits surrounded by word boundaries.

Example:

`Call 0712345678 today.`

The pattern can identify the phone number `0712345678`.

Regex can also replace phone numbers with a placeholder instead of deleting them:

`Call PHONE_NUMBER today.`

**Example 3: Replacing money**

Original:

`You have won £1,000!`

A regex pattern can identify the money expression and replace it with:

`You have won MONEY!`

This can preserve the fact that the message contains a monetary amount without keeping the exact amount.

**Limitations of Regex**

Regex can be useful for simple text cleaning, but broad patterns can remove information that is important for the task.

For example, removing all numbers may be harmful in financial analysis, SMS spam detection, medical text, or date-related text processing.

Regex rules should be designed according to the information the NLP task needs.

---

### 7. Case Study: The Effect of Preprocessing on NLP

Text preprocessing decisions can affect the performance of NLP systems because different techniques preserve or remove different information.

For example, consider a sentiment analysis system that classifies customer reviews as positive or negative.

Review:

`I don't think this jacket is worth the price. Not happy at all 😞`

Different preprocessing techniques may produce different results.

| Preprocessing technique | Result                                                              | Potential effect                                            |
| ----------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------- |
| Lowercasing             | `i don't think this jacket is worth the price. not happy at all 😞` | Makes text consistent while preserving most information     |
| Removing punctuation    | `i don't think this jacket is worth the price not happy at all 😞`  | May remove some emphasis or punctuation-related information |
| Removing stopwords      | May remove `not`                                                    | Can change negative sentiment                               |
| Stemming                | Reduces some words to stems                                         | May reduce vocabulary variations                            |
| Lemmatization           | Converts words to dictionary forms                                  | Preserves more meaningful word forms                        |
| Removing emojis         | Removes `😞`                                                        | May remove an important sentiment signal                    |

The review contains negative expressions such as `don't`, `not happy`, and the sad emoji.

If preprocessing removes negation or emojis, the model may lose information that helps identify negative sentiment.

For this reason, preprocessing should be selected according to the NLP task.

A sentiment analysis system may benefit from keeping negation words and emojis, while another task may not require them.

The goal of preprocessing is not to remove as much text as possible. The goal is to prepare text while preserving the information needed for the model.

