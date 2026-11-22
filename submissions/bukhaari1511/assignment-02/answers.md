# Assignment 2 – Preprocessing

**Student:** Bukhaari Abdirisaaq  
**GitHub Username:** bukhaari1511  
**Course:** NLP for Machine Learning Bootcamp

---

# Part A – Research

## 1. What Is Text Preprocessing?

Text preprocessing is the process of preparing raw text before it is used by a Natural Language Processing (NLP) system. It involves transforming text into a more consistent and useful format so that a computer can analyze it effectively.

NLP systems need preprocessing because human language contains many variations, including uppercase and lowercase letters, spelling differences, punctuation, extra spaces, abbreviations, and informal expressions. These variations can create unnecessary differences in the data and make it harder for machine learning models to recognize similar words or patterns.

However, excessive cleaning can remove information that is important for understanding the text. For example, consider the customer review:

"I am NOT happy with this product!"

If preprocessing removes the word "NOT" and all punctuation, the remaining text becomes "I am happy with this product". This changes the original meaning and can cause a sentiment analysis system to produce an incorrect prediction.

A real-life example is email spam detection. Messages containing words such as "URGENT", repeated exclamation marks, suspicious links, and phone numbers may indicate spam. Removing all capital letters, punctuation, links, and numbers could remove useful evidence.

Therefore, preprocessing should be selected according to the purpose of the NLP task rather than applying every cleaning technique automatically.

## 2. Text Normalization

Text normalization is the process of converting different written forms into a consistent representation. Its main purpose is to reduce unnecessary differences between text entries while preserving their meaning.

Common normalization techniques include:

**a. Lowercasing**

Lowercasing converts uppercase letters into lowercase letters.

Example:

- "HELLO" → "hello"
- "Python" → "python"
- "WELCOME" → "welcome"

This is useful in search engines because a search for "Python" can match documents containing "python", even when capitalization is different.

However, lowercasing can remove useful information. For example, "US" refers to the United States, while "us" is a personal pronoun. In a system that needs to distinguish these meanings, capitalization may be important.

**b. Whitespace Normalization**

Whitespace normalization removes unnecessary spaces, tabs, and line breaks.

Example:

Before: `"The    delivery     was    fast."`

After: `"The delivery was fast."`

This helps prevent the same sentence from being represented differently because of accidental spacing.

However, whitespace should not be changed blindly when processing documents where indentation, paragraph structure, or formatting carries meaning.

**c. Unicode Normalization**

Unicode normalization ensures that characters with equivalent visual representations are stored consistently.

For example, the character "é" can be represented using a single composed character or as the letter "e" followed by a combining accent mark. These representations may look identical but have different internal character sequences.

Unicode NFC normalization combines canonically equivalent representations into a consistent form.

NFKC can also convert certain compatibility characters into standard forms, but it may remove distinctions that are meaningful in some contexts.

A useful situation is processing international text where equivalent characters should match during searching or comparison.

A situation where normalization may remove useful information is processing names, foreign-language documents, or scientific text where accents, symbols, and special characters distinguish different terms.

## 3. Word-Level, Subword, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. A token may represent a complete word, a part of a word, or an individual character.

There are three common tokenization approaches.

### A. Word-Level Tokenization

Word-level tokenization divides text into complete words and sometimes separates punctuation marks.

Example:

Input: `"I love machine learning!"`

Tokens:

`["I", "love", "machine", "learning", "!"]`

Advantages:
- Easy to understand and interpret.
- Produces fewer tokens per sentence.
- Useful for traditional NLP tasks such as text classification and information retrieval.

Disadvantages:
- Can produce very large vocabularies.
- Unseen words may become unknown tokens.
- Different word forms may be treated as separate entries.

### B. Subword-Level Tokenization

Subword tokenization divides uncommon or long words into smaller meaningful or frequently occurring pieces.

For example:

`"unhappiness"` → `"un" + "happi" + "ness"`

The exact division depends on the tokenizer and its training data.

A well-known method is Byte Pair Encoding (BPE). It starts with smaller units and repeatedly combines frequently occurring sequences until it reaches a chosen vocabulary size.

Advantages:
- Handles rare and previously unseen words more effectively.
- Reduces the problem of unknown words.
- Can represent many words using a smaller vocabulary.

Disadvantages:
- Some words require several tokens.
- Token boundaries may not always correspond to meaningful linguistic units.

Subword tokenization is widely used in modern language models.

### C. Character-Level Tokenization

Character-level tokenization divides text into individual characters.

Example:

Input: `"NLP"`

Tokens:

`["N", "L", "P"]`

Advantages:
- Can represent unfamiliar words and spelling variations.
- Does not require a vocabulary containing every complete word.
- Useful in spelling correction and character-based language processing.

Disadvantages:
- Produces many tokens.
- Requires models to learn relationships between individual characters.
- May require more computational resources for longer texts.

### Comparison Table

| Feature | Word-level | Subword-level | Character-level |
|---|---|---|---|
| Token unit | Complete words | Word pieces | Characters |
| Vocabulary size | Large | Medium | Small |
| Unknown words | Possible | Less likely | Very unlikely |
| Tokens per sentence | Usually fewer | Medium | Many |
| Common applications | Traditional NLP | Modern language models | Spelling and character analysis |

**Vocabulary size** means the total number of unique token entries that a tokenizer can recognize.

For example, if a tokenizer contains 10,000 different tokens, its vocabulary size is 10,000.

Word-level tokenization generally requires a large vocabulary because it stores many complete words and their different forms. Character-level tokenization needs a much smaller vocabulary because it uses a limited set of characters. Subword tokenization provides a balance between the two.

## 4. Stopwords

Stopwords are common words that appear frequently in a language but may contribute relatively little information to certain NLP tasks.

In English, examples include:

- the
- is
- are
- and
- of
- to
- in

Stopword removal is the process of excluding selected words from text before further analysis.

### Benefits of Removing Stopwords

1. Reduces the number of tokens that a model needs to process.
2. Can reduce the size of text representations.
3. May improve information retrieval and document classification when frequent words are not useful for the task.
4. Can reduce unnecessary features in traditional machine learning models.

### Risks of Removing Stopwords

Some common words carry important meaning. Removing them can change the original message.

For example:

Original: "The customer is not satisfied."

After removing stopwords:

"customer satisfied"

The word "not" changes the meaning of the sentence. Removing it may cause a sentiment analysis system to interpret a negative review as positive.

Stopword lists must also depend on the language. A list created for English cannot automatically be applied to Somali, Arabic, or other languages because their grammar, vocabulary, and frequently used words are different.

The appropriate stopword list also depends on the NLP task. In information retrieval, removing certain frequent words may improve searching. In sentiment analysis, negation words should usually be preserved because they influence the emotional meaning of a sentence.

Therefore, stopword removal should be based on the language, dataset, and objective of the application.

## 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce different forms of words to a common representation. They help NLP systems recognize related words and reduce unnecessary vocabulary differences.

However, they use different methods.

### A. Stemming

Stemming is a rule-based process that removes prefixes or suffixes from words to produce a shorter stem.

For example:

- connected → connect
- connecting → connect
- studies → studi

The resulting stem does not always have to be a valid dictionary word.

**Porter Stemmer**

The Porter Stemmer was introduced by Martin Porter in 1980. It uses a sequence of suffix-removal rules to reduce English words to their stems.

It examines word endings and applies transformations according to conditions involving the remaining word.

Advantages:
- Relatively fast.
- Simple to implement.
- Useful for information retrieval and traditional text classification.

Disadvantages:
- May produce stems that are not valid English words.
- Does not always understand the grammatical role or meaning of a word.

**Lancaster Stemmer**

Lancaster is another rule-based stemming algorithm. It applies more aggressive suffix-removal rules than Porter in many cases.

It can produce shorter stems, but aggressive reduction may sometimes combine words that should remain different.

**Snowball Stemmer**

Snowball is a family of stemming algorithms developed by Martin Porter for several languages. Its English algorithm is a more developed approach to stemming and is available in NLTK.

### B. Lemmatization

Lemmatization reduces words to their dictionary base form, known as a lemma.

Unlike stemming, lemmatization uses linguistic information and vocabulary knowledge to produce valid words.

Examples:

- wolves → wolf
- better → good
- ran → run
- connecting → connect

A common approach is WordNet-based lemmatization.

WordNet is a lexical database that organizes English words into groups of related meanings and provides linguistic information.

In NLTK, the `WordNetLemmatizer` can use part-of-speech information to determine the appropriate base form.

For example:

- `lemmatize("running", pos="v")` → `run`
- `lemmatize("wolves", pos="n")` → `wolf`

### Comparison

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix removal | Linguistic and dictionary-based |
| Output | May not be a real word | Usually a valid dictionary form |
| Speed | Generally faster | Generally slower |
| Context | Limited grammatical awareness | Can use part of speech |
| Example | studies → studi | studies → study |

Stemming is preferred when processing speed and reducing vocabulary size are important, such as in large-scale information retrieval.

Lemmatization is preferred when grammatical accuracy and meaningful word forms are important, such as in sentiment analysis, document understanding, and linguistic applications.

## 6. Regular Expressions in Text Cleaning

Regular expressions, commonly called regex, are patterns used to search, identify, extract, replace, or remove specific sequences of characters in text.

Python provides the `re` module for working with regular expressions.

Regex is useful in NLP because raw text may contain unnecessary punctuation, repeated spaces, phone numbers, dates, email addresses, and other patterns that need special handling.

### Example 1: Removing Non-Alphabetic Characters

Pattern:

`[^a-z\s]`

Explanation:

- `[]` defines a character set.
- `^` at the beginning of the set means characters not included in the set.
- `a-z` represents lowercase English letters.
- `\s` represents whitespace characters.

This pattern matches any character that is not a lowercase English letter or whitespace.

It can remove punctuation, digits, and symbols from text.

### Example 2: Identifying Phone Numbers

Pattern:

`r"\b\d{10}\b"`

Explanation:

- `\b` identifies a word boundary.
- `\d` matches a digit.
- `{10}` requires exactly ten digits.

This pattern can identify a ten-digit phone number such as `0712345678`.

It can be used with `re.sub()` to replace the phone number with a general label such as `PHONE_NUMBER`.

### Risk of Overly Broad Patterns

An overly broad regular expression may remove important information.

For example, using `[^a-z\s]` on a spam message removes the currency symbol, money amount, phone number, and exclamation marks.

These details may be important evidence for a spam detector.

Therefore, replacing specific patterns with meaningful labels is sometimes better than deleting them completely.

## 7. Case Study: Comparing Preprocessing Techniques for Twitter Sentiment Analysis

**Research paper:**

Symeon Symeonidis, Dimitrios Effrosynidis, and Avi Arampatzis (2018), "A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis."

### Research Objective

The researchers investigated how different text preprocessing techniques affect the performance of sentiment analysis systems.

Twitter messages contain informal language, abbreviations, numbers, repeated punctuation, and other writing variations. These characteristics make social media text difficult to process using traditional NLP systems.

### Methods Compared

The researchers evaluated 16 preprocessing techniques using:

- Two Twitter sentiment analysis datasets.
- Four machine learning algorithms:
  - Linear Support Vector Classification.
  - Bernoulli Naive Bayes.
  - Logistic Regression.
  - Convolutional Neural Networks.

They compared individual preprocessing methods and combinations of multiple methods.

The evaluation considered classification accuracy and the number of features produced.

### Main Findings

The study found that preprocessing techniques do not have equal effects on model performance.

Some important findings were:

1. Lemmatization improved classification accuracy in their experiments.
2. Removing numbers and replacing contractions also improved accuracy.
3. Removing punctuation did not consistently improve accuracy.
4. Replacing repeated punctuation could be useful when techniques were combined.
5. The interaction between preprocessing techniques was important, meaning that a method's effect could depend on other methods used in the pipeline.

### Conclusion

The study demonstrates that preprocessing should be evaluated experimentally rather than selected based on assumptions.

A technique that improves one dataset may not improve another dataset or machine learning model.

For this reason, NLP developers should compare different preprocessing combinations using validation data before selecting a final pipeline.

**Reference:** Symeonidis et al. (2018), Expert Systems with Applications, DOI: 10.1016/j.eswa.2018.06.022.

---

# Part B – Practical

The practical section uses Python, NLTK, and the `re` module to investigate different preprocessing techniques.

## Task 1: Lowercasing

Lowercasing converts uppercase letters into lowercase letters. In the experiment, words such as `URGENT`, `WON`, and `NOW` became `urgent`, `won`, and `now`.

This makes different capitalization forms more consistent. However, a spam detector may lose capitalization patterns because excessive uppercase letters can indicate urgency or suspicious promotional language.

## Task 2: Whitespace Normalization

Whitespace normalization reduces repeated spaces and removes unnecessary spaces from the beginning and end of text.

The `re.sub(r"\s+", " ", T4).strip()` operation converts multiple whitespace characters into one space. This produces a cleaner and more consistent text representation without changing the words.

## Task 3: Regex Cleaning

The pattern `[^a-z\s]` removes every character that is not a lowercase English letter or whitespace. In T1, this removes exclamation marks, the currency symbol, the money digits, commas, the phone number, and the sentence punctuation.

The custom replacement rules preserve important information by replacing the money amount with `MONEY` and the phone number with `PHONE_NUMBER`. Repeated exclamation marks are also reduced to a single exclamation mark instead of being completely removed.

## Task 4: Tokenization

Tokenization divides text into smaller units called tokens. Python's `split()` separates text mainly at whitespace, while NLTK's `word_tokenize()` also separates punctuation and handles contractions differently.

The number of tokens depends on the tokenizer used. Vocabulary size is the number of unique tokens in the processed text, which is calculated using `len(set(tokens))`.

## Task 5: Stopwords and Negation

Default NLTK stopword removal can remove important negation words such as `not`, which may change the meaning of a customer's review.

The custom stopword list keeps `not`, `n't`, `no`, and `never`. This is more suitable for sentiment analysis because negative expressions are necessary for understanding whether a customer is satisfied or dissatisfied.

## Task 6: Stemming

The experiment compares Porter, Lancaster, and Snowball stemming algorithms using eleven English words.

### Stemming Comparison Table

The following table shows the expected NLTK results. Verify it against the actual printed output before submission.

| Original word | Porter | Lancaster | Snowball |
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

**Explanation:**

The three stemmers sometimes produce the same stem for different original words. For example, `connection`, `connected`, and `connecting` are reduced to `connect` by all three algorithms.

Some stemming results are not complete English dictionary words. For example, `university` becomes `univers`, demonstrating that stemming focuses on reducing word forms rather than producing valid dictionary entries.

## Task 7: Lemmatization

WordNetLemmatizer converts words into their dictionary base forms. Its default behavior assumes that words are nouns unless a different part of speech is supplied.

Providing the correct part-of-speech hint can improve the result because verbs, nouns, and adjectives have different grammatical forms.

For example, `ran` becomes `run` when the verb hint is supplied, while `better` becomes `good` when the adjective hint is supplied.

Words expected to change when using the appropriate POS hint include `connected`, `connecting`, `organized`, `better`, and `ran`.

## Task 8: Sentence Splitting

Sentence splitting divides a paragraph into individual sentences. NLTK's `sent_tokenize()` uses sentence-boundary information to identify where one sentence ends and another begins.

For T3, the expected result is two sentences. The abbreviation `Mr.` and the time expression `a.m.` should not incorrectly create additional sentence boundaries.

## Task 9: POS Tagging and Named Entity Recognition

POS tagging assigns grammatical categories to individual tokens, such as nouns, verbs, adjectives, and proper nouns.

Named Entity Recognition identifies important entities in text, including people, organizations, and geographical locations.

### Entity Table

The expected entities in T3 and their correct real-world labels are:

| Entity | Correct label | Explanation |
|---|---|---|
| Mr. Smith | PERSON | A person's name |
| London | GPE | A geographical location |
| Friday | DATE | A day of the week |
| Apple | ORGANIZATION | A company in this sentence |

**Model evaluation:** NLTK's `ne_chunk()` does not necessarily recognize every expected entity or assign the correct label. In particular, its standard named-entity categories do not include a dedicated DATE category. The actual predictions must be compared with the expected labels using the printed output.

## Task 10: Full Preprocessing Pipeline

The full pipeline applies five operations in sequence: lowercasing, regex cleaning, tokenization, stopword removal, and lemmatization.

The first experiment uses NLTK's default English stopword list. The second experiment uses a custom list that preserves negation words.

The default approach may remove `not`, which can change the customer's negative complaint. The custom approach preserves negation and therefore retains more of the original sentiment.

## Spam Detection: Recommended Preprocessing Steps

For a spam detection system, I would use whitespace normalization, carefully selected text normalization, tokenization, and targeted regex replacement. I would replace phone numbers and money amounts with markers such as `PHONE_NUMBER` and `MONEY` rather than deleting them, because their presence may provide useful spam evidence. I would avoid blindly removing punctuation, numbers, capitalization patterns, URLs, and repeated exclamation marks because these features can help identify suspicious messages. I would also evaluate whether stemming or stopword removal improves performance instead of applying them automatically.

---

# References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

2. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

3. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725. https://doi.org/10.18653/v1/P16-1162

4. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2018.06.022

5. *Comparison of Text Preprocessing Methods*. (2023). *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

6. Unicode Consortium. *Unicode Normalization FAQ*. https://www.unicode.org/faq/normalization.html

7. Python Software Foundation. *Regular Expression Operations – Python Documentation*. https://docs.python.org/3/library/re.html

8. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

9. NLTK Project. *Natural Language Toolkit Documentation*. https://www.nltk.org/

