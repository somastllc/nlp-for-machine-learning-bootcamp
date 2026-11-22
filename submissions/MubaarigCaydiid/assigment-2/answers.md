# Assignment 2: Text Preprocessing

## 1. What Is Text Preprocessing?

**Text preprocessing** is the process of transforming raw text into a
cleaner, more consistent, and structured format before it is used by a
Natural Language Processing (NLP) system or machine learning model.

Raw text may contain uppercase and lowercase letters, punctuation,
repeated words, emojis, URLs, unnecessary whitespace, and other
inconsistencies. Preprocessing helps prepare this text for analysis and
modeling.

However, preprocessing must be performed carefully. Removing too much
information can change the meaning of a sentence and reduce model
performance.

### Example: Sentiment Analysis

Consider the following sentences:

``` text
Sentence 1: I love this product.
Sentence 2: I don't love this product.
```

The word `don't` changes the meaning of the second sentence. If it is
removed during preprocessing, the two sentences may appear to express
similar sentiments, even though their meanings are different.

**Key point:** A preprocessing operation should preserve information
that is important to the NLP task.

------------------------------------------------------------------------

## 2. Text Normalization

**Text normalization** is the process of converting text into a
consistent format by reducing unnecessary variations. Common
normalization techniques include lowercasing, whitespace normalization,
and Unicode normalization.

### 2.1. Lowercasing

**Lowercasing** converts uppercase letters into lowercase letters.

**Before:**

``` python
["HELLO", "Hello", "hello"]
```

**After:**

``` python
["hello", "hello", "hello"]
```

Lowercasing can help a model treat differently capitalized versions of
the same word as one form. This may reduce the size of the vocabulary
and simplify text processing.

However, lowercasing is not appropriate for every task. For example,
`US` may refer to the United States, while `us` is a pronoun. Converting
both to lowercase removes this distinction.

**Key point:** Apply lowercasing only when capitalization is not
important to the task.

### 2.2. Whitespace Normalization

**Whitespace normalization** removes or standardizes unnecessary spaces,
tabs, and line breaks.

**Before:**

``` python
text = "I   love    NLP"
```

**After:**

``` python
text = "I love NLP"
```

This produces more consistent text and prevents unnecessary whitespace
from affecting text processing.

### 2.3. Unicode Normalization

**Unicode** is a standard that allows computers to represent characters
and symbols from many writing systems.

Unicode normalization converts text into a standardized Unicode
representation. Some characters that look identical can be represented
internally in different ways. Normalization helps make these
representations consistent.

Unicode normalization and lowercasing are different operations:

-   **Unicode normalization:** Standardizes equivalent character
    representations.
-   **Lowercasing:** Converts uppercase letters into lowercase letters.

For example, lowercasing may remove the distinction between `Apple` (a
company name) and `apple` (a fruit). Unicode normalization does not
perform this kind of case conversion.

### 2.4. Replacing vs. Deleting

When cleaning text, **replacing** information and **deleting**
information are two different decisions.

**Example: Spam Classification**

  Operation              Result
  ---------------------- -----------------------------------------------
  Original text          Call 0634445566 or visit http://win-prize.com
  Replace with markers   Call `PHONE_NUMBER` or visit `URL`

**Why replacement can be useful:** Instead of deleting the phone number
and URL, we can replace them with markers such as `PHONE_NUMBER` and
`URL`. This preserves information about the types of content in the
message without requiring the model to learn every individual phone
number or web address.

This approach may help a spam classifier identify patterns commonly
found in spam messages.

### 2.5. Information Loss

Removing punctuation, emojis, numbers, or URLs without considering their
meaning can cause **information loss**. Some of these elements may
contain useful clues for an NLP task.

**Example: Spam Detection**

  -----------------------------------------------------------------------
  Operation                           Result
  ----------------------------------- -----------------------------------
  Original text                       CONGRATULATIONS!!! You have won
                                      \$5,000 💰 Call 0634445566 or visit
                                      http://win-prize.com

  Unnecessary deletion                congratulations you have won call
                                      or visit
  -----------------------------------------------------------------------

In this example, the currency amount, emoji, punctuation, phone number,
and URL have been removed. Some of these elements may be useful signals
for identifying a spam message.

**Key point:** Text cleaning should remove irrelevant information
without destroying useful signals.

------------------------------------------------------------------------

## 3. Tokenization

**Tokenization** is the process of dividing text into smaller units
called **tokens**. Depending on the tokenizer, tokens can be individual
characters, complete words, or subword units.

The three common approaches are:

1.  Character-level tokenization
2.  Word-level tokenization
3.  Subword-level tokenization

### 3.1. Character-Level Tokenization

**Character-level tokenization** divides text into individual
characters.

In Python, a string can be converted into a list of characters using
`list()`.

**Before:**

``` python
text = "SOMAST"
```

**After:**

``` python
tokens = list(text)
print(tokens)
```

**Output:**

``` python
["S", "O", "M", "A", "S", "T"]
```

Each character becomes a separate token.

### 3.2. Word-Level Tokenization

**Word-level tokenization** divides text into complete words. The
Natural Language Toolkit (NLTK) provides a function called
`word_tokenize()` for this purpose.

**Before:**

``` text
I love machine learning
```

**After:**

``` python
["I", "love", "machine", "learning"]
```

Each word becomes a separate token. In practice, tokenizers may also
treat punctuation as separate tokens.

### 3.3. Subword-Level Tokenization

**Subword-level tokenization** divides words into smaller units. These
units may represent meaningful word parts or frequently occurring
character sequences.

One algorithm used for subword tokenization is **Byte-Pair Encoding
(BPE)**.

**Illustrative example:**

``` text
unhappiness → un + happiness
```

The exact segmentation depends on the tokenizer and its learned
vocabulary. The example above is illustrative rather than a guaranteed
output from every BPE tokenizer.

Subword tokenization is useful because it allows models to represent
words that are not present as complete entries in the vocabulary.

### 3.4. Vocabulary

A **vocabulary** is the collection of unique tokens recognized by a
tokenizer or extracted from a text dataset. Repeated tokens are included
only once in the vocabulary.

**Before:**

``` python
tokens = ["I", "love", "AI", "I", "love", "Python"]
```

**After:**

``` python
vocabulary = ["I", "love", "AI", "Python"]
```

### 3.5. Vocabulary Size

**Vocabulary size** is the total number of unique tokens in a
vocabulary.

In the example above, the vocabulary size is **4** because there are
four unique tokens.

  Tokenization type   Typical vocabulary size   Token unit
  ------------------- ------------------------- ----------------------------------
  Word-level          Often large               Complete words
  Subword-level       Usually moderate          Word parts or frequent sequences
  Character-level     Usually small             Individual characters

The actual vocabulary size depends on the language, dataset, and
tokenizer configuration.

Many Large Language Models (LLMs) use subword or byte-level
tokenization. For example, GPT-2 uses byte-level Byte-Pair Encoding
(BPE), which helps it represent a wide range of text.

------------------------------------------------------------------------

## 4. Stopwords

**Stopwords** are frequently occurring words that may contribute
relatively little to certain NLP tasks. Examples in English include:

``` text
the, is, am, are, a, an, and, of, to, in
```

Stopword removal is the process of removing selected stopwords from
text. It can reduce the number of tokens that a model or text-processing
system needs to analyze.

### Example

**Original:**

``` text
The student is learning Machine Learning.
```

**After stopword removal:**

``` text
student learning Machine Learning
```

### 4.1. Benefits and Risks

  -----------------------------------------------------------------------
  Benefits                            Risks
  ----------------------------------- -----------------------------------
  Can reduce the number of tokens     May remove words that carry
  being processed.                    important meaning.

  Can simplify some text analysis     May change the sentiment of a
  tasks.                              sentence.

  Can reduce less useful information  May remove important contextual
  in some datasets.                   information.
  -----------------------------------------------------------------------

Consider these two sentences:

``` text
Sentence 1: I do not like this movie.
Sentence 2: I like this movie.
```

If the word `not` is removed, the sentences may appear to have the same
meaning. However, their sentiments are opposite.

**Key point:** Stopword removal should be based on the requirements of
the task, not applied automatically.

### 4.2. Task and Language Dependence

The usefulness of stopword removal depends on the NLP task and the
language being processed.

-   **Search engines:** Removing selected stopwords may sometimes
    improve search efficiency.
-   **Sentiment analysis:** Words such as `not` are important because
    they can reverse the meaning or sentiment of a sentence.
-   **Somali NLP:** English stopword lists are not directly suitable for
    Somali text. Somali NLP requires language-specific resources and
    careful consideration of the task.

------------------------------------------------------------------------

## 5. Stemming Algorithms and Lemmatization Approaches

Stemming and lemmatization are techniques used to reduce different forms
of a word to a simpler or more common representation. However, they work
differently.

### 5.1. Stemming

**Stemming** is the process of removing prefixes or suffixes from a word
to produce a stem.

A stemming algorithm generally applies predefined rules. It does not
necessarily understand the grammatical role or meaning of a word. As a
result, the output may not always be a valid dictionary word.

**Example:**

``` text
connecting  → connect
connected   → connect
connection  → connect
```

### 5.2. Lemmatization

**Lemmatization** converts a word into its base dictionary form, known
as a **lemma**. It can use vocabulary and grammatical information, such
as the word's part of speech.

**Examples:**

``` text
better  → good
running → run
```

The correct lemma depends on the context and grammatical role of the
word.

### 5.3. Stemming vs. Lemmatization

  -----------------------------------------------------------------------
  Feature                 Stemming                Lemmatization
  ----------------------- ----------------------- -----------------------
  Main approach           Removes word endings or Converts a word to its
                          other affixes using     dictionary base form.
                          rules.                  

  Dictionary use          Usually does not        May use a dictionary or
                          require a dictionary.   lexical database.

  Grammatical information Usually does not        Can consider
                          consider grammatical    grammatical
                          context.                information, such as
                                                  part of speech.

  Output                  May produce a form that Usually produces a
                          is not a valid word.    meaningful dictionary
                                                  form.

  Typical priority        Speed and simplicity.   Linguistic accuracy.
  -----------------------------------------------------------------------

**Examples:**

  Original word   Stemming   Lemmatization
  --------------- ---------- ---------------
  Studies         Studi      Study
  Running         Run        Run
  Better          Better     Good
  Caring          Care       Care

The exact output depends on the algorithm, language, and grammatical
information provided. For example, `better → good` requires the system
to recognize the relevant adjective meaning.

### 5.4. A. Porter Stemming Algorithm

The **Porter Stemmer** is a well-known stemming algorithm developed by
Martin Porter in 1980. It removes word suffixes according to a sequence
of predefined rules.

**Examples:**

  Original word   Porter Stemmer output
  --------------- -----------------------
  connected       connect
  connecting      connect
  relational      relat
  studies         studi

**When is the Porter Stemmer useful?**

It may be suitable when:

-   Processing speed is important.
-   Large collections of text need to be processed.
-   Building search engines or information retrieval systems.
-   A dictionary-valid output is not required.

**Limitation:** The Porter Stemmer can produce forms that are not valid
dictionary words. For example, `studies` becomes `studi`.

### 5.5. B. WordNet-Based Lemmatization

**WordNet** is a large lexical database of English words and their
semantic relationships.

The NLTK Python library provides `WordNetLemmatizer`, which uses WordNet
to perform lemmatization.

For example, `running` can be converted to `run` when it is identified
as a verb. If a part of speech (POS) is not specified, NLTK's
`WordNetLemmatizer` treats the word as a noun by default.

**Conclusion:** Stemming is often preferred when speed and simplicity
are important. Lemmatization is generally preferred when producing
linguistically meaningful base forms is important.

------------------------------------------------------------------------

## 6. Regular Expressions in Text Cleaning

### 6.1. What Are Regular Expressions?

**Regular Expressions (Regex)** are patterns used to search for,
identify, extract, remove, or replace specific parts of text.

In NLP preprocessing, Regex can be used to identify:

-   URLs
-   Email addresses
-   Multiple whitespace characters
-   Numbers
-   Punctuation
-   Unwanted text patterns

Python provides the `re` module for working with regular expressions.

### 6.2. How Can an Overly Broad Regex Damage Text?

An overly broad Regex pattern can accidentally remove important
information because it may match more characters or text than intended.

**Original text:**

``` text
Visit https://example.com for more information.
```

If a Regex pattern is designed incorrectly, it may remove the URL or
additional text that should have been preserved.

**Possible output:**

``` text
Visit  for more information.
```

This example shows how removing a URL can leave unnecessary whitespace.
In other cases, an overly broad pattern may remove meaningful words or
punctuation as well.

**Key point:** Regex patterns should be tested against different
examples before being applied to an entire dataset.

------------------------------------------------------------------------

## 7. Case Study: The Impact of Preprocessing on Text Classification

### Research Paper

  ---------------------------------------------------------------------------------
  Field                               Details
  ----------------------------------- ---------------------------------------------
  Title                               The Impact of Preprocessing on Text
                                      Classification

  Authors                             Alper Kürşat Uysal and Serkan Günal

  Publication                         Information Processing & Management

  Volume and issue                    Volume 50, Issue 1

  Year                                2014

  Pages                               104--112

  DOI                                 <https://doi.org/10.1016/j.ipm.2013.08.006>
  ---------------------------------------------------------------------------------

### 7.1. What Was the Research About?

The research investigated how different text preprocessing techniques
influence the performance of text classification systems.

**Text classification** is an NLP task in which a computer automatically
assigns text documents to predefined categories.

For example, an email classification system may identify whether an
incoming message belongs to the spam or non-spam category.

The researchers aimed to understand whether different preprocessing
techniques improve classification accuracy and whether their
effectiveness changes depending on the language and type of text.

### 7.2. What Preprocessing Techniques Were Compared?

The researchers investigated four commonly used preprocessing
techniques:

  -----------------------------------------------------------------------
  Technique                           Description
  ----------------------------------- -----------------------------------
  Tokenization                        Dividing text into smaller units,
                                      such as words.

  Stopword removal                    Removing selected frequently
                                      occurring words.

  Lowercase conversion                Converting uppercase letters into
                                      lowercase letters.

  Stemming                            Reducing related word forms to
                                      common stems.
  -----------------------------------------------------------------------

Rather than testing only one fixed preprocessing pipeline, the
researchers evaluated different combinations of these techniques.

### 7.3. What Data and Languages Were Used?

The study examined two text domains and two languages.

  Category       Data used
  -------------- ------------------------------------
  Text domains   Email documents and news documents
  Languages      English and Turkish

This allowed the researchers to examine whether preprocessing methods
performed differently depending on the language and type of text.

The study also considered different feature dimensions to investigate
the relationship between preprocessing, feature reduction, and
classification performance.

### 7.4. What Were the Results?

The experimental findings showed that preprocessing techniques do not
have the same effect in every situation.

The researchers found that selecting appropriate combinations of
preprocessing techniques could improve classification accuracy. However,
applying all preprocessing techniques automatically was not necessarily
the best choice.

The effectiveness of each technique depended on factors such as:

1.  The language being processed.
2.  The type of text document.
3.  The combination of preprocessing methods.
4.  The number of features used in classification.

### 7.5. What Can We Learn From This Research?

This study demonstrates that preprocessing decisions should be based on
the requirements of the NLP task rather than following one fixed
cleaning method.

For example, removing stopwords or applying stemming may help one
classification dataset but may not provide the same improvement in
another dataset.

The research supports the idea that preprocessing should be tested and
evaluated before being applied to a complete NLP system.

### Case Study Conclusion

The research by Uysal and Günal (2014) demonstrates that text
preprocessing can significantly influence text classification
performance. It also highlights the importance of language and text
domain when selecting preprocessing techniques.

The main lesson is that effective preprocessing requires choosing
suitable techniques for a specific dataset and NLP task instead of
applying every cleaning operation without evaluation.

------------------------------------------------------------------------

## References

1.  Porter, M. F. (1980). An algorithm for suffix stripping. *Program,
    14*(3), 130--137. https://doi.org/10.1108/eb046814

2.  Bird, S., Klein, E., & Loper, E. (2009). *Natural Language
    Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

3.  Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine
    translation of rare words with subword units. *Proceedings of the
    54th Annual Meeting of the Association for Computational
    Linguistics*, 1715--1725. https://doi.org/10.18653/v1/P16-1162

4.  Unicode Consortium. (n.d.). *Unicode Standard Annex #15: Unicode
    Normalization Forms*. https://www.unicode.org/reports/tr15/

5.  Python Software Foundation. (n.d.). *Regular Expression Operations
    --- Python Documentation*. https://docs.python.org/3/library/re.html

6.  NLTK Project. (n.d.). *WordNetLemmatizer API Documentation*.
    https://www.nltk.org/api/nltk.stem.wordnet.html

7.  Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on
    text classification. *Information Processing & Management, 50*(1),
    104--112. https://doi.org/10.1016/j.ipm.2013.08.006
