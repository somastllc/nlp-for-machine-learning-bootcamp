 # Part A – Research: Text Preprocessing

## 1. Definition and Importance of Text Preprocessing

Text preprocessing is the process of transforming raw textual data into a structured and consistent format before it is used by a Natural Language Processing (NLP) system. Raw text may contain inconsistent capitalization, unnecessary whitespace, punctuation, spelling variations, special characters, and other irregularities.

NLP systems need preprocessing because computers do not directly understand human language in the same way people do. Many traditional NLP and machine learning methods require text to be divided into tokens and represented as numerical features. Preprocessing helps organize the text, reduce unnecessary variation, and make useful patterns easier for algorithms to identify.

However, more cleaning does not always produce better results. Removing information that appears unnecessary can destroy important meaning. The correct preprocessing method depends on the purpose of the NLP application.

**Real-life example:**

Consider a customer review:

"I am NOT satisfied with this product!"

If preprocessing removes the word "NOT" and the exclamation mark, the remaining text may look like:

"I am satisfied with this product"

The meaning has changed from negative to positive. A sentiment analysis model could therefore classify the review incorrectly.

This example demonstrates that preprocessing should preserve meaningful information rather than simply remove as many characters and words as possible.

In conclusion, effective preprocessing balances data cleaning with information preservation.

## 2. Text Normalization

Text normalization is the process of converting different representations of text into a consistent form. It helps NLP systems recognize that different representations may communicate the same information.

Common normalization techniques include:

### 2.1 Lowercasing

Lowercasing converts uppercase letters into lowercase letters.

Example:

- "COMPUTER"
- "Computer"
- "computer"

After lowercasing, all three become:

`computer`

This is useful in search engines and document classification because it prevents capitalization differences from creating unnecessary separate terms.

However, lowercasing can remove useful information. For example, "US" can refer to the United States, while "us" is a pronoun. In some applications, capitalization helps identify proper nouns and abbreviations.

### 2.2 Whitespace Normalization

Whitespace normalization removes unnecessary spaces, tabs, and line breaks.

Example:

Before:

`Natural     Language       Processing`

After:

`Natural Language Processing`

This is useful when combining text collected from different websites, documents, or databases. It creates consistent spacing and prevents formatting differences from affecting tokenization.

### 2.3 Unicode Normalization

Unicode allows characters to have different internal representations even when they look identical.

For example, the character "é" can be represented as:
- A single precomposed character.
- The letter "e" followed by a combining accent mark.

Unicode normalization provides standard forms for handling such differences.

The four main Unicode normalization forms are:

| Form | Description |
|---|---|
| NFC | Canonical decomposition followed by composition |
| NFD | Canonical decomposition |
| NFKC | Compatibility decomposition followed by composition |
| NFKD | Compatibility decomposition |

NFC is commonly appropriate for general text because it makes canonically equivalent representations consistent. NFKC can be useful for matching identifiers and compatibility characters, but it may remove distinctions that carry information.

**Example where normalization helps:**

A search system may receive two versions of a name containing "é". Unicode NFC normalization can make their internal representations consistent, allowing reliable comparison and matching.

**Example where normalization removes useful information:**

Converting all text to lowercase may remove the distinction between "Apple" as a company name and "apple" as a common noun. Similarly, compatibility normalization may convert certain special presentation characters into ordinary characters.

Therefore, normalization should be selected according to the language, data, and application.

## 3. Word-Level, Subword, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. A token may represent a complete word, a subword unit, or an individual character.

There are three important tokenization approaches.

### 3.1 Word-Level Tokenization

Word-level tokenization divides text into complete words.

Example:

Input:

`Natural Language Processing is useful.`

Output:

`["Natural", "Language", "Processing", "is", "useful"]`

Advantages:
- Tokens are easy for humans to understand.
- Common words can be represented as single units.
- Useful for traditional text classification and word-frequency analysis.

Disadvantages:
- The vocabulary can become very large.
- New or misspelled words may not exist in the vocabulary.
- Different forms of the same word may be treated as separate tokens.

### 3.2 Subword-Level Tokenization

Subword tokenization divides words into smaller meaningful units. Common words may remain complete, while rare or unfamiliar words are divided into smaller pieces.

Example:

`unhappiness`

Possible subword representation:

`["un", "happiness"]`

Another possible representation:

`["un", "happi", "ness"]`

The exact result depends on the tokenizer and its learned vocabulary.

Common subword approaches include:
- Byte Pair Encoding (BPE)
- WordPiece
- Unigram Language Model

Advantages:
- Handles many unfamiliar words.
- Reduces the need for extremely large vocabularies.
- Can represent related word forms using shared subword units.

Disadvantages:
- Some words may be divided into pieces that are difficult to interpret.
- Tokenization can be more complicated than simple word splitting.

Subword tokenization is widely used in modern transformer-based language models.

### 3.3 Character-Level Tokenization

Character-level tokenization divides text into individual characters.

Example:

Input:

`CAT`

Output:

`["C", "A", "T"]`

Advantages:
- Can represent unusual and previously unseen words.
- Useful for spelling correction and character-level language modeling.
- Does not require a separate vocabulary entry for every complete word.

Disadvantages:
- A sentence produces many more tokens.
- Individual characters may contain little semantic information.
- Models may need more processing to understand relationships between characters.

### 3.4 Comparison

| Feature | Word-level | Subword-level | Character-level |
|---|---|---|---|
| Basic unit | Complete word | Word pieces | Character |
| Vocabulary size | Usually large | Usually moderate | Usually small |
| Unknown words | More likely | Can split into known pieces | Can represent with known characters |
| Sequence length | Usually shorter | Medium | Usually longer |
| Main advantage | Easy to understand | Good balance of coverage and vocabulary size | Flexible for unusual words |
| Common use | Traditional NLP | Transformer language models | Spelling and character modeling |

### 3.5 Vocabulary Size

Vocabulary size is the number of unique tokens that a tokenizer or NLP system can recognize or represent.

For example, consider:

`cat dog cat bird dog`

The tokens are:

`["cat", "dog", "cat", "bird", "dog"]`

The unique tokens are:

`{"cat", "dog", "bird"}`

Therefore, the vocabulary size is **3**.

The choice of tokenizer affects vocabulary size:

- Word-level tokenization may require thousands or millions of complete word entries.
- Subword tokenization can represent many words using a smaller collection of reusable pieces.
- Character-level tokenization generally requires a relatively small character vocabulary.

A larger vocabulary can increase memory requirements, while a very small vocabulary can produce longer token sequences. The best choice depends on the language and NLP application.

## 4. Stopwords: Benefits and Risks of Removing Them

Stopwords are words that occur frequently in a language and may contribute relatively little information to certain NLP tasks.

Examples in English include:

- the
- is
- a
- an
- and
- of
- to

Stopword lists are not universal. They are usually selected according to the language, domain, and purpose of the application.

### 4.1 Benefits of Removing Stopwords

Removing selected stopwords can:

1. Reduce the number of tokens that a system processes.
2. Reduce the number of features in some text representations.
3. Help emphasize important content words.
4. Reduce unnecessary information in certain document classification and information retrieval tasks.

For example, in a simple topic classification system, words such as "the" and "is" may provide less information about the topic than words such as "computer", "medicine", or "education".

### 4.2 Risks of Removing Stopwords

Removing stopwords can damage meaning because common words are not always unimportant.

For example:

- "I like this product."
- "I do not like this product."

If the word "not" is removed, the second sentence may become similar to the first one.

Stopword removal can also damage:
- Question answering, where words such as "who", "what", and "where" are important.
- Machine translation, where grammatical words help preserve sentence structure.
- Sentiment analysis, where negation words change emotional meaning.
- Search queries and phrase matching, where word order and function words matter.

### 4.3 Why Stopword Lists Depend on Language and Task

Different languages have different grammatical structures and common words. A stopword list designed for English cannot automatically be applied to Somali, Arabic, Turkish, or other languages.

Even within one language, different tasks require different decisions.

For example:

| NLP task | Stopword consideration |
|---|---|
| Topic classification | Some frequent grammatical words may be removed |
| Sentiment analysis | Negation words should generally be preserved |
| Question answering | Question words must be preserved |
| Machine translation | Grammatical words are important |
| Search engines | Removal depends on query and search requirements |

Therefore, stopword removal should be evaluated rather than applied automatically.

## 5. Stemming and Lemmatization

Stemming and lemmatization are text normalization techniques that reduce different forms of words.

Although they have similar goals, they use different methods.

### 5.1 Stemming

Stemming removes or modifies word endings using predefined rules to produce a shorter stem.

Example:

| Original word | Possible stem |
|---|---|
| connected | connect |
| connecting | connect |
| studies | studi |
| happiness | happi |

A stem does not necessarily have to be a valid English dictionary word.

### 5.2 Porter Stemming Algorithm

One well-known stemming algorithm is the Porter Stemmer, introduced by Martin Porter in 1980.

The algorithm uses a sequence of suffix-removal rules. It examines word endings and applies transformations according to conditions on the remaining stem.

For example, related forms such as:

- CONNECT
- CONNECTED
- CONNECTING
- CONNECTION

may be reduced to a shared stem.

The original algorithm was developed for information retrieval, where grouping related word forms can improve matching and reduce vocabulary size.

**When stemming is preferred:**

- Large-scale document indexing.
- Search systems.
- Traditional text classification.
- Applications where reducing vocabulary size is important and exact dictionary forms are not necessary.

One limitation is that stemming can produce incomplete or misleading word forms.

### 5.3 Lemmatization

Lemmatization converts a word into its dictionary base form, known as a lemma.

Unlike simple suffix removal, lemmatization uses linguistic knowledge and vocabulary resources. Some approaches also use grammatical information, such as part of speech.

Examples:

| Original word | Lemma |
|---|---|
| running | run |
| wolves | wolf |
| better | good |
| studies | study |

The result depends on the word's grammatical role. For example, the correct lemma of a word may differ depending on whether it is used as a noun or verb.

### 5.4 WordNet-Based Lemmatization

WordNet is a lexical database of English words organized into groups of related meanings called synsets.

A WordNet-based lemmatizer can use this lexical information to find dictionary forms. In NLTK, the `WordNetLemmatizer` uses noun as its default part of speech unless another POS is provided.

For example:

- `running` with verb POS can become `run`.
- `wolves` with noun POS becomes `wolf`.
- `better` with adjective POS can become `good`.

**When lemmatization is preferred:**

- Sentiment analysis.
- Applications requiring meaningful dictionary forms.
- Linguistic analysis.
- Tasks where grammatical information is important.

### 5.5 Comparison

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix reduction | Dictionary and linguistic analysis |
| Output | Stem | Dictionary lemma |
| Valid dictionary word | Not always | Usually |
| POS information | Usually not required | Can improve accuracy |
| Processing | Generally simpler and faster | Often more computationally involved |
| Preferred use | Search and indexing | Meaning-sensitive NLP tasks |

In conclusion, stemming is useful when simplicity and speed are important, while lemmatization is preferable when meaningful word forms and grammatical information matter.

## 6. Regular Expressions in Text Cleaning

Regular expressions (regex) are patterns used to identify, match, extract, replace, or remove particular sequences of characters.

In NLP preprocessing, regex is useful for handling phone numbers, email addresses, dates, repeated punctuation, whitespace, and unwanted character patterns.

Python provides the `re` module for working with regular expressions.

### 6.1 Example Pattern 1: Email Address

Pattern:

```python
r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
```

This pattern is designed to match common email address formats.

It contains:
- `[A-Za-z0-9._%+-]+` – one or more permitted characters before the @ symbol.
- `@` – the required at sign.
- `[A-Za-z0-9.-]+` – the domain name.
- `\.` – a literal dot.
- `[A-Za-z]{2,}` – a final domain suffix containing at least two letters.

Example:

`Contact us at support@example.com`

The pattern can identify:

`support@example.com`

This can be useful when extracting contact information or replacing email addresses with a placeholder.

It is a practical pattern rather than a complete validation standard for every possible email address.

### 6.2 Example Pattern 2: Phone Number

Pattern:

```python
r"\b\d{10}\b"
```

This pattern matches a sequence of exactly ten digits bounded by word boundaries.

Explanation:
- `\b` – word boundary.
- `\d` – a digit from 0 to 9.
- `{10}` – exactly ten repetitions.
- The final `\b` – another word boundary.

Example:

`Call 0712345678 today.`

The pattern identifies:

`0712345678`

It can be replaced with a placeholder:

```python
import re

text = "Call 0712345678 today."

cleaned = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", text)

print(cleaned)
```

Output:

`Call PHONE_NUMBER today.`

This example assumes a ten-digit number format. Real telephone numbers vary by country, so a production system should use a more appropriate pattern.

### 6.3 How Overly Broad Regex Can Damage Text

An overly broad regex can remove meaningful information along with unwanted characters.

For example:

```python
re.sub(r"[^a-z]", "", text.lower())
```

This pattern removes every character except lowercase English letters.

Consider:

`I don't like this product!`

The result may become:

`idontlikethisproduct`

The spaces and apostrophe have disappeared. The original token boundaries and contraction structure are lost.

It can also remove numbers, emojis, accented letters, and characters from other writing systems.

This is particularly harmful when processing multilingual text or sentiment data.

Therefore, regex patterns should be designed carefully and tested against realistic examples before being used on an entire dataset.

## 7. Case Study: The Impact of Preprocessing on Text Classification

### Research Paper

Uysal, A. K., and Günal, S. (2014). **The Impact of Preprocessing on Text Classification.** Information Processing & Management, 50(1), 104–112.

DOI: https://doi.org/10.1016/j.ipm.2013.08.006

### 7.1 Research Objective

The researchers investigated how different text preprocessing decisions affect the performance of text classification systems.

Their study focused on whether preprocessing methods consistently improve classification accuracy or whether their effectiveness changes depending on the data.

### 7.2 Task and Experimental Setup

The research examined text classification in two different domains:

- Email.
- News.

It also investigated two languages:

- English.
- Turkish.

The researchers compared combinations of commonly used preprocessing operations and evaluated their effects on classification accuracy and feature dimensionality.

This approach allowed them to examine whether the usefulness of a preprocessing operation depended on the language and type of documents.

### 7.3 Preprocessing Choices Compared

The study evaluated combinations of common preprocessing operations, including:

- Lowercasing.
- Stopword removal.
- Stemming.
- Different combinations of these operations.

The experiments also considered different feature dimensions, including 10, 20, 50, 100, 200, 500, 1,000, and 2,000 features.

The researchers used these comparisons to investigate how preprocessing choices affected the classification system under different experimental conditions.

### 7.4 Main Findings

The study found that preprocessing can significantly affect text classification performance.

However, the results did not support the idea that every preprocessing operation should always be enabled.

Instead, the effectiveness of preprocessing depended on factors such as:

1. The language being processed.
2. The document domain.
3. The combination of preprocessing operations.
4. The number of selected features.

For example, a technique that helps classify one type of document may not provide the same improvement when applied to another language or domain.

The research therefore emphasizes selecting suitable combinations of preprocessing methods rather than applying all possible cleaning operations automatically.

### 7.5 Importance of the Case Study

This research is important because it demonstrates that text preprocessing is not simply a technical cleaning step. It is part of the design of an NLP system.

A developer should test different preprocessing configurations and evaluate their effect on the actual task.

For example, removing stopwords may help a topic classification system, but removing negation words may damage a sentiment analysis system.

### 7.6 Conclusion of the Case Study

The main lesson from Uysal and Günal's research is that preprocessing decisions should be based on experimental evidence.

There is no single preprocessing pipeline that is best for every language and text classification problem.

---

# Overall Conclusion

Text preprocessing is an essential part of many Natural Language Processing systems. It prepares raw text for analysis by improving consistency, organizing tokens, and reducing unnecessary variation.

This research examined normalization, tokenization, vocabulary size, stopword removal, stemming, lemmatization, regular expressions, and experimental evidence about preprocessing.

One important conclusion is that preprocessing must preserve information that is relevant to the task. Removing negation, punctuation, capitalization, or special characters without considering their meaning can damage the quality of an NLP system.

The research also demonstrates that tokenization and normalization choices affect vocabulary size and how models represent language. Word-level, subword-level, and character-level approaches each have advantages and limitations.

Finally, the case study by Uysal and Günal (2014) shows that preprocessing performance depends on language, domain, and the combination of methods used.

Therefore, an effective NLP developer should understand each preprocessing technique, select methods according to the task, and evaluate the results instead of assuming that more cleaning always produces better performance.

# References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Chapters 2 and 5.  
   https://nlp.stanford.edu/IR-book/

2. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.  
   https://doi.org/10.1108/eb046814

3. Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112.  
   https://doi.org/10.1016/j.ipm.2013.08.006

4. Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode Normalization Forms*.  
   https://www.unicode.org/reports/tr15/

5. Hugging Face. (n.d.). *Tokenization Algorithms*. Transformers Documentation.  
   https://huggingface.co/docs/transformers/en/tokenizer_summary

6. Hugging Face. (n.d.). *Tokenizers Course: Tokenization*.  
   https://huggingface.co/docs/course/main/en/chapter2/4

7. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.  
   https://www.nltk.org/book/

8. Natural Language Toolkit (NLTK). (n.d.). *WordNet: Sample Usage and Documentation*.  
   https://www.nltk.org/howto/wordnet

9. Natural Language Toolkit (NLTK). (n.d.). *Porter Stemmer Documentation*.  
   https://www.nltk.org/api/nltk.stem.porter.html

10. Unicode Consortium. (n.d.). *FAQ: Normalization*.  
    https://www.unicode.org/faq/normalization.html
      
      
      







# NLP Assignment 2 – Text Preprocessing Report

**Student:** Ahmed Ali  
**Assignment:** NLP for Machine Learning Bootcamp  
**Topic:** Text Preprocessing Using NLTK and Python

---

## Task 1 – Lowercase

Lowercasing is the process of converting all uppercase letters into lowercase letters.

I used Python's `.lower()` method to convert the text into lowercase.

**Example:**

Before:
```text
URGENT!!! You have WON a £1,000 gift card.
```

After:
```text
urgent!!! you have won a £1,000 gift card.
```

**Explanation:**

Lowercasing makes words consistent. For example, "WON" and "won" become the same word. This helps reduce unnecessary differences in text data.

## Task 2 – Whitespace Normalization

Whitespace normalization removes unnecessary spaces from text.

I used Python's `split()` and `join()` methods to clean the extra spaces.

**Before:**
```text
   The   delivery   was   fast,   and the  packaging   was   great.
```

**After:**
```text
The delivery was fast, and the packaging was great.
```

**Explanation:**

Removing extra spaces makes text cleaner and easier to process. It also creates consistent spacing between words.

## Task 3 – Regular Expressions (Regex)

Regular expressions are patterns used to search, match, remove, or replace specific text.

I used Python's `re` module to clean T1 and replace important information.

**Original text:**
```text
URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
```

**Cleaned text:**
```text
URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!
```

**Explanation:**

I used regex to:
- Replace the phone number with `PHONE_NUMBER`.
- Replace the money amount with `MONEY`.
- Reduce multiple exclamation marks to one.

Regex is useful for removing unwanted characters and protecting sensitive information.

## Task 4 – Tokenization

Tokenization is the process of splitting text into smaller units called tokens.

I compared Python's `split()` method with NLTK's `word_tokenize()` function.

**T2 using split():**
```text
['I', "don't", 'think', 'this', 'jacket', 'is', 'worth',
'the', 'price.', 'Not', 'happy', 'at', 'all', '😞']
```

Token count: **14**

**T2 using word_tokenize():**
```text
['I', 'do', "n't", 'think', 'this', 'jacket', 'is',
'worth', 'the', 'price', '.', 'Not', 'happy', 'at',
'all', '😞']
```

Token count: **16**

**T4:**

Token count: **9**

Vocabulary size: **8**

**Explanation:**

The `split()` method separates text using whitespace. The `word_tokenize()` function performs more detailed tokenization. For example, it separates "don't" into "do" and "n't", and separates punctuation into individual tokens.

Vocabulary size means the number of unique tokens in the text.

## Task 5 – Stopwords and Negation

Stopwords are common words that are often removed during text preprocessing.

I used NLTK's English stopword list to remove common words from T2. Then, I created a custom stopword list that preserves negation words.

The negation words I preserved were:

- not
- n't
- no
- never

**Explanation:**

Removing negation words can change the meaning of a sentence.

For example:

- "happy" means a positive feeling.
- "not happy" means a negative feeling.

NLTK's default stopword list may remove important negation words. The custom stopword list preserves them, helping maintain the customer's original meaning.

**Conclusion:**

The custom stopword list is more suitable for sentiment analysis because it keeps important negative expressions.

## Task 6 – Stemming

Stemming reduces words by removing suffixes or applying predefined rules.

I compared three stemming algorithms:

1. Porter Stemmer
2. Lancaster Stemmer
3. Snowball Stemmer

I applied them to the given word list:

```python
WORDS = [
    "connection",
    "connected",
    "connecting",
    "organization",
    "organized",
    "university",
    "universal",
    "flies",
    "wolves",
    "better",
    "ran"
]
```

**Explanation:**

Different stemming algorithms may produce different results for the same word.

Some different words can also be reduced to the same stem. For example, if `connected` and `connecting` produce the same stem, they have been merged by the stemmer.

A stem does not always have to be a valid English dictionary word. It may only be a shortened part of a word.

**Conclusion:**

Stemming reduces vocabulary size, but its results may not always be meaningful English words.

## Task 7 – Lemmatization

Lemmatization converts words into their dictionary or base forms.

I used NLTK's `WordNetLemmatizer` in two ways:

1. Default lemmatization.
2. Lemmatization with the correct part-of-speech (POS) hints.

The POS hints used were:

| POS | Meaning |
|---|---|
| `n` | Noun |
| `v` | Verb |
| `a` | Adjective |

Examples of expected transformations:

| Word | POS | Expected lemma |
|---|---|---|
| connected | v | connect |
| connecting | v | connect |
| organized | v | organize |
| flies | v | fly |
| wolves | n | wolf |
| better | a | good |
| ran | v | run |

**Explanation:**

Without a POS hint, WordNetLemmatizer uses noun as its default. Providing the correct POS helps it identify the correct base form.

**Conclusion:**

Lemmatization is generally more linguistically meaningful than stemming because it uses vocabulary and grammatical information.

## Task 8 – Sentence Splitting

Sentence splitting separates a paragraph into individual sentences.

I used NLTK's `sent_tokenize()` function on T3.

**T3:**
```text
Mr. Smith arrived in London on Friday.
He met the Apple team at 9 a.m.
```

**Expected result:**

Sentence 1:
```text
Mr. Smith arrived in London on Friday.
```

Sentence 2:
```text
He met the Apple team at 9 a.m.
```

Total sentences: **2**

**Explanation:**

The abbreviations "Mr." and "a.m." should not create incorrect sentence boundaries. NLTK correctly handles these abbreviations in this example.

## Task 9 – POS Tagging and Named Entity Recognition

POS tagging assigns grammatical labels to words. Named Entity Recognition (NER) identifies important entities such as people, places, organizations, and dates.

I used:

- `pos_tag()` for part-of-speech tagging.
- `ne_chunk()` for named entity recognition.

**T3:**
```text
Mr. Smith arrived in London on Friday.
He met the Apple team at 9 a.m.
```

**Expected entity table:**

| Entity | Expected label | Meaning |
|---|---|---|
| Smith | PERSON | Person's name |
| London | GPE | Geographic location |
| Friday | DATE | Day |
| Apple | ORGANIZATION | Company |

**Explanation:**

Apple refers to a company in this sentence, not a fruit.

The NLTK NER model may fail to recognize some entities or assign incorrect labels. Therefore, I compared its actual predictions with the expected labels.

**Conclusion:**

NER is useful for extracting important information from text, but its predictions are not always perfect because understanding context can be difficult.

## Task 10 – Full Preprocessing Pipeline

I created a complete preprocessing pipeline for T2 using five stages:

1. Lowercase
2. Regex cleaning
3. Tokenization
4. Stopwords removal
5. Lemmatization

I ran the pipeline twice.

### Run 1 – Default Stopwords

I used NLTK's default English stopword list.

This may remove negation words such as `not` and `n't`. Removing these words can change the meaning of the customer's complaint.

### Run 2 – Custom Stopwords

I used a custom stopword list that preserves:

- not
- n't
- no
- never

This keeps important negative expressions in the processed text.

**Explanation:**

The original customer message is:

```text
I don't think this jacket is worth the price.
Not happy at all 😞
```

The expression "not happy" communicates negative sentiment. If the word "not" is removed, the remaining word "happy" may incorrectly suggest positive sentiment.

The regex stage also removes the sad emoji in the pipeline used here, so this is another piece of sentiment information that could be preserved in a more advanced pipeline.

**Conclusion:**

The custom stopword version preserves the customer's complaint better because it keeps negation words. This is important for sentiment analysis and other NLP applications.

---

# Overall Conclusion

In this assignment, I practiced important text preprocessing techniques using Python and NLTK.

I learned how to:
- Convert text to lowercase.
- Normalize whitespace.
- Clean text using regular expressions.
- Tokenize words and sentences.
- Remove stopwords while preserving negation.
- Compare different stemming algorithms.
- Apply lemmatization with POS hints.
- Perform POS tagging and Named Entity Recognition.
- Build a complete text preprocessing pipeline.

These techniques help prepare raw text for Natural Language Processing and Machine Learning tasks.

**Final observation:** Text preprocessing should remove unnecessary information without losing important meaning. Negation, named entities, and sentiment-related information must be handled carefully.
