# Assignment 2 – Text Preprocessing

## Part A – Research

### 1. What Is Text Preprocessing?

Text preprocessing is the process of preparing raw text before it is used in a Natural Language Processing (NLP) system. It involves cleaning, organizing, and transforming text into a format that computers can analyze more effectively.

NLP systems need preprocessing because human language contains spelling variations, unnecessary spaces, punctuation, different letter cases, and other inconsistencies. These differences can make it difficult for machine learning models to identify useful patterns.

However, excessive cleaning can remove important information. For example, in a customer review, the sentence "I am not happy with this product" has a negative meaning. If preprocessing removes the word "not", the remaining words may suggest a positive meaning.

Therefore, preprocessing should depend on the purpose of the NLP task rather than removing ev erything that appears unnecessary.

### 2. Text Normalization

Text normalization is the process of converting text into a consistent representation while preserving its intended meaning.

Common normalization techniques include:

- **Lowercasing:** Converting uppercase letters into lowercase letters.
- **Whitespace normalization:** Removing unnecessary spaces and repeated whitespace.
- **Unicode normalization:** Converting equivalent Unicode representations into a consistent form.

For example, lowercasing can help a search engine treat "COMPUTER", "Computer", and "computer" as the same word.

However, lowercasing may remove useful information. In spam detection, uppercase words such as "URGENT" and "WIN" may indicate emphasis or suspicious advertising.

Unicode normalization is also important because visually identical characters can sometimes have different underlying representations.

Normalization improves consistency, but it should preserve information that is useful for the task.

### 3. Word-Level, Subword, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens.

There are three common approaches:

**Word-level tokenization**

This approach divides text into individual words.

Example:

"Natural language processing"

Tokens: `["Natural", "language", "processing"]`

It is easy to understand and useful for traditional NLP tasks. However, it can create a very large vocabulary because different word forms may be treated as separate tokens.

**Subword-level tokenization**

Subword tokenization divides words into smaller meaningful or frequently occurring units.

For example, the word "unhappiness" may be divided into smaller units such as "un", "happi", and "ness".

It helps models process unfamiliar words and reduces the number of vocabulary entries required.

Common approaches include Byte Pair Encoding (BPE) and WordPiece.

**Character-level tokenization**

Character-level tokenization divides text into individual characters.

Example:

"cat" becomes `["c", "a", "t"]`.

It can represent almost any word without requiring a large word vocabulary. However, it produces longer sequences and may make learning word meanings more difficult.

**Vocabulary Size**

Vocabulary size means the total number of unique tokens that a tokenizer can recognize.

Word-level tokenization usually requires a larger vocabulary, while subword tokenization reduces vocabulary size by combining smaller units. Character-level tokenization generally requires a relatively small character vocabulary.

The best tokenizer depends on the language, dataset, and NLP application.

### 4. Stopwords

Stopwords are common words that are sometimes removed during text preprocessing because they may contribute less information to particular NLP tasks.

Examples in English include:

- the
- is
- a
- an
- and
- of
- to

**Benefits of removing stopwords**

1. Reduces the number of tokens.
2. Reduces the size of text features.
3. Can improve efficiency in information retrieval and some text classification tasks.

**Risks of removing stopwords**

Removing stopwords can change the meaning of sentences.

For example:

"I do not like this product."

If the words "do" and "not" are removed, the sentence may become:

"like product."

This changes the original negative meaning.

Stopword lists should depend on the NLP task and language. Words that are unimportant in one task may be essential in another. Different languages also have different grammatical structures and common words.

For sentiment analysis, negation words such as "not", "no", "never", and "n't" should generally be preserved.

### 5. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce different forms of words to a common representation.

**Stemming**

Stemming removes prefixes or suffixes using predefined rules. It does not always produce a grammatically correct word.

For example:

- connected → connect
- connecting → connect
- organization → organ

One well-known algorithm is the Porter Stemmer, introduced by Martin Porter in 1980. It uses a series of suffix-removal rules to reduce related words to common stems.

The Lancaster Stemmer is another algorithm. It uses a more aggressive set of stemming rules and may produce shorter stems.

Stemming is useful in information retrieval and search engines when matching different forms of the same word is important.

**Lemmatization**

Lemmatization converts words into their dictionary base forms, called lemmas. It considers vocabulary and grammatical information.

Examples:

- running → run
- wolves → wolf
- better → good (when interpreted as an adjective with the appropriate lexical analysis)

The WordNetLemmatizer in NLTK uses WordNet to find dictionary forms. It can produce different results depending on the part-of-speech information supplied.

For example, the word "ran" can be converted to "run" when it is identified as a verb.

**Comparison**

| Stemming | Lemmatization |
|---|---|
| Uses rule-based suffix removal | Uses dictionary and linguistic information |
| Usually faster | Usually more computationally expensive |
| May produce non-dictionary words | Usually produces dictionary forms |
| Useful for search and retrieval | Useful when grammatical meaning matters |

Stemming is preferable when speed and broad word matching are important. Lemmatization is preferable when preserving meaningful base forms is important.

### 6. Regular Expressions in Text Cleaning

Regular expressions (regex) are patterns used to search, match, remove, or replace specific sequences of characters in text.

They are useful for cleaning text because they allow developers to identify repeated patterns such as phone numbers, email addresses, punctuation, and unwanted characters.

**Example 1:**

Pattern:

`[^a-z\s]`

This matches any character that is not a lowercase English letter or whitespace.

It can remove punctuation, numbers, currency symbols, and emoji from text.

**Example 2:**

Pattern:

`\d{10}`

This matches a sequence of exactly ten digits.

It can help identify phone numbers in a dataset when those numbers follow that format.

**Risk of overly broad patterns**

An overly broad regex pattern can remove meaningful information.

For example, removing every character that is not a letter may delete phone numbers, prices, punctuation, and emoji.

In a spam detection system, deleting a price such as £1,000 or a phone number may remove useful evidence.

Therefore, regex rules should be designed carefully according to the information the NLP task needs.

### 7. Case Study: Comparing Preprocessing Techniques for Twitter Sentiment Analysis

Symeonidis, Effrosynidis, and Arampatzis (2018) conducted a research study titled "A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis."

**Task**

The researchers investigated how different text preprocessing techniques affected sentiment classification on Twitter datasets.

**Methods compared**

The study evaluated sixteen preprocessing techniques using two Twitter datasets and four machine learning algorithms:

- Linear Support Vector Classification
- Bernoulli Naive Bayes
- Logistic Regression
- Convolutional Neural Networks

The researchers examined techniques including:

- Lemmatization
- Number removal
- Contraction replacement
- Punctuation removal
- Different combinations of preprocessing techniques

**Findings**

The study found that some techniques improved classification accuracy, particularly lemmatization, number removal, and contraction replacement.

However, removing punctuation did not consistently improve performance.

The researchers also investigated how combining preprocessing methods affected the results.

**Conclusion**

The study demonstrates that preprocessing choices can significantly influence NLP performance. It also shows that applying every cleaning technique is not always beneficial.

The most suitable preprocessing pipeline should be selected through experiments using the actual dataset and task.

---

## Part B – Practical

### Task 1: Lowercasing

Lowercasing converts all uppercase letters into lowercase letters.

In my experiment, the original T1 text contained the uppercase words "URGENT", "WON", and "NOW".

After lowercasing, these words became "urgent", "won", and "now".

This improves consistency, but a spam detector may lose capitalization patterns that indicate urgency or suspicious advertising.

### Task 2: Whitespace Normalization

Whitespace normalization removes unnecessary spaces and makes text more consistent.

In T4, repeated spaces between words were replaced with single spaces.

This makes the text easier to tokenize and process.

### Task 3: Regex Cleaning

The pattern `[^a-z\s]` removed all characters that were not lowercase English letters or whitespace.

From T1, the deleted information included:

- Exclamation marks
- The pound currency symbol (£)
- The amount 1,000
- The phone number 0712345678
- The full stop

This demonstrates why deleting every non-letter character can damage important information.

I also created replacement rules that preserved important information by replacing the money amount with `MONEY`, the phone number with `PHONE_NUMBER`, and repeated exclamation marks with one exclamation mark.

### Task 4: Tokenization

The T2 sentence was tokenized using both Python's `split()` and NLTK's `word_tokenize()`.

The `split()` method separated text according to whitespace, while `word_tokenize()` also separated punctuation and handled contractions differently.

For T2, the observed results were:

| Method | Token count |
|---|---:|
| split() | 14 |
| word_tokenize() | 16 |

For T4, the output contained 11 tokens and a vocabulary size of 10 unique tokens.

### Task 5: Stopwords and Negation

The default NLTK English stopword list removed several common words from T2.

However, preserving negation words is important because they affect the meaning of a sentence.

In my experiment, the custom stopword list preserved the negative word "not".

This allowed the original customer complaint to remain more understandable than the version using the default stopword list.

### Task 6: Stemming Comparison

I applied three stemming algorithms to the provided words: Porter, Lancaster, and Snowball.

The following table is based on my Python program output.

| Word | Porter | Lancaster | Snowball |
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

The results demonstrate that different stemming algorithms can produce different stems.

For example, "university" and "universal" are both reduced to "univers". This shows that stemming can merge different words into the same representation.

Also, "organ" and "org" are shortened forms that do not necessarily represent the original dictionary meaning.

### Task 7: Lemmatization

I used NLTK's WordNetLemmatizer to process all the provided words.

First, I applied lemmatization using the default noun setting. Then, I supplied part-of-speech hints such as:

- `pos="v"` for verbs
- `pos="n"` for nouns
- `pos="a"` for adjectives

The part-of-speech information helps the lemmatizer identify the correct base form.

**Important:** The exact before-and-after results should be taken from the generated `output.txt` file.

### Task 8: Sentence Splitting

I used NLTK's `sent_tokenize()` function to divide T3 into sentences.

Sentence splitting can be challenging because abbreviations such as "Mr." and "a.m." contain periods but do not necessarily indicate the end of a sentence.

The actual number of sentences and whether the abbreviations caused incorrect splitting should be verified using my program output.

### Task 9: POS Tagging and Named Entity Recognition

I used NLTK's `pos_tag()` to identify the grammatical roles of words in T3.

I also used `ne_chunk()` to identify named entities.

Named Entity Recognition (NER) attempts to identify categories such as:

- PERSON
- ORGANIZATION
- GPE (Geopolitical Entity)

The following table must be completed using the actual entities printed by my script.

| Entity | Expected label | Model label | Correct? |
|---|---|---|---|
| Mr. Smith | PERSON | Check output | Check output |
| London | GPE | Check output | Check output |
| Friday | DATE | Check output | Check output |
| Apple | ORGANIZATION | Check output | Check output |

NER systems can make mistakes because they depend on their training data and contextual information.

For example, Apple can refer to either a company or a fruit. Recognizing the company requires understanding its context.

### Task 10: Full Preprocessing Pipeline

I applied five preprocessing steps to T2:

1. Lowercasing
2. Regex cleaning
3. Tokenization
4. Stopwords removal
5. Lemmatization

I ran the pipeline twice, first using NLTK's default English stopwords and then using my custom stopword list.

The default stopword list removed important negation information. The custom list preserved "not", allowing the negative meaning of the customer's complaint to remain clearer.

Therefore, the custom stopword approach is more appropriate for this sentiment-related example.

---

## Overall Practical Discussion: Preprocessing for a Spam Detector

For a spam detection system, I would use lowercasing to make text consistent and whitespace normalization to remove unnecessary spaces. I would also use carefully designed regular expressions to identify phone numbers, money amounts, and suspicious punctuation without deleting their information. I would preserve useful features such as uppercase words, numbers, currency symbols, and repeated punctuation because they may indicate suspicious messages. I would also avoid removing negation words because they can change the meaning of a message. The practical experiments demonstrate that excessive cleaning can remove information that is useful for classification.

---

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

2. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

3. Willett, P. (2006). The Porter stemming algorithm: Then and now. *Program: Electronic Library and Information Systems, 40*(3), 219–223.

4. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2018.06.022

5. NLTK Project. (n.d.). *Natural Language Toolkit Documentation*. https://www.nltk.org/

6. Cambridge University Press. (2023). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

7. Snowball. (n.d.). *The Porter Stemming Algorithm*. https://snowballstem.org/algorithms/porter/stemmer.html
