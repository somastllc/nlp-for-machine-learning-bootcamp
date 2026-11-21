# Assignment 2 – Preprocessing

**Course:** NLP Foundations  
**Assignment:** 2 – Preprocessing

## Part A – Research

### 1. What is text preprocessing? Why do NLP systems need it, and why can more cleaning make results worse?

Text preprocessing is the process of preparing raw text before giving it to a Natural Language Processing (NLP) system. It can include making text lowercase, fixing spaces, splitting text into tokens, handling punctuation, and reducing word forms.

NLP systems need preprocessing because people write the same words in different ways. For example, `FREE`, `Free`, and `free` may be treated as different features by a basic word-based model. Lowercasing can make them consistent. However, too much cleaning can remove useful information. In a spam message, capital letters and repeated exclamation marks may show urgency. In a customer review, removing `not` can change a negative opinion into a positive one.

**Real-life example:** In the sentence “I am not happy with this service,” removing the word `not` leaves “happy service.” A sentiment system may then misunderstand the customer's complaint.

### 2. Explain text normalization. Give one situation where it helps and one where it removes useful information.

Text normalization means changing different forms of text into a consistent form. Common examples are lowercasing, correcting repeated spaces, and Unicode normalization. Unicode normalization makes characters that look the same use a consistent computer representation.

Normalization helps when a dataset contains `Hello`, `HELLO`, and `hello`. Lowercasing changes all three to `hello`, reducing unnecessary differences. Whitespace normalization also changes several spaces into one space.

However, normalization can remove useful information. For example, lowercasing changes `US` (the country abbreviation) into `us` (a pronoun). In spam detection, lowercasing can also remove the visual emphasis in words such as `URGENT`. Therefore, the correct choice depends on the task.

### 3. Compare word-level, subword and character-level tokenization. Explain vocabulary size and the effect of the tokenizer.

Tokenization divides text into smaller units called tokens. A word-level tokenizer treats words and often punctuation as separate tokens. For example, `I love NLP!` may become `I`, `love`, `NLP`, and `!`. It produces fewer tokens for a sentence, but a vocabulary can become large because many different word forms must be stored.

A character-level tokenizer treats each character as a token. It has a small vocabulary because it only needs characters, numbers, spaces and symbols, but it produces many tokens for each sentence. It can represent new or misspelled words because it works with their characters.

A subword tokenizer splits some words into smaller learned pieces. Common words may stay whole, while rare or long words are divided into parts. It usually has a medium-sized vocabulary and can represent many words using known pieces. Sennrich, Haddow and Birch (2016) showed that subword units can help machine translation handle rare and unknown words.

**Vocabulary size** means the number of different token entries a tokenizer knows. A word-level tokenizer usually has a larger vocabulary, a character tokenizer has a small vocabulary, and a subword tokenizer is usually between them. The tokenizer affects both the number of tokens in a sentence and the number of unique entries the model must store.

### 4. What are stopwords? Discuss the benefits and risks of removing them.

Stopwords are common words that appear often in a language, such as `the`, `is`, `and`, and `at`. Removing them can reduce the number of features and help a model focus on other words. This can be useful in some search and text-analysis tasks.

However, stopword removal can also remove important meaning. For example, `not` changes “good” into “not good.” If a sentiment system removes `not`, it may misunderstand the review. Words that are common in one language may not be common or unimportant in another language. Also, the useful words depend on the task: a stopword list for topic classification may not be suitable for sentiment analysis. For this reason, stopword removal should be tested instead of applied automatically.

### 5. Compare stemming and lemmatization. Explain one well-known algorithm and one lemmatization approach.

Stemming reduces words by removing or changing endings using rules. The result is called a stem and may not be a real dictionary word. For example, the Porter stemmer can reduce `studies` to `studi`. Porter (1980) introduced a sequence of suffix-stripping rules. Stemming is often fast and useful when a system needs to group related word forms, such as in information retrieval.

Lemmatization finds a word's dictionary form, called its lemma. It uses language knowledge and may use the grammatical role of the word. NLTK's `WordNetLemmatizer` uses WordNet and assumes a noun by default unless a part-of-speech hint is supplied. For example, `running` may remain `running` by default but become `run` when the verb hint is used.

Stemming may be preferred when speed and simple word grouping are important. Lemmatization may be preferred when meaningful dictionary forms are important. Both should be evaluated for the task because neither is always necessary.

### 6. How are regular expressions used in text cleaning? Give two patterns and explain the risk of a broad pattern.

A regular expression (regex) describes a pattern in text. Python's `re` module can use regex to find, remove, or replace text such as phone numbers, URLs, repeated spaces, and punctuation.

Two examples are:

| Regex pattern | What it matches | Example use |
|---|---|---|
| `\b\d{9,10}\b` | A complete group of 9 or 10 digits | Replace a phone number with `PHONE_NUMBER` |
| `!+` | One or more exclamation marks together | Change `!!!` into one `!` |

An overly broad pattern can delete information that the NLP task needs. For example, `[^a-z\s]` removes every character that is not a lowercase English letter or whitespace. It can delete money amounts, phone digits, punctuation, apostrophes and emojis. This can damage a spam detector because money and phone numbers may be useful spam signals.

### 7. Case study: preprocessing choices and text classification

Uysal and Günal (2014) studied how preprocessing affects text classification. They compared combinations of common preprocessing steps using email and news text in English and Turkish. They examined classification accuracy and the number of features, and considered how the text domain and language affected the results.

The study found that the best results did not come from automatically applying every cleaning step or removing every step. The useful combination depended on the language and the type of text. This supports the idea that preprocessing must match the NLP task. For example, removing numbers may be acceptable in one news-classification task but harmful in spam detection, where phone numbers and money amounts may be important.



## References

1. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.
2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725. https://doi.org/10.18653/v1/P16-1162
3. Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/
5. NLTK Project. (n.d.). *NLTK documentation: Stemming, tokenization, tagging and WordNet*. https://www.nltk.org/
