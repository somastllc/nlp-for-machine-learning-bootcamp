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

## Part B – Practical

**Important note:** The practical tables and exact model outputs must be taken from running `preprocessing_practice.py` with the course NLTK data installed. The script prints all required results. Do not invent or manually type model output. After running it, check `output.txt` and use the actual results from your computer in the tables below.

### Task 1 – Lowercasing

Lowercasing T1 changes uppercase words such as `URGENT` and `WON` into lowercase. This makes the words more consistent for a word-based model. However, a spam detector may lose the uppercase emphasis, which can be a useful sign of an urgent or suspicious message.

### Task 3 – Regex cleaning

The pattern `[^a-z\s]` removes characters that are not lowercase English letters or whitespace. In T1, this removes the pound sign, digits in the money amount and phone number, comma, periods and exclamation marks. The replacement rules are safer because they replace the phone number with `PHONE_NUMBER`, the amount with `MONEY`, and repeated exclamation marks with one `!`, preserving the type of information.

### Task 5 – Stopwords and negation

The default English stopword list removes common words and can also remove `not`. This is risky because T2 contains the complaint “Not happy at all.” The custom list keeps `not`, `n't`, `no`, and `never`, so the negative meaning is more likely to remain in the tokens.

### Task 6 – Stemming comparison

Use the exact output printed under **6) STEMMING COMPARISON** in `output.txt` to complete this table.

| Word | Porter | Lancaster | Snowball |
|---|---|---|---|
| connection | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| connected | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| connecting | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| organization | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| organized | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| university | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| universal | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| flies | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| wolves | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| better | Copy from output.txt | Copy from output.txt | Copy from output.txt |
| ran | Copy from output.txt | Copy from output.txt | Copy from output.txt |

The stemmers may merge different words into the same result. For example, the three related words `connection`, `connected`, and `connecting` are expected to be reduced to a common stem by common English stemmers. Some stems, such as `organ` or `univers`, are not ordinary dictionary words. This shows that stemming cuts words by rules and does not always produce a real word.

### Task 7 – Lemmatization

The default `WordNetLemmatizer` treats words as nouns. Supplying a part-of-speech hint can change the result; for example, a verb hint can change `connected` or `ran` to its base verb form, and an adjective hint can change `better` to `good`. Use the actual default and hinted results printed by the script, because the WordNet data and the supplied part-of-speech hints determine the output.

### Task 8 – Sentence splitting

The script uses NLTK's `sent_tokenize` to split T3 into sentences. The expected sentence count is two: the first sentence ends after `Friday.` and the second ends after `a.m.`. The tokenizer should keep `Mr.` and `a.m.` as abbreviations rather than treating each abbreviation period as a sentence boundary; confirm the exact result in `output.txt`.

### Task 9 – POS tagging and named entity recognition

Copy the actual POS tags and NER entities from the script output. NLTK's named entity model may recognize `Smith` as a person and `London` as a location, but it may not identify every entity correctly. In this sentence, `Apple` refers to a company, and `Friday` is a day. The model can miss or mislabel entities, so compare its result with the meaning of the sentence.

| Entity in T3 | Correct label | Model label from output.txt | Correct? |
|---|---|---|---|
| Mr. Smith | PERSON | Fill from output.txt | Check against PERSON |
| London | GPE / location | Fill from output.txt | Check against GPE |
| Friday | DATE / day | Fill from output.txt | Check whether recognized |
| Apple | ORGANIZATION | Fill from output.txt | Check against ORGANIZATION |

The model does not necessarily recognize every person, place, organization, or day. In particular, NLTK's standard named-entity labels may not include a separate day/date category in the same way as a dedicated date recognizer.

### Task 10 – Full pipeline

The full pipeline applies lowercase, regex cleaning, tokenization, stopword removal, and lemmatization in order. With the default stopword list, the word `not` may be removed, which weakens or changes the customer's complaint. The custom stopword list keeps negation words, so it better preserves the negative meaning of T2. The version that keeps negation is more suitable for customer-sentiment analysis.

### Which preprocessing steps would I use for a spam detector, and which would I avoid?

For a spam detector, I would normalize repeated spaces and use lowercasing carefully, but I would also keep information about uppercase emphasis. I would replace phone numbers and money amounts with markers instead of deleting them, and I would keep useful punctuation patterns such as repeated exclamation marks in a separate feature. I would avoid the broad regex `[^a-z\s]` because it deletes digits, currency symbols and punctuation. I would also avoid removing negation words and emojis without checking whether they help the model. I would test each choice using validation data before selecting the final pipeline.

## References

1. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.
2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725. https://doi.org/10.18653/v1/P16-1162
3. Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/
5. NLTK Project. (n.d.). *NLTK documentation: Stemming, tokenization, tagging and WordNet*. https://www.nltk.org/
