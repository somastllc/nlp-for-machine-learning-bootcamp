# Assignment 2 – Preprocessing

## Part A – Research

### 1. Text Preprocessing: Definition, Necessity, and Risks of Over-Cleaning
Text preprocessing is the set of transformations applied to raw, unstructured text before training a model or running text analytics so that linguistic units are represented consistently and reliably. Natural language data contains irregularities—such as mixed casing, extra spaces, punctuation, symbols, numbers, emojis, and informal abbreviations—that increase vocabulary sparsity and introduce noise into statistical and machine learning models.

However, applying more cleaning is not always beneficial. Over-aggressive cleaning can strip away signals that carry meaning or intent. For example, in **spam or phishing detection**, promotional emails or SMS messages often rely on all-caps urgency cues (such as `"URGENT"` or `"WON"`), currency amounts (`"£1,000"`), phone numbers or URLs, and repeated exclamation marks (`"!!!"`). Removing punctuation, numbers, or capitalization indiscriminately can erase the very features that distinguish spam from normal correspondence.

### 2. Text Normalization and Its Trade-Offs
Text normalization maps variant surface forms of text to a standard canonical form. Common normalization techniques include:
* **Lowercasing:** Converting alphabetic characters to lowercase so words differing only in case share a common form.
* **Whitespace normalization:** Collapsing repeated spaces, tabs, or line breaks into single spaces and trimming boundary whitespace.
* **Unicode normalization:** Converting characters with multiple Unicode representations (such as composed vs. decomposed accents or compatibility characters like full-width symbols or ligatures) into a canonical form (such as NFC or NFKC).

A helpful case is **search and indexing**, where lowercasing and whitespace normalization allow a query like `"  running   shoes "` to match indexed text containing `"Running shoes"`. Conversely, **lowercasing can remove useful information** in tasks such as Named Entity Recognition or acronym disambiguation, where capitalization helps distinguish proper names or abbreviations (such as `"US"` or `"Apple"`) from ordinary words.

### 3. Tokenization Approaches and Vocabulary Size
Tokenization segments raw text into smaller processing units called tokens:
* **Word-level tokenization:** Splits text into word-like units (and often separate punctuation marks), which is intuitive and interpretable but struggles with out-of-vocabulary (OOV) words and morphological variants.
* **Subword tokenization:** Breaks words into frequent stems, affixes, or character n-grams (common in modern neural language models), balancing compact vocabularies with the ability to represent rare or unseen words from known pieces.
* **Character-level tokenization:** Treats every individual character as a token, resulting in a very small base alphabet and zero OOV characters, though sequences become much longer and capturing word-level semantics requires more model capacity.

**Vocabulary size** is the count of unique token types in a tokenizer's lexicon or in a processed text collection. Word-level tokenizers tend to produce large vocabularies that grow as new words and inflections appear, character-level tokenizers produce very small vocabularies, and subword tokenizers sit in between by keeping common words intact while splitting rare ones into subword units.

### 4. Stopwords: Benefits, Risks, and Task/Language Dependence
Stopwords are high-frequency function words—such as articles, prepositions, conjunctions, and auxiliary verbs—that occur widely across documents and often contribute less topical specificity in bag-of-words retrieval or topic modeling. Removing stopwords can reduce feature dimensionality, speed up computation, and decrease storage requirements.

The risk of removing stopwords is losing grammatical structure, phrase identity, or sentiment polarity. In sentiment or review analysis, dropping negation terms like `"not"` or `"n't"` can invert the polarity of a statement. Moreover, the appropriate stopword list depends on both the **task** (e.g., search/indexing vs. sentiment classification or syntactic/semantic parsing) and the **language**, because each language has its own function words, clitics, and morphological structure.

### 5. Stemming vs. Lemmatization
Both stemming and lemmatization reduce inflected or derived words toward a base form, but they differ in mechanism and linguistic precision:
* **Stemming:** Applies rule-based suffix-stripping heuristics without consulting a morphological lexicon. Well-known English stemmers include the **Porter stemmer** (a classic rule-based stemmer), **Snowball** (an improved, more consistent successor to Porter), and **Lancaster** (a more aggressive stemmer). Stemming is fast and simple, making it useful when speed and aggressive conflation matter more than linguistic validity, though it can over-stem or produce non-words.
* **Lemmatization:** Uses morphological analysis and a lexical resource (such as **WordNet** in English) to return a valid dictionary base form (lemma), often guided by part-of-speech (POS) information. Lemmatization is preferred when linguistically valid base forms and grammatical accuracy matter, such as information extraction, semantic analysis, or downstream NLP pipelines where word meaning must stay intact.

### 6. Regular Expressions in Text Cleaning
Regular expressions (regex) define search patterns used to detect, extract, replace, or remove specific character sequences in text.
* **Example 1 — Stripping non-alphabetic/non-space characters (`[^a-z\s]`):** Matches any single character that is not a lowercase ASCII letter (`a-z`) or whitespace (`\s`), allowing removal of digits, punctuation, and symbols.
* **Example 2 — Collapsing repeated whitespace (`\s+`):** Matches one or more consecutive whitespace characters (spaces, tabs, newlines), allowing normalization to a single space (`" "`).

An overly broad pattern can damage text unintentionally. For example, an aggressive regex that strips all punctuation or all non-letter characters can corrupt domain tokens like version numbers, currency amounts, email addresses, or contractions (`don't`), altering the meaning of the text.

### 7. Case Study on Preprocessing Choices in NLP
In empirical NLP research on text classification and preprocessing pipelines—such as studies examining preprocessing choices for sentiment analysis and short-text classification (for instance, **Alam & Yao, 2019**, *"The impact of preprocessing steps on the accuracy of machine learning algorithms in sentiment analysis"*, or **Krouska et al., 2016**, *"The effect of preprocessing techniques on Twitter sentiment analysis"*)—researchers compare combinations of steps such as lowercasing, punctuation/symbol removal, stopword removal, stemming, and lemmatization across classifiers.

These studies consistently show that **no single preprocessing pipeline is universally optimal across all datasets and models**. While removing noise and normalizing text often helps linear or bag-of-words models reduce sparsity, removing negation words or overly stripping punctuation/emoticons can hurt sentiment classification accuracy, and the impact of stemming versus lemmatization varies depending on the dataset domain and learning algorithm.

### References
* Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft).
* Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python*. O'Reilly Media.
* Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
* Alam, T., & Yao, S. (2019). The impact of preprocessing steps on the accuracy of machine learning algorithms in sentiment analysis. *Computational and Mathematical Organization Theory*, 25(3), 310–335.
* Krouska, A., Troussas, C., & Virvou, M. (2016). The effect of preprocessing techniques on Twitter sentiment analysis. *IISA 2016*, 1–5.

---

## Part B – Practical

### Task 6 Table: Stemming Comparison

| Word | Porter | Lancaster | Snowball |
| :--- | :--- | :--- | :--- |
| `connection` | `connect` | `conne` | `connect` |
| `connected` | `connect` | `connect` | `connect` |
| `connecting` | `connect` | `connect` | `connect` |
| `organization` | `organ` | `org` | `organ` |
| `organized` | `organ` | `org` | `organ` |
| `university` | `univers` | `univers` | `univers` |
| `universal` | `univers` | `univers` | `univers` |
| `flies` | `fli` | `fli` | `fli` |
| `wolves` | `wolv` | `wolv` | `wolv` |
| `better` | `better` | `bet` | `better` |
| `ran` | `ran` | `ran` | `ran` |

### Task 9 Table: Named Entities in `T3`

| Entity | Label (`ne_chunk`) | Correct? |
| :--- | :--- | :--- |
| `Smith` | `PERSON` | Yes |
| `London` | `GPE` | Yes |
| `Friday` | `PERSON` | No (it is a day of the week / temporal expression, not a person) |
| `Apple` | `ORGANIZATION` | Yes |

### Short Answers (Tasks 1, 3, 5, 6, 7, 8, and 10)

* **Task 1 (Lowercasing):** Lowercasing `T1` converts uppercase text into lowercase (`urgent!!! you have won a £1,000 gift card. call 0712345678 now!`). A spam detector could lose emphasis cues from all-caps words like `URGENT`, `WON`, and `NOW`, which spammers frequently use to create artificial urgency. It can also blur distinctions between capitalized proper names/brands and regular words.
* **Task 3 (Regex Cleaning):** Applying `[^a-z\s]` to lowercased `T1` removes exclamation marks (`!`, `!`, `!`), the currency symbol (`£`), digits and commas (`1`, `,`, `0`, `0`, `0`, `0`, `7`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`), and the period (`.`), leaving only letters and spaces. Replacing patterns with `PHONE_NUMBER`, `MONEY`, and a single `!` preserves the presence of a phone number, money amount, and emphasis without deleting those signals outright.
* **Task 5 (Stopwords and Negation):** Removing NLTK's default English stopwords from `T2` drops `"don't"` (as split tokens) and `"not"`, leaving tokens such as `['think', 'jacket', 'worth', 'price', '.', 'happy', 'all', ]`. Using the negation-keeping stopword list retains `"don't"` and `"not"`, preserving the negative sentiment of the statement. Without those negation tokens, the remaining words sound much closer to a positive opinion about the jacket being happy/worthwhile.
* **Task 6 (Stemming):** Both Porter and Snowball merge `"connection"`, `"connected"`, and `"connecting"` into `"connect"`, and also merge `"organization"` and `"organized"` into `"organ"`, while Lancaster merges `"connection"`, `"connected"`, and `"connecting"` into `"conne"` and `"organization"` and `"organized"` into `"org"`. Examples of stems that are not standard dictionary words in the output include `"univers"` (from `"university"` and `"universal"`), `"fli"` (from `"flies"`), and `"wolv"` (from `"wolves"`).
* **Task 7 (Lemmatization):** When POS hints (`n`, `v`, `a`) are supplied to `WordNetLemmatizer`, `"flies"` (with `pos="v"`) changes from `"flies"` to `"fly"`, `"better"` (with `pos="a"`) changes from `"better"` to `"good"`, and `"ran"` (with `pos="v"`) changes from `"ran"` to `"run"`. Other words in the list stay the same because the default noun lookup already matches their base or uninflected form in WordNet.
* **Task 8 (Sentence Splitting):** `sent_tokenize` splits `T3` into **2** sentences (`"Mr. Smith arrived in London on Friday."` and `"He met the Apple team at 9 a.m."`). Neither `"Mr."` nor `"a.m."` wrongly breaks the text into extra sentences because NLTK's sentence tokenizer recognizes those abbreviations.
* **Task 10 (Full Pipeline):** Running the pipeline with default NLTK stopwords removes negation cues (`"not"` and `"n't"`), whereas the run with the negation-keeping stopword list keeps `"not"` and `"n't"` through the filtering and lemmatization stages. Therefore, the **negation-keeping version** better preserves the customer's complaint.

### Spam Detector Preprocessing Recommendation

For a spam detector, I would use lowercasing (or case-aware features alongside lowercasing), whitespace normalization, and targeted placeholder replacement (`PHONE_NUMBER`, `MONEY`, and normalized punctuation/exclamation signals) rather than blind deletion. As shown in my `T1` outputs, stripping `[^a-z\s]` deletes `£1,000`, `0712345678`, and `!!!` completely, whereas custom regex replacement keeps those spam-indicative patterns visible as structured tokens (`MONEY`, `PHONE_NUMBER`, `!`). I would avoid aggressive character stripping that erases monetary amounts, contact numbers, or exclamation patterns, and I would be cautious about removing stopwords or punctuation blindly if doing so discards urgency or negation cues that help separate spam from legitimate messages.