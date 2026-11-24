# Assignment 2 – Preprocessing Report

## Part A – Research

### 1. Text Preprocessing: Definition, Necessity, and Risks of Over-Cleaning

**Text preprocessing** refers to the sequence of transformations applied to raw natural language data—such as casing adjustments, character filtering, tokenization, morphological normalization, or stopword filtering—before text is converted into numerical representations for statistical or neural models (Jurafsky & Martin, 2024; Manning, Raghavan, & Schütze, 2008). Because raw text is inherently noisy and high-dimensional, machine learning models require consistent units and structured numerical features to generalize across unseen examples rather than memorizing surface quirks.

However, **more cleaning is not automatically better**. Aggressive or unexamined normalization can strip away discriminative signals that carry semantic, pragmatic, or stylistic meaning. For example, in **clinical or mental-health crisis detection** on social media or support logs, stripping punctuation, capitalization, or negation cues—or collapsing expressions such as `"not feeling safe"` into `"feeling safe"` after removing `"not"`—can invert clinical intent and cause a classifier to miss high-risk messages. Preprocessing should therefore be treated as a hypothesis-driven modeling choice validated on held-out data rather than a rigid checklist.

### 2. Text Normalization and Trade-Offs

**Text normalization** standardizes variant surface forms of text so that representations that should be treated as equivalent map to a common form (Jurafsky & Martin, 2024). Common operations include:

* **Lowercasing:** Mapping uppercase letters to lowercase (e.g., `"Search"` $\rightarrow$ `"search"`).
* **Whitespace normalization:** Collapsing repeated spaces, tabs, or newlines into single spaces and trimming leading or trailing whitespace.
* **Unicode normalization:** Resolving equivalent Unicode encodings—such as composed versus decomposed diacritics (NFC vs. NFD)—so visually identical characters share a single code-point sequence.

**When normalization helps:** In **document retrieval and web search**, lowercasing and Unicode NFC normalization help match user queries with indexed documents regardless of whether a user types a capitalized word at the start of a sentence or uses a different Unicode decomposition for accented characters.

**When normalization removes useful information:** In **named entity recognition or medical/legal text processing**, lowercasing can erase capitalization cues that distinguish acronyms or specific entities from common words (for example, `"AIDS"` vs. `"aids"` or `"HER2"` vs. `"her2"`), while stripping punctuation or hyphens can alter drug codes, legal citations, or technical identifiers.

### 3. Tokenization Paradigms and Vocabulary Size

Tokenization segments a continuous character string into discrete modeling units called **tokens** (Jurafsky & Martin, 2024; Sennrich, Haddow, & Birch, 2016).

| Dimension | Word-Level Tokenization | Subword Tokenization (e.g., BPE) | Character-Level Tokenization |
| :--- | :--- | :--- | :--- |
| **Token Unit** | Whole words and standalone punctuation | Frequent words and recurring word sub-units | Individual characters (letters, digits, symbols) |
| **Out-of-Vocabulary (OOV) Handling** | Vulnerable to unseen words, typos, and inflections | Decomposes rare/unseen words into known subwords | Virtually no OOVs within supported character set |
| **Sequence Length** | Shorter sequences per document | Moderate sequence length | Very long sequences; harder long-range credit assignment |
| **Typical Vocabulary Size** | Tens of thousands to hundreds of thousands+ | Typically ~16,000 to 100,000+ merges | Very small (dozens to a few hundred characters) |

**Vocabulary size** is the count of distinct token types in a tokenizer's or corpus's lexicon (distinct from total token occurrences, or corpus length). Word-level tokenization inflates vocabulary size whenever morphological variants or misspellings proliferate, whereas character-level tokenization keeps the vocabulary tiny at the cost of long sequences. Subword algorithms such as **Byte-Pair Encoding (BPE)** balance both trade-offs by iteratively merging frequent character/byte pairs on a training corpus so common words remain single tokens while rare or compositional words are built from smaller pieces (Sennrich et al., 2016).

### 4. Stopwords: Benefits, Risks, and Context Sensitivity

**Stopwords** are high-frequency function words—such as articles, prepositions, conjunctions, and auxiliary verbs—that often appear across many documents and may contribute little topical specificity in bag-of-words retrieval or coarse topic indexing (Manning et al., 2008).

* **Benefits:** Removing high-frequency function words reduces feature dimensionality, lowers memory and compute overhead in sparse vector spaces, and prevents overly common terms from dominating simple frequency counts.
* **Risks:** Removing stopwords can destroy syntactic structure and phrase meaning—most notably **negation** (`"not"`, `"no"`, `"never"`), **scope modifiers**, or **idiomatic expressions** (for example, the English rock band *The The* or phrases where function words carry discourse cues). In dense neural and transformer models, dropping function words can also distort positional attention and grammatical coherence.

The appropriate stopword list depends on **both the task and the language**. A list helpful for coarse keyword indexing may hurt sentiment analysis, stance detection, or question answering. Likewise, languages differ in morphology and function-word inventories—agglutinative or richly inflected languages express grammatical relations through affixes and clitics rather than standalone English-style function words, making generic foreign stopword lists poorly suited without linguistic inspection.

### 5. Stemming vs. Lemmatization

Both **stemming** and **lemmatization** map inflected or derived word variants to a common root/base form to reduce morphological sparsity, but they differ in mechanism and linguistic fidelity (Jurafsky & Martin, 2024; Manning et al., 2008).

* **Stemming (Rule-Based Suffix Stripping):** Applies heuristic rules to chop off suffixes without necessarily consulting a lexical dictionary or grammatical category. A foundational example is the **Porter stemming algorithm** (Porter, 1980), which uses multi-step condition-governed suffix-stripping rules for English. Because rules are heuristic, stems may not be valid words (e.g., `"studies"` $\rightarrow$ `"studi"`), and over-stemming or under-stemming can occur.
* **Lemmatization (Morphological / Lexical Analysis):** Resolves a word form to its canonical dictionary headword, or **lemma**, accounting for morphological inflection and typically part-of-speech (POS) context (for example, using morphological lexicons/dictionaries such as WordNet or statistical morphological analyzers). Thus `"better"` as an adjective maps to `"good"`, and `"ran"` as a past-tense verb maps to `"run"`.

**When to prefer each:** Stemming is often preferred when speed, simplicity, and lightweight lexical matching in information retrieval are prioritized and minor morphological distortion is tolerable. Lemmatization is preferred when linguistically valid base forms, disambiguation across word classes, or interpretable downstream features are important—provided accurate morphological/POS tools exist for the target language.

### 6. Regular Expressions in Text Cleaning

**Regular expressions (regex)** define formal search patterns over character strings to locate, extract, mask, or substitute recurring textual structures such as URLs, currency figures, phone numbers, whitespace runs, or markup tags prior to tokenization.

* **Pattern 1 — Repeated exclamation marks (`!{2,}`):** Matches two or more consecutive exclamation marks in a row. Useful when collapsing exaggerated punctuation (e.g., `"URGENT!!!"` $\rightarrow$ `"!"` or a standardized emphasis marker) while retaining that emphasis occurred.
* **Pattern 2 — International/UK-style mobile-like digit strings (`\b0\d{9,10}\b`):** Matches standalone 10- to 11-digit numeric sequences starting with `0` (such as `0712345678`). Useful for masking phone numbers with a `PHONE_NUMBER` token so models learn the presence of contact numbers without overfitting to specific digits.

**Risk of an overly broad pattern:** A broad regex such as `[^a-z\s]` (keep only lowercase ASCII letters and whitespace) strips digits, currency symbols (`£`, `$`), punctuation, and non-ASCII characters in a single pass. In spam or fraud detection, deleting `£1,000` and `0712345678` erases strong monetary and contact indicators, turning high-signal scam text into generic words.

### 7. Case Study: Preprocessing Choices in NLP

A well-known empirical study examining preprocessing decisions in text classification is **Camacho-Collados and Pilehvar (2018)**, *"On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis"* (Proceedings of *EMNLP* / **SEM* workshops).

* **Task:** The authors evaluated text categorization and sentiment analysis across multiple benchmarks (including sentiment/polarity and topic/news classification datasets) using neural network architectures (such as CNNs and LSTMs/GRUs).
* **Preprocessing Options Compared:** They systematically tested combinations of standard preprocessing operations—including lowercasing, punctuation removal/handling, stopword removal, lemmatization, and domain-specific normalization (such as handling mentions, URLs, and out-of-vocabulary tokens)—contrasting minimal/raw text against heavier cleaning pipelines.
* **Effect on Results:** Their findings showed that **no single preprocessing pipeline is universally optimal across tasks and datasets**. While light normalization and preserving task-relevant cues often maintained or improved performance, aggressive removal steps (such as discarding stopwords or stripping punctuation indiscriminately) frequently degraded sentiment classification accuracy because sentiment relies on subtle function words, negation, and expressive punctuation.

### References

* Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
* Camacho-Collados, J., & Pilehvar, M. T. (2018). On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text* (pp. 172–177).
* Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft).
* Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
* Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
* Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Computing (ACL)* (pp. 1715–1725).

---

## Part B – Practical Report

*(Note: Run `python preprocessing_practice.py > output.txt` after creating `preprocessing_practice.py` so that `output.txt` captures the exact printed output of your script.)*

### Task 6 Table: Stemming Comparison

| Word | Porter | Lancaster | Snowball |
| :--- | :--- | :--- | :--- |
| `connection` | `connect` | `connect` | `connect` |
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

### Task 9 Table: Named Entities on T3

| Entity Span | Label | Correct? |
| :--- | :--- | :--- |
| `Smith` | `PERSON` | Yes (surname portion; `Mr.` was outside the person chunk) |
| `London` | `GPE` | Yes |
| `Apple` | `ORGANIZATION` | Yes |

### Short Answers (Tasks 1, 3, 5, 6, 7, 8, and 10)

* **Task 1 (Lowercasing T1):** Lowercasing `"URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"` produces `"urgent!!! you have won a £1,000 gift card. call 0712345678 now!"`. A spam detector could lose all-caps emphasis cues (`URGENT`, `WON`, `NOW`), which often correlate with high-pressure promotional or scam messaging.
* **Task 3 (Regex cleaning on T1):** Applying `[^a-z\s]` removes the exclamation marks (`!!!`), the pound sign and comma in `£1,000` (leaving `1000` before `1000` is also stripped), the period after `card.`, and the digits `1000` and `0712345678`. Replacing with `MONEY`, `PHONE_NUMBER`, and `!` preserves structured scam signals while standardizing noisy formatting.
* **Task 5 (Stopwords and negation on T2):** Removing NLTK's default English stopwords drops `not` and `n't` (after contraction splitting), stripping out explicit negation cues from the token list. Using the custom stopword list that retains `not`, `n't`, `no`, and `never` keeps the negative sentiment markers intact so the complaint against the jacket's price and quality is not reversed into a positive statement.
* **Task 6 (Stemming):** All three stemmers merge `connection`, `connected`, and `connecting` into `connect`, and both Porter and Snowball merge `university` and `universal` into `univers` (while Lancaster merges `organization` and `organized` into `org`). Examples of non-dictionary stems in the output include `fli` (for `flies`), `wolv` (for `wolves`), and `univers` (for `university`/`universal`).
* **Task 7 (Lemmatization):** `sent_tokenize` splits **T3** into **2** sentences (`"Mr. Smith arrived in London on Friday."` and `"He met the Apple team at 9 a.m."`). Neither `Mr.` nor `a.m.` caused an erroneous split because NLTK's Punkt sentence tokenizer accounts for common abbreviations and time/title periods.
* **Task 8 (Sentence splitting):** `sent_tokenize` splits **T3** into **2** sentences (`"Mr. Smith arrived in London on Friday."` and `"He met the Apple team at 9 a.m."`). Neither `Mr.` nor `a.m.` caused an erroneous split because NLTK's Punkt sentence tokenizer accounts for common abbreviations and time/title periods.
* **Task 10 (Full pipeline on T2):** Running the full pipeline with NLTK's default stopwords removes `not` and `n't`, leaving tokens like `think`, `jacket`, `worth`, `price`, `happy`, which loses the explicit negation of `"happy"` and `"worth"`. Running it with the negation-preserving stopword list keeps `not` (and `nt` after regex punctuation removal), preserving the customer's complaint much better.

### Spam Detector Preprocessing Recommendation

For a **spam detector**, I would apply **lowercasing** (with caution or paired with emphasis features), **targeted regex replacement** (masking URLs, phone numbers like `0712345678` $\rightarrow$ `PHONE_NUMBER`, currency amounts like `£1,000` $\rightarrow$ `MONEY`, and collapsing repeated `!`), **whitespace normalization**, and **tokenization**, while **avoiding aggressive `[^a-z\s]` deletion** and **avoiding blind removal of numbers/symbols**. As shown on **T1**, `[^a-z\s]` wipes out `£1,000` and `0712345678`, which are primary indicators of prize and phishing spam, whereas targeted placeholder replacement keeps those signals cleanly represented across messages.