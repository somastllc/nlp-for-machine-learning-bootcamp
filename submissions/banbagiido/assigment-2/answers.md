# Assignment 2 – Preprocessing

## Part A – Research

### 1. Text preprocessing

**Definition.** Text preprocessing is the set of transformations applied to raw text before it is given to an NLP model. Typical transformations are changing case, splitting text into tokens, standardising spaces and characters, filtering words, and reducing words to a base form (Jurafsky & Martin, 2024, ch. 2; Manning, Raghavan & Schütze, 2008, ch. 2).

**Why it is needed.** Raw text is noisy and inconsistent. A model that counts words sees `Free`, `FREE` and `free` as three different strings, and sees `prize!` and `prize` as two different tokens. This spreads the evidence across many rare features and makes learning harder, especially with small datasets. Preprocessing reduces this surface variation so the model can learn from more examples of the same idea.

**Why more cleaning can hurt.** Every cleaning step throws away some information, and some of that information may be what the task needs. Uysal and Gunal (2014) tested combinations of preprocessing steps for text classification and concluded that no single combination was best everywhere; the right choice depended on the domain and the language. Cleaning is therefore a trade-off, not a free improvement.

**Real-life example.** Consider an SMS spam filter. The message "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!" has strong spam signals: capital letters, repeated exclamation marks, a money amount and a phone number. A pipeline that lowercases everything and deletes all non-letters leaves "urgent you have won a gift card call now", which looks much like a normal message. The cleaning removed exactly the evidence the classifier needed.

### 2. Text normalization

Normalization means converting different written forms of the same thing into one standard form. Common operations are lowercasing, collapsing repeated whitespace, and Unicode normalization. In Unicode, the same visible character can be stored in more than one way (for example `é` as one code point, or `e` plus a combining accent). The Unicode Standard defines normalization forms (NFC, NFD, NFKC, NFKD) so that equivalent strings compare as equal (Unicode Consortium, UAX #15).

- **Where it helps:** In a product-review search system, users type "iPhone", "IPHONE" and "iphone". Lowercasing lets one query match all documents. Similarly, NFC normalization makes the Somali or French words typed with different accent encodings match.
- **Where it removes useful information:** In named entity recognition, capitalization helps identify names. Lowercasing turns "Apple" (company) and "apple" (fruit), or "US" and "us", into the same token, and the model loses a strong clue.

### 3. Word-level, subword and character-level tokenization

| Aspect | Word-level | Subword | Character-level |
|---|---|---|---|
| Unit | Whole word (split by rules or spaces) | Frequent pieces learned from data (e.g. BPE, WordPiece, Unigram) | Single character (or byte) |
| Vocabulary size | Very large, grows with the corpus | Medium, fixed by design (commonly tens of thousands) | Very small |
| Unknown words | Out-of-vocabulary problem | Rare words split into known pieces | No unknown words |
| Sequence length | Shortest | Slightly longer than words | Longest |
| Needs training | No | Yes, on a large corpus | No |

**Vocabulary size** is the number of distinct token entries a tokenizer or representation can produce. It is not the number of tokens in a text: the sentence "the cat saw the dog" has 5 tokens but only 4 vocabulary entries.

**Effect of the tokenizer choice.** A word-level tokenizer needs an entry for every word form, so the vocabulary becomes huge and still fails on new words. A character-level tokenizer has a tiny vocabulary but makes sequences very long and puts meaning at a very low level. Subword methods are a compromise: Sennrich, Haddow and Birch (2016) applied byte-pair encoding to represent rare and unseen words as sequences of subword units, and the target vocabulary size is chosen by the designer. This is why most modern language models use subword tokenization (see also Kudo & Richardson, 2018).

### 4. Stopwords

**Definition.** Stopwords are very frequent words in a language (such as *the*, *is*, *and* in English) that are often assumed to carry little topical information. In information retrieval they were traditionally dropped to save space and reduce noise (Manning et al., 2008, ch. 2).

**Benefits.** Removal reduces the number of features and can make topic-based tasks (search, topic modelling, keyword extraction) focus on content words.

**Risks.** Some stopwords carry meaning. Removing *not* from "this movie is not good" gives "movie good" and reverses the sentiment. Phrases and bigrams such as "to be or not to be" can also disappear. Schofield, Magnusson and Mimno (2017) studied the effect of stopword removal on topic models and found that the benefit is often smaller than people assume, which supports the idea of testing before removing.

**Why the list depends on task and language.** A list built for English search is wrong for sentiment analysis, where negations and intensifiers matter. Lists also differ for each language, because function words, morphology and writing systems differ; an English list cannot be used for Somali. In practice the list should be checked and edited for the task, or replaced by statistical weighting such as TF-IDF.

### 5. Stemming and lemmatization

**Stemming** reduces a word to a stem by removing or changing endings using rules, without needing a dictionary. The best known algorithm is the **Porter stemmer** (Porter, 1980), which applies several ordered steps of suffix-stripping rules. It is fast, but the stem may not be a real word (`studies` → `studi`) and different words can collapse together. More aggressive algorithms such as Lancaster cut even more, and Snowball (Porter2) is an improved, multi-language version.

**Lemmatization** maps a word to its dictionary form (lemma), using a vocabulary and morphological analysis (Manning et al., 2008, ch. 2). A well-known approach is NLTK's `WordNetLemmatizer`, which looks words up in the WordNet lexical database (Miller, 1995; Bird, Klein & Loper, 2009). It needs a part-of-speech hint to be correct: *better* → *good* only if it is known to be an adjective. Modern toolkits such as spaCy use trained models to find the lemma from context.

**When to prefer each.**

- *Stemming* is a good choice when speed matters and the exact form is not shown to users, for example in search indexing or a quick baseline.
- *Lemmatization* is preferred when real words are needed, when irregular forms matter (`wolves` → `wolf`, `ran` → `run`), or when results are read by people. It is slower and needs language resources and often POS tags.
- For many modern models neither is needed; they should be tested on validation data.

### 6. Regular expressions in text cleaning

A regular expression (regex) is a compact pattern language for matching strings. In cleaning, regexes are used to find and replace things such as URLs, numbers, repeated spaces or punctuation (Jurafsky & Martin, 2024, ch. 2).

**Example 1:** `https?://\S+` matches `http://` or `https://` followed by any non-space characters, so it matches a whole web link. Replacing it with `URL` keeps the fact that a link appeared without keeping the exact address.

**Example 2:** `\s+` matches one or more whitespace characters. Replacing it with a single space collapses repeated spaces, tabs and line breaks.

**How a broad pattern damages text.** The pattern `[^a-z\s]` deletes every character that is not an English lowercase letter or whitespace. Applied to "Win $500 now, don't miss it" it removes `$500` (useful spam evidence), turns `don't` into `don t`, and would also destroy accented letters and any non-English alphabet. A pattern should therefore be as specific as the goal requires.

### 7. Case study

**Paper:** Uysal, A. K., & Gunal, S. (2014). *The impact of preprocessing on text classification.* Information Processing & Management, 50(1), 104–112.

- **Task:** Text classification, tested on e-mail and news datasets in two languages (English and Turkish).
- **Preprocessing options compared:** tokenization, stopword removal, lowercase conversion and stemming, tested in all combinations with a standard classifier and feature selection.
- **Findings:** The authors report that the best choice depended on the domain and the language, and that no single combination was always best; some steps helped in one dataset and did not help or hurt in another. They concluded that the combination should be chosen carefully for each setting and not applied automatically. *(Before submitting, read the paper yourself and check these details against its abstract and results tables, and add the exact numbers you find there.)*

### References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly Media.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft), chapter 2: Regular Expressions, Text Normalization, Edit Distance.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *EMNLP 2018: System Demonstrations.*
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press, chapter 2.
- Miller, G. A. (1995). WordNet: a lexical database for English. *Communications of the ACM, 38*(11), 39–41.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.
- Schofield, A., Magnusson, M., & Mimno, D. (2017). Pulling out the stops: Rethinking stopword removal for topic models. *EACL 2017.*
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL 2016.*
- Unicode Consortium. *Unicode Standard Annex #15: Unicode Normalization Forms.* https://unicode.org/reports/tr15/
- Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112.

---

## Part B – Practical

> Fill every `<<...>>` below by copying from your own `output.txt` after running
> `python preprocessing_practice.py > output.txt`. Do not type results from memory.

### Task 1 – Lowercasing
<<2–4 sentences: what a spam detector loses. Use the "Fully-capitalised words" line: URGENT, WON, NOW lose their shouting signal.>>

### Task 3 – Regex cleaning
<<2–4 sentences: list everything `[^a-z\s]` deleted (from the "Characters deleted" list), and say why the replace version (`PHONE_NUMBER`, `MONEY`, `!`) keeps more evidence.>>

### Task 5 – Stopwords and negation
<<2–4 sentences: compare the default result with the negation-keeping result. Mention whether `not` and `n't` survived, and note that NLTK's list contains `not` but not `n't`.>>

### Task 6 – Stemming

| Word | Porter | Lancaster | Snowball |
|---|---|---|---|
<<one row per word, copied from the Section 6 table in output.txt>>

<<2–4 sentences: one pair of different words merged into the same stem (see "Different words merged"), and one stem that is not a real word (see "not dictionary words").>>

### Task 7 – Lemmatization
<<2–4 sentences: which words changed only with the POS hint (see "Words that changed only with the hint").>>

### Task 8 – Sentence splitting
<<2–4 sentences: number of sentences, and whether `Mr.` or `a.m.` caused a wrong split.>>

### Task 9 – Entities

| Entity | Label from ne_chunk | Correct? |
|---|---|---|
<<one row per entity from output.txt; correct label should be Smith = PERSON, London = GPE, Apple = ORGANIZATION>>

<<Missed items: check the "Missed" list. Was Friday recognised? Was every person, place and organisation found?>>

### Task 10 – Full pipeline
<<2–4 sentences: compare version A (default stopwords) and version B (negation-keeping). Which one keeps the complaint? Check the `'not' survived?` lines. Also note what the regex did to `don't`.>>

### Spam detector paragraph
<<One paragraph. Suggested direction, using your outputs as evidence: keep or mark URLs, money, phone numbers and repeated `!` (Task 3 replace version); be careful with lowercasing because of URGENT/WON/NOW (Task 1); avoid the delete-everything regex `[^a-z\s]`; do not remove negation words (Tasks 5 and 10); stemming/lemmatization optional and tested on validation data.>>
