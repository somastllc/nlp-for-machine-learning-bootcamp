# Assignment 2 – Preprocessing

**Name:** Sagal Mohamed Yusuf

---

# Part A – Research

## 1. What is text preprocessing, and why can more cleaning hurt?

Text preprocessing is the set of transformations applied to raw text before a model sees it. Typical steps are normalization, tokenization, filtering and reducing words to a base form. Raw text is noisy and inconsistent: "Good", "good" and "GOOD!!!" are different strings to a computer. Preprocessing reduces this variation and shrinks the feature space, so there is less sparsity and the model can learn from fewer examples.

Cleaning can hurt because every step discards information, and what counts as noise depends on the task. A survey chapter on the text classification pipeline (arXiv:2501.00174) notes that preprocessing can inadvertently delete important data, such as relevant stopwords, or introduce errors, for example when stemming merges words with different meanings. The right choices therefore depend on the specific model and dataset.

**Real-life example:** In sentiment analysis of product reviews, the review "This phone is not good" becomes "phone good" if the stopword list contains *not*. A negative review now looks positive. A search engine has a similar problem: aggressive stopword removal can turn the query "to be or not to be" into nothing at all.

## 2. Text normalization

Normalization maps different surface forms of the same thing to one canonical form. Common steps:

- **Lowercasing:** "The" and "the" become one token.
- **Whitespace normalization:** collapse tabs, newlines and repeated spaces.
- **Unicode normalization:** the Unicode Standard defines forms such as NFC and NFD (UAX #15). "é" can be stored as one precomposed character or as "e" plus a combining accent. Without normalization, two visually identical words are different strings.

**When it helps:** Searching for "café" fails to match documents that store it in the decomposed form, until both are converted to the same Unicode form. Lowercasing likewise helps in topic classification, where "Economy" at the start of a sentence should count as "economy".

**When it removes useful information:** Lowercasing erases the difference between "US" (the country) and "us" (the pronoun), or "Apple" (the company) and "apple" (the fruit). Named-entity recognition relies on capitalization as a signal. Capital letters also carry emphasis in sentiment analysis ("this is AWFUL").

## 3. Word-level, subword and character-level tokenization

**Vocabulary size** is the number of distinct token types the model recognizes. It determines the size of the embedding table and the output layer. Words missing from the vocabulary are *out-of-vocabulary (OOV)* and are usually replaced by an `<UNK>` symbol.

| | Word-level | Subword (BPE, WordPiece, Unigram) | Character-level |
|---|---|---|---|
| Unit | Whole words | Frequent word pieces | Single characters/bytes |
| Typical vocabulary | Very large (tens to hundreds of thousands) | Fixed, moderate (often ~30k-50k) | Very small (hundreds) |
| OOV problem | Severe for rare words, names, typos | Mostly solved (rare words split into pieces) | None |
| Sequence length | Short | Medium | Very long |
| Weakness | Cannot handle unseen words; poor for morphologically rich languages | Splits can be linguistically arbitrary | Long sequences, harder to capture word meaning |

Byte-pair encoding (BPE) was adapted for neural machine translation by Sennrich et al. (2016). It starts from characters and repeatedly merges the most frequent adjacent pair, until a chosen vocabulary size is reached. That target size is a hyperparameter you set, so subword tokenizers give direct control over vocabulary size. Word-level tokenization produces a vocabulary size that depends on the corpus, and it grows as the data grows. Character-level tokenization fixes the vocabulary but makes sequences much longer.

## 4. Stopwords

Stopwords are very frequent function words (such as *the, of, and, is*) that carry little topical content. A stopword list is simply a list of words to filter out.

**Benefits:** Removing them reduces dimensionality and noise, speeds up training, and can sharpen keyword-based methods like TF-IDF or topic models.

**Risks:** They can carry meaning. Negations (*not, never*) matter for sentiment, and prepositions matter for relations ("flight **from** Nairobi **to** Mogadishu"). Modern contextual models (BERT-style) use the full sentence, so removing stopwords can hurt them.

**Why the list depends on task and language:** A list built for English news fails on Somali or Arabic, which have different function words and morphology. Even within one language, the list should match the domain. Schofield, Magnusson and Mimno (2017) studied this for topic models and found that removing stopwords is often less critical than commonly assumed. HaCohen-Kerner et al. (2020) report that for Jewish law documents written in Hebrew-Aramaic, word unigrams that included stopwords gave better classification results than unigrams without them.

## 5. Stemming vs. lemmatization

**Stemming** chops affixes using rules, with no dictionary. The best-known algorithm is the **Porter stemmer** (1980), which applies five ordered phases of suffix-stripping rules with conditions on the remaining stem. It turns "caresses" into "caress" and "relational" into "relat". Its output is often not a real word ("ponies" becomes "poni"). It has two typical errors: *over-stemming* ("university" and "universe" both become "univers") and *under-stemming* (related words that stay apart).

**Lemmatization** returns the dictionary form (lemma) using vocabulary and morphological analysis. A common approach is the **WordNet lemmatizer** (as in NLTK), which looks words up in WordNet (Miller, 1995) given a part-of-speech tag. It maps "was" to "be" and "better" (adjective) to "good". It needs POS information, which adds cost and can introduce tagging errors. A survey of text mining techniques (arXiv:1707.02919) notes that lemmatization requires determining each word's part of speech first, which is error-prone, so stemming is often preferred in practice.

| | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix stripping | Dictionary + morphology + POS |
| Output | Stem (may not be a word) | Valid lemma |
| Speed | Fast | Slower |
| Best for | Information retrieval, large noisy corpora, speed | Tasks needing readable or accurate word forms (linguistic analysis, some classification) |

## 6. Regular expressions in text cleaning

A regular expression (regex) is a compact pattern language for matching strings. In cleaning, regexes remove or replace URLs, mentions, HTML tags, repeated whitespace and so on (see Friedl, 2006; Jurafsky & Martin, 2025).

**Pattern 1:** `https?://\S+` matches "http://" or "https://" followed by any non-whitespace characters, so it captures a whole URL. It is often replaced by a placeholder like `<URL>`.

**Pattern 2:** `\s+` matches one or more whitespace characters (spaces, tabs, newlines). Replacing it with a single space is a basic whitespace normalization.

**An overly broad pattern:** `[^a-zA-Z\s]` removes everything that is not an English letter or whitespace. This deletes digits ("COVID-19" becomes "COVID"), emojis (which carry sentiment) and accented letters ("café" becomes "caf"). It also damages languages with other conventions. In Somali spelling, the apostrophe marks a glottal stop (for example *go'aan*, "decision"), so stripping punctuation would produce a different string, "goaan". Test every regex on a sample of real data before applying it to the whole corpus.

## 7. Case study: Uysal & Günal (2014)

**Task:** Text classification, evaluated on two domains (e-mail and news) and two languages (Turkish and English).

**Options compared:** The authors tested tokenization, stopword removal, lowercasing and stemming, and evaluated all possible combinations of them. They examined how these combinations affect classification accuracy, text domain, text language and dimension reduction.

**Result:** Choosing a suitable combination of preprocessing tasks, rather than enabling or disabling them all, can significantly improve accuracy, depending on the domain and language. So there is no universal best pipeline. The same step can help in one domain or language and make no difference in another.

**Supporting study:** HaCohen-Kerner et al. (2020, PLOS ONE) ran a similar systematic comparison on four benchmark corpora. For three datasets, stopword removal was the only single preprocessing method that significantly improved on the baseline of 1,000 word unigrams. This is another good source for your discussion, because it varies the dataset and the results change.

---

# Part B – Practical

All results below come from my script `preprocessing_practice.py` (saved output in `output.txt`).

## Task 1: Lowercasing

After lowercasing T1, a spam detector can lose the capitalization pattern. The script found three all-caps words in T1 (`URGENT`, `WON`, `NOW`), and after lowercasing they look like ordinary words, so the "shouting" style that is typical of spam disappears. A safer option is to count the all-caps words (here 3) as a feature before lowercasing.

## Task 3: Regex cleaning

The pattern `[^a-z\s]` deleted 21 characters in five runs: `!!!`, `£1,000`, `.`, `0712345678` and `!`. That includes the pound sign, the comma, the full stop and all the digits. The result, `urgent you have won a  gift card call  now`, has lost the money amount and the phone number (the most informative parts of this spam message) and has double spaces where they were. My replacement rules gave `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`, which keeps the fact that an amount and a phone number were present, without the specific values, and shortens `!!!` to a single `!`.

## Task 5: Stopwords and negation

With NLTK's default list the T2 tokens became `["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`, because `Not` was removed. With my negation-keeping list, `Not` stayed: `["n't", 'think', 'jacket', 'worth', 'price', '.', 'Not', 'happy', '😞']`. Without `Not`, the phrase "Not happy" becomes just "happy", so the second half of the review now reads as positive. Only two of my four negation words were actually in NLTK's list (198 words became 196), so `n't` and `never` were never being removed, and `n't` survives in both versions.

## Task 6: Stemming

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

All three stemmers merge `university` and `universal` into `univers`, even though the words have different meanings (over-stemming). Merging `connection`, `connected` and `connecting` into `connect` is the useful kind of merge. The stems `univers`, `fli` and `wolv` are not real words (WordNet found no entry), and Lancaster also produces `org`. Porter and Snowball turn `organization` into `organ`, which is a real word but the wrong meaning, and Lancaster turns `better` into `bet`.

## Task 7: Lemmatization

Five words changed only when I gave the part-of-speech hint: `connected`, `connecting`, `organized`, `better` and `ran`. Without a hint the lemmatizer treats every word as a noun, so verbs and adjectives stay as they are. With the hint they became `connect`, `connect`, `organize`, `good` and `run`. `flies` and `wolves` were already lemmatized correctly (`fly`, `wolf`) because they are nouns by default. Unlike the stemmers, the lemmatizer handles irregular forms such as `ran` and `better`.

## Task 8: Sentence splitting

`sent_tokenize` gave 2 sentences: "Mr. Smith arrived in London on Friday." and "He met the Apple team at 9 a.m." Neither `Mr.` nor `a.m.` caused a wrong split, because the sentence tokenizer knows common abbreviations. The final period of `a.m.` correctly ends the second sentence. One small quirk appeared later: `word_tokenize` split the text into `a.m` and a separate `.`.

## Task 9: POS tagging and NER

| Entity | Label | Correct? |
|---|---|---|
| Mr. | PERSON | Partly (the honorific was split off as its own entity instead of staying with "Smith") |
| Smith | PERSON | Yes |
| London | GPE | Yes |

The model missed three items. `Apple` was only tagged as a proper noun (NNP) and was not recognized as an ORGANIZATION. `Friday` (a day) and `9 a.m.` (a time) were also missed, because `ne_chunk` has no DATE or TIME labels. So the person and the place were recognized, but not the organization or the day.

## Task 10: Full pipeline

With NLTK's default stopwords the final result was `["n't", 'think', 'jacket', 'worth', 'price', 'happy']`. With my negation-keeping list it was `["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']`. The second version keeps the customer's complaint, because `not` stays in front of `happy`; the default version reads as a mildly positive review. Both versions lost the sad emoji at the regex stage, which removed another sentiment signal, and lemmatization changed nothing here.

## Preprocessing for a spam detector

For a spam detector I would use whitespace normalization and `word_tokenize` (Task 2 and Task 4 show these are safe and keep punctuation as separate tokens), and I would replace rather than delete patterns: my rules in Task 3 turned the amount and phone number into `MONEY` and `PHONE_NUMBER`, so the detector still sees that a spam message contains both, while the deleting pattern removed them completely. I would extract the all-caps count (3 in T1) and the exclamation marks before lowercasing, because Task 1 shows lowercasing hides the shouting signal. I would avoid the broad deleting regex `[^a-z\s]` (21 characters lost from T1) and heavy stemmers such as Lancaster, which turned `organization` into `org` and `better` into `bet` (Task 6); I would use light stemming or lemmatization only if testing shows it helps. Stopword removal is not essential for spam, and Task 5 shows the default list can silently remove words like `not` that change meaning, so I would test it before using it.

---

# References

- Friedl, J. E. F. (2006). *Mastering Regular Expressions* (3rd ed.). O'Reilly.
- HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020). The influence of preprocessing on text classification using a bag-of-words representation. *PLOS ONE, 15*(5), e0232525. https://doi.org/10.1371/journal.pone.0232525
- Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39-41.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130-137.
- Schofield, A., Magnusson, M., & Mimno, D. (2017). Pulling out the stops: Rethinking stopword removal for topic models. *EACL*.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL*.
- Unicode Consortium. (n.d.). *Unicode Standard Annex #15: Unicode Normalization Forms*.
- Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104-112. https://doi.org/10.1016/j.ipm.2013.08.006
- (Survey chapters used for background) *The Text Classification Pipeline: Starting Shallow going Deeper*, arXiv:2501.00174; *A Brief Survey of Text Mining: Classification, Clustering and Extraction Techniques*, arXiv:1707.02919.
