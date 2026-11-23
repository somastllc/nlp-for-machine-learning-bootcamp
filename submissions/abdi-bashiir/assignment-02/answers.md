# Assignment 2 – Preprocessing

## Part A – Research

### 1. What is text preprocessing?

Text preprocessing is the set of transformations applied to raw text before an NLP system uses it, for example changing case, fixing whitespace, splitting text into tokens, or removing selected words and symbols (Jurafsky & Martin, 2024; Manning et al., 2008). Raw text is messy: the same idea can be written in many surface forms, and a model that counts words will treat each form as a separate feature. Preprocessing reduces this variation so that the model sees more examples of each pattern.

More cleaning can make results worse because every step also discards information. Imagine a classifier that sorts delivery complaints into "problem" and "no problem". If the cleaning removes common words, "the package never arrived" can become "package arrived", which now looks like a happy customer. The cleaning removed the single word that carried the meaning. A step is only useful if the variation it removes is irrelevant to the task, so each step should be tested on validation data.

### 2. What is text normalization?

Normalization maps different written forms of the same thing to one standard form. Common steps are lowercasing, collapsing extra whitespace, and Unicode normalization. Unicode normalization (Unicode Standard Annex #15) matters because one visible character can be stored in more than one way: "é" can be a single code point or an "e" followed by a combining accent. Normal forms such as NFC make these identical, so the two spellings are not counted as different words (Davis & Whistler, 2023).

Helps: in a search engine, a user may type "Résumé", "résumé" or "RESUME", and normalization lets all of them match the same documents. Removes useful information: in named entity recognition, lowercasing turns "Apple" (the company) and "apple" (the fruit) into one string, and capital letters are one of the clues the recogniser uses to find names.

### 3. Word-level, subword, and character-level tokenization

A **word-level** tokenizer treats each word (and usually each punctuation mark) as a token. Sequences are short, but the vocabulary grows with every new spelling, name and inflected form, and any word not seen in training is out-of-vocabulary. A **character-level** tokenizer uses single characters, so the vocabulary is tiny and nothing is ever unknown, but sequences become very long and single characters carry little meaning. A **subword** tokenizer sits between the two: frequent words stay whole and rare words are split into smaller learned pieces, for example "unbelievably" into "un", "believ", "ably". Byte Pair Encoding (Sennrich et al., 2016), WordPiece and the SentencePiece library (Kudo & Richardson, 2018) are well-known methods; they learn their pieces from a corpus until a chosen vocabulary size is reached.

**Vocabulary size** is the number of distinct token types a tokenizer can produce; it is not the number of tokens in a text. The tokenizer choice largely decides it: a character tokenizer needs only the alphabet and symbols (a few hundred entries), a word tokenizer may need hundreds of thousands of entries for a large corpus, and a subword tokenizer lets the designer set it directly, commonly in the tens of thousands. A larger vocabulary gives shorter sequences but makes the model's embedding table bigger and leaves rare entries with little training data; a smaller vocabulary does the opposite.

### 4. What are stopwords?

Stopwords are very frequent, mostly grammatical words (in English "the", "of", "and") that carry little topical information. Information retrieval traditionally removed them to shrink the index and save time (Manning et al., 2008). Benefits: fewer features, faster processing, and less noise for tasks driven by content words such as topic search. Risks: they can carry meaning in phrases and negation ("to be or not to be" is almost all stopwords), and modern systems that use word order or context often work better with them kept. Manning et al. also note that good compression and weighting methods mean removal is often unnecessary.

The right list depends on the task: a sentiment classifier must keep "not" and "never", while a topic model may drop them. It also depends on the language, because frequent function words and word forms differ. A Somali list would contain words such as "iyo" and "waa", an English list would not, and languages with rich morphology attach function elements to words, so a list copied from English does not transfer.

### 5. Stemming and lemmatization

**Stemming** cuts affixes from words using rules so that related forms share one key. The best-known algorithm is Porter's (1980): it applies five ordered groups of suffix-rewriting rules, each rule conditioned on the "measure" (roughly the number of syllables) of the remaining stem. For example it maps "caresses" to "caress" and "relational" to "relat". It is fast and needs no dictionary, but the output is often not a real word and unrelated words can collide. Lancaster is more aggressive and Snowball (Porter2) is a refined version of Porter.

**Lemmatization** returns the dictionary form (lemma) of a word, using a lexicon and, ideally, the part of speech. NLTK's WordNetLemmatizer uses WordNet (Miller, 1995) and its morphological function: with the right part of speech it maps "mice" to "mouse" and "better" (adjective) to "good", which no suffix rule can do. It is slower, depends on a lexicon, and can give wrong results without a correct part-of-speech tag.

Stemming is preferred when speed matters and exact forms do not, such as indexing for search. Lemmatization is preferred when readable, linguistically valid forms matter or irregular forms are common, for example in text analysis or when preparing features for a classifier.

### 6. Regular expressions in text cleaning

Regular expressions describe text patterns that can be searched for or replaced (for example with Python's `re.sub`). In cleaning they find and normalise structured pieces of text such as emails, dates or repeated characters.

- `\b[\w.+-]+@[\w-]+\.[\w.]+\b` matches an email address: a local part, "@", a domain and a dot-suffix. Replacing it with `EMAIL` hides personal data but keeps the fact that an address appeared.
- `\d{4}-\d{2}-\d{2}` matches an ISO date such as 2026-11-22, which can be replaced with `DATE`.

An overly broad pattern damages text. `\d+` (delete all digits) turns "COVID-19" into "COVID-" and "Boeing 747" into "Boeing", and `[^\w\s]` (delete all punctuation) turns "C++" into "C" and "don't" into "dont". Patterns should be tested on real examples before they are applied to a whole corpus.

### 7. Case study: Twitter sentiment analysis

Symeonidis, Effrosynidis, and Arampatzis (2018) studied how preprocessing choices affect Twitter sentiment classification. They compared 16 preprocessing techniques on two Twitter datasets, SS-Twitter and SemEval, using four classifiers: Linear SVC, Bernoulli Naive Bayes, Logistic Regression, and a Convolutional Neural Network (CNN). They evaluated both classification accuracy and the number of features.

Their experiments found that the effect depended on the preprocessing technique and the classifier. Lemmatization, removing numbers, and replacing contractions improved accuracy in their experiments, while removing punctuation did not show the same benefit. In their combination study, replacing numbers and replacing repeated punctuation were especially significant, which shows that steps interact. The lesson is to test preprocessing choices on the actual dataset instead of assuming that more cleaning is always better.

## Part B – Practical Report

The script `preprocessing_practice.py` processes the supplied texts and words. Its complete printed output is saved in `output.txt`; every result below was copied from that file.

### Task 1 – Lowercasing

Lowercasing T1 changes `URGENT` and `WON` to lowercase but keeps digits and punctuation. A spam detector loses the emphasis shown by capital words, which can be a sign of an aggressive message.

### Task 3 – Regex cleaning

`[^a-z\s]` deletes every character that is not a lowercase letter or whitespace: `!`, `,`, `.`, `£` and the digits `0`–`8` that appear in T1. The result is `urgent you have won a  gift card call  now`, so the amount and the phone number are completely gone. The replacement rules instead produce `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`, which keeps the fact that an amount and a phone number appeared.

### Task 5 – Stopwords and negation

NLTK's default list contains `not` and `no` but not `n't`. With the default list the tokens become `["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`: `Not` was removed, so "Not happy" now reads as plain `happy` and the complaint is reversed. With my negation-keeping list the result is `["n't", 'think', 'jacket', 'worth', 'price', '.', 'Not', 'happy', '😞']`, which keeps `Not happy`. In both versions `n't` survives, but it is detached from `do` and does not replace the missing `Not`.

### Task 6 – Stemming

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

All three stemmers merge `university` and `universal` into `univers`, although the words have different meanings; All three also merge `organization` and `organized` (into `organ` for Porter and Snowball, and `org` for Lancaster). Several stems are not real words: `univers`, `fli`, `wolv`. Lancaster cuts `better` to `bet`, which changes the meaning completely.

### Task 7 – Lemmatization

The lemmatizer treats every word as a noun unless it gets a hint, so five words changed only when the hint was given: `connected` → `connect`, `connecting` → `connect`, `organized` → `organize`, `better` → `good` and `ran` → `run`. `flies` → `fly` and `wolves` → `wolf` were already correct as nouns, and `connection`, `organization`, `university` and `universal` did not change in either run.

### Task 8 – Sentence splitting

`sent_tokenize` returned 2 sentences: `Mr. Smith arrived in London on Friday.` and `He met the Apple team at 9 a.m.`. `Mr.` and `a.m.` did not cause wrong breaks. However, `word_tokenize` split the final `a.m.` into `a.m` and `.`, so the abbreviation lost its last period in the POS output.

### Task 9 – POS tagging and named entity recognition

`ne_chunk` found three entity chunks. It returned `Mr.` and `Smith` as two separate PERSON chunks instead of one name.

| Entity | Label from the model | Correct? |
|---|---|---|
| Mr. | PERSON | Partly: it is a title, and it was split from "Smith" |
| Smith | PERSON | Correct |
| London | GPE | Correct |

The model missed two items. `Apple` was tagged `NNP` but not grouped as an entity, although here it is a company (ORGANIZATION). `Friday` was also tagged `NNP` with no entity; `ne_chunk` has no DATE label, so a day is never recognized.

### Task 10 – Full pipeline

Both versions lowercase the text and the regex removes the period and the 😞 emoji, giving `i don't think this jacket is worth the price not happy at all`. Version A (default stopwords) ends with `["n't", 'think', 'jacket', 'worth', 'price', 'happy']`, so `not` is lost and the review looks positive. Version B (negation kept) ends with `["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']`, which keeps `not happy` and therefore the customer's complaint. Both versions lose the 😞 emoji, which also carried negative sentiment.

### Preprocessing choices for a spam detector

I would normalize whitespace, and replace phone numbers, money amounts and repeated `!` with markers. In my output, `[^a-z\s]` turned T1 into `urgent you have won a  gift card call  now`, deleting the £ amount and the whole phone number, while the replacement rules gave `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`, which keeps the evidence. I would not delete all digits and punctuation for the same reason. Lowercasing removed the capitals in `URGENT` and `WON`, so I would count fully capitalised words as a feature before lowercasing. I would avoid stopword removal that drops `not`, as T2 showed, and I would test each step on validation data before keeping it.

## References

1. Davis, M., & Whistler, K. (2023). *Unicode Standard Annex #15: Unicode Normalization Forms*. Unicode Consortium. https://unicode.org/reports/tr15/
2. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
3. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *Proceedings of EMNLP: System Demonstrations*, 66–71. https://doi.org/10.18653/v1/D18-2012
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
5. Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39–41. https://doi.org/10.1145/219717.219748
6. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814
7. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL*, 1715–1725. https://doi.org/10.18653/v1/P16-1162
8. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022
