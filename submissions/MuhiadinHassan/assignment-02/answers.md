# Assignment 2 – Preprocessing

## Part A – Research

### 1. What text preprocessing does

Text preprocessing is the set of operations that converts raw text into a form suitable for a particular NLP task. It can include normalization, tokenization, removal or replacement of selected text, and linguistic analysis such as stemming or lemmatization. These steps make text more consistent for search or modeling, but each step changes the evidence available to the system.

For example, an email spam detector may learn that all-caps words, repeated exclamation marks, suspicious phone numbers, or unusual currency amounts occur often in unwanted messages. Converting every character to lowercase and deleting every digit or punctuation mark could make the text simpler, but it would also remove useful spam signals. Preprocessing should therefore be tested against the task and its data; “cleaner” text is not automatically better (Symeonidis et al., 2018).

### 2. Text normalization

Normalization maps different surface forms to a consistent representation. **Lowercasing** maps `London` and `london` to the same string; **whitespace normalization** can replace runs of spaces, tabs, or line breaks with one space; and **Unicode normalization** maps canonically equivalent character sequences into a consistent form. For example, a letter with an accent may be encoded as one precomposed character or as a base letter followed by a combining mark. Unicode NFC normalization makes canonically equivalent strings comparable. The Unicode Consortium notes that compatibility forms such as NFKC can collapse additional distinctions and may lose information, so the form should match the use case (Unicode Consortium, 2026).

Lowercasing can help a case-insensitive search match `NLP` and `nlp` and can reduce a model’s vocabulary. It can also remove useful case clues: `US` may mean the country, while `us` is a pronoun; all-caps words may signal emphasis or spam. Likewise, aggressive Unicode compatibility normalization may be useful for loose identifier matching but inappropriate when exact typography or distinctions matter.

### 3. Word-level, subword, and character-level tokenization

**Word-level tokenization** divides text into words and often separates punctuation. It gives readable, short sequences, but a word absent from the training vocabulary becomes an out-of-vocabulary item. **Subword tokenization** breaks rare or unseen words into frequent pieces; methods such as byte-pair encoding build a vocabulary of these units from text. It can represent many word forms while keeping a manageable vocabulary, though one word may become several tokens. **Character-level tokenization** uses individual characters as tokens. Its vocabulary is small and it can represent any spelling, but sequences are much longer and the model must learn patterns from lower-level pieces (Jurafsky & Martin, 2026).

Vocabulary size means the number of distinct token types available to a tokenizer or model. A word tokenizer may have a large vocabulary; a subword tokenizer typically uses a fixed, smaller inventory than a full word list; and a character tokenizer generally has only the characters and symbols in its alphabet. Tokenizer choice trades off vocabulary size, sequence length, and how naturally word structure is represented.

### 4. Stopwords and why removal depends on the task

Stopwords are tokens that a particular system chooses to filter, commonly very frequent function words such as articles, pronouns, and prepositions. Removing them can shrink a bag-of-words feature space and reduce unhelpful matches in some retrieval settings. They are not universally meaningless, however: in sentiment analysis, deleting `not` can reverse the apparent polarity of “not happy”; in question answering, words such as `who` and `where` carry task-relevant information; and in authorship or spam detection, frequent words and writing habits can themselves be predictive.

A stopword list must fit the task and language. Lists differ across languages because grammatical systems and frequent words differ, and a list built for general English may not fit informal posts, dialects, or a specialized domain. Empirical studies have found that preprocessing choices can help one dataset or model and hurt another, so stopword removal should be evaluated rather than applied automatically (Samad et al., 2020; Symeonidis et al., 2018).

### 5. Stemming and lemmatization

**Stemming** applies rules that strip endings to produce a stem; the result need not be a dictionary word. Martin Porter’s well-known English algorithm applies ordered suffix-stripping rules, for example mapping related forms such as `connected` and `connecting` toward `connect` (Porter, 1980). NLTK also provides the more aggressive Lancaster stemmer and the Snowball English stemmer. Stemming is usually fast and can be useful in a broad search index where matching related forms matters more than preserving exact grammar, but it may over-merge words or create awkward stems.

**Lemmatization** maps a word to a dictionary lemma, usually using lexical information and sometimes its part of speech. NLTK’s `WordNetLemmatizer` uses WordNet and defaults to noun treatment; passing a POS hint can make verb forms such as `ran` map to `run`, or the adjective `better` map to `good` (Bird, Klein, & Loper, 2009; NLTK Project, n.d.). Lemmatization is preferable when readable, linguistically plausible base forms matter and the required lexical resource is available. It may be slower and can return the wrong form if the POS hint is wrong.

### 6. Regular expressions for cleaning

Regular expressions describe text patterns that can be searched, replaced, or removed. For example, `\b\d{3}-\d{3}-\d{4}\b` matches a phone number shaped like `123-456-7890`, while `!{2,}` matches a run of two or more exclamation marks and can replace it with a single `!`. Regex rules are useful for predictable formats such as phone numbers, URLs, or repeated punctuation.

An overly broad rule such as `[^a-z\s]` removes every character except lowercase English letters and whitespace. Applied to a message, it deletes numbers, currency symbols, punctuation, emoji, and non-ASCII letters together. That may erase a phone number that indicates spam, the currency in an offer, punctuation that conveys urgency, or the emoji expressing sentiment. Replacing a recognized pattern with a meaningful tag is often safer than deleting it.

### 7. Case study: comparing Twitter preprocessing choices

Symeonidis, Effrosynidis, and Arampatzis (2018) studied three-way sentiment classification of Twitter posts. They compared 16 preprocessing techniques across two Twitter sentiment datasets and four classifiers (Linear SVC, Bernoulli Naive Bayes, logistic regression, and a convolutional neural network), measuring classification accuracy and the resulting feature counts.

Their comparisons show that preprocessing effects were not uniform. The paper reports that lemmatization, removing numbers, and replacing contractions improved accuracy in their experiments, while simply removing punctuation did not. Their ablation and combination analysis also found that technique interactions mattered, including the handling of numbers and repeated punctuation. The practical lesson is not that one fixed recipe always wins, but that preprocessing must be evaluated with the intended data and model.

## Part B – Practical

The output below was generated by running `preprocessing_practice.py` with NLTK 3.10.4 and its required NLTK data. The repository available for this task did not contain the assignment’s `code/download-nltk-data.py`, so NLTK’s downloader was used to obtain the tokenizer, stopword, WordNet, POS-tagger, and named-entity resources. The complete captured output is in `output.txt`.

### Task 1 – Lowercasing

The script lowercased T1 from `URGENT!!! ... WON ... NOW!` to `urgent!!! ... won ... now!`. A spam detector could lose the all-caps emphasis in `URGENT`, `WON`, and `NOW`, which may be useful signals. Lowercasing keeps the words but removes that capitalization pattern.

### Task 2 – Whitespace

The script printed T4 with `repr()` before and after normalization. Runs of spaces became single spaces and the leading spaces were removed; see the exact before-and-after strings in `output.txt`.

### Task 3 – Regex cleaning and replacement

With `[^a-z\s]`, the output deleted `!` (4 occurrences), `£` (1), comma (1), period (1), and all 14 digits in the amount and phone number. The replacement rules produced `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`, retaining the meaning of the matched patterns as tags.

### Task 4 – Tokenization

For T2, Python `split()` produced 14 tokens, while NLTK `word_tokenize()` produced 16 because it split the contraction into `do` and `n't` and separated the period. For T4, NLTK produced 11 tokens and 10 unique case-sensitive tokens; the full list appears in `output.txt`.

### Task 5 – Stopwords and negation

The default NLTK English list removed `not` from T2; in this NLTK version, it did not remove the separate token `n't`. The custom list explicitly protects `not`, `n't`, `no`, and `never`, so the printed result retains both negative markers present in this sentence. This matters because dropping `not` can make a complaint such as “Not happy” look positive or neutral.

### Task 6 – Stemming table

The table below is copied from the script output. All three stemmers merge `connection`, `connected`, and `connecting` into `connect`. They can also produce non-dictionary stems: for example, `flies` becomes `fli` and `wolves` becomes `wolv`.

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

### Task 7 – Lemmatization

Without a POS hint, `WordNetLemmatizer` treats words as nouns, so several inflected verbs remain unchanged. With the selected POS hints, `connected`, `connecting`, `organized`, and `ran` become `connect`, `connect`, `organize`, and `run`; `better` becomes `good` as an adjective. These changes show why lemmatization needs grammatical context.

### Task 8 – Sentence splitting

NLTK split T3 into two sentences. In this run it kept `Mr. Smith` together and did not split at `a.m.`; the exact sentences are printed in `output.txt`.

### Task 9 – POS tagging and named entities

The table reflects the actual `ne_chunk` output. NLTK split the title and surname into separate person chunks, and it recognized London as a geopolitical entity.

| Recognized text | NLTK label | Correct? | Note |
|---|---|---|---|
| Mr. | PERSON | No, incomplete | Title was separated from the person’s name. |
| Smith | PERSON | Partly | A person surname, but the full span should be `Mr. Smith`. |
| London | GPE | Yes | Correctly identified as a place. |

The model missed `Friday` as a date and `Apple` as an organization. This illustrates that an NER model can miss entities or assign incomplete spans even when the POS tagger marks the tokens as proper nouns. `Apple` is especially context-dependent: the sentence means the company, not the fruit.

### Task 10 – Full pipeline

The pipeline lowercased T2, removed punctuation and the emoji while preserving apostrophes, tokenized it, filtered stopwords, and lemmatized the remaining tokens. The default list removes `not`, while the custom list keeps `not` and `n't`, so the custom-list version retains the complaint’s negative meaning more clearly. The exact text after each stage is in `output.txt`.

### Preprocessing choices for a spam detector

For spam detection, I would normalize repeated whitespace and apply conservative case handling while also preserving or separately recording signals such as all-caps words, repeated exclamation marks, phone numbers, currency amounts, URLs, and emoji. The script’s T1 output demonstrates why I would replace detected phone and money patterns with tags instead of deleting all digits and symbols: the broad regex erases the offer amount and phone number, while tagged replacements preserve their presence. I would test stopword removal and stemming against labeled spam examples before using them, and I would avoid removing negation or punctuation without checking whether model accuracy changes.

## References

Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O’Reilly Media. https://www.nltk.org/book/

Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

NLTK Project. (n.d.). *NLTK documentation: Stemmers and WordNetLemmatizer*. https://www.nltk.org/howto/stem.html ; https://www.nltk.org/api/nltk.stem.wordnet.html

Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

Samad, M. D., Khounviengxay, N. D., & Witherow, M. A. (2020). Effect of text processing steps on Twitter sentiment classification using word embedding. *Proceedings of the 2020 IEEE International Conference on Big Data*, 4512–4518. https://arxiv.org/abs/2007.13027

Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022

Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode normalization forms*. https://www.unicode.org/reports/tr15/
