# Assignment 2 — Text Preprocessing

> **Research and practical report draft.** Verify the sources and revise the explanations in your own words. The tables marked **TO COMPLETE** must be filled from the actual `output.txt` produced on your machine. They are intentionally not invented.

## Part A — Research

### 1. What is text preprocessing?
Text preprocessing is the set of operations used to convert raw written language into a representation that an NLP system can process consistently. Depending on the application, it can include Unicode handling, segmentation, normalization, annotation, and removal or replacement of noisy strings. It matters because text often contains inconsistent spelling, punctuation, formatting, and linguistic forms. Yet aggressive cleaning can discard signals relevant to the target task: in spam detection, the capitalization, exclamation marks, phone number, and currency amount in an urgent promotional message may help identify a scam. Removing all nonletters makes these signals disappear. Modern neural models can also exploit raw surface features, so preprocessing should be evaluated rather than assumed beneficial [1, 2].

### 2. Text normalization
Normalization maps multiple written forms to a more consistent form. Lowercasing reduces separate vocabulary entries for `London` and `london`; whitespace normalization collapses runs of spaces; Unicode normalization can make canonically equivalent sequences compare equal. Unicode NFC composes characters where possible, whereas NFKC additionally applies compatibility mappings [3]. Normalization helps when duplicate spellings otherwise fragment search results. However, lowercasing `US` to `us` can erase a country/pronoun distinction, and compatibility normalization can erase typographic distinctions that matter for certain tasks. Choose transformations based on data and evaluation.

### 3. Tokenization and vocabulary
| Method | Unit | Strength | Limitation | Typical vocabulary impact |
|---|---|---|---|---|
| Word-level | Whole words | Readable, meaningful units | Unseen words and many inflections | Often large; can have out-of-vocabulary items |
| Subword | Learned word fragments | Can represent rare words by composing pieces | One word may become several tokens | Controlled, fixed learned vocabulary |
| Character-level | Individual characters | Handles arbitrary spellings with a small symbol set | Long sequences and weaker direct word semantics | Usually smallest |

**Vocabulary size** means the number of distinct token types supported by a tokenizer, not the number of tokens in one text. Subword methods such as byte-pair encoding repeatedly merge frequent symbol pairs to learn a compact reusable inventory. Tokenizer design affects sequence length, multilingual coverage, model efficiency, and handling of unfamiliar words [1, 4].

### 4. Stopwords
Stopwords are frequent words such as articles, pronouns, and auxiliaries that some pipelines remove to reduce dimensionality or noise. Removal can improve a simple bag-of-words retrieval or classification pipeline by suppressing very common terms, but it can destroy meaning in sentiment analysis: `not happy` is very different from `happy`. A stopword list suitable for English is not automatically suitable for Somali, Arabic, or another language, because grammar, morphology, and common function words differ. Even within one language, legal search, question answering, sentiment analysis, and topic modeling require different choices [1, 2].

### 5. Stemming and lemmatization
Stemming heuristically strips or rewrites word endings, sometimes producing non-dictionary forms. The Porter algorithm uses ordered suffix-removal rules and measures of stem structure; Lancaster is generally more aggressive, while Snowball provides language-specific stemming algorithms [5, 6]. Lemmatization instead aims to return a dictionary base form using lexical and grammatical information. NLTK's WordNet lemmatizer uses WordNet's morphological processing and benefits from a part-of-speech hint; for example, `ran` can become `run` when interpreted as a verb. Stemming is attractive for lightweight information retrieval; lemmatization is often preferred when readable canonical forms or grammatical distinctions matter [1, 5, 7].

### 6. Regular expressions
Regular expressions are concise patterns for finding or replacing character sequences. `\b0\d{9}\b` matches a ten-digit number beginning with zero when bounded as a word; in the provided example it can identify a phone-like string, although it is not a universal phone-number validator. `£\s?\d[\d,]*(?:\.\d+)?` matches a pound amount such as `£1,000`, allowing an optional space and decimal part. Replacing matches with `PHONE_NUMBER` and `MONEY` retains the *type* of information without retaining the exact value. Conversely, `[^a-z\s]` deletes all characters except lowercase ASCII letters and whitespace, destroying numbers, punctuation, emoji, and many non-English letters [2, 8].

### 7. Case study: preprocessing in sentiment classification
Camacho-Collados and Pilehvar (2018) systematically investigated preprocessing choices for neural text classification, including how text cleaning, stopword removal, and linguistic normalization affect downstream tasks. Their results show that common preprocessing steps do **not** universally improve accuracy: the effect varies with task and model, and some conventional cleaning choices can hurt neural classifiers. The practical lesson is to compare preprocessing configurations on a held-out validation set rather than applying every possible cleaning rule by default [9].

## Part B — Practical

**Reproducibility:** Run `python preprocessing_practice.py > output.txt` after installing dependencies and NLTK resources. Fill in the values below using the actual file. Do not type or guess model output. NLTK resource versions may change tokenization, POS tags, and NER labels.

### Task 1 — Lowercasing
Lowercasing standardizes words such as `URGENT` and `WON`, but loses their original uppercase emphasis. For spam detection, capitalization can be a predictive feature; a good compromise is to preserve an uppercase-count feature before normalizing.

### Task 3 — Regex cleaning
The pattern `[^a-z\s]` deletes punctuation, digits, the pound sign, and other non-ASCII-letter characters from lowercased T1. Record the exact deleted characters from `output.txt`. Replacing the phone number and money amount with category markers retains useful spam-related cues without keeping their exact values.

### Task 5 — Stopwords and negation
Compare the default and customized token lists in `output.txt`. In this NLTK configuration, default stopwords can remove `not` and `don't`/`n't` depending on tokenization, weakening the negative review. Preserving negation helps retain the customer's dissatisfaction.

### Task 6 — Stemming
Copy the script-generated comparison into this table:

| Word | Porter | Lancaster | Snowball |
|---|---|---|---|
| connection | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| connected | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| connecting | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| organization | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| organized | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| university | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| universal | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| flies | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| wolves | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| better | TO COMPLETE | TO COMPLETE | TO COMPLETE |
| ran | TO COMPLETE | TO COMPLETE | TO COMPLETE |

Identify a merged pair from the script's `MERGED GROUPS` output and a non-dictionary stem from the table. Explain why these are acceptable for matching but potentially harmful when meaning matters.

### Task 7 — Lemmatization
Copy the `DEFAULT` and `WITH POS` values from `output.txt` and identify exactly which words changed only after the hint was supplied. WordNet's default noun assumption may leave verb past tenses unchanged; POS hints can recover the intended verb lemma. Interpret ambiguous words according to the explicit POS assumptions in the script.

### Task 8 — Sentence splitting
Record the sentence count from `output.txt` and check whether the abbreviations `Mr.` and `a.m.` caused false sentence boundaries. Sentence tokenizers use learned punctuation context rather than splitting at every period; results may depend on tokenizer resources and versions.

### Task 9 — POS tagging and named-entity recognition
Use the **actual** `ENTITIES` lines in `output.txt` to fill this table. Compare against the context: `Smith` is a person, `London` is a place, `Apple` is an organization, and `Friday` is a day (which may not be emitted by `ne_chunk`). Do not assume every mention will be detected or labeled correctly.

| Detected entity | NLTK label | Correct in context? | Explanation |
|---|---|---|---|
| TO COMPLETE | TO COMPLETE | TO COMPLETE | TO COMPLETE |

**Missed entities:** TO COMPLETE after reviewing `output.txt`.

### Task 10 — Full pipeline
Compare both `FINAL` outputs in `output.txt`. The negation-preserving version is more appropriate for sentiment classification if it retains the negative meaning of the complaint. Note that this simple regex also removes the sad emoji, another useful sentiment signal, so preserving emoji or mapping them to sentiment markers may improve the pipeline.

### Spam detector: recommended preprocessing
For a spam detector, I would normalize excessive whitespace and use selective lowercasing while also recording the original capitalization and punctuation counts. I would replace phone numbers and money amounts with category tokens because T1 contains potentially informative contact and reward cues; I would not simply delete every digit, currency symbol, or exclamation mark. I would also test keeping stopwords, negation, and emojis rather than removing them automatically, since they can signal persuasion or sentiment. The `output.txt` comparison of deletion versus replacement provides direct evidence of what information is lost and retained.

## References
1. Jurafsky, D., & Martin, J. H. *Speech and Language Processing*, 3rd ed. draft. https://web.stanford.edu/~jurafsky/slp3/
2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly. https://www.nltk.org/book/
3. Unicode Consortium. *Unicode Standard Annex #15: Unicode Normalization Forms*. https://unicode.org/reports/tr15/
4. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL 2016*. https://aclanthology.org/P16-1162/
5. Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137. https://doi.org/10.1108/eb046814
6. Snowball. *Snowball stemming algorithms*. https://snowballstem.org/
7. NLTK. *WordNetLemmatizer documentation*. https://www.nltk.org/api/nltk.stem.wordnet.html
8. Python Software Foundation. *Regular expression operations (`re`)*. https://docs.python.org/3/library/re.html
9. Camacho-Collados, J., & Pilehvar, M. T. (2018). On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis. *EMNLP 2018*. https://aclanthology.org/W18-5406/
