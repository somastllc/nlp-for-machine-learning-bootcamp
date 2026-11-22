# Assignment 2 – Preprocessing

## Part A – Research

### 1. What is text preprocessing, and why does it matter?

Text preprocessing is the process of converting raw written language into a form that an NLP system can analyze consistently. Depending on the model and task, it may involve normalizing characters, separating text into tokens, identifying special strings, or reducing variation between related word forms. Its purpose is not simply to make text look clean: it is to give the model relevant, usable information while avoiding unnecessary differences in how people write [1, 2].

For example, a customer-support system may receive the messages “Order #4521 is LATE!!!” and “order #4521 is late.” Lowercasing can help a traditional classifier recognize their shared vocabulary. However, deleting the order number prevents a support system from identifying the affected order, and deleting the exclamation marks may remove a clue about the customer's frustration. More cleaning can therefore make results worse when it removes information needed for the task. The appropriate preprocessing should be evaluated against model performance, not the apparent neatness of the text [2, 6].

### 2. Text normalization: benefits and information loss

Text normalization makes alternative written forms more consistent. **Lowercasing** turns capital letters into lowercase ones; **whitespace normalization** replaces irregular runs of spaces, tabs, or line breaks with a consistent separator; and **Unicode normalization** standardizes different character sequences that represent the same text. For example, Unicode NFC can make a letter represented by one precomposed character consistent with the equivalent representation built from a base letter and combining accent [1, 2].

Normalization helps when a basic search tool should find the same product whether someone enters “Notebook”, “NOTEBOOK”, or “notebook”. On the other hand, lowercasing may remove useful distinctions, such as between “US” (an abbreviation for a country) and “us” (a pronoun). Flattening whitespace can also damage structured text, including poetry or tables, where layout has meaning. A preprocessing rule must therefore be selected for the intended application [1, 2].

### 3. Word, subword, and character tokenization

Tokenization divides input text into units called **tokens**, which the model then represents or processes. The size of a token depends on the tokenizer. **Vocabulary size** is the number of distinct token types recognized by the tokenizer or representation, not the number of token occurrences in a particular sentence [1, 3].

| Approach | How it divides text | Typical vocabulary effect | Strength | Limitation |
|---|---|---|---|---|
| Word-level | Whole words, often with punctuation handled separately | Potentially very large, especially for languages with many word forms | Easy to interpret; useful in classical NLP | Rare or unseen words can be out of vocabulary |
| Subword | Frequent words may stay whole; unfamiliar words are split into learned pieces | Controlled, moderate-sized vocabulary | Can represent many rare forms using known pieces | Splits are not necessarily linguistically meaningful |
| Character-level | Individual characters | Usually small vocabulary | Handles unfamiliar spellings and characters | Creates long token sequences and makes longer-range patterns harder to learn |

For illustration, “unhappiness” might be one word token, several learned subword tokens, or individual letters. The exact subword division depends on the tokenizer's training data. SentencePiece is a documented approach for training subword tokenizers, including BPE and unigram-based models [3]. These trade-offs matter for morphologically rich and lower-resource languages such as Somali: a word-only vocabulary may grow quickly, while a suitably trained subword model can reuse pieces across different forms.

### 4. Stopwords: benefits, risks, and language dependence

**Stopwords** are frequently occurring words that a system may choose to exclude from its features. In some traditional information-retrieval or topic-modeling workflows, removing words that appear in almost every document can reduce feature counts and computational cost. Yet a word's frequency does not guarantee that it is unimportant [1, 2].

For instance, “The service is good” and “The service is **not** good” express different evaluations. Removing “not” can turn a complaint into a misleading positive-looking fragment. Pronouns, question words, and function words can likewise be essential for authorship analysis, question answering, and language understanding. A suitable stopword list must account for the **task**, **domain**, **language**, and tokenizer: English stopwords should not be applied blindly to Somali text, and an English tokenizer may split a contraction such as “don't” into tokens including `n't`. In many applications, keeping stopwords and comparing validation results is safer than removing them automatically [1, 2, 6].

### 5. Stemming and lemmatization

Both methods reduce variation among related word forms, but they use different principles. **Stemming** applies rules to shorten words toward a common stem, which need not be a valid dictionary word. **Lemmatization** attempts to identify a linguistically valid base form, or **lemma**, often using a vocabulary and grammatical information [1, 4, 5].

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Method | Rules for trimming/changing affixes | Lexical and morphological analysis |
| Example | `flies` → `fli` (Porter, in my practical) | `wolves` → `wolf` (WordNet, in my practical) |
| Output | May not be a word | Usually a valid dictionary form |
| Advantages | Simple; relatively fast; can merge related forms for search | More interpretable; can preserve intended word forms |
| Risks | Overstemming may combine words with different meanings | Requires suitable linguistic resources and often POS information |

A well-known method is **Porter's stemming algorithm**, which applies successive conditional suffix transformations, described by Porter (1980) [4]. A well-known lemmatization approach is the **WordNet-based lemmatizer** provided by NLTK: it searches lexical word forms, and its default part of speech is noun (`n`). With appropriate hints, `better` as an adjective becomes `good`, and `ran` as a verb becomes `run` [5]. Stemming may be preferred in a lightweight keyword-retrieval baseline; lemmatization can be preferable when readable lexical forms matter, assuming resources exist for the language. Neither is universally better, and both should be tested rather than applied by habit.

### 6. Regular expressions for cleaning text

A **regular expression (regex)** specifies a pattern used to find, validate, or replace character sequences. In Python, `re.sub(pattern, replacement, text)` replaces matched spans. Regex is useful for detecting repeated punctuation, contact details, URLs, or other predictable surface patterns [1].

| Pattern | What it matches | Example application |
|---|---|---|
| `\b\d{10}\b` | A sequence of exactly ten digits bounded by word boundaries | Replace a ten-digit phone-like string with `PHONE_NUMBER` |
| `!{2,}` | Two or more consecutive exclamation marks | Change `!!!` to a single `!` |

The phone-number pattern is only an illustrative rule, not a universal phone-number validator: some countries use different lengths and formats. A broad expression such as `[^a-z\s]` replaces every character that is not a lowercase English letter or whitespace. If applied to a spam message, it can delete prices, phone numbers, emojis, punctuation, and letters from other languages, removing potential signals rather than only noise. A safer alternative for some tasks is to replace a meaningful span with a descriptive placeholder [1, 2].

### 7. Research case study: preprocessing for sentiment analysis

The research article **“Evaluating the Effectiveness of Text Pre-Processing in Sentiment Analysis”** by Palomino and Aider (2022), published in *Applied Sciences*, examined whether different preprocessing pipelines changed the quality of sentiment classification on Twitter data [6]. The researchers compared combinations of preprocessing components—including tokenization, spelling correction, lemmatization, and stopword removal—using sentiment-analysis methods including **VADER**, **TextBlob**, and a **Naive Bayes** classifier.

A key feature of the experiment was comparing complete **flows**, not treating each cleaning rule as automatically beneficial. In particular, one flow removed lemmatization, another also omitted spelling correction, and a further flow kept stopwords rather than removing them. The study found that preprocessing order mattered for the Naive Bayes approach; lemmatization did not notably improve sentiment-analysis quality; and VADER performed best among the compared flows when stopwords were retained [6]. These results should not be generalized as universal rankings of preprocessing operations: they depend on the model and dataset. The practical lesson is to choose preprocessing by measuring the downstream task's results, especially when negation and other function words affect sentiment.

## Part B – Practical

**Execution note.** The practical interpretations below are based on the actual successful terminal output of `preprocessing_practice.py` that I ran using NLTK and Python `re`. The separate `output.txt` reproduces that run. The word and text examples are those supplied in the assignment. The source code is submitted separately as `preprocessing_practice.py`.

### Task 1 – Lowercasing T1

Original: `URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!`  
Lowercased: `urgent!!! you have won a £1,000 gift card. call 0712345678 now!`

The operation changed the uppercase letters to lowercase while preserving the words, phone number, price and punctuation. This can make related word forms consistent, but a spam detector may lose capitalization signals such as the all-caps urgency in `URGENT`, `WON`, and `NOW`. Those signals could help distinguish suspicious promotional messages from ordinary writing.

### Task 2 – Whitespace normalization T4

Using `repr()`, the original text started with extra spaces and contained multiple spaces between words. The normalized output was `'The delivery was fast, and the packaging was great.'`. The meaning of the short review was preserved while its spacing became consistent.

### Task 3 – Regex cleaning T1

The regex `[^a-z\s]` replaced all non-lowercase-English-letter, non-whitespace characters with spaces. As the output shows, it lost the pound symbol, the digits of `1,000`, the ten-digit phone number, and the punctuation (`!`, `,`, and `.`). Those characters and numbers may be useful clues in spam classification.

My more targeted rules replaced the phone number with `PHONE_NUMBER`, the price with `MONEY`, and repeated exclamation marks with a single `!`. The result, `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`, preserves the presence of contact and money information without retaining the exact values. Unlike broad deletion, this keeps several signals a spam detector could use.

### Task 4 – Tokenization and vocabulary

For T2, Python `split()` produced **14 tokens**, while NLTK `word_tokenize()` produced **16**. NLTK separated `don't` into `do` and `n't`, and split `price.` into `price` and `.`. T4 produced **11 tokens** and a **vocabulary size of 10**; capitalization means `The` and `the` count as different token types in the reported result.

### Task 5 – Stopwords and negation

With default NLTK stopwords, the T2 tokens included `"n't"` but not `not`: `['n't', 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`. With a modified stopword set preserving `not`, `n't`, `no`, and `never`, the result additionally included `not`: `['n't', 'think', 'jacket', 'worth', 'price', '.', 'not', 'happy', '😞']`. The modified version expresses the customer's negative judgment more clearly through `not happy`; the default result still retains `n't`, so it does **not** lose all negation. This shows why it is important to inspect the tokenizer's actual tokens rather than assuming every negation word was removed.

### Task 6 – Stemming comparison

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

The three algorithms merged `connection`, `connected`, and `connecting` into `connect`, which is useful when grouping related words. However, `university` and `universal` also merged into `univers`, despite referring to different concepts. Stems such as `fli`, `wolv`, and `univers` are not standard dictionary words. Lancaster also produced the very short `org` from `organization`, showing how different algorithms apply different degrees of reduction.

### Task 7 – Lemmatization with POS hints

| Word | Default lemma | POS hint | Lemma with hint |
|---|---|---|---|
| connection | connection | n | connection |
| connected | connected | v | connect |
| connecting | connecting | v | connect |
| organization | organization | n | organization |
| organized | organized | v | organize |
| university | university | n | university |
| universal | universal | a | universal |
| flies | fly | v | fly |
| wolves | wolf | n | wolf |
| better | better | a | good |
| ran | ran | v | run |

The words that changed **only after adding a POS hint** were `connected` → `connect`, `connecting` → `connect`, `organized` → `organize`, `better` → `good`, and `ran` → `run`. In contrast, `flies` and `wolves` were already reduced to `fly` and `wolf` with the default noun setting. These outputs demonstrate that specifying grammatical role can change the result and that the default setting may not interpret a form as the intended verb or adjective.

### Task 8 – Sentence splitting T3

The tokenizer returned **two sentences**: `Mr. Smith arrived in London on Friday.` and `He met the Apple team at 9 a.m.`. It did not incorrectly split after the title `Mr.` or inside the abbreviation `a.m.`. This result is more appropriate than a simple split at every period.

### Task 9 – POS tagging and named entity recognition

The actual NLTK `ne_chunk()` output contained the following recognized entities:

| Detected text | NLTK label | Correct? | Explanation |
|---|---|---|---|
| Mr. | PERSON | Partly | A title, incorrectly separated from the full person's name |
| Smith | PERSON | Partly | Correct person type, but the name `Mr. Smith` was split into two entities |
| London | GPE | Yes | London is a geographic/political location |

**Entities the model missed:** `Apple` was not labeled as an organization even though the context refers to a company. `Friday` was not labeled as a day/date. The model recognized the person category for `Smith`, but failed to combine `Mr.` and `Smith` into the full name `Mr. Smith`. The POS tagger separately produced `NNP` for proper nouns such as `Smith`, `London`, `Friday`, and `Apple`; a POS tag is not the same as a recognized entity.

### Task 10 – Full preprocessing pipeline T2

The pipeline applied lowercase → regex → tokenize → stopword removal → lemmatization twice. The default output after lemmatization was `["n't", 'think', 'jacket', 'worth', 'price', 'happy']`; the negation-preserving output was `["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']`. The latter explicitly retains `not happy`, so it communicates the customer's complaint more clearly, although the default version still contains the negative fragment `n't`. Both pipelines removed the sad emoji `😞`, and both lost some grammatical context, illustrating why a seemingly cleaner result is not always more informative.

### Which preprocessing steps would I use for a spam detector?

For a spam detector, I would normalize unnecessary whitespace and test lowercasing as a baseline, but I would consider keeping a separate feature for unusual capitalization. Based on Task 3, I would replace money amounts and phone numbers with markers such as `MONEY` and `PHONE_NUMBER` rather than deleting them, because their presence can be informative. I would also preserve or summarize repeated exclamation marks, since the original `URGENT!!!` communicates promotional urgency that blanket regex cleaning removes. I would avoid automatically deleting numbers, punctuation, URLs, and negation words, and I would assess stemming and stopword removal against validation results before including them. The T2 output also reminds me that emojis and other non-letter characters may matter to a classifier; the best pipeline is the one that retains useful predictive signals, not the one that removes the most characters.

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., online manuscript released August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/
2. Chai, C. P. (2022). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213
3. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. https://aclanthology.org/D18-2012/
4. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814
5. NLTK Project. (n.d.). *WordNetLemmatizer API documentation*. https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html
6. Palomino, M. A., & Aider, F. (2022). Evaluating the effectiveness of text pre-processing in sentiment analysis. *Applied Sciences, 12*(17), 8765. https://doi.org/10.3390/app12178765

**Practical evidence:** `preprocessing_practice.py` and its captured terminal output in `output.txt`.
