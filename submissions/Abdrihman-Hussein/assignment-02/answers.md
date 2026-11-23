# Assignment 2 – Preprocessing

**Student:** Abdrihman Hussein

## Part A – Research

### 1. Text Preprocessing: Definition, Purpose, and the Danger of Over-Cleaning

**Text preprocessing** is the set of transformations applied to raw text before it is converted into features or passed to a model. It is not one operation but a pipeline: normalisation, cleaning, tokenisation, stopword removal, stemming or lemmatisation, and sometimes sentence splitting or part-of-speech tagging. Bird et al. (2009) describe preprocessing as turning unstructured text into a form that later processing can use, and Jurafsky and Martin (2025) place it at the beginning of almost every NLP system.

NLP systems need preprocessing for four reasons. First, real text is noisy: it contains HTML, punctuation, emoji, repeated spaces, spelling variants, and mixed languages. Second, models need a fixed vocabulary and numerical features, so the text must be divided into units and mapped to numbers. Third, preprocessing reduces sparsity: if `connection`, `connected` and `connecting` are three separate features, each is seen less often during training. Fourth, it reduces cost: fewer tokens and a smaller vocabulary mean less memory and faster training. Uysal and Günal (2014) showed that the *combination* of preprocessing tasks affects classification accuracy and dimension reduction, and that the best combination depends on the domain and the language.

More cleaning is not automatically better, because every cleaning operation deletes or merges information. A real-life example is the SMS scam in **T1**:

> `URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!`

If we lowercase the message and delete every character that is not a lowercase letter or a space, the result is:

> `urgent you have won a  gift card call  now`

The amount `£1,000`, the phone number `0712345678`, and the repeated exclamation marks are gone. Those are exactly the features a spam detector needs: money, a call-to-action phone number, and shouting. The same problem appears in sentiment analysis. Removing the stopword `not` turns `Not happy at all` into `happy at all`, which reverses the customer's meaning. Camacho-Collados and Pilehvar (2018) also found that lowercasing can merge `Apple` the company with `apple` the fruit, and that preprocessing choices produce statistically significant differences in classifier accuracy. Preprocessing is therefore a **task-dependent decision**, not a competition to remove as much as possible.

| Operation | Usually helps when… | Can destroy… |
|---|---|---|
| Lowercasing | Matching `The`/`the` in topic classification or search | Case cues such as `US` vs `us`, `Apple` vs `apple`, or `URGENT` in spam |
| Punctuation removal | Cleaning noisy web text or bag-of-words models | Negation (`don't`), sentence boundaries, `!!!` emphasis, emoji sentiment |
| Number removal | Topic classification where numbers are noise | Prices, phone numbers, dates, times, measurements |
| Stopword removal | Information retrieval and large bag-of-words models | Negation, pronouns needed for coreference, short complaint texts |
| Stemming | IR where recall matters more than readability | Word identity: `university` and `universal` both become `univers` |

### 2. Text Normalisation

**Text normalisation** is the process of mapping different surface forms of the same piece of content to one canonical form. It includes lowercasing, whitespace normalisation, Unicode normalisation, punctuation standardisation, and the expansion of non-standard words such as numbers, dates, abbreviations and currency amounts (Sproat et al., 2001).

**Lowercasing** maps `Python`, `PYTHON` and `python` to one token. **Whitespace normalisation** collapses runs of spaces, tabs and newlines into a single space and removes leading and trailing spaces. **Unicode normalisation** is more technical. The same visible character can be stored as different code-point sequences: `é` can be one precomposed code point (U+00E9) or `e` followed by a combining acute accent (U+0065 U+0301). The Unicode Standard defines four normal forms (Unicode Consortium, 2024):

| Form | What it does | Example |
|---|---|---|
| **NFC** | Canonical composition: combines base letters and accents | `e` + `◌́` → `é` |
| **NFD** | Canonical decomposition: separates accents from base letters | `é` → `e` + `◌́` |
| **NFKC** | Compatibility composition: also folds compatibility characters | `ﬁ` → `fi`, `Ａ` → `A` |
| **NFKD** | Compatibility decomposition | `²` → `2`, `ﬁ` → `f` + `i` |

**A situation where normalisation helps.** A search engine indexes product names typed on different keyboards. One user writes `café` with a precomposed accent, another writes `cafe` with a combining accent, and a third writes `ＣＡＦＥ` in full-width characters. Without NFC/NFKC and case folding, these become different strings and the search misses relevant documents. Normalisation makes them comparable.

**A situation where normalisation removes useful information.** Lowercasing is harmful for named-entity recognition and for word-sense distinctions. `Apple` in `Apple team` is a company, while `apple` is a fruit; `US` is a country, while `us` is a pronoun; `Turkey` is a country, while `turkey` is a bird. In sentiment analysis, ALL CAPS and repeated punctuation (`GREAT!!!`) are intensity cues. In code or password analysis, case is part of the meaning. NFKC can also change mathematical notation or superscripts that matter in scientific text. The lesson is that normalisation should be chosen for the task, not applied blindly.

### 3. Word-level, Subword and Character-level Tokenisation

**Tokenisation** divides a text into the units that the model will treat as symbols. The choice of unit determines the **vocabulary size** — the number of distinct token types the model can represent, usually written `|V|`. Vocabulary size matters because it fixes the size of the embedding matrix (`|V| × d`), controls how often each token is seen during training, and decides how the model handles words it has never seen.

| Level | Unit | Typical vocabulary | Strengths | Weaknesses |
|---|---|---|---|---|
| **Word-level** | Whole words, usually split on whitespace and punctuation | 50,000–1,000,000+ | Intuitive; one token carries a full word meaning | Large vocabulary; out-of-vocabulary (OOV) words; poor with morphology and typos |
| **Subword** | Frequent character sequences learned by BPE, WordPiece or SentencePiece | 8,000–50,000 | Fixed vocabulary; rare words split into reusable pieces; handles morphology and OOV | Sequences get longer; tokens may not be meaningful words |
| **Character-level** | Individual characters or bytes | 30–300 | Tiny vocabulary; no OOV; robust to typos and spelling variation | Very long sequences; weaker word-level meaning; more compute per word |

**Word-level tokenisation** is the simplest. It works well for English, which has relatively clear word boundaries, and when a large training corpus covers most words. Its weakness is the OOV problem: a word that was not in the training vocabulary has no representation. It also treats `connect`, `connected` and `connecting` as unrelated features, which is costly for morphologically rich languages.

**Subword tokenisation** sits between words and characters. Byte-Pair Encoding (BPE) starts from characters and repeatedly merges the most frequent adjacent pair into a new symbol (Sennrich et al., 2016). Frequent words such as `the` remain one token, while rare words such as `unhappiness` are split into pieces such as `un` + `happi` + `ness`. SentencePiece learns subwords directly from raw text and does not require pre-tokenisation by whitespace, which makes it suitable for many languages (Kudo & Richardson, 2018). The vocabulary is fixed in advance, so every input can be represented and there is no true OOV problem.

**Character-level tokenisation** uses each character as a token. The vocabulary is tiny, and any string can be represented, including misspellings and novel words. The cost is sequence length: a 1,000-word document becomes roughly 5,000–6,000 character tokens, which makes training slower and makes it harder for the model to learn word-level meaning.

**How the choice affects vocabulary size.** A word-level model with a 100,000-word vocabulary still fails on a new product name or a dialect spelling. A subword model with a 32,000-piece vocabulary can represent that name as several pieces and still keep the matrix manageable. A character model with a 100-character vocabulary can represent anything but needs much longer sequences. The trade-off is therefore **vocabulary size versus sequence length versus coverage**: smaller units give better coverage and a smaller vocabulary, but longer sequences and weaker word-level semantics.

### 4. Stopwords

**Stopwords** are words that are very frequent but carry little topical information on their own, such as `the`, `is`, `at`, `of`, `and`, `to`. The idea comes from information retrieval, where a **stop list** (or negative dictionary) filters out words that would make poor index terms (Fox, 1990). NLTK's English stopword list contains 198 words, and it includes not only articles and prepositions but also negation words such as `not` and `no`.

**Benefits of removing stopwords.** Removal shrinks the index or feature matrix, speeds up processing, and reduces noise in bag-of-words and topic models. In a large search collection, removing `the` and `of` can cut the index size substantially without hurting retrieval. In a topic classifier, removing function words can make the remaining content words easier to learn.

**Risks of removing stopwords.** The risks are greatest in short texts and in tasks where function words carry meaning:

- **Negation.** Removing `not`, `no` or `never` can reverse the meaning. `not happy` becomes `happy`.
- **Sentiment and emotion.** Words such as `but`, `very`, `too` and `never` carry intensity or contrast.
- **Questions and intent.** `how`, `why`, `when` and `where` distinguish question types in a chatbot or QA system.
- **Coreference.** Pronouns such as `he`, `she`, `it` and `they` are needed to link mentions across sentences.
- **Short messages.** In a tweet or SMS, almost every word is a stopword, so removal can leave too little signal.

**Why the right list depends on the task and the language.** An information-retrieval system can afford an aggressive list; a sentiment analyser cannot remove negation; a machine-translation system usually keeps function words because they are needed to generate grammatical output. Language matters too. NLTK's English list does not contain Somali function words, and applying it to Somali removes nothing useful. Conversely, a Somali list built from frequency counts may remove words that carry grammatical or sentiment meaning. Agglutinative languages attach information to word stems, so a list designed for English may not transfer at all. Code-switched text needs a list that covers every language in the message.

Our own output shows the danger concretely. With NLTK's default list, T2 becomes `["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`. The second `Not` is removed, so `Not happy at all` becomes `happy`, which looks positive. The clitic `n't` survives only because `word_tokenize` splits `don't` into `do` + `n't`, and NLTK's list contains `don't` but not `n't`. With a negation-keeping list, the result is `["n't", 'think', 'jacket', 'worth', 'price', '.', 'Not', 'happy', '😞']`, and the complaint is preserved.

### 5. Stemming versus Lemmatisation

**Stemming** removes affixes from a word using rules, without checking a dictionary. The result is a **stem**, which may not be a real word. **Lemmatisation** uses a dictionary and morphological analysis to return the **lemma**, the base form that would appear in a dictionary. Lemmatisation normally needs a part-of-speech tag, because the lemma depends on the word's role: `running` as a verb lemmatises to `run`, while `running` as a noun may stay `running`.

| Aspect | Stemming | Lemmatisation |
|---|---|---|
| Method | Rule-based suffix stripping | Dictionary and morphological analysis |
| Output | A stem, not necessarily a word | A valid lemma |
| Needs POS? | No | Usually yes |
| Speed | Fast | Slower |
| Example | `studies` → `studi` | `studies` (verb) → `study` |
| Risk | Over-stemming and non-words | Wrong lemma if the POS is wrong |

**A well-known stemming algorithm: Porter.** Porter (1980) described a suffix-stripping algorithm for English information retrieval. It treats a word as a sequence of consonant (`c`) and vowel (`v`) groups and defines the **measure** `m` as the number of `VC` sequences in the stem. A rule such as `(m > 0) FULNESS → FUL` removes `-fulness` only when the remaining stem has a non-zero measure. The algorithm works in five steps: Step 1 handles plurals and past participles, Steps 2–4 handle derivational suffixes, and Step 5 cleans up a final `e` or double consonant. For example, `caresses → caress`, `ponies → poni`, and `relational → relate`. The Porter stemmer is fast and simple, and it became the standard baseline for IR.

**A well-known lemmatisation approach: WordNet.** WordNet is a lexical database that groups words into synonym sets and records morphological relations (Miller, 1995; Fellbaum, 1998). NLTK's `WordNetLemmatizer` uses WordNet's morphology files to map an inflected form to its lemma. Because the mapping depends on the part of speech, the caller should supply `pos="n"`, `"v"`, `"a"` or `"r"`. In our output, `better` stays `better` with the default noun assumption, but becomes `good` with `pos="a"`; `ran` stays `ran` by default but becomes `run` with `pos="v"`. That is the difference between a dictionary lookup and blind suffix stripping.

**When each is preferred.** Stemming is preferred in large-scale information retrieval and search, where speed and recall matter and the stem does not need to be readable. Lemmatisation is preferred when the output must be interpretable or when the task depends on real word identity: sentiment analysis, question answering, named-entity work, and any pipeline that already has POS tags. Stemming can also **over-stem**: in our output, `university` and `universal` both become `univers`, merging two different meanings. Lancaster is even more aggressive: `better` becomes `bet`, which is a different word. For a task where word identity matters, lemmatisation is the safer choice; for a first-pass search index, Porter or Snowball is often enough.

### 6. Regular Expressions in Text Cleaning

A **regular expression** (regex) is a pattern language for matching strings. In text cleaning it is used to find and replace predictable noise: HTML tags, URLs, email addresses, phone numbers, currency amounts, repeated punctuation, emoji, and extra whitespace. A pattern is built from **character classes** (`[a-z]`, `\d`, `\s`), **quantifiers** (`+`, `*`, `{9}`), **anchors** (`\b`, `^`, `$`), and **groups** (`(...)`). Python's `re` module provides `re.sub()` for replacement and `re.findall()` for inspection (Friedl, 2006; Bird et al., 2009).

**Example pattern 1: `[^a-z\s]`.** The square brackets define a character class, `^` negates it, `a-z` means any lowercase letter, and `\s` means whitespace. The whole pattern therefore matches **any character that is not a lowercase letter or a whitespace character**. On the lowercased T1 it matches `!`, `£`, `,`, `.` and every digit. If we replace those matches with an empty string, the result is:

> `urgent you have won a  gift card call  now`

This is useful for stripping punctuation and numbers from a bag-of-words model, but it also deletes the amount and the phone number.

**Example pattern 2: `\b0\d{9}\b`.** `\b` is a word boundary, `0` matches a literal zero, `\d` matches any digit, and `{9}` requires exactly nine more digits. The pattern matches a ten-digit number that starts with `0`, such as the UK-style mobile number `0712345678`. Replacing it with `PHONE_NUMBER` keeps the fact that a phone number was present without keeping the actual digits. A second useful pattern is `£\d[\d,]*`, which matches a currency amount such as `£1,000` and can be replaced with `MONEY`. A third is `!{2,}`, which matches two or more exclamation marks and can be collapsed to a single `!`.

**How an overly broad pattern can damage text.** The pattern `[^a-z\s]` is broad: it deletes *everything* that is not a letter or a space. On T1 it removes the currency amount, the phone number, the sentence-ending period and all exclamation marks. A spam detector that uses this cleaning step loses its strongest evidence. The same problem occurs with patterns such as `<.*>` (which can swallow everything between the first `<` and the last `>` on a line), `\S+` (which matches every non-space token), or `\d+` (which removes times, dates and measurements as well as noise). A safer design is to **replace** important patterns with typed placeholders before applying a general cleaner, or to restrict the cleaner to a known set of characters. The order of operations matters too: if the phone number is deleted before the phone-number rule runs, the rule has nothing left to match.

### 7. Case Study: Preprocessing Choices for Text Categorisation and Sentiment Analysis

**Paper:** Camacho-Collados, J., & Pilehvar, M. T. (2018). *On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis.* Proceedings of the 2018 EMNLP Workshop BlackboxNLP, 40–46.

| Aspect | Description |
|---|---|
| **Task** | Topic categorisation (assigning a topic to a document) and polarity detection (positive/negative sentiment) |
| **Preprocessing options compared** | Vanilla tokenisation only; lowercasing; lemmatisation; multiword grouping (treating expressions such as `United States` as one token) |
| **Data** | BBC, 20News, Reuters and Ohsumed for topic categorisation; RTC, IMDB, PL05, PL04 and the Stanford Sentiment Treebank for polarity detection |
| **Models** | A standard CNN and a CNN with an LSTM layer |
| **Main finding** | Simple tokenisation was generally adequate. Lemmatisation and multiword grouping did not help in general, and the effect of lowercasing was mixed. Preprocessing choices produced an average variability of about ±2.4%, with statistically significant differences in seven of the nine datasets. |
| **When preprocessing mattered most** | When training data were limited, and on the domain-specific Ohsumed medical dataset, where lowercasing and lemmatisation improved accuracy by more than six points because they reduced sparsity. |

The study is useful because it isolates the preprocessing step while holding the model fixed. Its practical message is that there is no universal "best" pipeline. A simple tokenised input was competitive in most settings, but a specialised domain with many rare terms benefited from normalisation that reduced sparsity. The authors also note that lowercasing can increase ambiguity: in their running example, `Apple` the company and `apple` the fruit become the same token. This supports the argument in Question 1: cleaning should be evaluated on the task, not assumed to help.

A second study points in the same direction. Uysal and Günal (2014) evaluated all combinations of common preprocessing tasks on email and news data in both Turkish and English. They found that the best combination depends on the domain and the language, and that enabling every preprocessing step is not the same as choosing the right ones. Together, the two studies show that preprocessing is an empirical design decision.

## Part B – Practical

### Setup

The script is `preprocessing_practice.py`. It uses NLTK and Python's `re` module. The NLTK data used were `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker`, `maxent_ne_chunker_tab` and `words`. The full printed output was saved with:

```text
python preprocessing_practice.py > output.txt
```

On this Windows machine, PowerShell 5.1 writes `>` output as UTF-16, so the committed `output.txt` was saved as UTF-8 after setting `[Console]::OutputEncoding` to UTF-8. This keeps the emoji in T2 intact.

Every result below is copied from `output.txt`; none of it was typed by hand.

### Task 1 – Lowercasing

```text
Before: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
After:  urgent!!! you have won a £1,000 gift card. call 0712345678 now!
```

**Answer.** Lowercasing merges `WON`, `Won` and `won`, which reduces vocabulary size and helps a bag-of-words model. A spam detector, however, loses a useful signal: `URGENT`, `WON` and `NOW` are in capitals because the sender is shouting, and that style is itself evidence of spam. The safest design is to keep the original text for a case-sensitive feature (such as "fraction of uppercase letters") and use the lowercased copy only for word matching.

### Task 2 – Whitespace Normalisation

```text
Before: '   The   delivery   was   fast,   and the  packaging   was   great.'
After:  'The delivery was fast, and the packaging was great.'
```

`repr()` makes the difference visible: the original has runs of three and two spaces and leading spaces, while the normalised version has exactly one space between tokens and no leading or trailing space.

### Task 3 – Regex Cleaning

Deleting with `[^a-z\s]`:

```text
Lowercased T1:       'urgent!!! you have won a £1,000 gift card. call 0712345678 now!'
After [^a-z\s]:     'urgent you have won a  gift card call  now'
Deleted characters:  ['!', '!', '!', '£', '1', ',', '0', '0', '0', '.', '0', '7', '1', '2', '3', '4', '5', '6', '7', '8', '!']
Unique deleted:      ['!', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '£']
```

The pattern deleted three exclamation marks, the pound sign, the comma, the full stop, and every digit. In other words, it deleted the price, the phone number, and the emphasis.

Replacing instead of deleting:

```text
Replaced T1:         URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!
```

**Answer.** `[^a-z\s]` removes everything that is not a lowercase letter or whitespace, so it deletes punctuation, currency symbols and all digits; on T1 that means the `!`, `£`, `,`, `.` and the digits of `£1,000` and `0712345678`. Deleting them is harmful for spam detection because the money amount, the phone number and the shouting are the strongest evidence in the message. Replacing them with `MONEY`, `PHONE_NUMBER` and a single `!` keeps the evidence while still normalising the text.

### Task 4 – Tokenisation

```text
T2 split():         ['I', "don't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price.', 'Not', 'happy', 'at', 'all', '😞']
T2 split() count:   14
T2 word_tokenize(): ['I', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', '.', 'Not', 'happy', 'at', 'all', '😞']
T2 token count:     16
T4 tokens:          ['The', 'delivery', 'was', 'fast', ',', 'and', 'the', 'packaging', 'was', 'great', '.']
T4 token count:     11
T4 vocabulary:      [',', '.', 'The', 'and', 'delivery', 'fast', 'great', 'packaging', 'the', 'was']
T4 vocab size:      10
T4 vocab (lower):   9
```

`split()` gives 14 tokens because it keeps `price.` as one token and `don't` as one token. `word_tokenize` gives 16 because it separates the full stop and splits `don't` into `do` + `n't`. For T4, the token count is 11 and the case-sensitive vocabulary size is 10; if the tokens are lowercased, `The` and `the` merge and the vocabulary size falls to 9. Vocabulary size is therefore sensitive to both the tokeniser and the normalisation applied before it.

### Task 5 – Stopwords and Negation

```text
Default stopwords:       ["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']
Keeping negation words:  ["n't", 'think', 'jacket', 'worth', 'price', '.', 'Not', 'happy', '😞']
```

**Answer.** With NLTK's default list, `I`, `do`, `this`, `is`, `the`, `not`, `at` and `all` are removed. The result still contains `n't` by accident, because `word_tokenize` splits `don't` into `do` + `n't` and NLTK's list contains `don't` but not the bare clitic `n't`. The important loss is the second negation: `Not happy at all` becomes `happy`, which makes the complaint look positive. The custom list removes the same function words but keeps `not`, `n't`, `no` and `never`, so the result contains both `n't` and `Not` and the negative meaning survives.

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

**Words merged by Porter:**

```text
connect <- ['connection', 'connected', 'connecting']
organ   <- ['organization', 'organized']
univers <- ['university', 'universal']
```

**Porter stems that are not real English words:**

```text
university -> univers
universal  -> univers
flies      -> fli
wolves     -> wolv
```

**Answer.** A stemmer merges different words that share a stem: Porter reduces `connection`, `connected` and `connecting` to `connect`, and `organization` and `organized` to `organ`. This is useful for search, where a query for `connection` should also find documents containing `connected`. The cost is over-stemming: `university` and `universal` both become `univers`, even though they mean different things. Several stems are not real words at all (`univers`, `fli`, `wolv`), which is acceptable for an index but bad for a human-readable output. Lancaster is the most aggressive of the three: it reduces `better` to `bet`, which is a different word.

### Task 7 – Lemmatisation

| Word | Default (noun) | With POS hint | Hint |
|---|---|---|---|
| connection | connection | connection | n |
| connected | connected | connect | v |
| connecting | connecting | connect | v |
| organization | organization | organization | n |
| organized | organized | organize | v |
| university | university | university | n |
| universal | universal | universal | a |
| flies | fly | fly | n |
| wolves | wolf | wolf | n |
| better | better | good | a |
| ran | ran | run | v |

**Changed only when the POS hint was given:** `connected`, `connecting`, `organized`, `better`, `ran`.

**Answer.** The default lemmatiser assumes every word is a noun, so it leaves verbs and adjectives unchanged: `connected`, `connecting`, `organized` and `ran` keep their surface form, and `better` stays `better`. When the correct part of speech is supplied, `connected` and `connecting` become `connect`, `organized` becomes `organize`, `ran` becomes `run`, and `better` becomes `good`. The words that changed only with the hint are therefore exactly the non-nouns. This shows why POS information is important for lemmatisation: without it, the lemmatiser is reliable only for nouns.

### Task 8 – Sentence Splitting

```text
Sentences: ['Mr. Smith arrived in London on Friday.', 'He met the Apple team at 9 a.m.']
Count:     2
```

**Answer.** `sent_tokenize` returned two sentences, which is correct. It did not split after `Mr.` because it recognises common abbreviations, and it did not split after `a.m.` for the same reason. A naive splitter that cut at every full stop would produce four fragments and would separate `Mr.` from `Smith`. The example shows why sentence splitting needs an abbreviation-aware model rather than a simple character rule.

### Task 9 – POS Tagging and Named Entity Recognition

```text
POS tags: [('Mr.', 'NNP'), ('Smith', 'NNP'), ('arrived', 'VBD'), ('in', 'IN'), ('London', 'NNP'), ('on', 'IN'), ('Friday', 'NNP'), ('.', '.'), ('He', 'PRP'), ('met', 'VBD'), ('the', 'DT'), ('Apple', 'NNP'), ('team', 'NN'), ('at', 'IN'), ('9', 'CD'), ('a.m', 'NN'), ('.', '.')]
Entities: [('Mr.', 'PERSON'), ('Smith', 'PERSON'), ('London', 'GPE')]
```

**Entity table:**

| Entity | Model label | Correct? |
|---|---|---|
| Mr. | PERSON | Yes, but only part of a name |
| Smith | PERSON | Yes |
| London | GPE | Yes |
| Apple | — (not tagged) | No: should be ORGANIZATION |
| Friday | — (not tagged) | No: should be DATE |
| 9 a.m. | — (not tagged) | No: should be TIME |

**What the model missed.** The model split `Mr. Smith` into two separate `PERSON` chunks instead of one full name. It did not recognise `Apple` as an organisation, even though `Apple team` makes the company sense clear. It left `Friday` as a plain noun because NLTK's chunker has no `DATE` label, and it left `9 a.m.` as a number plus a noun because it has no `TIME` label. The tags themselves are reasonable (`NNP` for proper nouns, `VBD` for past-tense verbs, `CD` for the number), but the entity layer is limited by the label set and by the training data. A production system would use a richer tagger and a domain-specific entity recogniser.

### Task 10 – Full Pipeline on T2

**Pipeline A — NLTK default stopwords:**

```text
Lowercased:         "i don't think this jacket is worth the price. not happy at all 😞"
Regex cleaned:      "i don't think this jacket is worth the price not happy at all"
Tokens:             ['i', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', 'not', 'happy', 'at', 'all']
Stopwords removed:  ["n't", 'think', 'jacket', 'worth', 'price', 'happy']
Lemmatized:         ["n't", 'think', 'jacket', 'worth', 'price', 'happy']
Final text:         n't think jacket worth price happy
```

**Pipeline B — negation-keeping stopwords:**

```text
Lowercased:         "i don't think this jacket is worth the price. not happy at all 😞"
Regex cleaned:      "i don't think this jacket is worth the price not happy at all"
Tokens:             ['i', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', 'not', 'happy', 'at', 'all']
Stopwords removed:  ["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']
Lemmatized:         ["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']
Final text:         n't think jacket worth price not happy
```

**Answer.** Pipeline A removes `not` but keeps `n't` by accident, so it produces `n't think jacket worth price happy`. The word `happy` now appears without the negation that belonged to it, and a classifier could read the message as positive. Pipeline B keeps both negation cues and produces `n't think jacket worth price not happy`, so the complaint survives. The comparison shows that the stopword list, not the tokeniser or the lemmatiser, is the step that decides whether the sentiment is preserved.

### Which Preprocessing Steps for a Spam Detector?

For a spam detector I would use **light normalisation and evidence-preserving cleaning**, not the aggressive pipeline used for topic classification. Lowercasing should be applied only to a copy of the text, while the original casing is kept as a feature, because `URGENT`, `WON` and `NOW` are exactly the kind of shouting that distinguishes spam from ordinary messages. Regex should **replace** phone numbers, money amounts and URLs with typed placeholders such as `PHONE_NUMBER`, `MONEY` and `URL`, rather than deleting them: the T1 output shows that `[^a-z\s]` removes `£1,000`, `0712345678` and `!!!`, which are the strongest spam cues in the message. Tokenisation should keep punctuation and emoji as separate tokens, or at least record their counts, because `!!!` and `😞` carry information. I would avoid removing stopwords wholesale: words such as `you`, `have`, `now` and `free` are common in spam, and the T2 experiment shows that stopword removal can delete negation. Stemming and lemmatisation are also risky here, because they merge different words (`university`/`universal` → `univers`) and can destroy the exact tokens that a spam filter has learned to trust. In short, the spam pipeline should be **lowercase for matching, keep the original for case features, replace sensitive patterns instead of deleting them, keep punctuation and emoji, and use a carefully chosen stopword list only if it is validated on spam data**.

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing text with the Natural Language Toolkit.* O'Reilly Media. https://www.nltk.org/book/
2. Camacho-Collados, J., & Pilehvar, M. T. (2018). On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis. *Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, 40–46. https://aclanthology.org/W18-5406/
3. Fellbaum, C. (Ed.). (1998). *WordNet: An electronic lexical database.* MIT Press.
4. Fox, C. (1990). A stop list for general text. *ACM SIGIR Forum, 24*(1–2), 19–35. https://doi.org/10.1145/378881.378888
5. Friedl, J. E. F. (2006). *Mastering regular expressions* (3rd ed.). O'Reilly Media.
6. Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
7. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *Proceedings of EMNLP 2018: System Demonstrations*, 66–71. https://aclanthology.org/D18-2012/
8. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval.* Cambridge University Press. https://nlp.stanford.edu/IR-book/
9. Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39–41. https://doi.org/10.1145/219717.219748
10. Paice, C. D. (1990). Another stemmer. *ACM SIGIR Forum, 24*(3), 56–61. https://doi.org/10.1145/101306.101310
11. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814
12. Porter, M. F. (2001). *Snowball: A language for stemming algorithms.* http://snowball.tartarus.org/texts/introduction.html
13. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725. https://aclanthology.org/P16-1162/
14. Sproat, R., Black, A. W., Chen, S., Kumar, S., Ostendorf, M., & Richards, C. (2001). Normalization of non-standard words. *Computer Speech & Language, 15*(3), 287–333. https://doi.org/10.1006/csla.2001.0169
15. Unicode Consortium. (2024). *Unicode Standard Annex #15: Unicode normalization forms.* https://www.unicode.org/reports/tr15/
16. Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006
