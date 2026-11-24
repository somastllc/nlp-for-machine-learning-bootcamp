## Part A – Research

### 1. Text Preprocessing: Definition, Necessity, and Trade-Offs

**Text preprocessing** is the systematic transformation of unstructured, raw natural language into a standardized, machine-readable representation suitable for downstream statistical or neural models (Jurafsky & Martin, 2024; Manning et al., 2008). Raw human writing contains typographical inconsistencies, formatting artifacts, inflectional variants, and non-linguistic symbols that inflate feature dimensionality and cause models to treat semantically identical expressions as unrelated inputs. Preprocessing reduces this surface sparsity so learning algorithms can generalize from finite training data.

However, **more aggressive cleaning can degrade model performance** whenever surface variations encode task-relevant signals. Every cleaning step acts as a lossy filter: if it discards features correlated with the target label, predictive accuracy drops. For example, in **financial sentiment analysis and social media stock prediction** (such as classifying tweets about earnings reports), stripping punctuation and casing removes vital cues: `"$AAPL to the moon!!!"` carries strong bullish emphasis through all-caps ticker formatting and repeated exclamation marks, whereas `"aapl to the moon"` flattens that intensity, and removing `$` conflates the ticker symbol `$AAPL` with ordinary words or strips currency indicators from price targets (Symeonidis et al., 2018).

---

### 2. Text Normalization

**Text normalization** maps surface-level orthographic variations onto a single canonical form so that equivalent strings share the same representation (Jurafsky & Martin, 2024). Three foundational normalization operations include:

* **Case folding (lowercasing):** Converts uppercase characters to lowercase (`"Apple"` $\rightarrow$ `"apple"`), merging sentence-initial words with mid-sentence occurrences.
* **Whitespace normalization:** Collapses consecutive spaces, tabs, and line breaks into a single space and trims leading/trailing padding.
* **Unicode normalization:** Resolves equivalent character encodings—for instance, converting decomposed sequences (`e` + combining acute accent `U+0301`) into precomposed characters (`é`, `U+00E9`) under Normalization Form C (NFC), or standardizing typographic curly quotes (`’`) to ASCII apostrophes (`'`) (Unicode Consortium, 2023).

**When normalization helps:** In **document retrieval and keyword search engines**, normalizing Unicode encodings (NFC) and folding case ensures that a user querying `"resume"` or `"résumé"` (typed with either a precomposed or decomposed `é`) matches all relevant CVs in an index without missing documents due to keyboard or encoding differences (Manning et al., 2008).

**When normalization removes useful information:** In **Named Entity Recognition (NER) and Information Extraction**, lowercasing destroys orthographic capitalization cues that distinguish proper nouns from common nouns. Without capitalization, `"Apple"` (the technology corporation) becomes indistinguishable from `"apple"` (the fruit), and `"WHO"` (World Health Organization) collapses into the interrogative pronoun `"who"`, substantially increasing entity boundary and classification errors (Lample et al., 2016).

---

### 3. Tokenization Paradigms and Vocabulary Size

**Tokenization** segments a continuous character stream into discrete processing units called **tokens**. **Vocabulary size** ($|V|$) refers to the total number of *unique* token types stored in a model's lookup table (as opposed to token count, which measures total token occurrences in a corpus).

| Dimension | Word-Level Tokenization | Subword Tokenization (e.g., BPE, WordPiece) | Character-Level Tokenization |
| :--- | :--- | :--- | :--- |
| **Segmentation Unit** | Whitespace- and punctuation-delimited words | Frequent whole words + recurring morphemes/character n-grams | Individual Unicode characters or bytes |
| **Typical Vocabulary Size ($|V|$)** | Very large ($100\text{k}\text{–}1\text{M}+$ types) | Moderate and fixed ($30\text{k}\text{–}64\text{k}$ types) | Very small ($100\text{–}500$ types for alphabetic scripts) |
| **Out-of-Vocabulary (OOV) Handling** | Poor: unseen words map to `<UNK>`, losing meaning | Strong: rare words decompose into known subword pieces | Complete: virtually zero OOV tokens |
| **Sequence Length** | Shortest (1 token per word) | Intermediate (~1.2–1.5 tokens per word) | Longest (5–7+ tokens per word) |
| **Primary Use Cases** | Classical bag-of-words, TF-IDF, Naive Bayes | Modern Transformers (BERT, GPT, RoBERTa) | Morphologically noisy text, OCR, spelling correction |

**How tokenizer choice governs vocabulary size:** Word-level tokenizers assign a distinct entry to every inflected form, compound, and typo, causing $|V|$ to grow unboundedly with corpus size (Heaps' Law) and forcing practitioners to prune rare words into an `<UNK>` bucket (Manning et al., 2008). Character-level tokenizers cap $|V|$ at the alphabet size, yielding tiny embedding tables at the cost of long sequences where models must learn spelling from scratch. Subword algorithms such as **Byte-Pair Encoding (BPE)** (Sennrich et al., 2016) solve this trade-off by iteratively merging the most frequent adjacent symbol pairs until reaching a predetermined budget (e.g., $32,000$ or $50,000$ tokens), keeping common words intact while representing rare words compositionally.

---

### 4. Stopwords: Benefits, Risks, and Contextual Design

**Stopwords** are high-frequency function words—such as determiners (`the`, `a`), prepositions (`in`, `on`), and auxiliary verbs (`is`, `was`)—that carry grammatical structure rather than topical content (Manning et al., 2008).

* **Benefits of removal:** Eliminating stopwords shrinks feature matrices, reduces memory footprint and training latency in bag-of-words classifiers, and prevents frequent uninformative words from dominating distance metrics in document clustering or search indexing.
* **Risks of removal:** Standard off-the-shelf stopword lists routinely include **negation markers** (`not`, `no`, `never`, `n't`), **modals** (`should`, `must`), and **intensifiers** (`very`, `too`). Removing them reverses polarity (`"not satisfied"` $\rightarrow$ `"satisfied"`), degrades question-answering systems (where `"to be or not to be"` loses every token), and breaks multi-word expressions in n-gram models (Saif et al., 2014).

**Why stopword lists depend on task and language:**
1. **Task dependency:** A topic-modeling pipeline (e.g., Latent Dirichlet Allocation) benefits from stripping function words and even domain-generic nouns (like `"patient"` in clinical notes), whereas a sentiment classifier or legal contract analyzer must preserve negations, modals, and prepositions (`"liable for"` vs. `"liable to"`).
2. **Language dependency:** Languages differ dramatically in morphological typology. Analytic languages like English express grammatical relations via standalone function words, whereas agglutinative or polysynthetic languages (such as Turkish, Finnish, or Somali) encode tense, possession, and negation through bound affixes attached to root stems, requiring language-specific stopword inventories and morphological analysis (Schofield et al., 2017).

---

### 5. Stemming vs. Lemmatization

Both **stemming** and **lemmatization** reduce inflectional variants to a shared base form to consolidate vocabulary, but they operate through fundamentally different mechanisms (Manning et al., 2008):

* **Stemming (Algorithmic Suffix Stripping):** A well-known example is the **Porter Stemmer** (Porter, 1980), which applies five sequential phases of deterministic heuristic rules (e.g., replacing `-ational` with `-ate`, or stripping `-ing` and `-ed` when the remaining stem contains a vowel). Because it relies purely on character patterns without a lexicon, stemming is fast but prone to **over-stemming** (merging unrelated words like `"universal"` and `"university"` into `"univers"`) and producing non-dictionary strings (`"flies"` $\rightarrow$ `"fli"`).
* **Lemmatization (Lexical and Morphological Analysis):** A standard approach is **WordNet-based morphological lemmatization** (such as NLTK's `WordNetLemmatizer`, built on the WordNet lexical database; Miller, 1995), which combines part-of-speech (POS) tags with inflectional rules and irregular-verb/noun exception dictionaries to return valid dictionary headwords (**lemmas**), mapping `"better"` (adjective) to `"good"` and `"ran"` (verb) to `"run"`.

| Criterion | Stemming (e.g., Porter) | Lemmatization (e.g., WordNet + POS) |
| :--- | :--- | :--- |
| **Mechanism** | Heuristic suffix stripping | Vocabulary lookup + morphological rules + POS context |
| **Output Validity** | Often truncated non-words (`"studi"`) | Valid dictionary lemmas (`"study"`) |
| **Computational Cost** | Very fast; minimal memory | Slower; requires lexicon and POS tagging |
| **Irregular Forms** | Fails (`"ran"` $\neq$ `"run"`, `"better"` $\neq$ `"good"`) | Succeeds when POS tag is supplied |

**When each is preferred:** **Stemming** is preferred in high-throughput information retrieval and search indexing where speed matters and human readability of internal index keys is unnecessary. **Lemmatization** is preferred in downstream NLP tasks requiring semantic precision, human-interpretable features (such as topic modeling or keyword extraction), or accurate handling of irregular verbs and comparative adjectives.

---

### 6. Regular Expressions in Text Cleaning

**Regular expressions (regex)** provide a formal syntax for matching, extracting, and substituting character patterns in raw text (Jurafsky & Martin, 2024).

1. **Pattern 1 — URL normalization:** `r"https?://[^\s<>\"']+"`
   * **What it matches:** Matches `http://` or `https://` followed by one or more non-whitespace characters excluding angle brackets or quotes, allowing a pipeline to replace arbitrary web links with a standardized placeholder (`<URL>`).
2. **Pattern 2 — Repeated character elongation normalization:** `r"(.)\1{2,}"` (replaced with `r"\1\1"`)
   * **What it matches:** Uses a capturing group `(.)` and backreference `\1{2,}` to match any character repeated three or more times consecutively (e.g., `"sooooo"` or `"amazzzzing"`), compressing noisy social-media elongations into `"soo"` or `"amazzing"` while preserving emphasis.

**How an overly broad pattern damages text:** Applying a naive ASCII-only filter such as `re.sub(r"[^a-zA-Z\s]", "", text)` deletes digits, currency symbols, punctuation, apostrophes, and all non-Latin scripts. In a customer-support or e-commerce dataset, this pattern turns `"Order #4092 cost $99.50 — didn't arrive!"` into `"Order  cost   didnt arrive"`, destroying order IDs, monetary figures, negation boundaries (`didn't`), and any multilingual user text written in Arabic, Somali diacritics, or accented Latin characters.

---

### 7. Case Study: Impact of Text Preprocessing on Sentiment Classification

* **Study:** Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). *A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis.* Expert Systems with Applications, 110, 298–310.
* **Task and Datasets:** The authors evaluated binary and ternary sentiment classification across two Twitter benchmark datasets (SS-Twitter and SemEval), testing four machine learning classifiers (Linear SVC, Bernoulli Naive Bayes, Logistic Regression, and Convolutional Neural Networks).
* **Preprocessing Options Compared:** The study systematically tested 16 preprocessing techniques—including lowercasing, removing punctuation, removing numbers, removing URLs/mentions, replacing negations (`not good` $\rightarrow$ `not_good`), lemmatization, Porter stemming, Snowball stemming, Lancaster stemming, and slang/spelling correction—both individually and in multi-step combinations.
* **Key Findings and Impact on Results:**
  1. **Destructive vs. beneficial steps:** Removing punctuation and numbers frequently *decreased* accuracy because emoticons (e.g., `:)`, `:(`) and exclamation marks carried strong sentiment signals on Twitter. Conversely, **replacing negations** and **expanding acronyms/slang** consistently improved F1-scores across classifiers.
  2. **Stemming vs. lemmatization:** Lemmatization outperformed aggressive stemmers (especially Lancaster stemming, which over-stemmed sentiment-bearing adjectives), while combining negation handling with lemmatization yielded the highest overall gains.
  3. **Classifier interaction:** Linear models (Linear SVC and Logistic Regression) were far more sensitive to vocabulary-reducing steps than neural models, confirming that preprocessing pipelines must be empirically validated against the specific model architecture and dataset.

---

### References

* Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
* Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. *Proceedings of NAACL-HLT 2016*, 260–270.
* Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
* Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39–41.
* Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
* Saif, H., Fernández, M., He, Y., & Alani, H. (2014). On stopwords, filtering and data sparsity for sentiment analysis of Twitter. *Proceedings of LREC 2014*, 810–817.
* Schofield, A., Magnusson, M., & Mimno, D. (2017). Pulling out the stops: Rethinking stopword removal for topic models. *Proceedings of EACL 2017*, 432–436.
* Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*, 1715–1725.
* Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications*, 110, 298–310.
* Unicode Consortium. (2023). *Unicode Standard Annex #15: Unicode Normalization Forms*. https://unicode.org/reports/tr15/

---

## Part B – Practical Report

### Task 6: Stemmer Comparison Table

| Word | Porter Stemmer | Lancaster Stemmer | Snowball Stemmer |
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

### Task 9: Named Entity Recognition (NER) Evaluation Table (`T3`)

*Input sentence (`T3`):* `"Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."`

| Entity Span | NLTK `ne_chunk` Label | Correct? | Notes |
| :--- | :--- | :--- | :--- |
| `Mr.` | `PERSON` | Partial / No | Honorific title tagged separately from `Smith` due to sentence boundary split at `Mr.` |
| `Smith` | `PERSON` | Yes | Correctly identified as a person's name |
| `London` | `GPE` | Yes | Correctly identified as a geo-political entity (city) |
| `Apple` | `PERSON` (or `GPE` depending on chunker version) | No | Misclassified; in this context `"the Apple team"` refers to an `ORGANIZATION` |

**What the model missed:** NLTK's default `ne_chunk` model does not recognize temporal or calendar expressions, so **`Friday`** (day of the week) and **`9 a.m.`** (time) were not labeled as entities at all. Additionally, it split **`Mr. Smith`** across two separate sentence trees instead of grouping the title and surname together, and it mislabeled **`Apple`** as a `PERSON` rather than an `ORGANIZATION`.

---

### Short Answers to Practical Questions

* **Task 1 (Lowercasing `T1`):** Lowercasing converts `"URGENT!!!"`, `"WON"`, and `"NOW!"` into `"urgent!!!"`, `"won"`, and `"now!"`. A spam detector loses capitalization-intensity features—specifically the ratio or presence of ALL-CAPS words (`URGENT`, `WON`, `NOW`)—which are strong indicators of urgency and promotional manipulation in unsolicited messages.
* **Task 3 (Regex Cleaning on `T1`):** Applying `r"[^a-z\s]"` to lowercased `T1` deleted every non-lowercase-letter and non-whitespace character: `'!'` (4 times), `'£'`, `'1'`, `','`, `'0'` (4 times), `'.'`, `'7'`, `'2'`, `'3'`, `'4'`, `'5'`, `'6'`, `'8'`, completely erasing the prize value (`£1,000`) and callback number (`0712345678`). Replacing them with `MONEY`, `PHONE_NUMBER`, and collapsing `!!!` to `!` removes unique digits that cause vocabulary sparsity while preserving the crucial structural signals that money and a phone number were offered.
* **Task 5 (Stopwords and Negation on `T2`):** With NLTK's default stopword list, `"don't"` (`"do"` + `"n't"`) and `"Not"` are removed, leaving `['think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`, which falsely makes the review look positive (`"think jacket worth price happy"`). Keeping negation words (`not`, `n't`, `no`, `never`) preserves `"n't"` and `"Not"`, retaining the customer's dissatisfaction (`"n't think jacket worth price ... Not happy"`).
* **Task 6 (Stemming Observations):** All three stemmers merge the distinct words **`university`** (an educational institution) and **`universal`** (meaning worldwide/all-encompassing) into the identical stem **`univers`**, and merge **`organization`** and **`organized`** into **`organ`** (Porter/Snowball) or **`org`** (Lancaster). Several resulting stems are not real English dictionary words, such as **`univers`**, **`fli`** (from `flies`), and **`wolv`** (from `wolves`).
* **Task 7 (Lemmatization with POS Hints):** Without a POS hint, `WordNetLemmatizer` assumes every word is a noun, so only plural nouns (`flies` $\rightarrow$ `fly`, `wolves` $\rightarrow$ `wolf`) changed by default. The words **`connected`** (`connect`), **`connecting`** (`connect`), **`organized`** (`organize`), **`better`** (`good`), and **`ran`** (`run`) changed only when given their accurate verb (`pos="v"`) or adjective (`pos="a"`) hints.
* **Task 8 (Sentence Splitting on `T3`):** NLTK's `sent_tokenize` produced **3 sentences** instead of 2: `['Mr.', 'Smith arrived in London on Friday.', 'He met the Apple team at 9 a.m.']`. While `a.m.` at the end of the text did not cause an erroneous split, `Mr.` *did* break the first sentence in the wrong place, isolating `'Mr.'` as its own sentence.
* **Task 10 (Full Pipeline Comparison on `T2`):** Running the full pipeline with NLTK's default stopwords outputs `['think', 'jacket', 'worth', 'price', 'happy']`, whereas the custom negation-preserving list outputs `['think', 'jacket', 'worth', 'price', 'not', 'happy']`. Only the **negation-keeping version** preserves the customer's complaint; the default pipeline deletes both the apostrophe in `"don't"` (turning it into `"don"` and `"t"`, both removed as stopwords) and the standalone word `"not"`, flipping the meaning to positive.

---

### Preprocessing Strategy for a Spam Detector

For a spam detection system, I would use **targeted regex placeholder replacement**, **whitespace normalization**, **word/subword tokenization**, and **selective case features** rather than aggressive text stripping. As shown in **Task 1** and **Task 3**, lowercasing `T1` erases ALL-CAPS urgency signals (`URGENT`, `WON`, `NOW`), and applying the broad regex `[^a-z\s]` completely wipes out `£1,000` and `0712345678`—two of the strongest indicators of financial bait and callback scams. Instead, using targeted regex rules (`£1,000` $\rightarrow$ `MONEY`, `0712345678` $\rightarrow$ `PHONE_NUMBER`, and `!!!` $\rightarrow$ `!`) prevents the model from overfitting to specific phone digits or dollar amounts while preserving the high-value signal that money, a phone number, and emphatic punctuation appeared. Furthermore, I would **avoid default stopword removal and aggressive stemming**: **Task 5** and **Task 10** prove that default stopword lists strip negations (`not`, `n't`) and flip sentence polarity, while **Task 6** shows stemmers conflating distinct concepts (`university` and `universal` $\rightarrow$ `univers`), which can obscure legitimate transactional messages (such as university or bank notices) from actual spam.
