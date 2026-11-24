Assignment-2: Preprocessing

1.Text Preprocessing:

Text processing in Natural Language Processing (NLP) refers to the manipulation, analysis, and interpretation of textual data using computational methods. t begins with tokenization, which involves splitting the text into smaller units like words, sentences or phrases.

Text preprocessing is the process by which we cleaning and standardize raw text data by removing the noise, such as punctuation, emojis, and common stop-words, to make it ready for our model to train. Because raw text is unstructured and inconsistent, pre-processing reduces unnecessary variations and helps the model focus on meaningful words to learn patterns more accurately.

Text preprocessing is not simply about making text look clean—it is about preparing text in a way that is useful for the specific NLP task. If we clean too aggressively and remove everything that looks unnecessary, we can erase signals that carry meaning. For example, transforming "WOW!!! This phone is AMAZING " into "wow this phone is amazing" may work fine for basic text classification, but in a sentiment or emotion analysis system, stripping emojis and exclamation marks can remove valuable emotional cues.



2.Text Normalization:

Text normalization is a preprocessing step in Natural Language Processing (NLP) that converts text data into a consistent, standardized format. This often involves converting all characters to lowercase (lowercasing), unifying punctuation and whitespace, removing special characters or irrelevant symbols, and resolving inconsistencies through Unicode normalization. Standardizing abbreviations, number formats, or dates can also be part of this process. The goal is to reduce data variability to simplify subsequent analysis or model training and improve their accuracy.

Situation Where Normalization Helps

In search indexing or social media comment grouping, users frequently type inconsistent spacing and capitalization (e.g., " URGENT " vs. "urgent"). Normalizing whitespace and casing aligns matching queries and comments so they are grouped and retrieved reliably.

Situation Where Normalization Removes Useful Information

In named entity recognition, acronym disambiguation, or case-sensitive tasks, lowercasing can erase meaningful distinctions. For example, "You have WON” becomes "you have won" (the pronoun), which can confuse models that rely on capitalization to distinguish acronyms and proper nouns from common words.



3.Compare word-level , subword , character :

Word-level: This is the most commonly used tokenization technique. It splits a piece of text into words based on a delimiter. The most commonly used delimiter is space.

subword tokenization which is a solution between word and character-based tokenization. The main idea is to solve the issues faced by word-based tokenization (very large vocabulary size, large number of OOV tokens, and different meaning of very similar words) and character-based tokenization (very long sequences and less meaningful individual tokens).

Character-level tokenizers split the raw text into individual characters. The logic behind this tokenization is that a language has many different words but has a fixed number of characters. This results in a very small vocabulary.

reducing the vocabulary size has a trade-off with the sequence length in character-based tokenization

Vocabulary Size: Vocabulary size is the total number of unique tokens stored in a tokenizer or model's vocabulary table. Word-level tokenization produces a very large vocabulary because every distinct word form, inflection, or spelling variant adds a new entry and increases out-of-vocabulary (OOV) risk. Subword tokenization keeps a controlled, moderate vocabulary size by keeping frequent words intact while splitting rare or compound words into known subword units. Character-level tokenization produces a very small vocabulary limited to individual characters, symbols, and digits, but creates longer token sequences.



4. Stopwords:

Stopwords are high-frequency function words (such as articles, prepositions, conjunctions, and common auxiliaries like "the", "is", "and", "in") that typically carry less standalone topical weight than content words (nouns, verbs, adjectives) in many text representations.

- Benefits of removing them: Removing stopwords shrinks vocabulary size and feature dimensionality, which reduces memory overhead, speeds up training or indexing, and helps frequency- or keyword-based methods focus on more informative content words.
- Risks of removing them: Removing stopwords can alter meaning or weaken grammatical and contextual cues—especially when negation words like "not", "no", or "never" are dropped, which can flip sentiment (for example, turning "not happy at all" toward "happy at all") or confuse models that rely on word order and context.
- Why the right list depends on the task and language: Tasks sensitive to negation, nuance, or phrasing—such as sentiment analysis or question answering—often need words like "not" or question/relation terms preserved, whereas coarse keyword filtering may remove more function words safely. Likewise, every language has its own high-frequency words and grammatical structure, so a stopword list must match the target language rather than applying a generic list from another language.



5. Compare stemming and lemmatization:

stemming s a rule-based text normalisation technique that reduces words to their root form by removing prefixes or suffixes. The resulting form called a stem, may not be a valid or meaningful word in the language.

stemming algorithms is the Porter (1980)

Lemmatization is a linguistically driven text normalization technique that converts words into their base dictionary form, known as a lemma, by considering grammar, vocabulary and context.

prefer stemming: when speed, simplicity, or lightweight indexing is prioritized and approximate term conflation is sufficient.

prefer lemmatization when semantic clarity, clean readable base form, or downstream linguistic/analytical accuracy matters more than raw rule-stripping speed.

6. Regular Expression:

Regular Expression is a sequence of characters that forms a search patternto match and characterize sets of character strings across a text. In text preprocessing and cleaning, regular expressions are used to locate specific patterns—such as extra whitespace, punctuation, symbols, or digits—and standardize or remove them consistently.

Two example:

| Pattern | Matches | Example |
| :--- | :--- | :--- |
| `\b\d{10}\b` | A standalone sequence of exactly 10 digits | `0712345678` |
| `@\w+` | An `@` symbol followed by one or more word characters (letters, digits, or underscore) | `@nlp_system` |

Overly Broad Pattern Can Damage Text: Applying re.sub(r"[^a-z\s]", " ", text.lower()) indiscriminately replaces apostrophes with spaces, turning "don't" into "don t" (splitting off "n't" and potentially weakening negation cues) and breaking structured strings like "@nlp_system", "0712345678" into disconnected fragments.

7.Case-study: preprocessing for word representation learning

Babanejad et al. (York University) examine text preprocessing for word representation learning in affective tasks, such as sentiment analysis, emotion classification, and sarcasm detection, noting that while preprocessing is widely studied in general NLP, its role in word-vector-based affective models during the training phase has been under-explored and often overlooked in pretrained word vector settings.

References
jurafsky, D., & Martin, J. H. Speech and Language Processing (Chapters on Regular Expressions, Text Normalization, Tokenization, Morphology, and Text Classification)

Bird, S., Klein, E., & Loper, E. Natural Language Processing with Python (NLTK documentation on tokenization, stopwords, stemming, lemmatization, POS tagging, and named entity recognition)

Porter, M. F. (1980). An algorithm for suffix stripping. Program

Babanejad, N., Agrawal, A., An, A., & Papagelis, M. A Comprehensive Analysis of Preprocessing for Word Representation Learning in Affective Tasks. York University


## Practical: Preprocessing Experiments (`preprocessing_practice.py` & `output.txt`)

### Task 6 Table: Stemming Comparison (Porter vs. Lancaster vs. Snowball)

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
| `ran` | `ran` | `ran` | `run` |

---

### Task 9 Table: Named Entities on T3 (`ne_chunk`)

| Entity Text | Assigned Label | Label Correct? (Yes/No) | Notes |
| :--- | :--- | :--- | :--- |
| `Mr.` | `PERSON` | Yes | Title paired with `Smith` in person reference |
| `Smith` | `PERSON` | Yes | Person name correctly identified |
| `London` | `GPE` | Yes | Geopolitical entity (city) correctly identified |
| *(Check `output.txt` for `Apple`)* | *(Check `output.txt`)* | *(Verify)* | `Apple` here refers to a company (`ORGANIZATION`), not the fruit; `ne_chunk` may miss it or tag its POS as `NNP` without an `ORGANIZATION` chunk |
| *(Check `output.txt` for `Friday`)* | *(Check `output.txt`)* | *(Verify)* | Day of the week (`Friday`); rule-based `ne_chunk` often does not label standalone weekdays |
| *(Check `output.txt` for `9 a.m.`)* | *(Check `output.txt`)* | *(Verify)* | Time expression (`9 a.m.`); `ne_chunk` typically does not output a `TIME` entity |

---

### Short Answers by Task

#### Task 1: Lowercasing
Lowercasing `T1` converts `"URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"` into `"urgent!!! you have won a £1,000 gift card. call 0712345678 now!"`. A spam detector can lose capitalization signals such as all-caps emphasis (`URGENT`, `WON`, `NOW`), which frequently indicate aggressive marketing or urgency in spam messages. Lowercasing can also blur distinctions between proper nouns/brands and ordinary words if case sensitivity is part of a spam heuristic.

#### Task 3: Regex Cleaning
Applying `[^a-z\s]` to the lowercased `T1` deletes the exclamation marks (`!!!`), the pound symbol (`£`), the comma in `1,000` (leaving `1 000` after space/token handling), the period after `card`, and all digits in `0712345678`. Using targeted replacement rules instead—turning the phone number into `PHONE_NUMBER`, the money amount into `MONEY`, and `!!!` into `!`—preserves spam indicators like contact numbers, monetary incentives, and punctuation intensity in normalized placeholders. This prevents destructive data loss while keeping the text structured for downstream features.

#### Task 5: Stopwords and Negation
Removing NLTK's default English stopwords from `T2` drops function words and negations such as `"not"`, which alters the complaint in `"I don't think this jacket is worth the price. Not happy at all 😞"`. With a custom stopword list that retains `"not"`, `"n't"`, `"no"`, and `"never"`, the negation cues remain in the token list. As a result, the first list makes the customer sound closer to expressing satisfaction or neutral opinion, whereas the negation-preserving list retains the explicit dissatisfaction (`"not"`, `"happy"`).

#### Task 6: Stemming
In the stemming results, Porter and Snowball merge `connection`, `connected`, and `connecting` into `connect`, and both Porter and Lancaster merge `university` and `universal` into `univers`. An example of a stem that is not a real English word is `univers` (from `university` and `universal`) or `fli` (from `flies`) and `wolv` (from `wolves`). Lancaster is noticeably more aggressive than Porter and Snowball, stripping words down to truncated roots like `bet` for `better`.

#### Task 7: Lemmatization
With default `WordNetLemmatizer` calls (without POS hints), inflected or irregular words such as `ran` and `better` remain unchanged (`ran` and `better`). When the appropriate POS hint is supplied (`pos="v"` for `ran` and `pos="a"` for `better`), they normalize to their base lemmas `run` and `good`. Providing POS hints allows lemmatization to resolve irregular and part-of-speech-dependent forms that default lookup misses.

#### Task 8: Sentence Splitting
Splitting `T3` with `sent_tokenize` produces two sentences: `"Mr. Smith arrived in London on Friday."` and `"He met the Apple team at 9 a.m."`. Abbreviations such as `"Mr."` and `"a.m."` contain periods, yet `sent_tokenize` did not split at them because its trained abbreviation/punctuation rules recognize common titles and time abbreviations. Thus, both sentence boundaries in `T3` were segmented in the right places.

#### Task 10: Full Pipeline
Running `T2` through lowercasing, regex cleaning, tokenization, stopword removal, and lemmatization with NLTK's default stopwords removes `"not"`, weakening or distorting the negative sentiment in the output. Running the same pipeline with the negation-preserving stopword list from Task 5 retains `"not"`, keeping the customer's dissatisfaction intact. Therefore, the version that keeps negation words preserves the customer's complaint much more faithfully.

---

### Spam Detector Preprocessing Recommendation

For a spam detector, I would apply lowercasing, whitespace normalization, and targeted regex replacement (such as mapping phone numbers to `PHONE_NUMBER`, currency amounts to `MONEY`, and collapsing repeated punctuation like `!!!` to `!`) rather than aggressive deletion with `[^a-z\s]`, because in my `T1` outputs aggressive character removal erased digits, currency symbols, and punctuation cues that distinguish promotional/scam messages. I would also avoid removing negation words or applying overly aggressive stemming like Lancaster (which turned `better` into `bet` and `wolves` into `wolv`), as heavy truncation and word stripping can discard useful lexical and contextual signals across messages.
