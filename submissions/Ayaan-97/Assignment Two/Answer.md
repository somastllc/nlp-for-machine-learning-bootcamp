## Research

### 1. Text Preprocessing

**Text preprocessing** is the process of preparing raw text so that an NLP system can process it more reliably. Raw text may contain inconsistent capitalization, extra spaces, HTML tags, punctuation, emojis, typos, and different forms of the same word.

Common preprocessing steps include:

| Step | Meaning | Example |
|---|---|---|
| Cleaning | Remove unwanted markup or noise | `<p>Hello</p>` → `Hello` |
| Normalization | Standardize text forms | `HELLO  World` → `hello world` |
| Tokenization | Split text into units | `I like tea` → `I`, `like`, `tea` |
| Punctuation/number handling | Keep, remove, or replace them | `Price: 500` → `Price: NUMBER` |
| Stopword filtering | Remove selected common words | `the`, `of`, `and` |
| Stemming/lemmatization | Reduce word forms | `running` → `run` |

NLP systems need preprocessing because computers treat different strings as different features. For example, `Colour`, `colour`, `COLOUR`, and `colour ` may be treated as separate tokens. This can increase vocabulary size and data sparsity, where the same concept is represented by many infrequent forms. Preprocessing can reduce variation, remove noise, and improve efficiency.

However, more cleaning does not always improve results. Cleaning can delete information that is important for a task. For example, the medical statement “No evidence of pneumonia” could become “evidence pneumonia” if the word `no` is removed as a stopword. This changes the meaning and may cause a medical classifier to produce a false positive.

**Conclusion:** preprocessing should be selected according to the task and dataset. Useful information should not be removed blindly.

### 2. Text Normalization

**Text normalization** converts different written forms into a consistent representation.

| Normalization step | What it does | Example |
|---|---|---|
| Lowercasing | Converts letters to lowercase | `Hello`, `HELLO` → `hello` |
| Whitespace normalization | Collapses repeated spaces and trims ends | `too   many spaces` → `too many spaces` |
| Unicode normalization | Standardizes equivalent character encodings | `é` represented as one character or as `e` + accent |
| Standardization | Replaces variants with a common form | `don't` → `do not`; URL → `URL` |

**Situation where normalization helps:** Unicode normalization can improve search. The character `é` may be stored as one code point or as `e` plus a combining accent. Although they look the same, they may be different strings. Converting them to the same Unicode form can help a search engine match them.

**Situation where normalization removes useful information:** lowercasing can hurt Named Entity Recognition (NER). In “Bill Gates founded Microsoft,” capitalization helps identify `Bill Gates` as a person and `Microsoft` as an organization. Lowercasing the sentence removes this signal. Similarly, `US` (the country) becomes `us` (a pronoun).

Normalization reduces variation, but it may remove information. The decision depends on the NLP task.

### 3. Tokenization and Vocabulary Size

**Tokenization** is the process of dividing text into smaller units called tokens. Tokens may be words, subwords, or characters.

**Vocabulary size** is the number of unique tokens in a dataset. For example, `the cat saw the dog` contains five tokens but four unique vocabulary items: `the`, `cat`, `saw`, and `dog`.

| Tokenization level | Example: `unhappiness` | Advantages | Disadvantages |
|---|---|---|---|
| Word-level | `unhappiness` | Readable; short sequences | Rare or misspelled words may become unknown |
| Subword-level | Example: `un` + `happi` + `ness` | Handles rare words and word variations | Splits may be less readable; tokenizer must be trained or selected |
| Character-level | `u n h a p p i n e s s` | Small vocabulary; handles unusual spelling | Long sequences and weaker word-level meaning |

Word-level tokenization generally creates a larger vocabulary. Character-level tokenization has a small vocabulary but creates longer sequences. Subword tokenization balances vocabulary size and coverage.

For example, if `unfriendable` was not seen during training:
- Word-level tokenization may represent it as an unknown token (`<UNK>`).
- Subword tokenization may build it from known pieces such as `un`, `friend`, and `able`.
- Character-level tokenization can represent it as individual characters.

Byte-Pair Encoding (BPE) starts with characters and repeatedly merges frequent neighboring pairs until a target vocabulary size is reached. Subword methods are useful for handling rare words and reducing unknown-token problems.

### 4. Stopwords

**Stopwords** are frequent words that may carry little topic-specific information in some tasks, such as `the`, `of`, `and`, `to`, and `is`.

#### Benefits of removing stopwords
- Reduces the number of features.
- Can reduce noise from very frequent words.
- Can make simple search and bag-of-words models more efficient.

#### Risks of removing stopwords
- Important meaning can be lost, especially negation such as `not`, `no`, and `never`.
- Grammar and context can be damaged.
- Useful words may be removed in sentiment analysis, question answering, or authorship analysis.

For example, `The film was not good` may become `film good` if `not` is removed. This reverses the sentiment.

| Task | Stopword handling |
|---|---|
| Search engine | Often removes common words such as `the`, `a`, and `of` |
| Sentiment analysis | Keeps negation words such as `not`, `no`, and `never` |
| Question answering | Keeps words such as `who`, `when`, `where`, and `how` |
| Authorship detection | Often keeps function words because they can reveal writing style |

Stopword lists should also be language-specific. An English list should not be applied directly to Somali, Arabic, or Swahili. Default software lists are starting points and should be inspected and adapted.

### 5. Stemming and Lemmatization

**Stemming** removes word endings using rules. It is fast, but the resulting stem may not be a real word.

**Lemmatization** returns a dictionary form, called a lemma, using vocabulary and grammatical information such as part of speech (POS).

#### Stemming examples

The following table compares common English stemmers. Exact outputs depend on the implementation and its rules.

| Original word | Porter | Lancaster | Snowball | Regexp |
|---|---|---|---|---|
| playing | play | play | play | play |
| jumped | jump | jump | jump | jump |
| kindness | kind | kind | kind | kindnes |
| beautifully | beauti | beauty | beauti | beautifully |
| nationalization | nation | nat | nation | nationalization |
| hoped | hope | hop | hope | hop |

**Regexp stemmer pattern used:** `ing|ed|s$` with a minimum stem length of 4. This is a simple custom suffix-removal rule, not a linguistic stemmer. It may produce incomplete or incorrect stems, such as `kindnes`, which demonstrates the limitations of regex-based stemming.

The Porter stemmer uses ordered suffix-stripping rules. It is fast and useful for vocabulary reduction, but it may merge words with different meanings or produce non-words. Lancaster is generally more aggressive. Snowball is a framework for stemming algorithms for multiple languages.

#### Lemmatization examples

WordNet-based lemmatization uses a lexical database and may require a POS hint.

| Word | Default (noun) | With POS hint |
|---|---|---|
| leaves | leaf | leave (`pos="v"`) |
| went | went | go (`pos="v"`) |
| mice | mouse | mouse (`pos="n"`) |
| better | better | good (`pos="a"`) |

In NLTK's WordNet lemmatizer, the default POS is noun. Therefore, supplying the correct POS can change the result. For example, `went` remains `went` by default but becomes `go` when the verb POS is supplied.

| | Stemming | Lemmatization |
|---|---|---|
| Method | Rules for removing suffixes | Lexicon and morphological analysis, often with POS |
| Output | May be a non-word | Usually a dictionary word |
| Speed | Very fast | Generally slower |
| Best use | Search and fast baseline models | Tasks where meaningful word forms matter |

Stemming may be preferred for large-scale search when speed and vocabulary reduction are important. Lemmatization is useful when valid word forms and meaning matter.

### 6. Regular Expressions in Text Cleaning

**Regular expressions (regex)** are patterns used to search for, extract, replace, or remove text. They can identify URLs, dates, phone numbers, money amounts, and repeated punctuation.

| Pattern | What it matches | Example |
|---|---|---|
| `https?://\S+` | HTTP/HTTPS URL followed by non-whitespace characters | `https://example.com/page` |
| `\b\d{4}-\d{2}-\d{2}\b` | Date-shaped text with four digits, two digits, and two digits | `2025-03-14` |
| `£[\d,]+` | Pound sign followed by digits and commas | `£1,000` |
| `\b0\d{9}\b` | Ten digits total, beginning with zero | `0712345678` |

An overly broad regex such as `[^a-z\s]` removes everything that is not a lowercase letter or whitespace. After lowercasing, applying it to:

`I can't use C++ on the café's COVID-19 server`

produces:

`i cant use c on the cafs covid server`

This damages the text:
- `can't` becomes `cant`.
- `C++` becomes `c`, losing the programming language.
- `café's` becomes `cafs`.
- `COVID-19` loses the number.

Regex cleaning should target specific unwanted patterns. Replacing a phone number with `PHONE_NUMBER` or an amount with `MONEY` can preserve useful information while making text cleaner.

### 7. Case Study

#### Main study: Uysal and Gunal (2014)

**Paper:** Uysal, A. K., & Gunal, S. (2014). *The impact of preprocessing on text classification*. Information Processing & Management, 50(1), 104–112.

| Item | Summary |
|---|---|
| Task | Text classification: assigning documents to categories |
| Data | Email and news domains, in Turkish and English |
| Options compared | Tokenization, stopword removal, lowercasing, and stemming in different combinations |
| Effect on results | The best combination depended on the domain and language. Preprocessing choices affected classification performance. |
| Conclusion | Preprocessing should be tested and selected for the particular task and dataset rather than applied automatically. |

Exact accuracy values should be checked directly in the paper before they are quoted.

#### Supporting study: Haddi et al. (2013)

**Paper:** Haddi, E., Liu, X., & Shi, Y. (2013). *The role of text pre-processing in sentiment analysis*. Procedia Computer Science, 17, 26–32.

The study examined preprocessing approaches for sentiment analysis, including cleaning, stopword handling, negation, and stemming. It supports the idea that suitable preprocessing can affect sentiment classification performance.

#### Survey: Chai (2023)

Chai, C. P. (2023). *Comparison of text preprocessing methods*. Natural Language Engineering, 29(3), 509–553. This paper is a survey of preprocessing methods rather than one single experiment. It provides context for how different methods are used and why preprocessing decisions depend on the corpus and task.

**Take-away:** preprocessing choices can change NLP results. The best approach depends on the task, language, and dataset.

---

## Part B — Practical Report

### Task 1 — Lowercasing

Lowercasing T1 changes:

`URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!`

into lowercase text. A spam detector may lose capitalization emphasis from `URGENT`, `WON`, and `NOW`. These words can be useful signals because spam messages often use capital letters to attract attention. Therefore, lowercasing should be evaluated rather than automatically applied.

### Task 3 — Regex Cleaning

The pattern `[^a-z\s]` deletes every character that is not a lowercase letter or whitespace. After lowercasing T1, it removes exclamation marks, the pound sign, commas, periods, and digits.

My replacement rules are:
- `£[\d,]+` → `MONEY`
- `\b0\d{9}\b` → `PHONE_NUMBER`
- `!{2,}` → `!`

These rules are better than deleting all non-letter characters because they preserve the fact that the message contains money, a phone number, and repeated exclamation marks.

### Task 5 — Stopwords and Negation

NLTK's default stopword list removes `not`, so T2 loses an important negative cue. The remaining words can suggest a positive meaning because `happy` remains.

My custom stopword list removes common stopwords but keeps `not`, `n't`, `no`, and `never`. This preserves the customer's complaint: the customer does not think the jacket is worth the price and is not happy.

### Task 6 — Stemming

| Original | Porter | Lancaster | Snowball |
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

Porter and Snowball merge `connection`, `connected`, and `connecting` into `connect`, which can help group related word forms.

However, `university` and `universal` both become `univers`, even though their meanings differ. Also, `flies` becomes `fli` and `wolves` becomes `wolv`, which are not normal English words. These examples demonstrate that stemming can create non-words and merge words with different meanings.

### Task 7 — Lemmatization

The default `WordNetLemmatizer` treats a word as a noun unless a POS hint is provided. Words such as `connected`, `connecting`, `organized`, and `ran` may change when the correct verb POS hint is supplied.

For example, `ran` remains `ran` by default but becomes `run` when `pos="v"` is used. This demonstrates why POS information is important for accurate lemmatization.

### Task 8 — Sentence Splitting

`sent_tokenize` splits T3 into two sentences. It correctly recognizes `Mr.` as an abbreviation and does not treat the period after `Mr.` as a sentence boundary. It also handles `a.m.` correctly, so the time expression does not create an extra sentence.

### Task 9 — POS Tagging and NER

| Entity | NLTK label | Correct? | Explanation |
|---|---|---|---|
| Smith | PERSON | Correct | Smith is a person's surname. |
| London | GPE | Correct | London is a geographic/political location. |
| Apple | ORGANIZATION | Correct | In this sentence, Apple refers to the company. |
| Friday | Usually not recognized | Missed | Friday is a day of the week and should ideally be recognized as a date/time expression. |

The NER system identifies `Smith` as a person and `London` as a location. It may identify `Apple` as an organization because the phrase `the Apple team` provides company context. However, `Friday` may be missed because it is a day of the week rather than a full date.

### Task 10 — Full Pipeline

The default-stopword pipeline removes `not`, weakening the customer's negative opinion. The negation-keeping pipeline retains `not` and `n't`, so it better preserves the complaint.

For a spam detector, I would use lowercasing only if evaluation shows that it helps, whitespace normalization, tokenization, and regex replacement of phone numbers, money amounts, URLs, and repeated punctuation. I would avoid deleting all punctuation, digits, capitalization, and negations because they can distinguish spam from normal messages.

In T1, deleting `!!!`, `£1,000`, and `0712345678` removes strong spam evidence. Replacing them with `!`, `MONEY`, and `PHONE_NUMBER` preserves the signal while making the text cleaner.

---

## Quick Summary for Revision

| Topic | Key idea |
|---|---|
| Preprocessing | Prepare raw text, but do not delete useful information |
| Normalization | Make variants consistent, such as case, spaces, and Unicode |
| Tokenization | Word = readable but larger vocabulary; subword = balanced; character = small vocabulary but long sequences |
| Stopwords | Remove for some search tasks; keep negation and task-relevant words |
| Stemming vs. lemmatization | Stemming is fast and rough; lemmatization aims for dictionary forms |
| Regex | Powerful, but broad patterns can damage text |
| Case study | Results depend on the task, language, and dataset |

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Chai, C. P. (2023). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553.
- Chapman, W. W., Bridewell, W., Hanbury, P., Cooper, G. F., & Buchanan, B. G. (2001). A simple algorithm for identifying negated findings and diseases in discharge summaries. *Journal of Biomedical Informatics, 34*(5), 301–310.
- Fellbaum, C. (Ed.). (1998). *WordNet: An electronic lexical database*. MIT Press.
- Fox, C. (1989). A stop list for general text. *ACM SIGIR Forum, 24*(1–2), 19–21.
- Friedl, J. E. F. (2006). *Mastering regular expressions* (3rd ed.). O'Reilly Media.
- Haddi, E., Liu, X., & Shi, Y. (2013). The role of text pre-processing in sentiment analysis. *Procedia Computer Science, 17*, 26–32.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and language processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
- Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. *Proceedings of ACL 2018*.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent tokenizer and detokenizer. *Proceedings of EMNLP 2018: System Demonstrations*.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.
- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39–41.
- Nothman, J., Qin, H., & Yurchak, R. (2018). Stop word lists in free open-source software packages. *Proceedings of the Workshop for NLP Open Source Software (NLP-OSS)*.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.
- Porter, M. F. (2001). *Snowball: A language for stemming algorithms*. http://snowball.tartarus.org/texts/introduction.html
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*.
- Unicode Consortium. (2023). *Unicode Standard Annex #15: Unicode normalization forms*. https://unicode.org/reports/tr15/
- Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112.
- Wu, Y., et al. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144.
- NLTK. *nltk.stem package documentation*. https://www.nltk.org/api/nltk.stem.html
- NLTK. *WordNetLemmatizer documentation*. https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html
- NLTK. *Snowball stemmer documentation*. https://www.nltk.org/api/nltk.stem.snowball.html