# Assignment 2 – Preprocessing: Answers

## Part A – Research

### 1. Text preprocessing

**Definition.** Text preprocessing is the set of transformations applied to raw text before it is given to an NLP model: for example normalizing case and spacing, splitting text into tokens, filtering words, and reducing words to base forms (Jurafsky & Martin, 2024; Manning, Raghavan & Schütze, 2008).

**Why systems need it.** Real text is noisy and inconsistent. A bag-of-words model treats "Prize", "PRIZE" and "prize" as three separate features, which spreads the evidence for one idea over several columns and makes sparse data even sparser. Preprocessing reduces this variation so that a model can learn from fewer, better-populated features.

**Why more cleaning can make results worse.** Every cleaning step assumes that the difference it removes is unimportant. When that assumption is false, useful signal is lost. Uysal and Gunal (2014) found that choosing suitable combinations of preprocessing steps, rather than turning them all on or all off, gave the best classification accuracy, and that the best choice depended on the domain and language.

**Real-life example.** A bank builds a filter for scam text messages: *"URGENT!!! You won £1,000. Call 0712345678 now!"* If the pipeline lowercases everything, deletes all digits and symbols, and strips punctuation, the message becomes *"urgent you won call now"*. The money amount, the phone number, the capitals and the repeated exclamation marks, which are the strongest scam signals, are gone, and the message looks like a normal one. A second example is sentiment analysis: removing the stopword "not" turns "not good" into "good".

### 2. Text normalization

**Definition.** Normalization maps different written forms of the same thing to one consistent form (Jurafsky & Martin, 2024). Three common types:

| Type | What it does | Example |
|---|---|---|
| Lowercasing | Converts capitals to lowercase | `Prize`, `PRIZE` → `prize` |
| Whitespace normalization | Collapses runs of spaces, tabs and line breaks | `I   love  NLP` → `I love NLP` |
| Unicode normalization | Converts equivalent character encodings to one standard form (NFC, NFD, NFKC, NFKD; Unicode Consortium, UAX #15) | `é` written as one character vs. `e` plus a combining accent become the same string |

**Where it helps.** In a search engine, a user typing "iphone" should match pages that write "iPhone" or "IPHONE". Lowercasing merges these. Unicode normalization matters for languages with accents, since two visually identical words can otherwise be different strings and never match.

**Where it removes useful information.** In named entity recognition or part-of-speech tagging, capitalization is a major clue: "Apple" (company) versus "apple" (fruit), or "US" (country) versus "us" (pronoun). Lowercasing also removes the "shouting" signal in spam or hate-speech detection.

### 3. Word-level, subword and character-level tokenization

| Aspect | Word-level | Subword | Character-level |
|---|---|---|---|
| Unit | Whole words (and punctuation) | Frequent word pieces | Single characters |
| Example for "unhappiness" | `unhappiness` | `un` + `happi` + `ness` (varies by tokenizer) | `u`,`n`,`h`,`a`,`p`,`p`,`i`,`n`,`e`,`s`,`s` |
| Vocabulary size | Very large (every word form) | Medium, chosen in advance (often tens of thousands) | Very small (alphabet, digits, symbols) |
| Unknown or misspelled words | Out-of-vocabulary problem | Built from known pieces | Always representable |
| Sequence length | Shortest | Slightly longer than words | Longest |
| Needs training? | No (rules or whitespace) | Yes, learned from a corpus | No |

Subword methods such as Byte-Pair Encoding (BPE) start from characters and repeatedly merge the most frequent adjacent pair into a new symbol, until a target vocabulary size is reached (Sennrich, Haddow & Birch, 2016). SentencePiece (Kudo & Richardson, 2018) is a widely used implementation that works directly on raw text.

**Vocabulary size** is the number of unique token types a tokenizer or model can represent. It is not the number of tokens in a text: in "the cat saw the dog" there are 5 tokens but 4 types.

**Effect of tokenizer choice.** Word-level tokenization yields the biggest vocabulary, especially in morphologically rich languages, where one root appears in many surface forms. Character-level yields the smallest vocabulary but long sequences and weaker word-level meaning. Subword tokenization is a compromise: the vocabulary size is set by the designer, and rare words are split into reusable pieces.

### 4. Stopwords

**Definition.** Stopwords are very frequent function words (in English: *the, is, and, of*) that carry little topical content and are often removed in information retrieval and classical text mining (Manning et al., 2008).

**Benefits.** Fewer features, smaller models, faster training and search, and less weight given to words that do not distinguish documents.

**Risks.**
- Removing negation (*not, no, never*) can reverse the meaning of a sentence.
- Removing phrase-level words damages expressions such as "to be or not to be" or the band name "The Who".
- For modern neural models, function words give grammatical context, so removing them can hurt.
- Schofield, Magnusson and Mimno (2017) studied stopword removal for topic models and found that removing stopwords before training is often unnecessary, because the models can handle frequent words or they can be filtered afterwards.

**Why the right list depends on task and language.** A word that is uninformative in one task can be crucial in another: "not" is useless for topic classification but essential for sentiment analysis. Lists also differ by language, since each language has its own function words, and published lists vary widely in size and quality across software packages, with inconsistencies that can affect results (Nothman, Qin & Yurchak, 2018). Domain matters too: in legal or medical text, a word that is common in that domain may need to be treated as a stopword even though it is not one in general English.

### 5. Stemming vs. lemmatization

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix stripping | Morphological analysis plus a dictionary (lexicon) |
| Output | A stem, not necessarily a word (`studies` → `studi`) | A lemma, a real dictionary form (`studies` → `study`) |
| Needs POS? | No | Usually helps (`better` → `good` only as an adjective) |
| Speed | Fast | Slower |
| Typical errors | Over-stemming (merging unrelated words) and under-stemming | Wrong lemma if POS is wrong or the word is not in the lexicon |

**Stemming algorithm: the Porter stemmer** (Porter, 1980). It removes English suffixes in a sequence of steps, each applying rewrite rules under conditions on the remaining stem, such as the number of consonant-vowel sequences. It is fast and widely used as a baseline. The Lancaster (Paice/Husk) stemmer is more aggressive (Paice, 1990).

**Lemmatization approach: dictionary-based lemmatization with WordNet.** WordNet is a large lexical database of English (Miller, 1995). The NLTK `WordNetLemmatizer` looks a word up in it and returns the dictionary form for the given part of speech. Other approaches include statistical and neural lemmatizers trained on annotated treebanks.

**When to prefer each.** Stemming suits large-scale search and retrieval, where speed matters and an unreadable stem does not affect users. Lemmatization is preferable when the output must be a readable word or when meaning matters, for example in text analysis, topic labelling, or tasks needing correct grammatical information. For morphologically rich or low-resource languages the choice depends on whether good language-specific tools exist.

### 6. Regular expressions in text cleaning

**Use.** A regular expression (regex) is a compact pattern language for matching strings (Friedl, 2006). In cleaning it is used to find and replace or delete things such as extra whitespace, URLs, email addresses, phone numbers, digits, or punctuation, typically with `re.sub()` in Python.

**Two example patterns.**

| Pattern | Matches | Example |
|---|---|---|
| `\s+` | One or more whitespace characters in a row | `"a   b"` → replace with one space → `"a b"` |
| `https?://\S+` | `http://` or `https://` followed by non-space characters, i.e. a URL | `"see http://x.com/win now"` → `"see URL now"` |

**How an overly broad pattern damages text.** The pattern `[^a-z\s]` deletes everything except lowercase English letters and whitespace. On "Win $500 now, don't miss café deals!" it also removes the `$500` (a spam clue), the apostrophe in "don't" (leaving "don t"), and the accented letter in "café", which breaks the word. In a Somali or other non-English corpus, the same rule can delete legitimate characters.

### 7. Case study: comparing preprocessing choices

**Reference.** Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006

**Task.** Text classification. The authors ask how preprocessing influences classification accuracy, and how this depends on text domain, language and feature dimension.

**Data.** Two domains (e-mail and news) in two languages (Turkish and English), using benchmark datasets.

**Preprocessing options compared.** The widely used steps tokenization, stopword removal, lowercase conversion and stemming, evaluated in all possible combinations, so that each step's contribution and the interactions among steps could be measured.

**Findings.** According to the abstract, picking an appropriate combination of preprocessing tasks, rather than switching all of them on or off, can significantly improve accuracy, and the best combination depends on the domain and language studied. The takeaway matches the lesson of Q1: preprocessing is an experimental choice, not a fixed recipe.

*Note: I based this summary on the published abstract and bibliographic records. Read the full paper's results section and add specific numbers or step-by-step findings before submitting.*

---

## Part B – Practical

> The outputs below must come from **your own run** of `preprocessing_practice.py`. Run `python preprocessing_practice.py > output.txt`, then fill in the bracketed parts from `output.txt`. Do not type results from memory.

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

All three stemmers merge `connection`, `connected` and `connecting` into `connect`; they also merge `university` and `universal` into `univers`. Lancaster is the most aggressive here: it reduces `organization` to `org` and `better` to `bet`, while Porter and Snowball produce `organ` and leave `better` unchanged. The WordNet check reports `fli`, `univers` and `wolv` as non-words for Porter and Snowball; Lancaster also reports `org`.

### Task 9 – Named entities in T3

| Entity | Label from `ne_chunk` | Correct? |
|---|---|---|
| Mr. | PERSON | No; this is a title, not the person's name. |
| Smith | PERSON | Yes |
| London | GPE | Yes; it is a place. |
| Apple | Not recognized | No; it refers to a company in this sentence. |
| Friday | Not recognized | No; it is a day. |

The model recognized Smith and London, but missed Apple and Friday. It also grouped `Mr.` as a separate `PERSON`, which is not a useful entity label for the title by itself.

### Short answers

**Task 1 – What could a spam detector lose by lowercasing T1?**
Lowercasing removes the capitalization in `URGENT`, `WON` and `NOW`. That emphasis can signal a promotional or spam message, so a detector should consider preserving capitalization as a feature rather than discarding it.

**Task 3 – What did `[^a-z\s]` delete, and why is that a problem?**
The output shows that the pattern deletes `!!!`, the `£1,000` amount, the phone number `0712345678`, the period and the final exclamation mark. This removes potentially useful spam signals such as the currency, amount, phone number and emphatic punctuation.

**Task 4 – Compare `split()` and `word_tokenize` on T2, and the T4 counts.**
`split()` returns 14 tokens and keeps `don't` and `price.` intact, while `word_tokenize()` returns 16 by splitting `don't` into `do` and `n't` and separating the period after `price`. The emoji is retained by both; T4 has 11 tokens and 10 case-sensitive vocabulary items, or 9 after lowercasing.

**Task 5 – How did the meaning change between the two stopword lists?**
The default stopword list removes `not`, so its output no longer clearly represents the negative sentiment in “Not happy”; it still retains `n't` because `n't` is not in NLTK's default list. The custom list preserves both `not` and `n't`, retaining more of the complaint's negation.

**Task 6 – Stemming observations.**
The stemmers merge `connection`, `connected` and `connecting` into `connect`, and merge `university` and `universal` into `univers`. Lancaster is most aggressive in this example, producing `org` and `bet`; the WordNet check also identifies stems such as `fli` and `wolv` as non-words.

**Task 7 – Lemmatization.**
The words that change when given a POS hint are `connected` → `connect`, `connecting` → `connect`, `organized` → `organize`, `better` → `good` and `ran` → `run`. The default treats each word as a noun, so it cannot correctly lemmatize these verb and adjective forms.

**Task 8 – Sentence splitting.**
The tokenizer returns two sentences, and neither `Mr.` nor `a.m.` causes an incorrect sentence split. It keeps “Mr. Smith arrived in London on Friday.” together as the first sentence.

**Task 10 – Which pipeline keeps the customer's complaint?**
The negation-keeping pipeline is more informative because its final list retains `not` as well as `n't`. The default-stopword pipeline removes `not`, weakening the negative sentiment in the customer's complaint.

### Which preprocessing steps for a spam detector?

I would normalize whitespace and replace recognizable values such as phone numbers and money amounts with labels like `PHONE_NUMBER` and `MONEY`, rather than deleting them: Task 3 shows that deletion removes the amount and number. I would be cautious about lowercasing and repeated-punctuation cleanup because capitalization and `!!!` may be spam signals, and I would avoid the broad `[^a-z\s]` rule and blind stopword removal because they can discard useful evidence, including negation. Stemming or lemmatization should be tested against a validation set, and the same preprocessing should be applied during both training and prediction.

---

## References

- Friedl, J. E. F. (2006). *Mastering Regular Expressions* (3rd ed.). O'Reilly Media.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *Proceedings of EMNLP 2018: System Demonstrations*.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39–41.
- Nothman, J., Qin, H., & Yurchak, R. (2018). Stop word lists in free open-source software packages. *Proceedings of the Workshop for NLP Open Source Software (NLP-OSS)*.
- Paice, C. D. (1990). Another stemmer. *ACM SIGIR Forum, 24*(3), 56–61.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.
- Schofield, A., Magnusson, M., & Mimno, D. (2017). Pulling out the stops: Rethinking stopword removal for topic models. *Proceedings of EACL 2017*.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*.
- Unicode Consortium. *Unicode Standard Annex #15: Unicode Normalization Forms.* https://unicode.org/reports/tr15/
- Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006
