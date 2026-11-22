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

### Table for task 6 (stemming)

| Word | Porter | Lancaster | Snowball |
|---|---|---|---|
| connection | [from output] | [ ] | [ ] |
| connected | | | |
| connecting | | | |
| organization | | | |
| organized | | | |
| university | | | |
| universal | | | |
| flies | | | |
| wolves | | | |
| better | | | |
| ran | | | |

Pair of different words merged into the same stem: [copy from the "merged" lines of your output, e.g. check `university` and `universal`].
A stem that is not a real word: [copy from the WordNet check line].

### Entity table for task 9 (NER on T3)

| Entity | Label from `ne_chunk` | Correct? |
|---|---|---|
| [from output] | [ ] | [ ] |

What the model missed: [check whether Smith, London, Apple and Friday were all found].

### Short answers (2–4 sentences each)

**Task 1 – What could a spam detector lose by lowercasing T1?** [Hint: look at what the original has that the lowercase version loses: capitalized words such as URGENT, WON and NOW. Explain why shouting is a spam clue.]

**Task 3 – What did `[^a-z\s]` delete, and why is that a problem?** [Use the "everything that was matched" line. Mention the `£`, the digits of the amount and phone number, and the punctuation.]

**Task 4 – Compare `split()` and `word_tokenize` on T2, and the T4 counts.** [Use your printed token counts. Explain how punctuation, the contraction "don't" and the emoji are handled differently, and how case changes the vocabulary size of T4.]

**Task 5 – How did the meaning change between the two stopword lists?** [Compare the two printed lists. Note whether negation words survived in the second list and how that changes the sentiment of "I don't think this jacket is worth the price. Not happy at all".]

**Task 6 – Stemming observations.** [Name your merged pair and your non-word stem from the table, and say which stemmer was most aggressive.]

**Task 7 – Lemmatization.** [List the words whose lemma changed only when you supplied the part-of-speech hint, from the "changed?" column. Explain why the default noun assumption fails for verbs and adjectives.]

**Task 8 – Sentence splitting.** [State the number of sentences from your output and whether `Mr.` and `a.m.` broke the sentences wrongly.]

**Task 10 – Which pipeline keeps the customer's complaint?** [Compare the final lemma lists of the two runs. Identify which one still contains the negation words and explain why that matters.]

### Which preprocessing steps for a spam detector?

[Write one paragraph using your own output as evidence. Suggested structure: (1) **Use** whitespace normalization, and regex *replacement* (PHONE_NUMBER, MONEY, URL) rather than deletion, because task 3 shows that deleting removes the strongest clues; (2) **be careful** with lowercasing, since capitalization and `!!!` are signals (task 1), so consider keeping a feature for capitalization or repeated punctuation; (3) **avoid** the aggressive `[^a-z\s]` rule and blind stopword removal, with evidence from tasks 3 and 5; (4) stemming or lemmatization are optional, so compare on validation data; (5) apply identical preprocessing at training and prediction time.]

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
