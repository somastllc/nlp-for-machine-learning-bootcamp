Assignment 2 – Preprocessing

Student: Alas_duqow
Files: answers.md, preprocessing_practice.py, output.txt

Citations use numbered references in square brackets, e.g. [1], and are listed in the References section at the end.

Part A – Research
1. Text preprocessing and the "more cleaning" problem

Definition. Text preprocessing turns raw, irregular text into a consistent form before a model or search index sees it. Jurafsky and Martin call the core of this text normalization: tokenizing words, standardizing word forms and segmenting sentences [1]. It also covers cleaning (markup, URLs, symbols), case and Unicode handling, stopword filtering, and stemming or lemmatization.

Why it is needed. A computer sees characters, not words. Without preprocessing, USA, U.S.A. and usa are three unrelated strings and "fast," differs from "fast". This splits the evidence for one word across many entries and makes data sparse. Normalization applied the same way to documents and queries lets different surface forms match [2], and it also shrinks vocabulary and processing cost.

Why more cleaning can hurt. Every step is lossy: it discards distinctions assumed to be noise, and what counts as noise depends on the task.

Lowercasing helps retrieval and speech recognition but is usually unwise for sentiment analysis, information extraction and machine translation, where US must stay different from us [1].
Collapsing periods so U.S.A. matches USA is sensible, but the same rule makes a query for C.A.T. match every cat [2].
Lowercasing lowered a CNN sentiment classifier's accuracy on IMDB reviews, and lemmatization did not help in general [3].
Testing all combinations of common steps showed that a combination suited to the domain and language beats switching everything on or off [4].

Example. A shop sorts reviews into "happy" and "unhappy". The review "I don't think this jacket is worth the price. Not happy at all 😞" goes through lowercasing, deletion of non-letters and stopword removal. The emoji is deleted and Not is removed as a stopword, leaving "n't think jacket worth price happy". The only negation left is an orphaned n't, so the text looks like a satisfied customer. I reproduced this in Practical tasks 5 and 10.

2. Text normalization

Normalization maps surface variants of the same content onto one canonical form.

Step	What it does	Benefit	Risk
Lowercasing	Woodchuck → woodchuck	Smaller vocabulary	Loses US vs us, Apple vs apple, shouted words
Whitespace normalization	Runs of whitespace → one space	Removes spurious differences	Can destroy layout in code or tables
Unicode normalization (NFC, NFD, NFKC, NFKD)	Makes equivalent encodings identical	Same visible text gets the same string	NFKC/NFKD can erase meaningful distinctions
Accent/punctuation handling	Strip diacritics, drop punctuation	Matches cafe/café	Can merge different words

Unicode. UAX #15 defines four normalization forms [5]. Canonically equivalent sequences represent the same abstract character (e.g. Ç as one character, or C plus a combining cedilla). NFD decomposes; NFC decomposes then recomposes. Compatibility forms (NFKC/NFKD) also merge things like the ligature ﬃ and ffi, and the standard warns they can remove formatting distinctions that matter [5].

Where it helps. The name Café typed on two devices may use different encodings of é. They look identical but fail an exact match; NFC plus lowercasing makes them equal [5]. Case folding likewise helps retrieval, where users rarely capitalize queries consistently [1].

Where it removes information. In spam detection, lowercasing erases shouted words such as URGENT (my Task 1 found three all-caps words). In named-entity recognition, lowercasing makes Apple the company look like apple the fruit [1]. NFKC rewrites m² as m2 (my own illustration of the warning in [5]).

3. Word, subword and character tokenization

Vocabulary size. The vocabulary V is the set of distinct token types, and |V| is their number. Word types (distinct words) differ from word instances (running words): "the cat saw the dog" has five instances but four types [6]. |V| fixes the size of the embedding table and output layer, and any word outside V is unknown to the model [7].

	Word-level	Subword (BPE, WordPiece, Unigram)	Character-level
Vocabulary	Very large, keeps growing	Chosen by the designer, tens of thousands	Tiny (256 for bytes)
Unknown words	<unk>	Rare; built from known pieces	None
Sequence length	Shortest	Medium	Longest
Meaning per token	High	Medium	Low
Weakness	Out-of-vocabulary words, huge table	Odd splits, depends on training corpus	Long sequences of low-meaning units
Word-level tokenization needs many rules (clitics like don't, abbreviations, hyphens, languages without spaces) [1]. Vocabulary has no ceiling: by Heaps' law |V| = kN^β, new words keep appearing as the corpus grows [6]. Every variant (love, loving, loved) needs its own entry [7].
Subword tokenization is the compromise used by modern models. In BPE, the process starts from characters and repeatedly merges the most frequent adjacent pair; the number of merges is the one hyperparameter [8]. Unseen words are split into known pieces. SentencePiece trains such tokenizers directly on raw text [9].
Character-level tokenization removes unknown words and keeps V tiny, but sequences are long and each unit carries little meaning [7]. Character-level CNNs can still be competitive on large classification datasets [10].

How the tokenizer sets vocabulary size.

GPT's BPE has 40,478 entries (478 base symbols + 40,000 merges). GPT-2's byte-level BPE has 50,257 (256 bytes + 50,000 merges + 1 end-of-text token); starting from bytes means nothing is ever unknown [7].
Practical subword vocabularies run from 50,000 to 200,000 entries [6].
Even for word-level systems the rules matter: in my Task 4, split() gave 14 tokens for T2 and word_tokenize gave 16.
Tokenizers trained mostly on English split other languages into more pieces: a 16-word Spanish sentence became 33 tokens against 18 for the English version [6]. I would expect this to matter more for under-represented languages (my own inference).
4. Stopwords

Definition. Stopwords are extremely frequent function words (the, of, is) assumed to add little to matching or meaning. The traditional recipe ranks terms by collection frequency and hand-filters the top ones [2].

Benefits. Smaller index or feature space, and less repetitive low-information material in frequency counts for tasks like topic labelling.

Risks.

Meaning-bearing function words: removing of/to hurts "President of the United States" and "flights to London", and titles like "To be or not to be" vanish entirely [2]. In sentiment analysis, removing not reverses polarity (my Task 5).
Little benefit today: with index compression and IDF weighting, web search engines generally use no stop list [2]. For topic models, stopword removal mostly changes which words appear at the top of topics, and removing corpus-specific stopwords after training is more transparent [11].
List quality: 52 English stop lists were found to differ from one another and to be internally inconsistent (one had hasn't but not hadn't), and a list may not match its tokenizer [12]. My Task 5 hit this: NLTK's list contains not but not n't, so after word_tokenize splits don't into do + n't, Not is removed and n't stays.

Task and language. A topic classifier may drop the, a sentiment classifier must keep not, phrase search needs of and to, and authorship attribution relies on function words. Each language has its own function words, so lists are language-specific (NLTK ships separate files; the English one has 198 entries in my version) and must match the language's tokenization and morphology.

5. Stemming vs. lemmatization

Stemming is a crude rule-based process that chops affixes so related forms share a representation. Lemmatization uses a vocabulary and morphological analysis to return the dictionary form (am, are, is → be) [1, 2].

	Stemming	Lemmatization
Method	Suffix-stripping rules	Dictionary look-up plus morphology, often POS-guided
Output	May not be a word (univers)	A real word (fly, good)
Irregular forms (ran, better)	Cannot handle	Handles them with the correct POS
Cost	Very fast, few resources	Slower, needs a lexicon and POS tags
Typical error	Over-stemming	Wrong lemma if POS is wrong

Porter stemmer. Porter's algorithm applies about five phases of rewrite rules, choosing the longest matching suffix and conditioning many rules on the stem's "measure" [13]. It turns replacement into replac but leaves cement alone [2]. Typical rules are ational → ate and sses → ss [1]. Known errors: organization → organ and policy → police (over-generalization), European/Europe (under-generalization) [1]. Porter later introduced Snowball and "Porter2", regarding over-stemming as the most harmful error [14]. The more aggressive Paice/Husk (Lancaster) stemmer applies an ordered rule table until none applies [15, 16].

WordNet lemmatizer. NLTK's WordNetLemmatizer uses WordNet's morphy, returns the shortest candidate lemma, and returns the word unchanged if none is found. The POS defaults to noun, so verbs and adjectives need the correct tag [17]. spaCy offers rule-based, look-up-table and trainable edit-tree lemmatization [18].

When to use which. Use stemming when speed and recall matter and output is only an internal matching key, e.g. search in morphologically rich languages, where gains are large; for English gains are modest [2]. Use lemmatization when output must be real words, irregular forms matter, or precision matters more than recall. Lemmatization did not generally improve a neural classifier [3], so neither is a default and both should be tested.

6. Regular expressions in text cleaning

A regular expression (regex) is a compact pattern language for finding and replacing text [6]. In Python, re.sub(pattern, replacement, text) replaces every match [19], for whitespace collapsing, stripping tags, URLs or mentions, and replacing numbers with placeholders.

Pattern	Matches	Use
\s+	One or more whitespace characters	Replace with one space
https?://\S+	http:// or https:// plus all following non-space characters	Remove or replace URLs with URL

Overly broad patterns. Mistakes are false positives (matching too much) or false negatives. [tT]he meant to find the also matches inside other and there; \b[tT]he\b fixes it [6]. A more destructive case is a catch-all like [^a-z\s], which deletes digits, currency signs, apostrophes, accents, non-Latin scripts and emoji. On the lowercased spam message in my Task 3 it deleted 21 characters, including £1,000, the phone number and every exclamation mark, the strongest spam clues. It also turns don't into dont, which no stopword list recognizes as negation.

7. Case study: preprocessing in neural text classification

Source. Camacho-Collados and Pilehvar (2018), BlackboxNLP workshop at EMNLP 2018 [3].

Setup. Nine benchmarks: topic categorization (BBC, 20News, Reuters, Ohsumed) and polarity detection (RTC, IMDB, PL05, PL04, Stanford Sentiment). Models: a CNN and a CNN+LSTM with 300-dimensional word2vec embeddings. Options: vanilla tokenization, lowercasing, lemmatization, and multiword grouping (e.g. United_States). Experiment 1 applied the same preprocessing to the dataset and the embedding corpus; experiment 2 preprocessed only the embedding corpus.

Findings.

Preprocessing changed accuracy by about ±2.4% on average for the CNN+LSTM, significantly on seven of nine datasets.
Plain tokenization was generally as good as or better than the elaborate options.
Lowercasing can hurt: on IMDB with the CNN, accuracy fell from 87.5 (vanilla) to 84.2 (lowercased).
Domain-specific data was the exception: on medical Ohsumed with the CNN+LSTM, vanilla scored 30.8 against 37.5 (lowercased) and 37.1 (lemmatized).
In experiment 2, embeddings trained on multiword-grouped text were best on most datasets with the CNN.

Takeaway. There is no universally best recipe; preprocessing must be held constant when comparing systems. This agrees with [4], where choosing a suitable combination rather than applying everything significantly raised accuracy for Turkish and English e-mail and news classification.

Practical

All results come from preprocessing_practice.py; the full output is in output.txt (python preprocessing_practice.py > output.txt).

Task 6 table – stems
Word	Porter	Lancaster	Snowball
connection	connect	connect	connect
connected	connect	connect	connect
connecting	connect	connect	connect
organization	organ	org	organ
organized	organ	org	organ
university	univers	univers	univers
universal	univers	univers	univers
flies	fli	fli	fli
wolves	wolv	wolv	wolv
better	better	bet	better
ran	ran	ran	ran
Task 9 table – named entities in T3
Entity	NLTK label	Expected	Correct?
Mr.	PERSON	part of the span "Mr. Smith"	no
Smith	PERSON	PERSON	yes
London	GPE	GPE	yes

Missed: Apple (ORGANIZATION), Friday (DATE), 9 a.m. (TIME). pos_tag tagged Apple and Friday as NNP, but ne_chunk left them unlabeled. It split Mr. and Smith into two chunks, and its label set has no DATE or TIME. word_tokenize also split the final a.m. into a.m and ..

Short answers

Task 1 – Lowercasing. My script found three all-caps words (URGENT!!!, WON, NOW!). Lowercasing erases this shouting, itself a spam signal. Safer: compute all-caps counts before lowercasing.

Task 3 – Regex cleaning. [^a-z\s] deleted 21 characters (!!!, £1,000, ., 0712345678, !) and left double spaces. My replacement rules gave URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!, keeping the presence of an amount and phone number without the digits. The cost: !!! collapses to !.

Task 5 – Stopwords and negation. NLTK's list (198 words) removes not and no but not never or n't. On T2 it left ["n't", "think", "jacket", "worth", "price", ".", "happy", "😞"], where happy stands alone. My list (NLTK's minus not, n't, no, never) kept Not as well. The first reads like praise; the second keeps the complaint.

Task 6 – Stemming. Porter and Snowball agreed on all eleven words; Lancaster differed on organization/organized (org) and better (bet), so it is more aggressive. All three merged university and universal into univers (over-stemming), and Porter/Snowball reduced organization to organ, the classic error in [1]. Several stems are not words (univers, fli, wolv, org), and irregular forms failed (ran stayed ran).

Task 7 – Lemmatization. Five words changed only with a POS hint: connected → connect, connecting → connect, organized → organize, better → good, ran → run. flies → fly and wolves → wolf were already correct as nouns. The lemmatizer gives real words and handles irregular forms, but only with the right POS.

Task 8 – Sentence splitting. sent_tokenize returned two sentences and did not break after Mr. or inside a.m.. A naive regex splitting after every period plus space gave three pieces and separated Mr. from Smith. Here a.m. is at the very end, so it was not a hard test.

Task 10 – Full pipeline. With NLTK's default stopwords the result was ["n't", "think", "jacket", "worth", "price", "happy"]; with my negation-keeping list it was ["n't", "think", "jacket", "worth", "price", "not", "happy"]. Only the second keeps the complaint. Both lose the 😞 (removed by the regex) and do, leaving n't without its verb. I used [^a-z'\s] here so don't keeps its apostrophe.

Which steps for a spam detector?

I would use whitespace normalization, word_tokenize, and replace-not-delete regex rules like Task 3b (MONEY, PHONE_NUMBER). I would extract style features (all-caps word count, exclamation count) from the raw text before lowercasing, then lowercase for word features. I would avoid [^a-z\s], which removed 21 characters including most of the spam evidence, and avoid the default stopword list, which drops Not but keeps n't (Tasks 5 and 10). Stemming is optional: it merges connect/connected/connecting but also university/universal, so I would compare accuracy with and without it on a validation set.

References

[1] D. Jurafsky and J. H. Martin, Speech and Language Processing (3rd ed. draft), Ch. 2: Regular expressions, text normalization, edit distance, draft of Dec. 29, 2021. https://web.stanford.edu/~jurafsky/slp3/

[2] C. D. Manning, P. Raghavan and H. Schütze, Introduction to Information Retrieval. Cambridge University Press, 2008, Secs. 2.2.2–2.2.4. https://nlp.stanford.edu/IR-book/

[3] J. Camacho-Collados and M. T. Pilehvar, "On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis," in Proc. 2018 EMNLP Workshop BlackboxNLP, 2018, pp. 40–46. https://doi.org/10.18653/v1/W18-5406

[4] A. K. Uysal and S. Günal, "The impact of preprocessing on text classification," Information Processing & Management, vol. 50, no. 1, pp. 104–112, 2014. https://doi.org/10.1016/j.ipm.2013.08.006

[5] Unicode Consortium, "Unicode Standard Annex #15: Unicode Normalization Forms," Version 16.0.0, 2024. https://www.unicode.org/reports/tr15/

[6] D. Jurafsky and J. H. Martin, Speech and Language Processing (3rd ed. draft), Ch. 2: Words and tokens, draft of Aug. 24, 2025. https://web.stanford.edu/~jurafsky/slp3/

[7] Hugging Face, "Summary of the tokenizers," Transformers documentation, n.d. https://huggingface.co/docs/transformers/tokenizer_summary

[8] R. Sennrich, B. Haddow and A. Birch, "Neural machine translation of rare words with subword units," in Proc. ACL 2016. https://aclanthology.org/P16-1162/

[9] T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing," in Proc. EMNLP 2018: System Demonstrations. https://aclanthology.org/D18-2012/

[10] X. Zhang, J. Zhao and Y. LeCun, "Character-level convolutional networks for text classification," in Advances in Neural Information Processing Systems 28 (NIPS 2015). arXiv:1509.01626

[11] A. Schofield, M. Magnusson and D. Mimno, "Pulling out the stops: Rethinking stopword removal for topic models," in Proc. 15th Conf. European Chapter of the ACL (Short Papers), 2017, pp. 432–436. https://aclanthology.org/E17-2069/

[12] J. Nothman, H. Qin and R. Yurchak, "Stop word lists in free open-source software packages," in Proc. Workshop for NLP Open Source Software (NLP-OSS), 2018, pp. 7–12. https://aclanthology.org/W18-2502/

[13] M. F. Porter, "An algorithm for suffix stripping," Program, vol. 14, no. 3, pp. 130–137, 1980. https://tartarus.org/martin/PorterStemmer/

[14] M. F. Porter, "Snowball: A language for stemming algorithms," 2001. https://snowballstem.org/texts/introduction.html

[15] C. D. Paice, "Another stemmer," ACM SIGIR Forum, vol. 24, no. 3, pp. 56–61, 1990.

[16] NLTK Project, "nltk.stem.lancaster," n.d. https://www.nltk.org/_modules/nltk/stem/lancaster.html

[17] NLTK Project, "nltk.stem.wordnet," n.d. https://www.nltk.org/_modules/nltk/stem/wordnet.html

[18] spaCy, "Lemmatizer," API documentation, n.d. https://spacy.io/api/lemmatizer

[19] Python Software Foundation, "re — Regular expression operations," n.d. https://docs.python.org/3/library/re.html