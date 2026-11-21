# Assignment 2 – Text Preprocessing

## Part A – Research

### 1. What is text preprocessing?

#### Definition. 
Text preprocessing is the set of transformations applied to raw text before it is given to an NLP model, so that the text is in a consistent, machine-readable form. Typical steps include normalization (case, whitespace, Unicode), tokenization, removal or replacement of noise (HTML, URLs, numbers), stopword filtering, and reducing words to a base form by stemming or lemmatization (Jurafsky & Martin, 2025, ch. 2; Bird, Klein & Loper, 2009, ch. 3).

#### Why NLP systems need it. 
A computer does not see "words"  it sees a sequence of characters or bytes. Without preprocessing:

the same word appears in many surface forms (Price, price, price., PRICE!), so a bag-of-words model treats them as unrelated features, which makes the feature space larger and sparser;
noise such as markup, repeated spaces or invisible Unicode characters creates fake tokens;
the model's input does not match the format it was trained on (for example, a pretrained tokenizer expects a particular Unicode form).

Uysal and Günal (2014) describe preprocessing as one of the key components of a text classification framework, together with feature extraction, feature selection and classification. Reducing the number of distinct features also lowers memory use and training time.

#### Why more cleaning can make results worse. 
Every cleaning step is a decision about which information is "noise". If the task depends on that information, cleaning removes the signal. Camacho-Collados and Pilehvar (2018) found that for a neural (CNN) classifier, simple tokenization was often enough, and heavier steps such as lemmatization did not consistently help. Saif et al. (2014) showed that removing words with a classic pre-compiled stopword list reduced Twitter sentiment-classification performance.

#### Real-life example. 

Consider a product-review sentiment system for an online shop. A review says: "Not good. NOT worth $80". A "maximal" cleaning pipeline (lowercase → delete non-letters → remove stopwords) produces good worth. The negation (not), the emphasis (capitals), the price and the angry emoji are all gone, and the remaining words look positive. The shop would wrongly count an angry customer as a happy one. Part B (task 10) reproduces exactly this effect on T2.

### 2. Text normalization

Normalization maps different written variants of "the same" text to one canonical form (Manning, Raghavan & Schütze, 2008, §2.2.3).

#### Step What it does Example
Lowercasing (case folding)	Converts all letters to one case	Apple → apple
Whitespace normalization	Collapses runs of spaces/tabs/newlines to one space and trims ends	"  a   b " → "a b"
Unicode normalization	Converts canonically or compatibly equivalent code-point sequences to one form (NFC, NFD, NFKC, NFKD)	é written as e + combining accent (2 code points) → single é (NFC); ﬁ ligature → fi (NFKC); full-width Ａ → A (NFKC)
Other	Expanding contractions, unifying quotes/dashes, number normalization	’ → '; don't → do not

Unicode normalization matters because two strings that look identical can be different byte sequences. The Unicode Standard Annex #15 defines four normalization forms: NFC/NFD handle canonical equivalence (same character, different encoding), while NFKC/NFKD also fold compatibility variants such as ligatures, superscripts and full-width letters (Unicode Consortium, 2024). In Python this is unicodedata.normalize("NFKC", s).

Where normalization helps. In a search engine, a user typing cafe or CAFE should find documents containing Café. Lowercasing plus Unicode normalization (and, optionally, accent stripping) lets all variants match the same index entry, increasing recall (Manning et al., 2008). In T4 (Part B task 4), The and the are counted as two different types; lowercasing would merge them into one.

#### Where normalization removes useful information.

Lowercasing: US (the country) becomes us (pronoun); Apple (company) becomes apple (fruit). Capitalization is a strong feature for named-entity recognition, and ALL-CAPS is a known signal of shouting or spam. In T1, lowercasing deletes the fact that 44% of the letters were capitals (see task 1 output).
NFKC: compatibility folding turns x² into x2 and ½ into 1⁄2, which changes meaning in scientific or recipe text.

### 3. Word-level, subword and character-level tokenization

Tokenization splits text into units (tokens) that the model treats as atomic symbols. The vocabulary is the set of distinct token types a model knows; vocabulary size |V| is the number of those types. In neural models each type needs one row in the embedding matrix and one output in the softmax, so |V| directly affects model size and speed.

	Word-level	Subword (BPE, WordPiece, Unigram)	Character / byte-level
Unit	Whole words separated by spaces/punctuation	Frequent words stay whole; rare words are split into frequent pieces (unhappiness → un, happi, ness)	Single characters or UTF-8 bytes
Typical vocabulary size	Very large (hundreds of thousands, grows with corpus)	Fixed by design, typically ~30k–250k	Very small (~100–300 for bytes/ASCII; larger for full Unicode characters)
Out-of-vocabulary (OOV) words	Common problem → mapped to <UNK>	Almost none: any word can be built from pieces	None
Sequence length	Shortest	Medium	Longest (many more tokens per sentence)
Weakness	Cannot handle new words, typos, rich morphology	Splits may not match real morphemes; tokenizer must be trained	Long sequences are slow; model must learn word structure itself
Word-level tokenizers (e.g., NLTK word_tokenize, based on Penn Treebank rules) are simple and interpretable, but every inflected form, typo and new name becomes a new type. Following Heaps' law, vocabulary keeps growing as the corpus grows (Manning et al., 2008, §5.1.1).
Subword tokenizers solve OOV. Byte-Pair Encoding (BPE) starts from characters and repeatedly merges the most frequent adjacent pair until a target number of merges is reached (Sennrich, Haddow & Birch, 2016). SentencePiece treats the input as a raw stream (including spaces) so it works for languages without spaces, such as Japanese (Kudo & Richardson, 2018). The designer chooses |V| (the number of merges).
Character/byte-level models (e.g., ByT5) need no learned vocabulary, are robust to spelling noise, and work for any language, but sequences become several times longer, which increases computation (Xue et al., 2022).

How the tokenizer affects vocabulary size. Word-level: |V| is determined by the data and is large and open-ended. Subword: |V| is a hyperparameter — small values give longer sequences, large values give shorter sequences but bigger embeddings. Character-level: |V| is tiny and closed. Even small choices matter: in Part B, split() gave 14 tokens for T2 but word_tokenize gave 16, because it separates do + n't and price + .. In T4, The and the count as two vocabulary types until the text is lowercased.

### 4. Stopwords

Definition. Stopwords are very frequent words that carry little topical content on their own — mostly function words such as articles, prepositions, pronouns and auxiliaries (the, of, is, he). The idea comes from information retrieval, where such words appear in almost every document and do not help distinguish one document from another (Manning et al., 2008, §2.2.2). NLTK's English list in our environment contains 198 words.

#### Benefits of removal

Smaller feature space and index; faster training and retrieval.
In bag-of-words topic classification, content words (jacket, price, delivery) remain more visible to the model.

#### Risks of removal

Negation is lost. NLTK's list contains not, no, nor and contractions such as don't, isn't. Removing them can reverse sentiment (Part B task 5: not happy → happy).
Phrases and names break: "To be or not to be" consists only of stopwords; The Who becomes who.
Syntax and modern models: parsers, machine translation and transformer models use function words to understand structure; removing them usually hurts these systems.
Stylometry/authorship: function-word frequencies are among the best features for identifying authors.

Saif et al. (2014) tested six stopword identification methods on six Twitter datasets with two classifiers. They found that classic pre-compiled lists negatively impacted sentiment classification, while a dynamic list built by removing words occurring only once kept performance high and reduced data sparsity.

#### Why the list depends on the task and language.

Task: for topic classification not is noise; for sentiment it is essential; for spam detection words like you, now and free may be important features even though they are frequent.
Domain: in a corpus of medical records, patient appears everywhere and behaves like a stopword; in a general corpus it does not.
Language: each language has its own function words, and morphology changes what a "word" is. In Turkish or Finnish, negation and possession are suffixes inside a word, so an English-style list does not apply. In Chinese, words are not separated by spaces, so stopword removal depends on segmentation first. A list must therefore be built (or adapted) for each language and task, ideally from corpus frequency plus task knowledge.

### 5. Stemming vs. lemmatization

Both reduce related word forms to a shared base so that connect, connected and connecting count as one feature.

#### Stemming	Lemmatization
Method	Rule-based suffix stripping, no dictionary	Dictionary (lexicon) + morphological analysis, often needs part of speech
Output	A stem, which may not be a real word (univers, fli)	A lemma, the dictionary form (fly, wolf, run)
Irregular forms	Not handled (ran stays ran)	Handled (ran → run, better → good with POS)
Speed	Very fast	Slower
Errors	Over-stemming (university/universal → univers) and under-stemming	Wrong or missing POS gives wrong lemma

#### Stemming algorithm – the Porter stemmer. 
Martin Porter (1980) designed an English suffix-stripping algorithm that applies five ordered steps of rewrite rules. Each word is viewed as a pattern of consonant groups (C) and vowel groups (V), [C](VC)^m[V], where m is the "measure". A rule such as (m>0) ATIONAL → ATE only fires if enough of the word would remain, which prevents very short stems. Step 1 handles plurals and -ed/-ing; later steps remove derivational suffixes such as -ization, -ful, -ness. Porter later wrote the improved Snowball ("Porter2") version and a framework for stemmers in other languages. The Lancaster (Paice/Husk) stemmer is a more aggressive iterative stemmer (Paice, 1990), which our output confirms (organization → org, better → bet).

#### Lemmatization approach – WordNet lemmatization. 
NLTK's WordNetLemmatizer uses the WordNet lexical database (Miller, 1995). It applies WordNet's morphy procedure: it tries a list of detachment rules for the given POS (e.g., for nouns -ies → -y, -ves → -f; for verbs -ed → ""), checks whether the result is a real WordNet entry, and also checks an exception list for irregular forms (ran → run, better → good). If no POS is given it assumes noun, which is why many verbs are not changed without a hint (Part B task 7). Statistical or neural lemmatizers (e.g., in spaCy or Stanza) instead predict the POS from context first.

#### When to prefer each

Stemming: information retrieval and search, or large-scale bag-of-words classification, where speed and higher recall matter more than readable output, and a few wrong merges are acceptable.
Lemmatization: when output must be real words (keyword extraction, chatbots, showing results to humans), for morphologically rich languages, or when precision matters (e.g., you do not want university and universal merged). Uysal and Günal (2014) found that the benefit of stemming depends on the domain and language, so it should be tested rather than assumed.

### 6. Regular expressions in text cleaning

A regular expression (regex) is a pattern language that describes a set of strings. In text cleaning, regexes are used to find (e.g., all emails), delete (e.g., HTML tags) or replace (e.g., numbers with a placeholder) parts of the text, and to drive rule-based tokenizers (Jurafsky & Martin, 2025, ch. 2; Friedl, 2006). Python provides them through the re module (re.sub, re.findall, re.split).

#### Example patterns

Pattern	What it matches	Use
https?://\S+	http:// or https:// followed by one or more non-space characters	Replace URLs with URL in social-media text
\b0\d{9}\b	A whole "word" of exactly 10 digits starting with 0 (e.g., 0712345678)	Replace local phone numbers with PHONE_NUMBER (used in Part B)
\s+	One or more whitespace characters	Collapse to a single space (Part B task 2)
[£$€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?	A currency symbol followed by an amount with optional thousands commas and decimals (£1,000, $9.99)	Replace money amounts with MONEY

#### How an overly broad pattern damages text. 
The pattern [^a-z\s] ("delete everything that is not a lowercase letter or space") looks like a quick way to remove punctuation, but it also deletes:

digits and currency (£1,000 and the phone number disappear completely — Part B task 3 shows 21 characters deleted from T1, leaving won a  gift card call  now),
apostrophes, turning don't into dont and it's into its (a different word),
every non-ASCII letter — café → caf, and Somali, Arabic or Chinese text would be erased entirely,
emoji, which carry sentiment.

Another classic case is the "Scunthorpe problem": a profanity filter using a substring pattern without word boundaries (\b) blocks innocent words that contain the bad string. Good practice is to make patterns as specific as possible, use word boundaries, prefer replacing with a placeholder over deleting, and test the pattern on real samples.

### 7. Case study – Uysal & Günal (2014), The impact of preprocessing on text classification

Source: Uysal, A. K., & Günal, S. (2014). Information Processing & Management, 50(1), 104–112.

#### Task. 
Text classification in two domains — e-mail (spam vs. legitimate) and news (topic categories) — and in two languages — English and Turkish (four datasets in total). Classification used a Support Vector Machine, evaluated with Micro-F1 across different numbers of selected features.

#### Preprocessing options compared. 
Four common steps — tokenization, stopword removal, lowercase conversion and stemming — were each switched on or off, and all possible combinations were evaluated, so the authors could see interactions rather than single steps in isolation.

#### Results.

There was no single best pipeline. The main conclusion is that choosing the right combination of steps, rather than turning all of them on or all off, can significantly improve accuracy, and the best combination depends on the domain and the language.
Lowercase conversion was generally beneficial in their experiments.
The effect of stopword removal and stemming varied: in some domain/language combinations they helped, in others they reduced accuracy. Stemming behaves differently for an agglutinative language like Turkish (where one root can carry many suffixes) than for English.
Preprocessing also changes the dimensionality of the feature space, so its effect interacts with feature selection.

#### Relevance to this assignment. 
The study supports the main lesson of Part B: preprocessing choices are task- and language-dependent and should be tested experimentally. The newer work of Camacho-Collados and Pilehvar (2018) reaches a similar conclusion for neural classifiers, finding high variability across preprocessing techniques and advising consistency between the preprocessing of training data, embeddings and evaluation data.

## Part B – Practical

All results below are copied from output.txt, produced by running python preprocessing_practice.py > output.txt (NLTK 3.10.4).

### Task 1 – Lowercasing

Before: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW! After: urgent!!! you have won a £1,000 gift card. call 0712345678 now!

What a spam detector could lose: the all-caps words URGENT, WON and NOW lose their "shouting" form, and the uppercase-letter ratio drops from 0.44 to 0.0. Excessive capitalization is a typical spam/urgency signal, so after lowercasing the detector can no longer tell NOW! from a normal now. If lowercasing is used, a feature such as "caps ratio" should be computed before it.

### Task 2 – Whitespace

Before: '   The   delivery   was   fast,   and the  packaging   was   great.' (67 characters) After: 'The delivery was fast, and the packaging was great.' (51 characters)

### Task 3 – Regex cleaning

3a. [^a-z\s] on lowercased T1: urgent you have won a  gift card call  now

Deleted (21 characters): ! ! ! (the three exclamation marks), £, 1, ,, 0 0 0 (the money amount £1,000), . (the full stop), all ten digits of 0712345678, and the final !.

3b. Replace instead of delete: URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!

Rules used: \b0\d{9}\b → PHONE_NUMBER, £[\d,]+ → MONEY, !+ → !.

Answer: The delete pattern removed the two strongest spam cues — the prize amount and the phone number — and left two double spaces where they had been. The replacement rules keep the fact that a money amount and a phone number were present while removing their specific values, which a classifier can learn from (any MONEY + PHONE_NUMBER + won is suspicious). Replacing is safer than deleting because it reduces vocabulary without losing the signal.

### Task 4 – Tokenization
Text	Tokenizer	Tokens	Count
T2	split()	I, don't, think, this, jacket, is, worth, the, price., Not, happy, at, all, 😞	14
T2	word_tokenize	I, do, n't, think, this, jacket, is, worth, the, price, ., Not, happy, at, all, 😞	16
T4	word_tokenize	The, delivery, was, fast, ",", and, the, packaging, was, great, .	11

T4 vocabulary size: 10 unique tokens (The and the are different types; was appears twice). split() keeps punctuation attached (price.) and keeps don't whole; word_tokenize separates punctuation and splits the contraction into do + n't.

### Task 5 – Stopwords and negation

Lowercased T2 tokens: i, do, n't, think, this, jacket, is, worth, the, price, ., not, happy, at, all, 😞

List	Result
NLTK default	n't, think, jacket, worth, price, ., happy, 😞
Own list (keeps not, n't, no, never, nor and all ...n't forms)	n't, think, jacket, worth, price, ., not, happy, 😞

Answer: With NLTK's default list, not is removed, so the second sentence "Not happy at all" becomes just happy — the meaning is reversed from negative to positive. Interestingly, n't survives in both versions because NLTK's list contains don't but not the separate token n't that word_tokenize produces, so the list and the tokenizer do not match. With the negation-keeping list, not happy is preserved and the complaint stays clear.

### Task 6 – Stemming
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

Answer: All three stemmers merge university and universal into univers, although they mean different things (over-stemming). They also map organization to organ (Porter/Snowball), which is a different real word (a body part or musical instrument). Stems that are not real words include fli, wolv and univers (Porter stems checked against WordNet in the script); Lancaster also produces the clipped stem org. Lancaster is the most aggressive (better → bet), while none of them can handle the irregular form ran.

### Task 7 – Lemmatization
Word	Default (noun)	POS hint	With hint	Changed only with hint?
connection	connection	n	connection	no
connected	connected	v	connect	yes
connecting	connecting	v	connect	yes
organization	organization	n	organization	no
organized	organized	v	organize	yes
university	university	n	university	no
universal	universal	a	universal	no
flies	fly	v	fly	no
wolves	wolf	n	wolf	no
better	better	a	good	yes
ran	ran	v	run	yes

Answer: connected, connecting, organized, better and ran changed only when the correct POS was given, because the default treats every word as a noun and these are verb/adjective forms. flies and wolves were already correct by default since fly and wolf exist as nouns. Unlike the stemmers, the lemmatizer always returns real words and keeps university and universal separate.

### Task 8 – Sentence splitting

#### sent_tokenize output:

Mr. Smith arrived in London on Friday.
He met the Apple team at 9 a.m.

Answer: We got 2 sentences, which is correct. The Punkt model knows Mr. is an abbreviation, so it did not split after it, and a.m. at the end correctly closes the second sentence. A naive split on every .  would have wrongly separated Mr. from Smith.

### Task 9 – POS tagging and NER

POS tags: Mr./NNP Smith/NNP arrived/VBD in/IN London/NNP on/IN Friday/NNP ./. He/PRP met/VBD the/DT Apple/NNP team/NN at/IN 9/CD a.m/NN ./.

Entity (from ne_chunk)	Label	Correct?
Mr.	PERSON	Partly — it is a title, and it should be one entity together with Smith
Smith	PERSON	Yes (but split from Mr.)
London	GPE	Yes

#### What the model missed:

Expected entity	Expected label	Found?
Mr. Smith	PERSON	Found, but as two separate PERSON chunks
London	GPE (place)	Yes
Friday	DATE	Missed
Apple	ORGANIZATION	Missed
9 a.m.	TIME	Missed

NLTK's ne_chunk only uses the labels PERSON, ORGANIZATION, GPE, LOCATION, FACILITY and GSP, so it cannot label dates or times at all. It also missed Apple as an organization, even though the POS tagger correctly tagged it as a proper noun (NNP). The tokenizer also split a.m. into a.m + ..

### Task 10 – Full pipeline (T2)
Stage	NLTK default stopwords	Negation-keeping stopwords
0. Original	I don't think this jacket is worth the price. Not happy at all 😞	same
1. Lowercase	i don't think this jacket is worth the price. not happy at all 😞	same
2. Regex ([^a-z\s'] removed)	i don't think this jacket is worth the price not happy at all	same
3. Tokenize	i, do, n't, think, this, jacket, is, worth, the, price, not, happy, at, all	same
4. Stopwords	n't, think, jacket, worth, price, happy	n't, think, jacket, worth, price, not, happy
5. Lemmatize	n't think jacket worth price happy	n't think jacket worth price not happy

Answer: The negation-keeping version keeps the complaint: it ends with not happy, while the default version ends with just happy, which looks positive. The default list keeps only the first n't (by accident, as explained in task 5) and loses the not of the second sentence. Both versions also lost the 😞 emoji at the regex stage, which was itself a clear negative signal — a better pipeline would replace it with a token such as EMOJI_SAD.

#### Which preprocessing steps for a spam detector?

For a spam detector I would use whitespace normalization, word tokenization and, above all, regex replacement with placeholders: in task 3b, PHONE_NUMBER, MONEY and a single ! keep the strongest spam cues of T1 while reducing the vocabulary. Lowercasing is useful for matching words such as won/WON, but only after extracting a capitalization feature, since task 1 shows it erases the 0.44 caps ratio and the shouting words URGENT, WON, NOW. I would avoid the broad delete pattern [^a-z\s] (task 3a deleted 21 characters, including the prize amount and the phone number) and avoid aggressive stopword removal, because spam relies on frequent words like you, now and call. Stemming or lemmatization adds little here — spam is detected more by patterns (money, phone numbers, urgency, punctuation) than by word roots — and Lancaster-style over-stemming (organization → org) could even merge unrelated words.

## References
Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media. https://www.nltk.org/book/
Camacho-Collados, J., & Pilehvar, M. T. (2018). On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP (pp. 40–46). ACL. https://doi.org/10.18653/v1/W18-5406
Friedl, J. E. F. (2006). Mastering Regular Expressions (3rd ed.). O'Reilly Media.
Jurafsky, D., & Martin, J. H. (2025). Speech and Language Processing (3rd ed. draft), Chapter 2: Regular Expressions, Tokenization, Edit Distance. https://web.stanford.edu/~jurafsky/slp3/
Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of EMNLP 2018: System Demonstrations (pp. 66–71). https://doi.org/10.18653/v1/D18-2012
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. https://nlp.stanford.edu/IR-book/
Miller, G. A. (1995). WordNet: A lexical database for English. Communications of the ACM, 38(11), 39–41. https://doi.org/10.1145/219717.219748
Paice, C. D. (1990). Another stemmer. ACM SIGIR Forum, 24(3), 56–61. https://doi.org/10.1145/101306.101310
Porter, M. F. (1980). An algorithm for suffix stripping. Program, 14(3), 130–137. https://doi.org/10.1108/eb046814
Saif, H., Fernández, M., He, Y., & Alani, H. (2014). On stopwords, filtering and data sparsity for sentiment analysis of Twitter. In Proceedings of LREC 2014 (pp. 810–817). https://aclanthology.org/L14-1265/
Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In Proceedings of ACL 2016 (pp. 1715–1725). https://doi.org/10.18653/v1/P16-1162
Unicode Consortium. (2024). Unicode Standard Annex #15: Unicode Normalization Forms. https://unicode.org/reports/tr15/
Uysal, A. K., & Günal, S. (2014). The impact of preprocessing on text classification. Information Processing & Management, 50(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006
Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models. Transactions of the ACL, 10, 291–306. https://doi.org/10.1162/tacl_a_00461