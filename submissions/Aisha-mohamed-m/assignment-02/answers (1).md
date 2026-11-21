# Assignment 2 – Preprocessing

## Part A – Research

### 1. What is text preprocessing?

**Text preprocessing** is the set of operations used to transform raw text into a form that an NLP system can process more consistently. Depending on the application, preprocessing can include normalization, tokenization, removal or transformation of unwanted material, stopword handling, stemming, lemmatization, and other task-specific transformations. Research shows that preprocessing is not merely a mechanical cleaning stage: it can change the distribution and information content of the original data and therefore affect downstream model performance. citeturn122158search0turn122158search1

NLP systems need preprocessing because raw language contains variation that can make learning or matching harder. For example, `Hello`, `hello`, and `HELLO` may be treated as different tokens by a case-sensitive system, while inconsistent whitespace or Unicode representations can create unnecessary variation. However, **more cleaning is not automatically better**. A cleaning step can remove information that is useful for the task. In a spam detector, for example, an SMS such as `URGENT!!! YOU WON £1,000` contains capitalization, repeated exclamation marks, and monetary information that may be predictive of spam. Deleting all punctuation, numbers, and capitalization could therefore remove useful evidence rather than simply removing noise.

### 2. Text normalization

**Text normalization** makes equivalent or near-equivalent text representations more consistent. Common examples include lowercasing, collapsing repeated whitespace, and Unicode normalization. Unicode defines four standard normalization forms: NFC and NFD for canonical equivalence, and NFKC and NFKD for compatibility equivalence. NFC is generally appropriate for ordinary text, while compatibility normalization can remove distinctions that may be meaningful in some contexts. citeturn856057search0turn856057search10

A useful example is Unicode normalization when the same accented word can be represented either as a precomposed character or as a base character plus a combining mark. Normalization makes these equivalent representations consistent, which helps comparison and search. On the other hand, lowercasing can remove useful information: in spam detection, `URGENT` or `FREE` in all capitals may carry a stronger signal than ordinary lowercase wording. Similarly, aggressive compatibility normalization can remove distinctions that a specialized application intentionally needs to preserve.

### 3. Word-level, subword, and character-level tokenization

| Tokenization level | Basic idea | Main advantage | Main limitation |
|---|---|---|---|
| **Word-level** | Splits text into words or word-like units | Tokens are linguistically interpretable and often short sequences | Large vocabulary and many out-of-vocabulary words |
| **Subword** | Splits words into reusable pieces such as prefixes, stems, and suffixes | Handles rare and unseen words while keeping useful word structure | A word may require several tokens |
| **Character-level** | Treats individual characters as tokens | Very small vocabulary and almost no unknown-word problem | Produces longer sequences and requires the model to learn word structure from characters |

**Vocabulary size** is the number of distinct token types available to a tokenizer/model. A word-level vocabulary may contain tens of thousands or more complete words. A subword vocabulary is usually deliberately limited to a manageable set of pieces. A character vocabulary is much smaller because it mainly contains the characters used by the language, although sequence lengths become much longer.

Subword tokenization became an important solution to the open-vocabulary problem. Sennrich, Haddow, and Birch showed that subword units allow neural translation systems to represent rare and previously unseen words through smaller reusable pieces. In Byte-Pair Encoding (BPE), frequent symbol pairs are repeatedly merged until a chosen vocabulary size is reached. citeturn122158search2turn122158search53

### 4. Stopwords

**Stopwords** are very frequent words that are often considered to carry relatively little topical information, such as `the`, `is`, `of`, and `to`. A stopword list is therefore a task-dependent resource rather than a universal linguistic truth.

Removing stopwords can reduce the number of features and sparsity in some traditional NLP systems. However, it can also remove grammatical information that is important for meaning. This is especially clear in sentiment analysis: `happy` and `not happy` have opposite meanings, so removing `not` can seriously damage the representation. Research on Twitter sentiment classification found that using precompiled stopword lists could actually reduce classification performance, while a more data-driven filtering strategy performed better in the studied settings. citeturn122158search5

The appropriate stopword list depends on both the **task** and the **language**. Search or topic classification may tolerate removing some high-frequency function words, while sentiment, question answering, authorship analysis, or negation-sensitive tasks may need them. Different languages also have different grammatical structures and frequency distributions, so an English list should not automatically be transferred to Somali, Arabic, or another language.

### 5. Stemming vs. lemmatization

**Stemming** usually applies heuristic rules to remove or modify word endings. It does not necessarily produce a valid dictionary word. The classic **Porter stemmer**, introduced by Martin Porter in 1980, is a suffix-stripping algorithm designed partly for information retrieval. It applies ordered rules to reduce related forms toward a common stem. citeturn386209search36turn386209search0

**Lemmatization** is more linguistically informed. Instead of simply cutting endings, it attempts to return a dictionary or lexical base form (lemma), normally using lexical resources and, when available, part-of-speech information. NLTK's `WordNetLemmatizer`, for example, uses WordNet's morphological information and accepts a POS argument such as noun, verb, or adjective. citeturn856057search6

Stemming is often preferred when speed and simple conflation of related word forms are more important than producing valid words, such as some information-retrieval pipelines. Lemmatization is preferable when interpretable base forms and grammatical distinctions matter, although it is usually more resource-dependent. The important point is that neither method should be applied automatically: the best choice depends on the downstream task.

### 6. Regular expressions in text cleaning

Regular expressions (regex) describe patterns that can be searched for, replaced, or removed from text. They are useful for predictable structures such as phone numbers, dates, repeated punctuation, URLs, or unwanted characters. Python's `re` module supports character classes, repetition operators, groups, anchors, and replacement operations. citeturn386209search8turn386209search9

Two examples are:

| Pattern | What it matches | Example |
|---|---|---|
| `\b\d{10}\b` | A standalone sequence of exactly 10 digits | `0712345678` |
| `£[\d,]+` | A pound sign followed by one or more digits/commas | `£1,000` |

An overly broad pattern can destroy information. For example, applying `[^a-z\s]` to a spam message deletes digits, punctuation, the pound sign, and the emoji. That may remove phone-number, money, urgency, or emotional signals that a classifier could have used.

### 7. Case study: preprocessing choices and NLP performance

Camacho-Collados and Pilehvar (2018) conducted an evaluation of preprocessing decisions for neural text classification. They compared choices including tokenization, lemmatization, lowercasing, and multiword grouping across standard text-classification and sentiment-analysis benchmarks. Their results showed that simple tokenization was generally a strong baseline, but preprocessing choices still produced meaningful variation in model performance; therefore, preprocessing should be treated as an experimental part of the pipeline rather than assumed to be harmless. citeturn122158search0

A second relevant study by Babanejad et al. (2020) examined preprocessing for word-vector-based affective tasks such as sentiment, emotion, and sarcasm analysis. The study emphasizes that preprocessing decisions can affect both the representation-learning stage and downstream affective tasks, reinforcing the idea that there is no universally optimal preprocessing recipe. citeturn122158search3

## References

1. Babanejad, N., Agrawal, A., An, A., & Papagelis, M. (2020). *A Comprehensive Analysis of Preprocessing for Word Representation Learning in Affective Tasks*. Proceedings of ACL. https://aclanthology.org/2020.acl-main.514/
2. Camacho-Collados, J., & Pilehvar, M. T. (2018). *On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis*. BlackboxNLP. https://aclanthology.org/W18-5406/
3. Porter, M. F. (1980). *An algorithm for suffix stripping*. Program, 14(3), 130–137. https://doi.org/10.1108/eb046814
4. Saif, H., Fernandez, M., He, Y., & Alani, H. (2014). *On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter*. LREC. https://aclanthology.org/L14-1265/
5. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. https://aclanthology.org/P16-1162/
6. Unicode Consortium. *Unicode Normalization Forms (UAX #15)*. https://unicode.org/reports/tr15/
7. NLTK Project. *NLTK documentation: tokenization, stemming, POS tagging, and WordNet*. https://www.nltk.org/

---

# Part B – Practical Report

> **Important:** The practical results below must be copied from the output produced by `preprocessing_practice.py`. The script is deliberately written to print every required intermediate result, so the submitted `answers.md` should use the exact output from your own environment rather than hand-typed results.

## 1. Lowercasing

Lowercasing makes `URGENT`, `WON`, and `NOW` look like their lowercase equivalents and can reduce vocabulary variation. However, the output also shows why this can be risky for spam detection: capitalization itself can be a useful signal because excessive ALL-CAPS text often indicates emphasis or promotional/spam style.

## 3. Regex cleaning

The pattern `[^a-z\s]` keeps only lowercase English letters and whitespace, so it deletes the digits, currency symbol, punctuation, and other non-letter characters from T1. This is potentially harmful because the deleted characters include the phone number, money amount, repeated exclamation marks, and sentence punctuation; the replacement version is safer because it converts important structured information into labels such as `PHONE_NUMBER` and `MONEY`.

**Deleted-character list and replacement output:** copy these directly from `output.txt`.

## 5. Stopwords and negation

Default stopword removal can delete words such as `not`, which changes the meaning of the complaint. The custom list keeps `not` and `n't`, so the resulting tokens preserve the negative polarity in `I don't think...` and `Not happy...`.

**Exact token lists:** copy them directly from `output.txt`.

## 6. Stemming

Porter, Lancaster, and Snowball are rule-based stemmers, but they do not necessarily produce the same result. The important observation is that a stemmer may merge related words successfully while also producing non-words such as shortened forms; therefore, stemming is a normalization strategy rather than a guarantee of valid dictionary lemmas.

**Stemming results:**

| Word | Porter | Lancaster | Snowball | Regexp |
|---|---|---|---|---|
| connection | connect | connect | connect | connect |
| connected | connect | connect | connect | connect |
| connecting | connect | connect | connect | connect |
| organization | organ | org | organ | organiz |
| organized | organ | org | organ | organiz |
| university | univers | univers | univers | university |
| universal | univers | univers | univers | universal |
| flies | fli | fli | fli | flie |
| wolves | wolv | wolv | wolv | wolve |
| better | better | bet | better | better |
| ran | ran | ran | ran | ran |

Porter, Lancaster, and Snowball merge several related forms, but Lancaster is more aggressive for some words. The simple Regexp stemmer follows only the suffix rules defined in the script and can produce non-words such as `flie` and `wolve`. For example, `connection`, `connected`, and `connecting` are all reduced to `connect` by the three standard stemmers.

## 7. Lemmatization

With the default setting, NLTK's WordNet lemmatizer treats the words as nouns, so some verb and adjective forms are not reduced correctly. Providing the correct POS hint gives WordNet the grammatical information needed to produce more appropriate lemmas, such as verb forms becoming their base verbs and `better` being interpreted as an adjective.

**Lemmatization results:**

| Word | Default | Verb (`v`) | Noun (`n`) | Adjective (`a`) |
|---|---|---|---|---|
| connection | connection | connection | connection | connection |
| connected | connected | connect | connected | connected |
| connecting | connecting | connect | connecting | connecting |
| organization | organization | organization | organization | organization |
| organized | organized | organize | organized | organized |
| university | university | university | university | university |
| universal | universal | universal | universal | universal |
| flies | fly | fly | flies | flies |
| wolves | wolf | wolves | wolf | wolves |
| better | better | better | better | good |
| ran | ran | run | ran | ran |

The changed examples show why POS information matters. For example, `connected`, `connecting`, and `organized` become `connect`, `connect`, and `organize` when treated as verbs. `better` becomes `good` when treated as an adjective, while `ran` becomes `run` with the verb POS hint. Additional examples from the script include `running -> run` and `studies -> study`.

## 8. Sentence splitting

`sent_tokenize` uses NLTK's trained Punkt sentence tokenizer. For T3, the expected structure is two sentences, and the abbreviations `Mr.` and `a.m.` should not normally be treated as sentence boundaries by the tokenizer.

**Sentence results:**

1. `Mr. Smith arrived in London on Friday.`
2. `He met the Apple team at 9 a.m.`

**Sentence count: 2**

The tokenizer correctly avoided splitting the text at `Mr.` and `a.m.`, because these are abbreviations rather than sentence boundaries.

## 9. POS Tagging and Named Entity Recognition

POS tagging assigns grammatical categories to tokens, while Named Entity Recognition (NER) identifies entities such as people, places, and organizations. For T3, NLTK produced the following POS tags:

| Token | POS Tag |
|---|---|
| Mr. | NNP |
| Smith | NNP |
| arrived | VBD |
| in | IN |
| London | NNP |
| on | IN |
| Friday | NNP |
| . | . |
| He | PRP |
| met | VBD |
| the | DT |
| Apple | NNP |
| team | NN |
| at | IN |
| 9 | CD |
| a.m | NN |
| . | . |

### Named Entity Results

| Entity | Expected Label | NLTK Result | Correct? |
|---|---|---|---|
| Mr. Smith | PERSON | `Mr.` and `Smith` detected separately as PERSON | Partially |
| London | GPE | GPE | Yes |
| Apple | ORGANIZATION | Not recognized as an organization | No |

### Entities Found

1. **Mr. Smith – PERSON:** NLTK recognized `Mr.` and `Smith` as PERSON entities, but it did not combine them into the complete name **Mr. Smith**.
2. **London – GPE:** NLTK correctly identified London as a geographical/political entity.

### Missed Entity

**Apple** was the main entity missed by NLTK. In the sentence, "Apple team" refers to **Apple, the company**, so the expected entity type is **ORGANIZATION**. However, NLTK tagged `Apple` as `NNP` and did not classify it as an organization.

### Conclusion

This example shows that NER is not always perfect. NLTK correctly identified London and parts of the person name, but it missed Apple as an organization and did not combine `Mr.` and `Smith` into one PERSON entity. Therefore, NER results should be checked using the context and meaning of the sentence rather than assuming that every prediction is correct.

## 10. Full pipeline

The full pipeline demonstrates why preprocessing order and choices matter. Lowercasing and regex cleaning reduce surface variation, tokenization creates units for later processing, stopword removal reduces common words, and lemmatization attempts to return base forms. The negation-preserving version should retain the negative word(s), so it better preserves the customer's complaint than the default-stopword version.

**Copy all five stages for both runs from `output.txt`.**

## Spam-detector recommendation

For a spam detector, I would use **moderate normalization rather than aggressive cleaning**. Lowercasing can be useful, but I would consider preserving an additional capitalization feature because the T1 example contains strong ALL-CAPS emphasis. I would normalize whitespace and use targeted regex rules to replace phone numbers and money amounts with informative placeholders instead of deleting them; I would also preserve useful punctuation such as repeated exclamation marks, or convert it into a feature. I would avoid blindly removing all stopwords because negation can change meaning, and I would avoid the broad `[^a-z\s]` rule because the practical output demonstrates that it deletes numbers, currency, punctuation, and other potentially predictive signals. Stemming or lemmatization should also be tested empirically rather than assumed to improve spam classification.
