# Assignment 2: Preprocessing

## Part A: Research

### 1. What is Text Preprocessing?

**Text preprocessing** is the process of transforming raw text into a form that is more suitable for an NLP system. It can include operations such as normalization, tokenization, removing or replacing certain characters, stemming, and lemmatization. The purpose is not simply to make text "clean", but to reduce unwanted variation while preserving information that is useful for the specific NLP task.

NLP systems often need preprocessing because the same meaning can appear in many different written forms. For example, a customer-support system might receive:

> "I can't access my account!!!"

and

> "I cannot access my account."

A basic text representation may treat many of their tokens as different. Normalization and tokenization can reduce some of this variation and make the data easier for a model to process.

However, **more cleaning does not always mean better results**. Some text characteristics contain useful information. For example, in a sentiment-analysis system, removing punctuation or capitalization may remove signals that help distinguish an ordinary statement from a strongly emotional one. Research on neural text classification has also found that simple tokenization can sometimes perform as well as or better than more complex preprocessing such as lemmatization and multiword grouping.

Therefore, preprocessing should be treated as a set of decisions that should be tested against the task, rather than as a fixed list of cleaning steps.

---

## 2. Text Normalization

**Text normalization** means converting different forms of text into a more consistent representation. Common forms include lowercasing, whitespace normalization, and Unicode normalization.

### Lowercasing

Lowercasing converts uppercase letters to lowercase. For example:

```text
Machine Learning
```

becomes:

```text
machine learning
```

This can reduce the number of different word forms in a vocabulary. It is especially useful when capitalization does not carry important information.

However, lowercasing can also remove useful information. For example, **"US"** and **"us"** can have different meanings. In named entity recognition, capitalization can also provide useful clues about names and organizations. A study of neural text classification found that lowercasing can reduce vocabulary size and sparsity, but can also increase ambiguity.

### Whitespace normalization

Whitespace normalization makes spaces, tabs, and line breaks more consistent. For example:

```text
The    system    is   working.
```

can become:

```text
The system is working.
```

This prevents accidental spacing differences from creating unnecessary variation in the input.

### Unicode normalization

Unicode allows some characters to be represented in more than one equivalent way. For example, a character such as **é** can be represented as a single precomposed character or as `e` plus a combining accent. Unicode normalization makes equivalent representations consistent. The Unicode Standard defines four main normalization forms: NFC, NFD, NFKC, and NFKD.

A situation where normalization helps is **text matching**. If two visually identical strings have different internal representations, normalization can make them compare as equal.

However, normalization can also remove useful information. Unicode notes that NFKC and NFKD remove compatibility distinctions, so they can lose information and should therefore be used carefully. Recent NLP research also shows that normalization can remove signals that are useful for some language-classification tasks.

The general principle is therefore to normalize only the differences that are irrelevant to the task.

---

## 3. Word-Level, Subword, and Character-Level Tokenization

**Tokenization** divides text into smaller units called tokens. The choice of tokenization determines what the NLP model treats as an individual unit.

| Tokenization | Description | Main advantage | Main limitation |
|---|---|---|---|
| **Word-level** | Uses whole words as tokens | Simple and produces relatively few tokens per sentence | Large vocabulary and problems with unseen words |
| **Subword-level** | Divides uncommon words into smaller learned pieces | Handles rare and unseen words better | Requires a trained tokenizer |
| **Character-level** | Uses individual characters as tokens | Very small vocabulary and can represent any word | Produces long sequences and loses some word-level structure |

### Word-level tokenization

A sentence such as:

```text
The student studies biology.
```

could become:

```text
The | student | studies | biology | .
```

The vocabulary can become very large because every different word form may become a separate entry. For example, `study`, `studies`, and `studying` can all be different tokens.

### Character-level tokenization

The same sentence can be represented character by character:

```text
T | h | e |   | s | t | u | d | e | n | t ...
```

The vocabulary is much smaller because it mainly consists of letters, numbers, punctuation, and other characters. However, the resulting sequences are much longer.

### Subword tokenization

Subword tokenization lies between the two. A word can be represented using smaller units that have been learned from a large corpus. One well-known approach is **Byte Pair Encoding (BPE)**. Sennrich et al. (2016) introduced BPE-based subword units for neural machine translation to address the problem of rare and unknown words. Their experiments showed that subword models could represent rare words more effectively than word-level systems with a back-off dictionary.

For example, a rare word might be divided approximately like this:

```text
un + help + ful
```

instead of requiring the complete word `unhelpful` to already exist in the vocabulary.

### Vocabulary size

**Vocabulary size** is the number of unique tokens that a tokenizer recognizes.

For example:

```text
Text: "cat cat dog"

Tokens: cat, cat, dog

Vocabulary: {cat, dog}

Vocabulary size = 2
```

The tokenizer directly affects vocabulary size. Word-level tokenization generally produces a larger vocabulary because every word form can become an entry. Character-level tokenization has a much smaller vocabulary, while subword tokenization usually uses a predefined vocabulary size.

Subword tokenization has become particularly important because it provides a compromise between word-level and character-level representations and helps models handle rare words.

---

## 4. Stopwords

**Stopwords** are very common words that are sometimes removed during text processing because they may contribute relatively little to certain tasks. In English, examples can include words such as `the`, `and`, `of`, and `to`. Stopword lists are commonly used in information retrieval and text analysis.

### Benefits of removing stopwords

Removing stopwords can:

- Reduce the number of features.
- Reduce the size of the data representation.
- Reduce some forms of noise.
- Make some keyword-based analyses focus more on content words.

For example:

```text
The report was about climate change.
```

could become:

```text
report climate change
```

For some topic or keyword-based tasks, the remaining words may still contain the main information.

### Risks of removing stopwords

The problem is that common words are not always meaningless. Some of them are important for grammar, meaning, and relationships between words.

For example:

```text
The product is reliable.

The product is not reliable.
```

Removing `not` can make the two sentences much more similar even though their meanings are opposite. Research has shown that negation words can have an important effect on sentiment classification.

Stopword removal can also hurt information retrieval when the words are part of an important phrase. Modern information-retrieval systems therefore do not always use traditional stopword lists, partly because weighting methods such as IDF already reduce the importance of very common terms.

### Why the list depends on the task and language

There is no universal stopword list that is correct for every application. A word that is unimportant in one domain may be important in another. Stopword lists also differ between languages because languages have different grammatical structures and different distributions of frequent words.

Research comparing English stopword lists found substantial differences between commonly used lists, including unexpected omissions and inclusions.

Therefore, stopword removal should be evaluated according to the **task, language, dataset, and model**, rather than automatically applying a standard list.

---

## 5. Stemming vs. Lemmatization

Both **stemming** and **lemmatization** attempt to reduce different forms of a word so that related forms can be treated more similarly.

### Stemming

**Stemming** removes or changes word endings according to predefined rules. It does not necessarily produce a real dictionary word.

For example:

```text
connected → connect
connection → connect
```

A famous algorithm is the **Porter stemming algorithm**, introduced by Martin Porter in 1980. It uses a sequence of suffix-removal rules to reduce English words to stems. The original algorithm was designed partly for information retrieval, where grouping related terms can reduce the number of terms and improve retrieval performance.

The advantage of stemming is that it is generally simple and fast. It can be useful when the main goal is to group related word forms, especially in traditional information-retrieval systems.

The disadvantage is that it can be too aggressive. A stem may not be an actual English word, and unrelated words can sometimes be reduced to the same stem. Porter himself noted that suffix stripping is intended to improve retrieval rather than perform perfect linguistic analysis.

### Lemmatization

**Lemmatization** attempts to convert a word into its **lemma**, or dictionary form. Unlike basic stemming, it uses linguistic information and can consider grammatical properties.

For example:

```text
cars → car
better → good
```

The second transformation requires knowledge about the grammatical and lexical relationship between the forms.

One common approach uses **WordNet**, an English lexical database in which words are organized according to lexical and semantic relationships.

Lemmatization can produce more linguistically meaningful results than stemming, but it generally requires more language-specific resources and processing.

### Comparison

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Main idea | Remove word endings using rules | Find the dictionary form |
| Output | May not be a real word | Usually a valid lemma |
| Linguistic knowledge | Relatively limited | More linguistic information |
| Speed | Usually faster | Usually more complex |
| Example | `studies → studi` | `studies → study` |
| Useful when | Fast term grouping is sufficient | Meaningful word normalization is important |

Neither method is automatically better. For a simple information-retrieval system, stemming may be preferable because it is fast and reduces vocabulary variation. For applications where linguistic interpretation matters, lemmatization may be more appropriate. The final choice should be tested on the target task.

---

## 6. Regular Expressions in Text Cleaning

A **regular expression**, or **regex**, is a pattern used to find particular sequences of characters in text. In NLP preprocessing, regex can be used to identify and replace URLs, numbers, repeated punctuation, whitespace, email addresses, and other patterns.

### Example 1: Detecting URLs

```regex
https?://\S+
```

This pattern matches URLs beginning with either `http://` or `https://`, followed by non-whitespace characters.

For example:

```text
Visit https://example.com/page
```

can be transformed into:

```text
Visit URL
```

Replacing the URL with a marker can preserve the information that a link was present without forcing the model to memorize every individual URL.

### Example 2: Detecting repeated punctuation

```regex
!+
```

This matches one or more consecutive exclamation marks.

For example:

```text
Amazing!!!!
```

could become:

```text
Amazing!
```

This can reduce unnecessary variation while preserving the presence of an exclamation mark.

### Risk of an overly broad pattern

A regex can become harmful when its pattern matches more text than intended.

For example:

```regex
[^a-zA-Z\s]
```

matches anything that is not an English letter or whitespace. It could therefore remove:

```text
$500
```

and also characters from other writing systems.

If the task is fraud detection, the amount `$500` could be important. Similarly, removing non-English characters would be especially damaging in multilingual NLP.

Therefore, regex should be designed around the actual information that the task needs to preserve.

---

## 7. Case Study: The Effect of Preprocessing on Neural Text Classification

### Study

Camacho-Collados and Pilehvar (2018) investigated how preprocessing choices affect neural NLP models in their paper **"On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis."** The study examined both **topic categorization** and **sentiment classification** using several standard datasets.

### Preprocessing options compared

The researchers compared four settings:

1. **Vanilla**, meaning tokenization only.
2. **Lowercasing**.
3. **Lemmatization**.
4. **Multiword grouping**, where selected multiword expressions were treated as single units.

They evaluated these choices using CNN and CNN+LSTM text-classification models across multiple topic-classification and polarity-detection datasets.

### Findings

The results showed that preprocessing could noticeably affect model performance. For the CNN+LSTM model, the average difference between the best and worst preprocessing settings was about **2.4 percentage points**, and statistically significant differences appeared in several datasets.

An important finding was that **simple tokenization was generally competitive**. It was within the range of the best-performing approach on 8 of the 9 datasets. More complex preprocessing such as lemmatization and multiword grouping did not consistently improve performance.

However, the results changed for the specialized **Ohsumed medical dataset**. There, lowercasing and lemmatization improved performance by more than six percentage points over simple tokenization for the CNN+LSTM setting. This suggested that preprocessing can become more important when the dataset is smaller or contains specialized vocabulary.

The researchers concluded that there is no single preprocessing strategy that is best for every dataset. Simple tokenization can be a strong baseline, while additional preprocessing may help in particular domains or data conditions. They also emphasized that preprocessing choices should be reported and kept consistent when comparing NLP systems.

### Summary of the case study

| Aspect | Details |
|---|---|
| **Task** | Topic categorization and sentiment classification |
| **Models** | CNN and CNN+LSTM |
| **Preprocessing compared** | Tokenization, lowercasing, lemmatization, multiword grouping |
| **Main finding** | Simple tokenization was generally competitive |
| **Important exception** | More preprocessing helped on the specialized medical dataset |
| **Main lesson** | Preprocessing should be selected and evaluated according to the dataset and task |

This case study directly supports the main idea of preprocessing: **more processing is not automatically better**. A preprocessing technique should be used when there is a reason to expect it to help, and its effect should be evaluated experimentally.

---

# Conclusion

Text preprocessing is an important part of an NLP pipeline because raw text contains many forms of variation that can make learning more difficult. Normalization can make equivalent forms consistent, tokenization determines the units processed by the model, stopword removal can reduce unnecessary features, and stemming and lemmatization can connect related word forms.

However, preprocessing can also remove useful information. Capitalization, punctuation, numbers, negation, and other features may be important depending on the application. The research case study also shows that a simple preprocessing approach can sometimes outperform more complicated alternatives.

Therefore, the main principle is not to clean text as much as possible, but to **choose preprocessing steps based on the NLP task, language, dataset, and model, and then evaluate their actual effect on performance**.


## Part B: Practical

The following results were produced by running `preprocessing_practice.py`. The complete console output is saved in `output.txt`.

### 1. Lowercasing

T1 changed from `URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!` to `urgent!!! you have won a £1,000 gift card. call 0712345678 now!`. A spam detector could lose capitalization signals such as `URGENT` and `WON`, which may indicate an aggressive promotional message. Therefore, lowercasing reduces variation but may remove useful spam evidence.

### 2. Whitespace normalization

T4 was normalized from `'   The   delivery   was   fast,   and the  packaging   was   great.   '` to `'The delivery was fast, and the packaging was great.'`. This removes accidental spacing differences while preserving the words.

### 3. Regex cleaning and replacement

The pattern `[^a-z\s]` changed the lowercased T1 to:

```text
urgent    you have won a        gift card  call            now 
```

It deleted the pound sign, digits, comma, full stop, exclamation marks, and other characters that are not lowercase English letters or whitespace. This can be harmful because the amount and phone number may be useful spam signals.

The replacement rules preserved that information using markers:

```text
urgent! you have won a MONEY gift card. call PHONE_NUMBER now!
```

This is better for a spam detector because it removes the exact value while retaining the facts that a money amount and a phone number were present.

### 4. Tokenization

The `split()` method produced 14 tokens because it keeps punctuation attached to words and keeps `don't` as one token. NLTK's `word_tokenize()` produced 16 tokens because it separated punctuation and split `don't` into `do` and `n't`. Tokenizing T4 produced 11 tokens and a vocabulary size of 10, because `was` appeared twice.

### 5. Stopwords and negation

Default stopword removal produced:

```text
["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']
```

The custom list that kept negation produced:

```text
["n't", 'think', 'jacket', 'worth', 'price', '.', 'Not', 'happy', '😞']
```

The default list removed `not`, while the custom list preserved it. Keeping negation is important because the complaint is negative: removing `not` can make `Not happy` look like simply `happy` and can change the sentiment interpretation.

### 6. Stemming

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

The Porter stemmer merges `connection`, `connected`, and `connecting` into `connect`. The stem `univers` from `universal` is not a normal English word. These results show that stemming can group related forms efficiently, but can also produce unnatural stems.

### 7. Lemmatization

| Word | Default | POS | With POS |
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

The words that changed only after a POS hint were `connected`, `connecting`, `organized`, `better`, and `ran`. POS information helps the lemmatizer distinguish a verb or adjective from the default noun interpretation.

### 8. Sentence splitting

The sentence tokenizer returned two sentences:

1. `Mr. Smith arrived in London on Friday.`
2. `He met the Apple team at 9 a.m.`

Neither `Mr.` nor `a.m.` caused an incorrect split. This demonstrates why a sentence tokenizer is more useful than simply splitting at every period.

### 9. POS tagging and named entities

The POS tagger produced tags including `Mr.`/`NNP`, `Smith`/`NNP`, `London`/`NNP`, `Friday`/`NNP`, and `Apple`/`NNP`. The NER model extracted the following entities:

| Text extracted | Model label | Is the label correct? | Comment |
|---|---|---|---|
| Mr. | PERSON | Partly | `Mr.` is a title, not a person by itself. |
| Smith | PERSON | Yes | It is a person's surname in this sentence. |
| London | GPE | Yes | London is a geographical/political place. |

The model missed the full person name `Mr. Smith` as one entity, the day `Friday`, and the organization `Apple`. In this context, `Apple` refers to the company, not a fruit. The model's output should therefore be checked rather than treated as perfect ground truth.

### 10. Full pipeline

With default stopwords, the final text was:

```text
think jacket worth price happy
```

With the negation-keeping list, the final text was:

```text
think jacket worth price not happy
```

The second version keeps the customer's complaint because it preserves `not`. The first version loses part of the negative meaning when `not` is removed. The regex stage also removed punctuation, the emoji, and the apostrophe before tokenization, so preprocessing choices affected the final result.

### Preprocessing choices for a spam detector

For a spam detector, I would use whitespace normalization and carefully designed regex rules that replace phone numbers, money amounts, URLs, and email addresses with markers. I would probably preserve capitalization and repeated punctuation as features because `URGENT`, `WON`, and `!!!` can signal spam. I would also preserve negation words and avoid blindly deleting numbers, currency symbols, emojis, or punctuation, because the output shows that these can carry useful evidence. Stemming, lemmatization, and stopword removal should be tested rather than assumed to improve accuracy.


# References

1. Camacho-Collados, J., & Pilehvar, M. T. (2018). *On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis*. Proceedings of the 2018 EMNLP Workshop BlackboxNLP, 40–46. Association for Computational Linguistics. https://aclanthology.org/W18-5406/

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://www-nlp.stanford.edu/IR-book/

3. Porter, M. F. (1980). *An Algorithm for Suffix Stripping*. Program, 14(3), 130–137. https://www.cs.toronto.edu/~frank/csc2501/Readings/R2_Porter/Porter-1980.pdf

4. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, 1715–1725. Association for Computational Linguistics. https://aclanthology.org/P16-1162/

5. Saif, H., Fernandez, M., He, Y., & Alani, H. (2014). *On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter*. Proceedings of LREC 2014, 810–817. https://aclanthology.org/L14-1265/

6. Nothman, J., Qin, H., & Yurchak, R. (2018). *Stop Word Lists in Free Open-source Software Packages*. Proceedings of the Workshop for NLP Open Source Software, 7–12. Association for Computational Linguistics. https://aclanthology.org/W18-2502/

7. Unicode Consortium. (2025). *Unicode Standard, Chapter 3: Conformance and Unicode Normalization*. https://www.unicode.org/versions/Unicode18.0.0/core-spec/chapter-3/

8. van der Goot, R. (2026). *From Bytes to Subwords: Challenges of Input Representations in NLP*. Findings of the Association for Computational Linguistics: ACL 2026, 10911–10919. Association for Computational Linguistics. https://aclanthology.org/2026.findings-acl.530/

9. Miller, G. A., Beckwith, R., Fellbaum, C., Gross, D., & Miller, K. J. (1990). *Introduction to WordNet: An On-line Lexical Database*. International Journal of Lexicography, 3(4), 235–244. https://academic.oup.com/ijl/article-abstract/3/4/235/923280