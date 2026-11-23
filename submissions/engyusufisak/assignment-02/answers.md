# Assignment 2 – Text Preprocessing

**Name:** Yusuf Isak  
**GitHub:** engyusufisak  
**Course:** NLP for Machine Learning Bootcamp

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is text preprocessing?](#1-what-is-text-preprocessing)
3. [Text normalization](#2-text-normalization)
4. [Word, subword and character tokenization](#3-word-subword-and-character-tokenization)
5. [Stopwords](#4-stopwords)
6. [Stemming and lemmatization](#5-stemming-and-lemmatization)
7. [Regular expressions for cleaning](#6-regular-expressions-for-cleaning)
8. [Case study: does preprocessing matter for neural models?](#7-case-study-does-preprocessing-matter-for-neural-models)
9. [Part B – Practical](#part-b--practical)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Introduction

In Assignment 1 I learned that a computer must change text into numbers before a model can use it. But before that, the text itself is usually messy. People write with capital letters, extra spaces, emojis, links, different spellings and many forms of the same word. **Text preprocessing** is the work we do to prepare this messy text.

In Part A of this assignment I research the main preprocessing steps: normalization, tokenization, stopwords, stemming, lemmatization and regular expressions. For each step I try to explain not only *how* it works but also *when it can hurt*, because the main thing I learned is that more cleaning is not always better. At the end I summarize a research paper that tested different preprocessing choices. In Part B I apply these steps in my own Python script.

Where I can, I also give Somali examples, because most preprocessing tools were made for English.

---

## 1. What is text preprocessing?

### 1.1 Definition

In my own words, **text preprocessing is the set of steps that change raw text into a cleaner and more consistent form before we give it to an NLP model.** Common steps are tokenization (cutting text into pieces), normalization (making different forms look the same), removing words or characters we don't need, and reducing words to a base form (Jurafsky & Martin, 2025, ch. 2).

### 1.2 Why NLP systems need it

- **Less noise.** Extra spaces, HTML tags or random symbols don't help the model, so removing them makes the data cleaner.
- **Smaller vocabulary.** If "Hargeysa", "HARGEYSA" and "hargeysa" are three different tokens, the model has to learn three things instead of one. Normalizing them makes the vocabulary smaller and gives each word more examples.
- **Consistency between training and prediction.** If training text is cleaned one way, new text must be cleaned the same way, or the model sees words it doesn't know.

### 1.3 Why more cleaning can make results worse

Every cleaning step **removes information**. If that information was useful for the task, the model gets worse. Manning, Raghavan & Schütze (2008, ch. 2) explain that search engines moved away from big stopword lists, because removing words made some searches impossible. The case study in question 7 also shows that a simple tokenizer often worked as well as, or better than, heavier preprocessing (Camacho-Collados & Pilehvar, 2018).

### 1.4 Real-life example: restaurant reviews on social media

Imagine a company that classifies customer comments about a restaurant as positive or negative. A customer writes:

> *"The food was sooo good 🔥🔥 but the service... NEVER again!!!"*

If we "clean" this strongly (remove emojis, remove punctuation, lowercase, remove stopwords), we may get something like `food sooo good service`. The 🔥 emojis (very positive), the capital letters in "NEVER" (strong emotion), the "!!!" and the word "never" itself (a negation) are all gone. The cleaned version looks **positive**, but the real review is **mixed**, with a strong complaint. So for sentiment, some "noise" is actually the signal.

---

## 2. Text normalization

**Normalization** means changing text so that things that should count as the same really look the same to the computer (Jurafsky & Martin, 2025, ch. 2).

| Step | What it does | Example |
| --- | --- | --- |
| **Lowercasing** | Changes all letters to small letters | `HARGEYSA` → `hargeysa` |
| **Whitespace normalization** | Changes many spaces, tabs and new lines into one space, and removes spaces at the start and end | `"Wax   ka  qabo\n"` → `"Wax ka qabo"` |
| **Unicode normalization** | Makes characters that look the same also have the same code inside the computer | `é` written as one character vs `e` + accent mark |

### 2.1 Unicode normalization (explained)

A computer stores every character as a code. The problem is that some characters can be stored in **more than one way**. For example, "café" can be stored with `é` as one character, or with `e` plus a separate accent mark. On screen they look identical, but in Python they are **not equal**. I tested this: the first version has 4 characters, the second has 5, and `a == b` is `False`.

The Unicode standard has four normalization forms: **NFC, NFD, NFKC and NFKD** (Unicode Consortium, 2026). NFC joins characters into one when possible, and NFD splits them into parts. The "K" forms (NFKC, NFKD) also change "compatibility" characters into normal ones. For example, the single ligature character `ﬁ` becomes the two letters `f` + `i`, and full-width letters like `Ｈｉ` become normal `Hi`. After NFC normalization, my two versions of "café" became equal.

### 2.2 One situation where normalization helps

**Search.** If a user searches `hargeysa` but the news article writes `Hargeysa`, the search will fail without lowercasing. Lowercasing both the query and the documents makes them match. Unicode normalization helps in the same way when text is copied from different websites, PDFs or phones that store the same letter differently.

### 2.3 One situation where normalization removes useful information

**Capital letters that change the meaning.** "Turkey" (the country) and "turkey" (the bird) become the same word after lowercasing. Also, names of people and places usually start with a capital letter, and tools that find names (NER) use this as a clue. If we lowercase first, the NER tool loses that clue. In sentiment or spam, words written in ALL CAPS also show strong emotion, and lowercasing removes it. Even NFKC can change meaning: it turns the superscript `²` into a normal `2`, so `x²` becomes `x2`.

---

## 3. Word, subword and character tokenization

**Tokenization** means cutting text into small units called tokens. There are three main levels:

| | **Word-level** | **Subword** | **Character-level** |
| --- | --- | --- | --- |
| Token is | A whole word | A piece of a word (frequent pieces) | One letter or symbol |
| Example: `gurigayga` ("my house") | `gurigayga` | e.g. `guri` + `gayga` (depends on the training data) | `g`,`u`,`r`,`i`,`g`,`a`,`y`,`g`,`a` |
| Vocabulary size | Very big (every different word) | Medium, **chosen by us** (e.g. 30,000) | Very small (letters, numbers, symbols) |
| Unknown words? | Yes, many new words are unknown | Rare, a new word can be built from pieces | Almost never |
| Length of a sentence | Short (few tokens) | Medium | Very long (many tokens) |

### 3.1 What vocabulary size means

**Vocabulary size** is the number of *different* tokens the tokenizer knows. It is not the length of a text. A model has to learn something (for example a vector) for every entry in the vocabulary, so a bigger vocabulary means a bigger model and more data needed.

### 3.2 How the tokenizer changes vocabulary size

- **Word-level:** every new form of a word is a new entry. In Somali, one word can have many forms with endings (`guri`, `guriga`, `guryaha`, `gurigayga`…), so a word-level vocabulary grows very fast. Words that never appeared in training become **unknown (out-of-vocabulary)**.
- **Character-level:** the vocabulary is tiny, because Somali uses the Latin alphabet plus some symbols. But a sentence becomes very long, and the model has to learn by itself how letters make words, which is harder.
- **Subword:** this is the middle way used by most modern models. A method like **Byte Pair Encoding (BPE)** starts from characters and repeatedly joins the most frequent pair into a new token, until the vocabulary reaches the size we chose (Sennrich, Haddow & Birch, 2016). Common words stay whole, and rare words are split into known pieces, so there are almost no unknown words. For example, BERT uses a subword (WordPiece) vocabulary of about 30,000 tokens (Devlin et al., 2019).

**My conclusion:** the tokenizer decides the balance between vocabulary size and sequence length. For a language with many word forms like Somali, subword tokenization is usually a good choice, but only if the tokenizer was trained on enough Somali text. Otherwise, it cuts Somali words into strange small pieces.

---

## 4. Stopwords

### 4.1 What are stopwords?

**Stopwords** are very common words that appear in almost every text and often carry little meaning for the task, like "the", "of", "and" in English or "iyo" (and), "ka", "ku" in Somali. A **stopword list** is a list of these words that we remove before processing (Manning, Raghavan & Schütze, 2008, ch. 2). NLTK's English list has 198 words (I checked this in Python).

### 4.2 Benefits of removing them

- The text becomes shorter, so processing is faster and uses less memory.
- The model focuses on the words that carry meaning (topic words), which can help for tasks like topic classification and keyword search.

### 4.3 Risks of removing them

- **Negation is lost.** NLTK's English list contains `not`, `no`, `nor`, `don't` and `against` (I checked these too). Removing them can turn a negative sentence into a positive one.
- **Phrases lose meaning.** Manning et al. (2008) give the example of searching for "To be or not to be" or song titles, where almost every word is a stopword.
- **Grammar and order are lost**, which matters for tasks like translation and question answering.

### 4.4 Why the list depends on the task and the language

- **Task:** for *topic classification*, removing "and" or "the" is usually fine. For *sentiment analysis*, words like "not" and "never" are very important and must be kept. For *authorship detection*, the small common words are actually the most useful, because each person uses them in their own style.
- **Language:** a stopword list only works for its language. NLTK has no Somali stopword list. If we use the English list on Somali text, it removes almost nothing useful. Also, Somali small words can have important meaning. For example, `ma` is a negation in *"Cuntadu ma fiicna"* ("The food is not good"). If someone puts `ma` in a Somali stopword list because it is very common, the sentence becomes "Cuntadu fiicna" and the meaning flips.

---

## 5. Stemming and lemmatization

Both methods try to reduce different forms of a word to one common form, so that, for example, "argued" and "arguing" count as the same word. But they do it differently.

| | **Stemming** | **Lemmatization** |
| --- | --- | --- |
| How | Cuts off endings using fixed rules | Looks up the word in a dictionary, using its grammar |
| Result | A **stem**, often not a real word (`ponies` → `poni`) | A **lemma**, a real dictionary word (`geese` → `goose`) |
| Needs | Only rules | A dictionary and usually the part of speech |
| Speed | Very fast | Slower |
| Irregular words | Can't handle them (`mice` stays different from `mouse`) | Handles them (`mice` → `mouse`) |

### 5.1 A stemming algorithm: the Porter stemmer

The **Porter stemmer** was published by Martin Porter in 1980 and is the most famous English stemmer (Porter, 1980). It removes suffixes in **five steps**, one after another. Each step has rules like "replace *-sses* with *-ss*" or "replace *-ies* with *-i*". The rules check how long the remaining stem is (Porter calls this the *measure*), so that short words are not cut too much. Some results I got with NLTK's Porter stemmer:

| Word | Porter stem | Comment |
| --- | --- | --- |
| caresses | caress | good |
| ponies | poni | not a real word, but still works for matching |
| argued, arguing | argu, argu | two forms joined ✔ |
| general, generate | gener, gener | **different meanings merged** ✘ |
| policy, police | polici, polic | similar words kept apart ✔ |

The "general / generate" example shows the main problem with stemming: **over-stemming**, when words with different meanings get the same stem.

### 5.2 A lemmatization approach: WordNet lemmatizer

The **WordNet lemmatizer** in NLTK uses **WordNet**, a big English dictionary that groups words and their meanings (Miller, 1995). It first applies simple rules to remove endings, then checks whether the result is a real word in WordNet. It also has an exception list for irregular forms. That is why I got `geese` → `goose` and `mice` → `mouse`. It works best when you tell it the part of speech. For example, `was` with `pos="v"` became `be`.

### 5.3 When to prefer each

- **Prefer stemming** when you need speed, have a lot of text, and only need to *match* words, not understand them. Search engines are a typical example: if a user searches "argued", it's fine to also find "arguing".
- **Prefer lemmatization** when the exact meaning matters or a person will read the output, for example in chatbots, question answering, or when showing keywords to users.
- **For Somali:** neither works out of the box. NLTK's Snowball stemmer supports 16 languages, including Arabic, but not Somali (I checked the list in Python). WordNet is English only. A Somali stemmer or lemmatizer would need rules for Somali endings like `-ka`, `-ga`, `-ta`, `-kii`, `-yada`.

---

## 6. Regular expressions for cleaning

A **regular expression (regex)** is a short pattern that describes a type of text. Python's `re` module can **find**, **remove** or **replace** every part of the text that matches the pattern (Jurafsky & Martin, 2025, ch. 2). In cleaning, regex is used to remove links, emails, HTML tags or extra symbols, or to replace them with a placeholder like `URL`.

### 6.1 Two example patterns

| Pattern | What it matches | My test |
| --- | --- | --- |
| `https?://\S+` | `http` or `https`, then `://`, then all characters until the next space, meaning a web link | `see https://bbc.com/somali today` → `see URL today` (with `re.sub(..., "URL", ...)`) |
| `[\w.+-]+@[\w-]+\.[\w.]+` | Letters/numbers/dots before `@`, a domain name, a dot and an ending, meaning an email address | Found `info@example.so` in `contact info@example.so now` |

Explanation of the symbols:

- `s?` means the `s` is optional.
- `\S+` means one or more characters that are not spaces.
- `\w` means a letter, digit or `_`.
- `[...]` means any one of these characters.
- `+` means one or more times.

### 6.2 How an overly broad pattern can damage text

A pattern that is too general deletes things we wanted to keep. Two examples I tested:

1. **Removing punctuation in Somali:** the pattern `[^\w\s]` ("anything that is not a letter, digit or space") looks safe. But in Somali, the apostrophe `'` is a real letter that shows a sound (a glottal stop), as in `ba'an` ("severe"). After the pattern, *"Abaar ba'an ayaa ka dhacday"* ("A severe drought happened") became *"Abaar **baan** ayaa ka dhacday"*. The word changed and its meaning is lost.
2. **Removing numbers:** `\d+` removes all digits. *"COVID-19 cases rose to 250"* became *"COVID- cases rose to"*. The disease name is broken and the important number is gone.

**Lesson:** always test a regex on real examples and look at the output before using it on the whole dataset. And think about replacing instead of deleting (for example `250` → `NUMBER`).

---

## 7. Case study: does preprocessing matter for neural models?

### 7.1 The paper

Camacho-Collados, J., & Pilehvar, M. T. (2018). *On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis.* Published at the BlackboxNLP workshop, EMNLP 2018.

### 7.2 The task

Two text classification tasks:

- **Topic categorization:** what is the article about?
- **Polarity detection (sentiment):** is the review positive or negative?

### 7.3 The data

They used **9 datasets**:

| Task | Datasets |
| --- | --- |
| Topic | BBC news (5 topics, 2,225 documents), 20 Newsgroups, Reuters, Ohsumed (medical, 23 topics) |
| Sentiment | Rotten Tomatoes snippets, IMDB movie reviews (50,000), two Pang & Lee movie datasets (PL04, PL05), Stanford sentiment phrases |

### 7.4 The preprocessing options compared

1. **Vanilla:** only tokenization, nothing else.
2. **Lowercased:** tokenization + lowercasing.
3. **Lemmatized:** tokenization + lemmatization.
4. **Multiword:** tokenization + joining common multi-word expressions into one token (like `New_York`).

They tested each option with two neural models: a **CNN** and a **CNN + LSTM**.

### 7.5 Results (accuracy %, CNN+LSTM model, selected datasets)

| Dataset | Vanilla | Lowercased | Lemmatized | Multiword |
| --- | ---: | ---: | ---: | ---: |
| BBC (news topics) | **97.0** | 96.4 | 95.8 | 96.2 |
| IMDB (sentiment) | **88.9** | 88.3 | 87.7 | **88.9** |
| PL04 (small sentiment) | 71.4 | **73.3** | 72.6 | 67.0 |
| Ohsumed (medical) | 30.8 | **37.5** | 37.1 | 29.0 |

### 7.6 Main findings

- **Simple tokenization (vanilla) was enough in most cases.** It was close to the best result on 8 of the 9 datasets.
- **The exception was the medical dataset (Ohsumed).** There, lowercasing and lemmatization improved accuracy by more than 6 points. I think this is because medical text has many rare technical words in different forms, so joining the forms gives the model more examples of each.
- **Lowercasing sometimes hurt.** With the CNN model, lowercasing lowered IMDB accuracy from 87.5 to 84.2. In sentiment, capital letters can show emotion.
- **Small datasets changed the most** with different preprocessing, and big datasets changed the least.
- The authors' conclusion: simple tokenization usually works as well as or better than heavier preprocessing for these neural models, except for special domains, and researchers should always report which preprocessing they used.

### 7.7 What I learn from it

This paper supports my answer to question 1: **more cleaning is not automatically better.** The best preprocessing depends on the task, the domain and the size of the data, so you have to test it. For Somali, I think lemmatization-like steps could help more than in English, because Somali words have many forms (like the medical words in Ohsumed). But this should be tested and not assumed.

---

## Part B – Practical

All results below come from my script [`preprocessing_practice.py`](preprocessing_practice.py). The full printed output is in [`output.txt`](output.txt) (created with `python preprocessing_practice.py > output.txt`).

### Task 1 – Lowercasing

`URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!` became `urgent!!! you have won a £1,000 gift card. call 0712345678 now!`. My script found three words that were written in capitals: `URGENT`, `WON` and `NOW`. After lowercasing, a spam detector can't see that the message is "shouting" at the reader, and shouting is a typical sign of spam. So lowercasing removes a useful spam signal.

### Task 2 – Whitespace

With `repr()` the extra spaces are visible. The text went from 67 characters (`'   The   delivery   was   fast, ...'`) to 51 characters (`'The delivery was fast, and the packaging was great.'`). Nothing useful was lost here.

### Task 3 – Regex cleaning

The pattern `[^a-z\s]` on the lowercased T1 deleted **21 characters**: all the `!` (4), the `£`, the comma and the full stop, and **all the digits** of `1,000` and `0712345678`. The result is `urgent    you have won a        gift card  call            now`. The money amount and the phone number are completely gone, but they are probably the strongest evidence that this is spam.

My own rules **replace instead of delete**:

- `\b0\d{9}\b` → `PHONE_NUMBER`
- `[£$€]\s?\d[\d,]*(\.\d+)?` → `MONEY`
- `!{2,}` → `!`

Result: `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`. Now the model still knows that the message *contains* money and a phone number, without having to learn every different amount or number.

### Task 4 – Tokenization

- **T2 with `split()`:** 14 tokens. `don't` and `price.` stay as one token each, with the punctuation stuck on.
- **T2 with `word_tokenize`:** 16 tokens. `don't` was split into `do` + `n't`, and `price` + `.` were separated. The emoji `😞` stayed as its own token in both.
- **T4:** 11 tokens but a **vocabulary size of 10**, because `was` appears twice. `The` and `the` count as two different entries. When I lowercased first, the vocabulary size dropped to 9.

### Task 5 – Stopwords and negation

| Version | Result |
| --- | --- |
| NLTK default list | `n't, think, jacket, worth, price, ., happy, 😞` |
| My list (keeps `not`, `n't`, `no`, `never`) | `n't, think, jacket, worth, price, ., Not, happy, 😞` |

With the default list, **"Not happy" became just "happy"**, so the second sentence now looks positive. With my list, `Not` stays and the complaint "not happy" is kept. Interesting detail: `n't` survived in **both** versions, because NLTK's list has `don't` but not the separate piece `n't` that `word_tokenize` creates. So the list and the tokenizer must fit together.

### Task 6 – Stemming

| Word | Porter | Lancaster | Snowball |
| --- | --- | --- | --- |
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

- **Good merge:** `connection`, `connected` and `connecting` all became `connect` with all three stemmers.
- **Different words merged (over-stemming):** `university` and `universal` both became `univers`, but a university (a school) and universal (for everyone) have different meanings. Also `organization` became `organ`, which is a real word with a completely different meaning (a body part).
- **Stems that are not real words:** `fli`, `wolv` and `univers` (all three stemmers), and Lancaster's `org`.
- Lancaster is the most aggressive: it changed `better` to `bet`, a real word with a totally different meaning. No stemmer changed `ran` to `run`, because stemmers only cut endings and can't handle irregular verbs.

### Task 7 – Lemmatization

| Word | Default | POS hint | With hint |
| --- | --- | --- | --- |
| connected | connected | v | **connect** |
| connecting | connecting | v | **connect** |
| organized | organized | v | **organize** |
| better | better | a | **good** |
| ran | ran | v | **run** |
| flies | fly | v | fly |
| wolves | wolf | n | wolf |
| connection, organization, university, universal | (no change) | n / a | (no change) |

**Five words changed only with the hint:** `connected`, `connecting`, `organized`, `better` and `ran`. This is because the lemmatizer treats every word as a noun by default, so verbs and adjectives are not reduced. `flies` and `wolves` already worked by default because they are plural nouns. Unlike the stemmers, the lemmatizer always gave real words, and it even knew the irregular forms `ran → run` and `better → good`.

### Task 8 – Sentence splitting

`sent_tokenize` found **2 sentences**, which is correct:

1. `Mr. Smith arrived in London on Friday.`
2. `He met the Apple team at 9 a.m.`

It did **not** break after `Mr.` and did not break inside `a.m.`. NLTK's sentence tokenizer (Punkt) has learned that these are abbreviations. A naive split on every `.` would have given wrong pieces like `Mr` and `a`.

### Task 9 – POS tagging and NER

Most POS tags were reasonable: `Smith`, `London`, `Friday` and `Apple` were all tagged `NNP` (proper noun), and `arrived` and `met` were `VBD` (past-tense verb). But `a.m` was tagged `NN` (common noun).

| Entity in text | NLTK label | Correct label | Correct? |
| --- | --- | --- | --- |
| Mr. | PERSON | not a separate entity (a title) | ✘ |
| Smith | PERSON | PERSON | ✔ |
| London | GPE (place) | GPE | ✔ |
| Friday | **missed** | DATE | ✘ |
| Apple | **missed** | ORGANIZATION | ✘ |

NLTK found 2 of the 4 real entities correctly. It made `Mr.` a separate PERSON instead of joining it with `Smith`. It missed `Apple` completely, even though here it means the company. It also missed `Friday`. NLTK's `ne_chunk` doesn't have a DATE label at all, so it can never find days or times.

### Task 10 – Full pipeline (T2)

| Stage | Default stopwords | My stopwords |
| --- | --- | --- |
| 1 lowercase | `i don't think this jacket is worth the price. not happy at all 😞` | same |
| 2 regex `[^a-z\s']` | `i don't think this jacket is worth the price  not happy at all` | same |
| 3 tokenize | `i, do, n't, think, this, jacket, is, worth, the, price, not, happy, at, all` | same |
| 4 stopwords | `n't, think, jacket, worth, price, happy` | `n't, think, jacket, worth, price, **not**, happy` |
| 5 lemmatize | same as 4 | same as 4 |

In my regex I kept the apostrophe (`'`) so that `don't` would survive, but the emoji 😞 was deleted. **My negation-keeping version keeps the complaint**, because it still has `not happy`. The default version ends with `... price, happy`, which looks almost positive. Both lost the sad emoji, which was also a clear sign of the customer's feeling.

### Which steps I would use for a spam detector

For a spam detector I would use **whitespace normalization** and **tokenization with `word_tokenize`**, because they remove noise without losing anything (Task 2, Task 4). I would use **regex replacement instead of deletion**: in Task 3, deleting with `[^a-z\s]` removed the money amount and the phone number, but my rules turned them into `MONEY` and `PHONE_NUMBER`, which are strong spam signals. I would turn `!!!` into `!`, or better, also keep a feature that counts exclamation marks. I would **avoid plain lowercasing**, or at least first save a feature like "number of words in capitals", because Task 1 showed that `URGENT`, `WON` and `NOW` were written in capitals and that shouting disappears after lowercasing. I would be careful with **stopword removal** and stemming. Spam messages are short, and words like "you" and "now" are part of the typical spam style. Stemming like `organization → organ` (Task 6) can mix up words for no real gain.

---

## Conclusion

In this assignment I learned that text preprocessing is a set of choices, not one fixed recipe. Normalization makes different forms match, but can remove useful signals like capital letters (Q2). The tokenizer controls the vocabulary size: word-level gives big vocabularies and unknown words, character-level gives long sequences, and subword methods like BPE are in between (Q3). Stopword removal makes text shorter, but can delete negation, and the list must fit the task and the language (Q4). Stemming is fast but cuts words roughly, while lemmatization gives real words but needs a dictionary (Q5). Regex is powerful for cleaning, but a pattern that is too broad can damage words, like the Somali apostrophe (Q6). The case study showed that heavy preprocessing often doesn't help modern neural models, except in special domains (Q7).

The most important point for me is that every step removes something, so before using a step I should ask: **is this noise, or is it evidence for my task?** For Somali, most ready-made tools (stopword lists, stemmers, WordNet) don't exist yet, so each step has to be checked carefully.

---

## References

Camacho-Collados, J., & Pilehvar, M. T. (2018). On the role of text preprocessing in neural network architectures: An evaluation study on text categorization and sentiment analysis. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP*. <https://arxiv.org/abs/1707.01780>

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). <https://aclanthology.org/N19-1423>

Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft), Chapter 2: Words and tokens. <https://web.stanford.edu/~jurafsky/slp3/>

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*, Chapter 2: The term vocabulary and postings lists. Cambridge University Press. <https://nlp.stanford.edu/IR-book/>

Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM, 38*(11), 39–41.

Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In *Proceedings of the 54th Annual Meeting of the ACL* (pp. 1715–1725). <https://aclanthology.org/P16-1162>

Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode normalization forms*. <<https://www.unicode.org/repo> rts/tr15/>
