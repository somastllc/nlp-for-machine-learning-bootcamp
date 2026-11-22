# Preprocessing

## Goal

Learn how to prepare raw, messy text for an NLP system without removing useful meaning: lowercasing, tokenization, punctuation handling, stopwords, stemming, lemmatization and regular expressions. The companion script is `code/preprocessing.py`.

## 1. What is text preprocessing?

**Text preprocessing** means applying rules or transformations to raw text before using it in an NLP system. A transformation might change uppercase letters to lowercase, identify words, normalize spaces, remove selected punctuation, or combine related word forms. Different tasks need different transformations.

Imagine that our spam dataset contains the following messages:

```text
FREE PRIZE!!!
Free prize!
free prize
```

A word-based system may initially treat `FREE`, `Free`, and `free` as different items, depending on its settings. By converting the text to lowercase, we can make them consistent. That can help a small classifier learn from more examples instead of treating every capitalization as a completely different feature.

However, notice the first message: `FREE PRIZE!!!`. The capital letters and repeated exclamation marks might be useful signs of an aggressive promotional message. If we remove them, we may lose information. **Preprocessing is a choice with benefits and costs**, not a set of rules that always makes text better.

### Common preprocessing techniques

| Technique | Simple meaning | Example | What to be careful about |
|---|---|---|---|
| **Lowercasing** | Make letters lowercase | `FREE` → `free` | Capitalization may matter |
| **Tokenization** | Identify pieces of text | `I love NLP!` → `I`, `love`, `NLP`, `!` | Different tokenizers split differently |
| **Punctuation handling** | Keep, remove, or standardize symbols | `hello!` → `hello` | `!`, URLs, and symbols may be useful |
| **Stopword removal** | Remove selected common words | Remove `the` from a sentence | Words such as `not` can change meaning |
| **Stemming** | Cut word endings using rules | `studies` → `studi` with Porter stemming | The result may not be a real word |
| **Lemmatization** | Find a word's dictionary form | `cars` → `car` | Needs suitable language knowledge |
| **Regex cleaning** | Find or replace text patterns | Replace repeated spaces | A broad rule can delete important text |

These are **options**, not mandatory steps. We can use some of them, all of them, or almost none of them, depending on the dataset and task.

## 2. Why do we need preprocessing?

Suppose our model is learning to identify messages about receiving a prize. Our dataset contains the phrases `Win a prize`, `WIN A PRIZE`, and `win a prize!`. To a human reader, they are clearly related. But a basic word-based representation may create separate features for differently capitalized or punctuated forms. Preprocessing can reduce unnecessary variation and make the data easier to learn from.

The opposite problem is also possible: a cleaning step can delete information the task needs. Section 5 shows how removing a single word such as `not` can reverse the meaning of a sentence.

A useful question before every preprocessing step is: **What difference are we trying to remove, and could that difference be useful to the model?** We should begin with simple processing, train a baseline model, and check whether additional cleaning improves performance on validation data.

## 3. Normalization: making text consistent

**Normalization** means making different written forms more consistent. It often includes lowercasing, standardizing whitespace, and carefully handling Unicode characters. Normalization is especially helpful when a model should treat minor writing differences as equivalent.

### Lowercasing

Lowercasing converts uppercase letters into lowercase letters. For example, `Python`, `PYTHON`, and `python` all become `python`. This can reduce the number of unique tokens in a word-based vocabulary because several versions of the same word are combined.

**Example:**

| Before | After lowercasing |
|---|---|
| `I LOVE Python` | `i love python` |
| `FREE`, `Free`, `free` | `free`, `free`, `free` (3 forms become 1 token) |

But lowercasing is not always best. In some applications, `US` (a country abbreviation) and `us` (a pronoun) have different meanings. Uppercase writing can also be a signal in spam messages. We should decide based on our task rather than applying lowercasing automatically.

### Whitespace normalization

Whitespace means spaces, tabs, and line breaks. People sometimes insert multiple spaces accidentally. A simple normalization rule can make them consistent.

**Example:**

```text
Before: "I   love    NLP"
After:  "I love NLP"
```

This is helpful for many text-classification tasks, but it may be unsuitable when document layout or formatting carries meaning.

### Unicode, numbers, URLs, and phone numbers

Unicode is the system computers use to represent characters from many writing systems. Some characters can have multiple equivalent encodings, so Unicode normalization can make their representations consistent.

**Example:**

| Written as | Characters stored |
|---|---|
| `café` (version A) | 4 characters: `c`, `a`, `f`, `é` |
| `café` (version B) | 5 characters: `c`, `a`, `f`, `e`, plus a separate accent mark |

The two look identical on screen but are different strings, so they become two different tokens. After Unicode normalization (NFC), both are stored as version A and become one token.

The same happens with quotes: a curly `’` in `don’t` and a straight `'` in `don't` are different characters, so `don’t` and `don't` become different tokens unless we normalize them. However, we should not casually delete accents, apostrophes, or non-English characters because those marks can be part of a language.

Numbers, URLs, email addresses, and phone numbers also need thought. For a spam classifier, a link like `https://example.com/win` or a phone number could be useful evidence. Removing all numbers and symbols might hurt the model. In another task, we might replace a specific phone number with a general marker such as `PHONE_NUMBER` so the model can learn that a number is present without memorizing every number.

**Example:**

| Choice | Result |
|---|---|
| Original | `Call 0612345678 or visit http://win-prize.com` |
| Replace with markers | `Call PHONE_NUMBER or visit URL` |
| Delete | `Call or visit` |

**Important:** Replacing and deleting are different decisions. Replacing a URL with a marker can preserve the fact that a link appeared; deleting it removes that information entirely.

### Emojis, slang, and abbreviations

Messages and reviews are full of informal writing: emojis such as 😂 or 👍, slang, and abbreviations such as `u` for “you,” `gr8` for “great,” or `pls` for “please.” A cleaning rule that keeps only letters deletes emojis completely, even though 😡 or 👍 can carry strong sentiment. Abbreviations create extra vocabulary entries for the same idea, so `u` and `you` are treated as different tokens.

There are three common choices. **Keep** emojis and abbreviations as they are and let the model learn from them. **Replace** them with words or markers, such as turning 👍 into `positive_emoji` or `u` into `you`. **Remove** them when they add noise for the task.

**Example:**

| Choice | Result |
|---|---|
| Original | `gr8 phone 👍 u should buy` |
| Keep | `gr8 phone 👍 u should buy` |
| Replace | `great phone positive_emoji you should buy` |
| Remove emojis | `gr8 phone u should buy` |

As with every other step, inspect real examples and test whether the choice helps.

## 4. Tokenization: dividing text into units

We discussed tokenization in the NLP Foundations lesson. A **token** is a unit of text that a tokenizer produces. It can be a whole word, part of a word, a character, or punctuation. In this bootcamp's classical NLP examples, we will mainly work with word-level tokens.

**Example:** the sentence `I love Python!`

| Method | Tokens | Count |
|---|---|---:|
| Split on spaces | `I` · `love` · `Python!` | 3 |
| NLTK `word_tokenize` | `I` · `love` · `Python` · `!` | 4 |

Splitting on spaces leaves the exclamation mark attached to `Python!`; the word tokenizer separates it. Neither result happens by magic; it depends on the tokenization rules.

### Three types of tokenizers

Tokenizers differ in **how big each token is**. There are three main types.

**1. Character-level tokenizer:** every single character, including spaces, is one token. This is an easy job: plain Python `list(text)` does it, with no special tool.

**2. Word-level tokenizer:** each word (and each punctuation mark) is one token. This is also easy: Python's `text.split()` splits on spaces, and NLTK's `word_tokenize` handles punctuation and contractions more carefully. This is what the classical models in this bootcamp use.

**Example:** the sentence `I love NLP!`

| Tokenizer type | Tool | Tokens | Count |
|---|---|---|---:|
| Character-level | Python `list(text)` | `I` · ` ` · `l` · `o` · `v` · `e` · ` ` · `N` · `L` · `P` · `!` | 11 |
| Word-level | NLTK `word_tokenize` | `I` · `love` · `NLP` · `!` | 4 |

**3. Subword tokenizer:** a word can be split into smaller pieces. Common words stay whole, while rare or long words are broken into parts. This is **not** an easy job, and we cannot do it by guessing which part of a word should be a token. A subword tokenizer must be **trained on a large corpus**: an algorithm such as **BPE** (Byte-Pair Encoding) counts which character sequences appear together most often and repeatedly merges them into new tokens, until it reaches a chosen vocabulary size. The result is a built tokenizer that is saved and reused. NLTK does not provide one; trained subword tokenizers are shared through libraries such as Hugging Face `transformers` (see Bonus 02).

Ready-made tokenizers exist for different languages:

- **English:** GPT-2's tokenizer is a well-known **byte-level BPE** (BBPE) tokenizer trained on English web text. Byte-level means it works on the bytes of the text, so it never meets an unknown character.
- **Somali:** we have built **[somast/somali-bpe-v2](https://huggingface.co/somast/somali-bpe-v2)**, a 48,000-token byte-level BPE tokenizer trained on a Somali NLP corpus.

**Example:** the published output of `somast/somali-bpe-v2` for one Somali sentence

```text
Input:  Caafimaadka carruurtu waa mudnaanteenna koowaad.
Tokens: Caafimaadka · Ġcarruurtu · Ġwaa · Ġmudnaan · teenna · Ġkoowaad · .
```

Most words stay whole, but the long word `mudnaanteenna` is split into `mudnaan` and `teenna`, pieces the tokenizer learned from its training corpus. The `Ġ` symbol marks a token that starts with a space.

| Question | Character-level | Word-level | Subword |
|---|---|---|---|
| **How hard to build** | Easy (Python) | Easy (Python or NLTK) | Hard: needs a large corpus and training |
| **Vocabulary size** | Very small (letters, digits, symbols) | Very large (every word form) | Medium (often 30K–50K pieces) |
| **New or misspelled word** | Always representable | Unknown; no token for it | Built from known pieces |
| **Tokens per sentence** | Many | Fewest | A few more than words |
| **Typical use** | Spelling-heavy tasks | Classical NLP (this bootcamp) | Modern language models |

**Vocabulary** is the collection of unique token entries recognized by a particular text representation. **Vocabulary size** is simply the number of entries in that collection. If a tokenizer recognizes 32,000 unique pieces, its vocabulary size is 32K. That does not tell us how many tokens one sentence or document will produce; the actual output count depends on the tokenizer and the text.

**Example:** the sentence `the cat saw the dog`

| Measure | Value |
|---|---|
| Tokens (occurrences) | `the` · `cat` · `saw` · `the` · `dog` = **5** |
| Vocabulary (unique tokens) | `{the, cat, saw, dog}` = **4** |

`the` appears twice, so it counts twice as a token but only once in the vocabulary.

### Punctuation and contractions

Consider the sentence:

```text
I can't recommend this.
```

Some tokenizers keep `can't` together, while others split it into pieces such as `ca` and `n't`. This matters because the sentence expresses a negative opinion. A cleaning rule that deletes the wrong part of the contraction can change what the model sees.

### Sentence splitting

Tokenization often divides text into words or subwords, while **sentence splitting** finds sentence boundaries. For example, `I like NLP. It is useful.` contains two sentences. However, a period does not always end a sentence; it can occur in decimals, abbreviations, and website addresses. NLTK's `sent_tokenize` uses a trained model of these cases instead of simply splitting at every period.

**Example:** `Dr. Ali paid $3.50 for the book. He left at 5 p.m. today.`

| Method | Result |
|---|---|
| Split at every period | `Dr` · `Ali paid $3` · `50 for the book` · `He left at 5 p` · `m` · `today` (wrong) |
| NLTK `sent_tokenize` | `Dr. Ali paid $3.50 for the book.` · `He left at 5 p.m. today.` (2 sentences) |
 Sentence splitting is useful for long reviews or articles, while many tasks, such as classifying a short SMS, can treat the whole text as one document.

## 5. Stopwords: should we remove common words?

**Stopwords** are words that appear very frequently in a language. In English, examples include `the`, `is`, `and`, and `at`. Some NLP pipelines remove selected stopwords to reduce the number of features or focus on other words.

**Example:**

```text
Before: The book is on the table
After:  book table
```

After removing NLTK's English stopwords, only `book` and `table` remain. We can still guess the topic, so this type of removal may be acceptable for some search or topic-analysis tasks.

But now consider these two reviews:

```text
This movie is good.
This movie is not good.
```

If we delete `not`, the two sentences may become nearly identical. That is a serious problem for **sentiment analysis**, where the word `not` changes the meaning from positive to negative. Other negation words, such as `never` and `no`, can matter too.

This is why we should not remove stopwords just because they are common. Common words can still carry essential meaning. In many classical NLP pipelines, **TF-IDF** already gives lower weights to words found in nearly every document, so explicit stopword removal is not always necessary.

### Example with NLTK

The teaching script demonstrates two cases. First, it uses NLTK's default English stopword list, which contains `not`. Then, it removes `not` from that list so the token survives preprocessing. The second version can preserve important negative meaning, but **preserving only `not` does not solve all negation problems**. We must also think about `no`, `never`, and contractions such as `don't`, depending on the task and tokenizer.

## 6. Stemming vs. lemmatization

These two techniques try to connect different forms of related words. They are useful because a word can appear in several forms even when the forms refer to a similar basic idea.

### What is stemming?

**Stemming** uses rules to remove or change word endings. For example, an English stemmer might reduce `studies` to `studi`. The result helps group related forms for a model, even though `studi` is not a normal English dictionary word.

Stemming is often relatively simple and fast. However, because it follows broad rules, it can sometimes cut too much or group words that should remain different. A stem is not guaranteed to be a valid word or to preserve every part of the original meaning.

### Types of stemmers in NLTK

NLTK provides several stemmers. They follow different rules, so the same word can produce different stems.

| Stemmer | How it works | Typical behavior |
|---|---|---|
| **Porter** | A classic set of English suffix-removal rules applied in steps | Moderate; the most widely used baseline |
| **Lancaster** | A more aggressive rule set that keeps cutting while rules match | Shorter stems; more words merge, including some that should stay different |
| **Snowball** | An improved version of Porter, also called Porter2, available for several languages | Usually a little more consistent than Porter |
| **RegexpStemmer** | Removes only the suffixes you list in a regular expression, such as `ing$` or `s$` | Fully under your control, but it knows nothing beyond your pattern |

**Example:** the same words through all four stemmers (real NLTK output; the Regexp stemmer uses the pattern `ing$|s$|e$|able$`):

| Word | Porter | Lancaster | Snowball | Regexp |
|---|---|---|---|---|
| `running` | `run` | `run` | `run` | `runn` |
| `studies` | `studi` | `study` | `studi` | `studie` |
| `fairly` | `fairli` | `fair` | `fair` | `fairly` |
| `generously` | `gener` | `gen` | `generous` | `generously` |
| `happiness` | `happi` | `happy` | `happi` | `happines` |
| `eating` | `eat` | `eat` | `eat` | `eat` |

Notice that Lancaster cuts `generously` all the way to `gen`, the same stem it gives `general` and `generate`, so three unrelated words merge. Snowball keeps `generous`, a real word. The Regexp stemmer leaves `fairly` and `generously` unchanged because its pattern has no rule for `ly`, and it produces `runn` because it only removes `ing`. No stemmer is “correct”; the useful one is the one that groups the right words for your task.

### What is lemmatization?

**Lemmatization** tries to find the dictionary form of a word, called a **lemma**. For example, `cars` can become `car`. With the correct grammatical information, `running` can be mapped to the base verb `run`.

Lemmatization usually relies on language resources and sometimes on **part-of-speech (POS)** information. If a tool does not know whether `running` is being used as a verb, noun, or adjective, the result may differ from what we expect. It is more linguistically informed than simple suffix removal, but that does not mean it always improves a model.

### Helping the lemmatizer with part of speech

NLTK's `WordNetLemmatizer` assumes every word is a **noun** unless told otherwise. As a noun, `running` stays `running`; with the hint `pos="v"` (verb), it becomes `run`. Likewise, `better` stays `better` by default but becomes `good` with `pos="a"` (adjective).

**Example:** (real NLTK output)

| Word | Default (noun) | With a POS hint |
|---|---|---|
| `running` | `running` | `run` (`pos="v"`) |
| `better` | `better` | `good` (`pos="a"`) |
| `studies` | `study` | `study` (`pos="v"`) |
| `cars` | `car` | `car` (`pos="n"`) |

`studies` and `cars` already work as nouns; `running` and `better` need the correct part of speech. In a full pipeline, a POS tagger can supply these hints automatically, which is one practical reason POS tagging matters.

| Question | Stemming | Lemmatization |
|---|---|---|
| **What does it do?** | Cuts or changes word endings | Looks for a dictionary form |
| **Does it always produce a real word?** | No | Usually yes |
| **Does it need language knowledge?** | Depends on the stemmer and its rules | Usually more language-specific resources |
| **Example** | `studies` → `studi` | `cars` → `car` |
| **Possible problem** | Cuts too aggressively | May choose the wrong lemma without context |

**Remember:** Stemming and lemmatization are *not mandatory*. A simple word or bigram model may already work well without them. The best approach is to compare models using validation results.

## 7. Regular expressions (regex): cleaning patterns

A **regular expression**, or **regex**, is a way to describe a text pattern. Instead of searching for one exact sentence, we can write a rule that matches certain kinds of characters or character sequences. Regex is often used to find numbers, URLs, repeated spaces, punctuation, or other text patterns.

The teaching script uses the function `re.sub()`, which finds a matching pattern and replaces each match with something else. The pattern `[^a-z\s]` means **any character that is not a lowercase English letter (`a` to `z`) or whitespace**. The code replaces each such character with a space.

For an English sentence, this can remove punctuation and digits:

```text
Before:            "Win $500 NOW!!!"
After lowercasing: "win $500 now!!!"
After regex:       "win      now   "
```

But this same rule can cause problems. It removes the `$500`, even though the amount might be useful for spam detection. It also removes apostrophes and characters outside the English alphabet. For multilingual projects, a simple “English letters only” regex is often too aggressive.

Regex can also be precise. **More examples:**

| Goal | Pattern | Before | After |
|---|---|---|---|
| Collapse repeated `!` | `!+` → `!` | `WIN NOW!!!` | `WIN NOW!` |
| Replace links with a marker | `https?://\S+` → `URL` | `Claim at http://win-prize.com today` | `Claim at URL today` |
| Replace phone numbers | `\b\d{9,10}\b` → `PHONE_NUMBER` | `Call 0612345678 now` | `Call PHONE_NUMBER now` |

**The lesson is not that regex is bad.** Regex is useful, but we need to understand exactly what each rule removes.

## 8. POS tagging and named entity recognition

Some NLP tools do more than clean text. Two important examples are **part-of-speech tagging** and **named entity recognition**. We introduce them here so we know what they are, even though our basic preprocessing pipeline does not require them.

**Part-of-speech (POS) tagging** identifies a word's grammatical role, such as noun, verb, or adjective. In `Amina reads books`, a POS tagger might identify `Amina` as a proper noun, `reads` as a verb, and `books` as a noun. This information can help lemmatization choose more appropriate word forms. In NLTK, `pos_tag` assigns tags such as `NNP` (proper noun), `VBZ` (verb) and `NNS` (plural noun).

**Example:** `Amina works at the United Nations in Nairobi.` (real NLTK output)

| Word | Tag | Meaning |
|---|---|---|
| `Amina` | `NNP` | Proper noun |
| `works` | `VBZ` | Verb, present tense |
| `at` | `IN` | Preposition |
| `the` | `DT` | Determiner |
| `United` `Nations` | `NNP` `NNPS` | Proper nouns |
| `in` | `IN` | Preposition |
| `Nairobi` | `NNP` | Proper noun |

**Named entity recognition (NER)** identifies names of people, organizations, places, or other entities inside text. For example, in `Amina lives in Muqdisho`, an NER system might identify `Amina` as a person and `Muqdisho` as a location. These are labels for **parts of a sentence**, which is different from assigning `spam` or `ham` to an entire message. In NLTK, `ne_chunk` takes POS-tagged tokens and groups them into entities such as `PERSON`, `ORGANIZATION` and `GPE` (a place such as a city or country). Its English model makes mistakes, especially with names it has rarely seen, so always read its output.

**Example:** the same sentence through `ne_chunk` (real NLTK output)

| Text span | Entity label | Correct? |
|---|---|---|
| `Amina` | `GPE` (place) | ❌ Amina is a person, but the model has rarely seen this name |
| `United Nations` | `ORGANIZATION` | ✅ |
| `Nairobi` | `GPE` (place) | ✅ |
| `Monday` | none | It is a date, which this model does not label |

## 9. Why preprocessing depends on the task

There is no single best preprocessing pipeline for every NLP application. A useful rule for one project may be harmful in another. For example, removing all numbers might be acceptable when studying broad document topics, but it could be a serious problem when identifying scam messages that promise `$5,000` or ask users to call a particular number.

| NLP task | Information we might want to keep | Why |
|---|---|---|
| **Spam detection** | URLs, money amounts, phone-number markers, repeated `!` | They may signal unwanted promotions |
| **Sentiment analysis** | `not`, `never`, `no`, intensifiers | They change how opinions are expressed |
| **Search** | Product names, technical words, sometimes word variants | People need relevant matches |
| **Named entity recognition** | Capitalization and punctuation | They can help identify names and boundaries |

**Start with a small, reasonable baseline.** Then add one preprocessing transformation, examine what it changes, and measure whether it helps. The right question is not “Did I clean the text?” but **“Did this cleaning make the NLP task work better?”**

## 10. Practical: the teaching script

The code for this lesson lives in `code/preprocessing.py`, our teaching script. Run `python code/download-nltk-data.py` once, then `python code/preprocessing.py` from the repository root.

The script follows this lesson in eleven numbered steps: lowercasing, whitespace normalization, regex cleaning, tokenization, stopwords and negation, the full pipeline, comparing stemmers, lemmatization with a part-of-speech hint, sentence splitting, POS tagging, and named entity recognition. Each step has a short comment explaining what to look for and prints its result under a heading, so students see every concept in the output. It is a teaching demonstration of an English pipeline, not a production cleaner.

## 11. Training and inference must use the same preprocessing

Remember the difference between **training** and **inference**. During training, an NLP system learns from text examples. During inference, it processes new text. Both stages must use a consistent text-processing policy.

Imagine that we lowercase all training messages, so the model learns the vocabulary token `prize`. But when the system receives a new message, we forget to lowercase it and pass `PRIZE` instead. Depending on the vectorizer, the new word may not match the feature the model learned. A small inconsistency can reduce performance.

The same problem applies to stopwords, punctuation, tokenization, and normalization. We should save the preprocessing rules together with the trained model so that both stages always match.

## 12. A careful preprocessing workflow

A sensible workflow begins with the task, not with a list of cleaning commands. First, inspect a small sample of raw text and note the common patterns: capital letters, links, prices, phone numbers, mixed languages, or unusual spelling. Next, build a simple baseline with minimal, explainable processing. Then introduce one change at a time and observe the result.

For example, test lowercasing first. Does it reduce unnecessary vocabulary differences? Does the model perform better or worse? Next, test whether stopword removal helps. Pay special attention to negation words. Then check whether removing punctuation loses useful information. Compare results on a validation set, and leave the final test set untouched until the approach is chosen.

**The order also matters.** Lowercasing before building the vocabulary merges capitalization variants. Removing punctuation before tokenization may split `can't` differently. Removing stopwords before creating two-word features (bigrams) may erase phrases such as `not good`.

**Example:** one message through every stage (real output of the teaching script, step 6)

| Stage | Result |
|---|---|
| Original | `Not every free offer is spam; meet me at 8 tonight!` |
| Lowercase | `not every free offer is spam; meet me at 8 tonight!` |
| Regex `[^a-z\s]` | `not every free offer is spam  meet me at   tonight` |
| Tokenize | `not` · `every` · `free` · `offer` · `is` · `spam` · `meet` · `me` · `at` · `tonight` |
| Remove stopwords | `every` · `free` · `offer` · `spam` · `meet` · `tonight` |
| Lemmatize | `every` · `free` · `offer` · `spam` · `meet` · `tonight` |

Two pieces of information were lost along the way: the regex deleted `8`, and the stopword list deleted `not`. The final tokens now suggest “every free offer is spam,” the opposite of the original message.

A good preprocessing pipeline is one that we can **explain, reproduce, inspect, and evaluate**.

## Lesson recap

- **Text preprocessing** prepares raw text for NLP, but not every cleanup step is useful.
- **Lowercasing** can merge spelling variants, while capitalization can sometimes carry meaning.
- **Tokenization** identifies the units of text the model will process; those units depend on the tokenizer.
- **Stopword removal** can reduce features but can also delete important words such as `not`.
- **Stemming** cuts words toward rough roots, and stemmers such as Porter, Lancaster, Snowball and Regexp cut differently; **lemmatization** aims for dictionary forms and works better with a part-of-speech hint.
- **Emojis, slang, and abbreviations** can be kept, replaced, or removed, depending on the task.
- **Regex** is useful for cleaning patterns, but an aggressive English-only rule can remove prices, numbers, punctuation, and characters from other languages.
- **POS tagging**, **NER**, and **sentence splitting** are useful NLP techniques, but a basic preprocessing pipeline does not require them.
- **Training and inference** must apply the same text transformations.
- **Evaluate each preprocessing choice** instead of assuming that more cleaning makes a better model.
