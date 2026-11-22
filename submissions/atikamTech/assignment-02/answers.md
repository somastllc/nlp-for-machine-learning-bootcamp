# Assignment 2 – Text Preprocessing

## Part A: Research Questions

### 1. What is text preprocessing?

Text preprocessing is the process of preparing raw text so that an NLP system can analyze it. It may include lowercasing, removing unnecessary spaces, tokenization, handling punctuation, removing selected stopwords, and stemming or lemmatization.

NLP systems need preprocessing because raw text often contains inconsistent formatting, spelling variations, extra spaces, and other details that can make analysis more difficult. However, more cleaning does not always mean better results. Removing important information can change the original meaning.

For example, in a customer review, the sentence “I am not happy” expresses negative sentiment. If preprocessing removes the word “not,” the remaining words may give the model a different impression. Therefore, preprocessing should depend on the purpose of the NLP task.

### 2. Text normalization

Text normalization changes different forms of text into a more consistent format.

Common examples include:

- **Lowercasing:** Changing uppercase letters to lowercase, such as `URGENT` to `urgent`.
- **Whitespace normalization:** Removing extra spaces and replacing them with a single space.
- **Unicode normalization:** Handling different Unicode representations of characters that may look identical but have different underlying representations.

Normalization can help when a system should treat `Apple` and `apple` as the same word, such as in a general word-frequency analysis. However, lowercasing can be harmful in tasks that depend on proper names or capitalization. For example, distinguishing `US` from `us` may matter in some contexts.

Unicode normalization is also useful when text comes from different sources, because visually similar characters can sometimes have different encoded representations.

### 3. Word-level, subword-level, and character-level tokenization

Tokenization divides text into smaller units called tokens.

| Tokenization type | Description | Vocabulary size |
|---|---|---|
| Word-level | Splits text into complete words | Usually large because different word forms are separate |
| Subword-level | Splits words into smaller meaningful or frequently occurring pieces | Usually smaller than word-level |
| Character-level | Splits text into individual characters | Usually small, but sequences become longer |

Word-level tokenization is easy to understand, but it can struggle with unfamiliar words. Subword tokenization can represent an unfamiliar word using smaller pieces that already exist in the vocabulary. Character-level tokenization can handle unusual spellings, but it produces longer sequences.

The choice of tokenizer affects the vocabulary, sequence length, and how a model represents words. For example, subword tokenization is useful for handling new or rare words without requiring every complete word to be in the vocabulary.

### 4. Stopwords

Stopwords are common words that are sometimes removed during text preprocessing because they may contribute little to a particular task. Examples in English include `the`, `is`, `and`, and `at`.

Removing stopwords can reduce the number of features and make some text-analysis tasks simpler. However, stopword removal can also remove important meaning. Words such as `not`, `no`, and `never` can change the meaning of a sentence.

For example:

- Original: `I am not happy`
- After removing `not`: `I am happy`

The meaning has changed. Stopword removal should therefore depend on the task and language. For sentiment analysis, keeping negation words is especially important. A stopword list designed for English may also be unsuitable for another language.

### 5. Stemming and lemmatization

Stemming and lemmatization both reduce related word forms, but they work differently.

**Stemming** removes word endings or applies rules to produce a shorter stem. The result does not always have to be a real dictionary word.

One well-known algorithm is the **Porter Stemmer**, which applies a sequence of suffix-removal rules. For example, `connected`, `connecting`, and `connection` can be reduced to `connect`.

**Lemmatization** uses vocabulary and grammatical information to return a word's base or dictionary form, called a lemma. For example, `running` can become `run`, and `better` can become `good` when the correct adjective information is supplied.

Stemming is often useful when reducing vocabulary size is important and an approximate stem is acceptable. Lemmatization is preferable when meaningful dictionary forms are important, although it may require more linguistic information.

### 6. Regular expressions (Regex)

Regular expressions are patterns used to find, match, remove, or replace parts of text. They are useful for cleaning predictable patterns such as phone numbers, currency values, and repeated punctuation.

**Example 1: Matching a phone number**

- Pattern: `\b\d{10}\b`
- Example text: `Call 0712345678 now`
- Match: `0712345678`

This pattern matches a sequence of exactly ten digits surrounded by word boundaries.

**Example 2: Matching repeated exclamation marks**

- Pattern: `!{2,}`
- Example text: `URGENT!!!`
- Match: `!!!`

This pattern matches two or more consecutive exclamation marks.

An overly broad regex can damage useful information. For example, the pattern `[^a-z\s]` removes every character that is not a lowercase English letter or whitespace. It can remove punctuation, numbers, currency symbols, apostrophes, and emoji. This may destroy important information, especially in financial messages, sentiment analysis, or spam detection.

### 7. Case study: Preprocessing for Twitter sentiment analysis

Symeonidis, Effrosynidis, and Arampatzis (2018) compared 16 preprocessing techniques using two Twitter datasets and four machine-learning algorithms for sentiment analysis. They evaluated how the techniques affected classification accuracy and the number of features.

Their findings showed that the effects of preprocessing techniques were different. Lemmatization, number removal, and replacing contractions improved accuracy in their experiments, while removing punctuation did not always help. They also examined how combining techniques affected results.

This case study shows that preprocessing choices should be tested for the specific NLP task rather than applying every cleaning method automatically. A technique that works for one dataset or model may not work equally well for another.

## Part B: Practical Work

The practical tasks were implemented in Python using NLTK and the `re` module. The results below are from running `preprocessing_practice.py`.

### Task 1: Lowercasing

The T1 text was converted to lowercase.

- Before: `URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!`
- After: `urgent!!! you have won a £1,000 gift card. call 0712345678 now!`

Lowercasing makes the text more consistent by treating uppercase and lowercase versions of a word similarly. However, a spam detector should not automatically remove all capitalization. Words such as `URGENT` and `NOW` may be useful signals because scammers often use capital letters to attract attention.

### Task 2: Whitespace normalization

The original T4 text contained multiple spaces between words.

- Before: `'   The   delivery   was   fast,   and the  packaging   was   great.'`
- After: `'The delivery was fast, and the packaging was great.'`

Whitespace normalization makes the text easier to process while keeping the words and sentence meaning unchanged.

### Task 3: Regex cleaning

The generic pattern `[^a-z\s]` was applied to the lowercase T1 text.

It removed characters including:
- Exclamation marks
- The pound sign
- Digits in the money amount
- The comma in the amount
- Digits in the phone number
- The full stop and other punctuation

This demonstrates why a general cleaning pattern can be too aggressive. The custom replacements preserved useful information by replacing the phone number with `PHONE_NUMBER`, the money amount with `MONEY`, and repeated exclamation marks with a single `!`.

### Task 4: Tokenization

The T2 text was tokenized using both Python's `split()` and NLTK's `word_tokenize()`.

| Method | Token count |
|---|---:|
| Python `split()` | 14 |
| NLTK `word_tokenize()` | 16 |

The difference occurs because `word_tokenize()` handles punctuation and contractions differently. For example, it separates `don't` into `do` and `n't`, and treats the full stop as a separate token.

For T4, the result was:

- Token count: **11**
- Vocabulary size: **10**

The vocabulary size is smaller than the token count because the word `was` appears more than once.

### Task 5: Stopword removal

The default NLTK English stopword list and a custom list that preserves negation were compared.

The default stopword removal result was:

`["n't", "think", "jacket", "worth", "price", ".", "happy", "😞"]`

The custom list preserved negation:

`["n't", "think", "jacket", "worth", "price", ".", "not", "happy", "😞"]`

The comparison shows that the default list removed `not`, while the custom list kept it. This is important because removing negation can change the meaning of a complaint or review.

### Task 6: Stemming comparison

The three NLTK stemmers produced the following results:

| Original word | Porter | Lancaster | Snowball |
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

The three stemmers produced similar results for several words, but they did not always produce the same stem. For example, Lancaster reduced `organization` to `org`, while Porter and Snowball produced `organ`.

The words `connection`, `connected`, and `connecting` were reduced to the same stem, `connect`. Some results, such as `univers`, `fli`, and `wolv`, are not complete dictionary words. This demonstrates that stemming focuses on reducing word forms rather than always producing a meaningful dictionary word.

### Task 7: Lemmatization

The WordNet lemmatizer was tested first with its default settings and then with part-of-speech (POS) information.

| Word | Default result | With POS hint |
|---|---|---|
| running | running | run (verb) |
| connected | connected | connect (verb) |
| flies | fly | fly (verb) |
| wolves | wolf | wolf (noun) |
| better | better | good (adjective) |
| ran | ran | run (verb) |

The default WordNet lemmatizer assumes a noun unless another POS is provided. This is why some words did not change initially. Supplying the correct POS allows the lemmatizer to identify the intended grammatical form and produce a more appropriate lemma.

### Task 8: Sentence tokenization

NLTK sentence tokenization divided T3 into two sentences:

1. `Mr. Smith arrived in London on Friday.`
2. `He met the Apple team at 9 a.m.`

The sentence count was **2**. The test also showed that `Mr.` and `a.m.` were preserved rather than incorrectly splitting the sentences at those abbreviations.

This demonstrates that sentence tokenization needs to recognize that some periods belong to abbreviations rather than sentence endings.

### Task 9: POS tagging and named entity recognition

POS tagging identified grammatical categories such as proper nouns, verbs, prepositions, and determiners.

The NLTK named entity recognition output identified these entities:

| Entity found | NLTK label |
|---|---|
| Mr. | PERSON |
| Smith | PERSON |
| London | GPE |

The output did not identify `Apple` as an organization or `Friday` as a date-related entity. It also split `Mr. Smith` into two separate PERSON entities instead of one complete name.

In this sentence, Apple refers to a company, not the fruit. This demonstrates that automated named entity recognition can make mistakes and may require human review or a more advanced model.

### Task 10: Full preprocessing pipeline

The full preprocessing pipeline was tested using T2. It applied lowercasing, regex cleaning, tokenization, stopword removal, and lemmatization.

The regex was adjusted to preserve apostrophes. As a result, `don't` remained unchanged during cleaning, and NLTK separated it into `do` and `n't` during tokenization.

With the default stopword list, the final tokens were:

`["n't", "think", "jacket", "worth", "price", "happy"]`

With the custom stopword list, which preserves negation, the final tokens were:

`["n't", "think", "jacket", "worth", "price", "not", "happy"]`

The comparison shows that the default stopword list removes `not`, while the custom list keeps it. Preserving negation is important for sentiment analysis because it helps the model understand whether a customer is satisfied or dissatisfied.

### Preprocessing for a spam detector

For a spam detector, preprocessing should preserve information that may help identify suspicious messages. T1 contains urgency, a money amount, a phone number, and repeated exclamation marks. Lowercasing makes the text more consistent, but removing all numbers and punctuation can remove useful spam indicators. Instead, replacing values with labels such as `MONEY` and `PHONE_NUMBER` can preserve their meaning without keeping the original details. The results show that preprocessing should be selected according to the task rather than removing as much text as possible.

## References

1. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

3. NLTK Project. (n.d.). *NLTK: Natural Language Toolkit documentation*. https://www.nltk.org/

4. Porter, M. F. (n.d.). *The Porter stemming algorithm*. Snowball. https://snowballstem.org/algorithms/porter/stemmer.html

5. Unicode Consortium. (n.d.). *Unicode normalization*. https://www.unicode.org/faq/normalization.html