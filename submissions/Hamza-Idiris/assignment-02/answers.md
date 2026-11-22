**# Assignment 2 -- Preprocessing**



**\*\*Student:\*\*** \Hamza Mohamed Idiris\\

**\*\*Course:\*\*** Natural Language Processing\\

**\*\*Assignment:\*\*** 2 -- Preprocessing\\



\------------------------------------------------------------------------



**# Part A -- Research**



**## 1. Text Preprocessing: Purpose and Risks**



Text preprocessing is the set of operations used to prepare raw text for

analysis or for use by an NLP model. Raw text collected from websites,

messages, reviews, or documents often contains inconsistent

capitalization, extra spaces, unusual symbols, spelling variations, and

formatting that may not be useful for a particular task. Preprocessing

can standardize these differences, divide text into tokens, and

transform words into forms that are easier for a system to analyse

(Palomino & Aider, 2022).



NLP systems need preprocessing because text from real sources is rarely

consistent. For example, the same word may appear in uppercase and

lowercase, sentences may contain repeated spaces, and social-media

messages may include emojis or informal contractions. Preparing this

material can reduce unnecessary variation and make the data more

suitable for tokenization, feature extraction, and machine-learning

models.



However, more cleaning is not automatically better. A cleaning rule can

remove information that is important to the task. For example, consider

a customer review that says, "I do not recommend this service." If

preprocessing removes the word "not" as a stopword, the remaining words

may suggest a positive recommendation even though the customer is

expressing a negative opinion. Similarly, removing all digits from a

message could remove a phone number or a monetary amount that is

important for fraud or spam detection. Preprocessing should therefore be

selected according to the goal of the NLP application rather than

applied as a fixed list of cleaning operations.



**## 2. Text Normalization**



Text normalization converts different forms of text into a more

consistent representation. It can include lowercasing, whitespace

normalization, Unicode normalization, and carefully selected

replacements. The purpose is to reduce differences that should not

affect the task.



**\*\*Lowercasing\*\*** changes uppercase letters into lowercase letters. For

example, \`DELIVERY\`, \`Delivery\`, and \`delivery\` become the same form.

This can reduce vocabulary size when capitalization is not meaningful.

However, case can carry information: \`US\` may refer to the United

States, while \`us\` is a pronoun. In named-entity recognition,

capitalization can also help identify names and organizations.



**\*\*Whitespace normalization\*\*** replaces repeated spaces, tabs, or line

breaks with a consistent separator and removes unnecessary spaces at the

beginning or end of a text. For example, \`The   service  was fast\`

becomes \`The service was fast\`. This is useful because accidental

spacing should not normally create different tokens.



**\*\*Unicode normalization\*\*** handles characters that may look identical but

have different underlying Unicode representations. Unicode provides

normalization forms such as NFC and NFKC. NFC combines canonically

equivalent character sequences into a consistent representation. NFKC

also applies compatibility transformations, which can make some visually

or functionally similar characters consistent. NFKC should be used

carefully because compatibility transformations may remove distinctions

that matter in specialist text, identifiers, or mathematical notation.



A useful situation for normalization is cleaning a collection of

customer reviews where repeated spaces and inconsistent capitalization

would otherwise create unnecessary differences. A situation where

normalization may remove useful information is a language-identification

or named-entity task in which capitalization distinguishes an

abbreviation from an ordinary word. The correct choice depends on the

data and the objective.



**## 3. Word-Level, Subword, and Character-Level Tokenization**



Tokenization divides a text sequence into smaller units called tokens. A

tokenizer determines what counts as a unit and therefore influences the

vocabulary and the way a model represents text.



  -------------------------------------------------------------------------------

  Tokenization      Description    Example for      Main advantage Main

  level                            \`unhappiness\`                   limitation

  ----------------- -------------- ---------------- -------------- --------------

  Word-level        Treats each    \`unhappiness\`    Tokens are     Rare or unseen

                    complete word                   easy for       words may not

                    as one token.                   people to      exist in the

                                                    interpret.     vocabulary.



  Subword-level     Splits words   \`un\`, \`happi\`,   Can represent  A word may be

                    into frequent  \`ness\`           unfamiliar     split into

                    pieces learned (illustrative)   words using    pieces that

                    from data or                    known pieces   are not

                    defined by an                   and control    meaningful by

                    algorithm.                      vocabulary     themselves.

                                                    size.          



  Character-level   Treats         \`u\`, \`n\`, \`h\`,   Can represent  Sequences

                    individual     \`a\`, \`p\`, \`p\`,   any spelling   become longer,

                    characters as  \`i\`, \`n\`, \`e\`,   made from the  and the model

                    tokens.        \`s\`, \`s\`         character set  must learn

                                                    and handles    word structure

                                                    unusual forms. from smaller

                                                                   units.

  -------------------------------------------------------------------------------



**\*\*Vocabulary size\*\*** is the number of distinct tokens available to a

tokenizer or model. In a word-level system, the vocabulary may contain

many complete words. A subword tokenizer usually has a controlled

vocabulary of reusable word pieces. A character-level vocabulary is

often much smaller because it contains letters, digits, punctuation, and

other characters rather than every complete word.



The tokenizer affects both vocabulary size and sequence length.

Word-level tokenization may produce short sequences but require a large

vocabulary. Character-level tokenization uses fewer token types but

produces longer sequences. Subword tokenization aims to balance these

trade-offs. The best choice depends on the language, the amount of

training data, and the task.



**## 4. Stopwords: Benefits and Risks**



Stopwords are words that occur frequently in a language and are

sometimes removed because they contribute less to a particular analysis.

Common English examples include articles, prepositions, and conjunctions

such as \`the\`, \`in\`, and \`and\`. Stopword lists are not universal

linguistic rules; they are practical lists designed for particular

applications.



Removing stopwords can reduce the number of features in a document

collection and make some information-retrieval or topic-analysis tasks

more efficient. For example, in a simple keyword search, words such as

\`the\` and \`of\` may contribute little to identifying the subject of a

document.



However, stopword removal can also damage meaning. Words such as \`not\`,

\`no\`, and \`never\` express negation. Removing them from a

sentiment-analysis example can reverse or weaken the meaning. In the

sentence "The delivery was not good," removing \`not\` leaves words that

may be interpreted as positive. Function words can also be important in

authorship analysis, grammar research, question answering, and other

tasks.



The correct stopword list depends on the task and the language. A list

created for English should not be applied to Somali or another language

without linguistic checking. Even within English, a list appropriate for

search may be unsuitable for sentiment analysis. A careful system should

test whether removing each group of words improves the intended task.



**## 5. Stemming and Lemmatization**



Stemming and lemmatization both reduce related word forms, but they use

different methods.



**\*\*Stemming\*\*** applies rules that remove or alter word endings to produce

a stem. A stem does not have to be a valid dictionary word. Stemming is

generally faster and can be useful when approximate matching is more

important than grammatical accuracy.



One well-known algorithm is the **\*\*Porter stemming algorithm\*\***,

introduced by Martin Porter (1980). It applies a sequence of

suffix-removal rules to English words. For example, it can reduce

inflected forms such as \`connected\` and \`connecting\` toward a shared

stem. The **\*\*Lancaster stemmer\*\*** is another rule-based stemmer and is

generally more aggressive, meaning it may shorten words more strongly.

NLTK also provides the **\*\*Snowball stemmer\*\***, an improved family of

stemming algorithms for several languages, including English.



**\*\*Lemmatization\*\*** attempts to return a word to its dictionary base form,

called a lemma. It uses vocabulary and linguistic information, and it

can use the word's part of speech. For example, the verb \`ran\` can be

mapped to \`run\`, while the noun \`wolves\` can be mapped to \`wolf\`. The

result is intended to be a valid lexical form rather than an arbitrary

stem.



NLTK's \`WordNetLemmatizer\` uses WordNet and accepts a part-of-speech

hint. Its default part of speech is noun (\`pos="n"\`). Providing the

correct hint can change the result: a verb such as \`ran\` should be

lemmatized with \`pos="v"\` to obtain \`run\`. A part-of-speech tagger can

help select an appropriate hint in a larger system, although tagging

errors can also affect lemmatization.



  -----------------------------------------------------------------------

  Aspect                  Stemming                Lemmatization

  ----------------------- ----------------------- -----------------------

  Method                  Rule-based removal or   Vocabulary and

                          transformation of word  morphological analysis

                          endings                &#x20;



  Output                  A stem that may not be  A dictionary lemma

                          a dictionary word      &#x20;



  Speed                   Usually faster          Usually more

                                                  computationally

                                                  involved



  Linguistic context      Often does not require  Can use part of speech

                          part of speech          and word meaning



  Suitable use            Fast search or          Tasks where readable,

                          approximate matching    linguistically valid

                                                  base forms matter

  -----------------------------------------------------------------------



A stemmer may be preferred for a large search index where speed and

matching related forms are important. Lemmatization may be preferred for

a language-analysis or classification task where valid base forms and

grammatical context matter. Neither method should be assumed to improve

every NLP task; the effect should be tested.



**## 6. Regular Expressions in Text Cleaning**



A regular expression (regex) is a pattern that describes a set of

strings. Python's \`re\` module can use regex patterns to find, remove,

split, or replace parts of text. Regex is useful for predictable

patterns such as phone numbers, repeated whitespace, dates, or unwanted

formatting.



Two examples are:



  --------------------------------------------------------------------------------------

  Regex pattern                          What it matches         Possible use

  -------------------------------------- ----------------------- -----------------------

  \`\b0\d{9}\b\`                           A standalone sequence   Finding a ten-digit

                                         beginning with \`0\`      phone-number pattern in

                                         followed by exactly     the supplied practice

                                         nine digits.            text.



  \`£\s?\d{1,3}(?:,\d{3})\*(?:\\.\d{2})?\`   A pound sign, optional  Finding a monetary

                                         whitespace, a number    amount such as \`£1,000\`

                                         with optional thousands or \`£25.50\`.

                                         groups, and optional    

                                         two-decimal digits.    &#x20;

  --------------------------------------------------------------------------------------



Regex can also normalize repeated punctuation. For example, \`!{3,}\`

matches three or more consecutive exclamation marks and can replace them

with one exclamation mark.



An overly broad pattern can damage text by deleting meaningful

information. For example, \`[^a-z\s]\` removes every character that is not

a lowercase English letter or whitespace. Applied to the supplied

message, it removes digits, punctuation, the pound sign, and other

symbols. This would erase the phone number and amount, and it could

remove emojis or characters from languages that do not use the English

alphabet. Replacing important patterns with labels is often safer than

deleting them.



**## 7. Case Study: Comparing Preprocessing Techniques for Twitter Sentiment Analysis**



Symeonidis, Effrosynidis, and Arampatzis (2018) studied how

preprocessing choices affect sentiment classification on Twitter data.

The task was sentiment analysis, in which a system predicts the

sentiment category of a tweet. The authors compared 16 commonly used

preprocessing techniques across two Twitter datasets and evaluated them

with four machine-learning algorithms: Linear Support Vector

Classification, Bernoulli Naive Bayes, Logistic Regression, and

Convolutional Neural Networks.



The researchers examined the effect of individual preprocessing methods

and also investigated how methods interacted when combined in a

pipeline. Their evaluation considered classification accuracy and the

number of features produced. According to their reported findings,

lemmatization, removing numbers, and replacing contractions improved

accuracy in their experiments, while removing punctuation did not

provide the same benefit. Their combination and ablation experiments

also showed that replacing numbers and reducing repeated punctuation

could matter when preprocessing methods were used together.



This case study demonstrates that preprocessing is not simply a matter

of removing as much text as possible. A technique may help one dataset

or model but provide little benefit in another. The study is relevant

because it compares multiple preprocessing choices for a real NLP task

and evaluates their effects rather than assuming that every cleaning

method is useful.



\------------------------------------------------------------------------



**# Part B -- Practical**



## Environment and reproducibility

This practical work was completed using Python 3.13, NLTK 3.10.4, and NumPy 2.5.5 on Windows. The required NLTK data packages were downloaded locally, including `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker_tab`, and `words`.

The repository did not contain a `requirements.txt` file or `code/download-nltk-data.py`, so NLTK and its data packages were installed/downloaded directly in the local environment.

From the repository root, the practical script was run and its complete output was saved with:

```powershell
$env:PYTHONIOENCODING = "utf-8"
cmd /c "set PYTHONIOENCODING=utf-8&& python submissions/Hamza-Idiris/assignment-02/preprocessing_practice.py > submissions/Hamza-Idiris/assignment-02/output.txt"
```

The results below are based on the generated `output.txt`.

## Practical results

### Task 1 -- Lowercasing

Original T1:

`URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!`

After lowercasing:

`urgent!!! you have won a £1,000 gift card. call 0712345678 now!`

Lowercasing changes uppercase letters to lowercase but leaves punctuation, numbers, and symbols unchanged. A spam detector may lose the uppercase emphasis in words such as `URGENT` and `WON`, which can be useful signals of promotional or urgent messages.

### Task 2 -- Whitespace normalization

Before:

`'   The   delivery   was   fast,   and the  packaging   was   great.'`

After:

`'The delivery was fast, and the packaging was great.'`

The repeated spaces and leading spaces were reduced to single spaces, and the leading/trailing whitespace was removed. The words and punctuation were preserved.

### Task 3 -- Regex cleaning

Applying `[^a-z\s]` to lowercased T1 produced:

`urgent you have won a  gift card call  now`

The pattern removed characters that were not lowercase English letters or whitespace. In this example, those included exclamation marks, the pound sign, digits, the comma, and the full stop. The phone number and monetary amount were therefore lost.

The custom replacements produced:

`urgent! you have won a MONEY gift card. call PHONE_NUMBER now!`

This is more informative because the phone number and currency amount are represented by meaningful labels instead of being deleted. The repeated exclamation marks were also reduced to one.

### Task 4 -- Tokenization

For T2, `split()` returned 14 tokens:

`['I', "don't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price.', 'Not', 'happy', 'at', 'all', '😞']`

`word_tokenize()` returned 16 tokens:

`['I', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', '.', 'Not', 'happy', 'at', 'all', '😞']`

The counts differ because `word_tokenize()` separates the contraction `don't` into `do` and `n't`, and separates the full stop from `price.`.

For T4, the token count was **11** and the vocabulary size (number of unique tokens) was **9**. The unique tokens were comma, full stop, `and`, `delivery`, `fast`, `great`, `packaging`, `the`, and `was`.

### Task 5 -- Stopwords and negation

After applying the NLTK English stopword list, the remaining tokens were:

`["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']`

After applying the custom stopword list that protects negation, the remaining tokens were:

`["n't", 'think', 'jacket', 'worth', 'price', '.', 'not', 'happy', '😞']`

The custom list retains `not` as well as the contraction token `n't`. Negation is important in sentiment analysis: removing it can change or weaken the meaning of a negative review. For example, `not happy` should not be treated as equivalent to `happy`.

### Task 6 -- Stemming comparison

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

All three stemmers merged different words into shared stems. For example, `connection`, `connected`, and `connecting` all became `connect`. `organization` and `organized` also shared a stem with Porter and Snowball (`organ`) and Lancaster (`org`).

Some generated stems, such as `organ`, `univers`, `fli`, and `wolv`, are not ordinary complete English dictionary words. This is expected because stemming uses rule-based suffix removal and does not guarantee a valid dictionary form. Lancaster was more aggressive for some words; for example, it reduced `better` to `bet`.

### Task 7 -- WordNet lemmatization

| Word | Default (noun) | POS hint | With hint |
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

The words that changed only after a part-of-speech hint was supplied were `connected`, `connecting`, `organized`, `better`, and `ran`. The grammatical role matters because the same spelling can have different base forms depending on whether it is used as a noun, verb, or adjective. For example, `ran` becomes `run` when it is identified as a verb.

### Task 8 -- Sentence splitting

NLTK returned **2 sentences**:

1. `Mr. Smith arrived in London on Friday.`
2. `He met the Apple team at 9 a.m.`

Neither `Mr.` nor `a.m.` caused an incorrect sentence boundary in this output. Both abbreviations remained within their intended sentences.

### Task 9 -- POS tagging and named entity recognition

The entities extracted from the actual NLTK output were:

| Entity text | NLTK label | Correct? | Explanation |
|---|---|---|---|
| Mr. | PERSON | Partially | It is a title associated with the following person's name, rather than a person's name by itself. |
| Smith | PERSON | Yes | Smith is the person's surname in the sentence. |
| London | GPE | Yes | London is a geographical/political location. |

The NER output did **not** identify `Apple` as an organization. It was left as an ordinary token in the tree, even though the sentence refers to the Apple team. It also did not label `Friday` as a date or time entity. Therefore, the model did not identify every relevant entity in the sentence. This demonstrates that NLTK's named-entity chunker can miss entities or assign incomplete spans, even when the intended meaning is clear from context.

### Task 10 -- Full preprocessing pipeline

With the default NLTK stopword list, the final tokens were:

`["n't", 'think', 'jacket', 'worth', 'price', 'happy']`

With the negation-preserving stopword list, the final tokens were:

`["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']`

The second version retains the word `not`, so it preserves more of the negative meaning of the customer's complaint. This comparison shows why stopword lists should be adapted to the NLP task rather than applied without checking their effect on meaning.

**## Preprocessing choices for a spam detector**



For a spam detector, I would normalize repeated whitespace and use

carefully designed regex rules to identify structured information such

as phone numbers, currency amounts, and suspicious links. I would

consider lowercasing for ordinary word matching, but preserve separate

signals for uppercase emphasis and repeated punctuation because these

may help identify promotional or urgent messages. I would avoid blindly

deleting all digits, symbols, punctuation, or stopwords: amounts,

contact numbers, exclamation marks, and wording can all be useful

evidence. I would compare preprocessing variants on a labelled

validation dataset before choosing the final pipeline.



\------------------------------------------------------------------------



**## References**



Bird, S., Klein, E., & Loper, E. (2009). \*Natural language processing

with Python: Analyzing text with the natural language toolkit\*. O'Reilly

Media. https\://www\.nltk.org/book/



Jurafsky, D., & Martin, J. H. (2024). *\*Speech and language processing\**

(3rd ed. draft). https\://web.stanford.edu/\\\~jurafsky/slp3/



Loper, E., & Bird, S. (2002). NLTK: The natural language toolkit. In

\*Proceedings of the ACL-02 Workshop on Effective Tools and Methodologies

for Teaching Natural Language Processing and Computational Linguistics\*

(pp. 63--70). Association for Computational Linguistics.

https\://doi.org/10.3115/1118108.1118117



Manning, C. D., Raghavan, P., & Schütze, H. (2008). \*Introduction to

information retrieval\*. Cambridge University Press.

https\://nlp.stanford.edu/IR-book/



Palomino, M. A., & Aider, F. (2022). Evaluating the effectiveness of

text pre-processing in sentiment analysis. *\*Applied Sciences, 12\**(17),

8765\. https\://doi.org/10.3390/app12178765



Porter, M. F. (1980). An algorithm for suffix stripping. \*Program,

14\*(3), 130--137. https\://doi.org/10.1108/eb046814



Unicode Consortium. (n.d.). \*Unicode Standard Annex #15: Unicode

normalization forms\*. https\://unicode.org/reports/tr15/



NLTK Project. (n.d.). *\*NLTK documentation\**. https\://www\.nltk.org/



Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative

evaluation of pre-processing techniques and their interactions for

Twitter sentiment analysis. *\*Expert Systems with Applications, 110\**,

298--310. https\://doi.org/10.1016/j.eswa.2018.06.022
