# Text Preprocessing in Natural Language Processing (NLP)

## Introduction

Text preprocessing is a set of operations used to prepare raw text for
analysis by a computer system. It may include normalization,
tokenization, removal of unwanted formatting, spelling correction,
stopword handling, stemming, and lemmatization. These operations can
make text more consistent and easier to process, but they are not
automatically beneficial in every situation.

The appropriate preprocessing strategy depends on the language, the
source of the text, the model, and the objective of the NLP task.
Research has shown that a preprocessing operation can improve one
dataset while reducing performance on another (Chai, 2023;
HaCohen-Kerner et al., 2020).

This assignment examines seven areas: the purpose of preprocessing, text
normalization, tokenization, stopwords, stemming and lemmatization,
regular expressions, and a published experimental case study.

------------------------------------------------------------------------

## 1. Text Preprocessing: Definition, Purpose, and Risks of Over-Cleaning

### 1.1 Definition

**Text preprocessing** is the process of transforming raw text into a
form that is more suitable for a particular NLP task. It includes
operations that standardize text, divide it into units, correct or
remove unwanted material, and prepare useful linguistic information for
later analysis.

Raw text collected from websites, emails, social media, documents, or
customer messages is rarely uniform. It may contain HTML tags,
inconsistent spacing, different letter cases, spelling errors, emojis,
URLs, punctuation, and unusual character encodings. A preprocessing
pipeline handles selected parts of this variation before the text is
passed to an NLP model.

A typical preprocessing pipeline may contain:

1.  **Text extraction:** obtaining the actual text from a file, webpage,
    or message.
2.  **Normalization:** standardizing selected text forms, such as
    Unicode representation or repeated whitespace.
3.  **Tokenization:** dividing text into words, subwords, or characters.
4.  **Noise handling:** removing or replacing irrelevant markup and
    unwanted artifacts.
5.  **Linguistic processing:** optionally applying stopword filtering,
    stemming, or lemmatization.
6.  **Representation:** converting the processed text into features or
    numerical vectors for a model.

Not every project needs every stage. For example, removing HTML tags may
be important when collecting webpage text but unnecessary for a dataset
that already contains plain SMS messages.

### 1.2 Why NLP systems need preprocessing

Preprocessing can help an NLP system in several ways:

-   **Consistency:** equivalent text forms can be represented in a
    consistent way.
-   **Noise reduction:** irrelevant webpage markup or duplicated
    formatting can be removed.
-   **Better matching:** normalization and morphological processing can
    help a search system match related word forms.
-   **Smaller feature spaces:** carefully selected filtering can reduce
    the number of distinct terms a traditional model must process.
-   **Improved model performance:** appropriate preprocessing can
    improve classification, retrieval, or other task-specific results.
-   **Data quality:** preprocessing can expose malformed records and
    make the dataset easier to inspect.

For example, consider customer messages copied from a website:

``` text
<div>Order received!</div>
<div>Order    delayed.</div>
```

If the task is to classify customer complaints, HTML tags can usually be
removed and repeated spaces can be standardized:

``` text
Order received!
Order delayed.
```

The resulting text is easier to tokenize and analyze.

### 1.3 Why more cleaning can make results worse

Cleaning is not the same as improving. A word or symbol that looks
unimportant to a general-purpose cleaning rule may carry essential
information for a particular task.

Consider these two product reviews:

-   "The battery is good."
-   "The battery is not good."

If preprocessing removes the word **not**, both sentences may become:

-   "The battery is good."
-   "The battery is good."

A sentiment classifier would lose the distinction between a positive and
a negative opinion. This is an example of destructive preprocessing: the
system removes information that is necessary for the task.

Other examples include:

-   Removing question marks when the task is to identify questions.
-   Removing emojis when analyzing social-media sentiment.
-   Removing numbers when extracting prices, dates, or medicine dosages.
-   Converting all text to lowercase when capitalization distinguishes a
    name, abbreviation, or code.
-   Removing punctuation when punctuation is important for identifying
    sentence boundaries or expressions.
-   Deleting words from a legal or medical document because they occur
    frequently.

### 1.4 Real-life example: Customer-service sentiment analysis

Imagine an online store that analyzes customer messages to identify
dissatisfied customers.

Original message:

> "The delivery was NOT fast, and the package arrived damaged!!!"

An aggressive cleaning pipeline might lowercase the text, remove
punctuation, and remove stopwords:

``` text
delivery fast package arrived damaged
```

This pipeline removes the uppercase emphasis and the negation word
"NOT". It may therefore weaken the evidence that the customer is
dissatisfied.

A more appropriate pipeline would preserve negation and useful emphasis
while removing only irrelevant formatting:

``` text
The delivery was not fast, and the package arrived damaged!!!
```

The system can then use the remaining words and punctuation to estimate
the customer's sentiment.

**Main principle:** preprocessing should preserve information needed by
the intended task. The goal is not to produce the shortest or
cleanest-looking text; it is to prepare text without destroying its
useful meaning.

Chai (2023) emphasizes that preprocessing decisions should be customized
to the corpus and NLP application. HaCohen-Kerner et al. (2020) also
demonstrate experimentally that different preprocessing combinations do
not have the same effect across datasets.

------------------------------------------------------------------------

## 2. Text Normalization

### 2.1 Definition

**Text normalization** is the process of converting different forms of
text into a consistent representation according to a defined set of
rules.

The purpose is to make text comparison and processing more reliable.
Normalization may involve lowercasing, whitespace standardization,
Unicode normalization, or carefully selected spelling and character
transformations.

Normalization should be distinguished from general cleaning:
normalization standardizes forms, whereas cleaning can also remove or
replace material considered irrelevant.

### 2.2 Common normalization operations

#### A. Lowercasing

Lowercasing converts uppercase letters into lowercase letters.

Before:

``` text
UNIVERSITY Library SERVICES
```

After:

``` text
university library services
```

**When it helps:** In a general document search system, a search for
"Library" should often match a document containing "library".
Lowercasing prevents capitalization alone from creating separate word
forms.

**When it can remove useful information:** In some datasets,
capitalization distinguishes abbreviations, proper names, or code
identifiers. For example, `US` may refer to the United States, while
`us` is a pronoun. In programming-related text, `FileName` and
`filename` may refer to different identifiers in a case-sensitive
language.

Therefore, lowercasing is useful when case distinctions are irrelevant
to the task, but it should not be applied automatically to every
dataset.

#### B. Whitespace normalization

Whitespace normalization standardizes spaces, tabs, and line breaks.

Before:

``` text
The    examination
starts
tomorrow.
```

After:

``` text
The examination starts tomorrow.
```

**When it helps:** A search or classification system should generally
not treat multiple accidental spaces as meaningful differences.

**When it can remove useful information:** Line breaks, indentation, and
spacing may be meaningful in source code, poetry, tables, or structured
documents. For example, indentation in Python code affects program
structure. Removing all line breaks from such data can damage its
meaning.

A common approach for ordinary prose is to replace runs of whitespace
with one space while retaining paragraph boundaries when they matter.

#### C. Unicode normalization

Unicode allows some visible characters to be represented by different
sequences of code points. For example, an accented character can
sometimes be represented as one precomposed character or as a base
character followed by a combining accent.

These forms may look identical to a reader but differ internally.
Unicode normalization converts equivalent sequences into a consistent
representation.

The Unicode Standard defines four principal normalization forms:

  -----------------------------------------------------------------------
  Form                    Meaning                 Typical use
  ----------------------- ----------------------- -----------------------
  NFC                     Canonical composition   General text storage
                                                  and comparison

  NFD                     Canonical decomposition Internal processing
                                                  that needs decomposed
                                                  characters

  NFKC                    Compatibility           Certain identifier and
                          composition             loose-matching tasks

  NFKD                    Compatibility           Specialized internal
                          decomposition           processing
  -----------------------------------------------------------------------

The Unicode Consortium identifies NFC as a suitable general form for
text. NFKC and NFKD remove some compatibility distinctions and can
therefore lose information; they should be selected only when that
behavior is appropriate (Unicode Consortium, n.d.).

**When it helps:** Two strings that look the same but use canonically
equivalent Unicode sequences can be normalized to the same
representation. This improves reliable comparison, searching, and
deduplication.

**When it can remove useful information:** Compatibility normalization
can collapse distinctions that matter in specialized text. For example,
formatting or compatibility characters may have a deliberate role in
mathematical, technical, or historical text. A system should not assume
that every visually similar character is interchangeable.

### 2.3 Normalization example

Suppose a university search system receives these queries:

``` text
"  Computer   Science "
"computer science"
"COMPUTER SCIENCE"
```

For a case-insensitive general search, lowercasing and whitespace
normalization can transform all three into:

``` text
computer science
```

This improves matching because differences in letter case and accidental
spaces no longer prevent retrieval.

However, if the system searches programming identifiers, lowercasing
could be harmful because capitalization may distinguish two separate
identifiers.

**Conclusion:** normalization is useful when it removes accidental
differences, but it must preserve distinctions that carry meaning in the
target domain.

------------------------------------------------------------------------

## 3. Word-Level, Subword, and Character-Level Tokenization

### 3.1 What is tokenization?

**Tokenization** is the process of dividing text into smaller units
called tokens. Depending on the method, a token can be a complete word,
a part of a word, or an individual character.

The tokenizer determines the units that a model receives as input. Its
design therefore affects the number of tokens produced, the vocabulary
size, the handling of rare words, and the amount of information
represented by each token.

### 3.2 Word-level tokenization

Word-level tokenization divides a sentence into complete words.

Example:

``` text
Text:    Students are learning programming.
Tokens:  ["Students", "are", "learning", "programming", "."]
```

**Advantages**

-   Tokens are easy for people to interpret.
-   Complete words often carry clear linguistic meaning.
-   It can work well for languages and datasets with relatively stable
    word boundaries.
-   It produces relatively short token sequences for ordinary text.

**Disadvantages**

-   A new or misspelled word may not exist in the vocabulary.
-   Morphological variants can become separate vocabulary entries.
-   Large text collections may require very large vocabularies.
-   Word boundaries are not equally straightforward in every writing
    system.

For example, if the vocabulary contains `connect` but not
`connectivity`, a strict word-level system may treat the latter as an
unknown word.

### 3.3 Subword-level tokenization

Subword tokenization divides words into smaller, reusable units. A
subword can be a frequent word part, a prefix, a suffix, or another
learned sequence of characters.

Example (illustrative segmentation):

``` text
Text:   unhelpful
Tokens: ["un", "help", "ful"]
```

The exact segmentation depends on the tokenizer and its learned
vocabulary. Common subword approaches include Byte Pair Encoding (BPE),
WordPiece, and Unigram language-model tokenization.

Sennrich, Haddow, and Birch (2016) demonstrated how subword units can
help neural machine translation handle rare and previously unseen words.
Their BPE-based approach allowed a model to represent words through
smaller units instead of relying only on a fixed list of complete words.

**Advantages**

-   Can represent many rare and unseen words using known pieces.
-   Usually requires a smaller vocabulary than a complete-word
    vocabulary.
-   Can capture recurring word parts.
-   Is useful for multilingual and morphologically complex text.

**Disadvantages**

-   A word may be divided into pieces that are difficult for humans to
    interpret.
-   Token sequences may be longer than word-level sequences.
-   Poorly chosen segmentation can split meaningful units.
-   Different tokenizers may segment the same word differently.

### 3.4 Character-level tokenization

Character-level tokenization divides text into individual characters.

Example:

``` text
Text:   cat
Tokens: ["c", "a", "t"]
```

**Advantages**

-   It can represent any word formed from the character inventory.
-   It avoids many complete-word out-of-vocabulary problems.
-   It can be useful for spelling correction and noisy text.
-   It can capture information about prefixes, suffixes, and character
    patterns.

**Disadvantages**

-   It produces much longer sequences than word-level tokenization.
-   Individual characters usually carry less meaning than complete
    words.
-   Models may need more computation and data to learn useful word-level
    patterns.
-   Character sequences can make long-range relationships harder to
    model.

### 3.5 Comparison of tokenization methods

  ----------------------------------------------------------------------------
  Feature            Word-level        Subword-level     Character-level
  ------------------ ----------------- ----------------- ---------------------
  Basic unit         Complete word     Part of a word or Individual character
                                       frequent sequence 

  Typical sequence   Shorter           Medium            Longer
  length                                                 

  Vocabulary size    Often large       Usually moderate  Usually small

  Rare-word handling Weak unless word  Can combine known Can represent words
                     is known          pieces            from known characters

  Interpretability   High              Moderate          Low at the lexical
                                                         level

  Common use         Traditional text  Modern language   Spelling and
                     classification    models and        character-sensitive
                     and search        translation       tasks
  ----------------------------------------------------------------------------

### 3.6 What does vocabulary size mean?

**Vocabulary size** is the number of distinct tokens that a tokenizer
recognizes and can represent directly.

For example, if a tokenizer has the following vocabulary:

``` text
["book", "student", "learn", "computer", "science"]
```

its vocabulary size is **5**.

The meaning of vocabulary size depends on tokenization:

-   A word-level vocabulary may contain tens or hundreds of thousands of
    complete words.
-   A subword vocabulary may contain a selected number of frequent word
    pieces.
-   A character-level vocabulary may contain letters, digits,
    punctuation, and other character units.

A larger vocabulary can represent more complete units directly, but it
requires more embedding parameters and may still fail to include rare or
new words. A smaller vocabulary reduces the number of distinct entries,
but it may split text into more tokens.

There is therefore a trade-off between vocabulary size, sequence length,
coverage, and model efficiency.

### 3.7 How tokenization choice affects vocabulary size

Consider the made-up word `reusable`.

-   **Word-level:** `["reusable"]` --- one vocabulary entry is needed if
    the word is included.
-   **Subword-level:** `["re", "use", "able"]` --- the model can reuse
    pieces that may also appear in other words.
-   **Character-level:** `["r", "e", "u", "s", "a", "b", "l", "e"]` ---
    the model uses individual character units.

The word-level approach may need separate entries for many word forms.
Subword tokenization reuses common pieces and usually offers a
compromise between vocabulary size and sequence length. Character-level
tokenization uses a small inventory but produces longer sequences.

**Conclusion:** no tokenization method is universally best. The choice
should reflect the language, vocabulary coverage, task, available data,
and computational constraints.

------------------------------------------------------------------------

## 4. Stopwords: Benefits, Risks, and Task Dependence

### 4.1 Definition of stopwords

**Stopwords** are words that are often very frequent in a language and
are sometimes removed from text because they contribute relatively
little to a particular information-retrieval or text-analysis task.

In English, common examples include:

-   the
-   is
-   are
-   and
-   of
-   to
-   in

A stopword list is a collection of terms selected for possible removal.
It is not a universal list of words that have no meaning. A word's
usefulness depends on the task and context.

### 4.2 Benefits of removing stopwords

Stopword removal can be useful in some traditional NLP applications:

1.  **Reduces the number of features:** removing frequent terms can
    reduce the size of a document-term matrix.
2.  **Reduces computational work:** fewer tokens may need to be stored
    or compared.
3.  **Improves keyword-focused retrieval:** in some search systems,
    common grammatical words do not help distinguish one document from
    another.
4.  **Can improve classification:** if frequent words add little useful
    information, removing them may help a classifier focus on more
    discriminative terms.

For example, in a simple topic-search task, the words `the`, `is`, and
`of` may occur in nearly every document. Removing them may make
topic-related words more prominent.

### 4.3 Risks of removing stopwords

Stopword removal can also destroy useful information.

**Risk 1: Negation**

The word `not` can reverse the meaning of a sentence:

-   "The service is reliable."
-   "The service is not reliable."

If `not` is removed, the two sentences become almost identical.

**Risk 2: Questions and intent**

Words such as `who`, `what`, `where`, and `how` can be essential in
question answering. Removing them may make it harder to identify what
the user is asking.

**Risk 3: Style and authorship**

In some text-classification tasks, function words can reveal writing
style. Removing them may eliminate useful evidence.

**Risk 4: Domain-specific meaning**

A word that is common in ordinary language may be important in a
specialized domain. In a medical corpus, a frequent word can still
contribute to identifying a type of instruction or relationship.

### 4.4 Why stopword lists depend on task and language

A stopword list should be developed or selected for the actual language
and application.

-   **Sentiment analysis:** negation words such as `not` should normally
    be retained.
-   **Question answering:** interrogative words such as `who`, `what`,
    and `where` should be preserved.
-   **Search:** some high-frequency function words may be removed,
    depending on the search engine's design.
-   **Authorship or style analysis:** grammatical words may be useful
    signals and should not automatically be removed.
-   **Somali NLP:** an English stopword list is not appropriate for
    Somali. A Somali-specific list must account for Somali vocabulary,
    grammar, and the purpose of the application.

Even within one language, a stopword list suitable for a general search
engine may be unsuitable for sentiment analysis.

Research confirms that stopword removal can improve some
text-classification datasets but reduce performance on others.
HaCohen-Kerner et al. (2020), for example, found that stopword removal
helped on three of their four benchmark datasets, but was not beneficial
for the Sentiment Labelled Sentences dataset.

**Conclusion:** stopword removal is an optional, task-dependent
decision. A researcher should test whether removing a word improves the
task rather than assuming that frequent words are always useless.

------------------------------------------------------------------------

## 5. Stemming and Lemmatization

### 5.1 Stemming

**Stemming** is a text-processing method that reduces related word forms
to a common stem by applying rules, often involving the removal or
modification of prefixes and suffixes.

A stem does not have to be a valid dictionary word.

Illustrative examples:

  Original word   Possible stem
  --------------- ---------------
  connected       connect
  connecting      connect
  connections     connect
  studies         studi

The output `studi` illustrates that a stemmer may produce a form that is
not a normal English word.

### 5.2 A well-known stemming algorithm: Porter Stemmer

The **Porter stemming algorithm**, introduced by Martin Porter in 1980,
is a widely known rule-based algorithm for English stemming.

It applies a sequence of suffix-reduction rules. Rather than
understanding a word's full meaning, it examines spelling patterns and
removes or changes endings according to conditions.

For example, the algorithm may reduce related forms such as:

``` text
connected
connecting
connection
```

toward a common stem such as:

``` text
connect
```

The exact result depends on the algorithm's rules and the input word.

**Advantages of Porter stemming**

-   Relatively fast.
-   Does not require a large dictionary.
-   Useful for grouping related word forms in information retrieval.
-   Easy to apply to large collections of English text.

**Limitations**

-   It may produce non-dictionary forms.
-   It does not fully understand grammatical context.
-   It can incorrectly combine words that are not linguistically
    related.
-   The original algorithm is designed for English and should not be
    assumed to work correctly for every language.

Willett (2006) reviews the Porter algorithm and documents its wide use
in information retrieval and related research.

### 5.3 Lemmatization

**Lemmatization** is the process of reducing an inflected word to its
dictionary or base form, known as a lemma.

Unlike basic stemming, lemmatization attempts to use linguistic
information such as vocabulary, morphology, and part of speech.

Examples:

  -----------------------------------------------------------------------
  Word form                           Lemma
  ----------------------------------- -----------------------------------
  am                                  be

  are                                 be

  running                             run

  studies                             study

  better                              good (when treated as the
                                      comparative form of the adjective)
  -----------------------------------------------------------------------

The correct lemma may depend on the grammatical role of a word. For
example, `meeting` can be a noun or a form of the verb `meet`; a
lemmatizer may need context or part-of-speech information to choose the
correct analysis.

### 5.4 A well-known lemmatization approach: WordNet

**WordNet** is a large lexical database of English that organizes words
into sets of synonyms called synsets and records semantic and lexical
relationships.

A WordNet-based lemmatization approach uses lexical and morphological
information to identify a valid base form.

For example, the WordNet lemmatizer in the Natural Language Toolkit
(NLTK) can use a part-of-speech category:

``` python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running", "v"))
# run

print(lemmatizer.lemmatize("cars", "n"))
# car
```

The part-of-speech argument matters. A lemmatizer may produce a
different result when a word is treated as a noun rather than a verb.
NLTK's WordNet lemmatizer uses WordNet's morphological analysis and
returns the shortest lemma found for the specified form; if no lemma is
found, it can return the original word.

**Advantages of lemmatization**

-   Usually produces meaningful dictionary forms.
-   Can use grammatical categories.
-   Helps group inflected forms without creating arbitrary stems.
-   Useful when preserving linguistic interpretability is important.

**Limitations**

-   Often requires a dictionary or morphological resource.
-   Can be slower or more resource-intensive than simple stemming.
-   Accuracy depends on language coverage and part-of-speech
    information.
-   A resource developed for English cannot simply be assumed to work
    for Somali or another language.

### 5.5 Comparison between stemming and lemmatization

  -----------------------------------------------------------------------
  Feature                 Stemming                Lemmatization
  ----------------------- ----------------------- -----------------------
  Main operation          Applies reduction rules Finds a dictionary base
                                                  form

  Linguistic knowledge    Usually limited         Often uses lexical and
                                                  grammatical information

  Output                  May not be a valid word Normally a valid lemma

  Speed                   Usually faster          Often slower

  Resources               Rules may be sufficient Usually needs a lexicon
                                                  or morphological model

  Example                 studies → studi         studies → study
                          (Porter-style result)   

  Common use              Large-scale information Linguistically informed
                          retrieval               text analysis
  -----------------------------------------------------------------------

### 5.6 When should each method be preferred?

**Prefer stemming when:**

-   Processing a large collection of documents.
-   Speed and simplicity are important.
-   The task is broad keyword matching.
-   Exact dictionary forms are not essential.

**Prefer lemmatization when:**

-   Linguistically meaningful base forms are important.
-   The task involves detailed language analysis.
-   Grammatical distinctions matter.
-   A suitable dictionary or morphological model is available.

Neither method should be applied automatically. For example, stemming
can improve document matching by grouping related forms, but it may also
merge words that should remain distinct. Lemmatization may preserve more
linguistic structure, but it can be less effective if the required
language resources are incomplete.

------------------------------------------------------------------------

## 6. Regular Expressions in Text Cleaning

### 6.1 Definition

A **regular expression (regex)** is a pattern used to search, identify,
extract, replace, or validate sequences of characters in text.

In NLP preprocessing, regular expressions are useful for finding
repeated whitespace, HTML-like fragments, URLs, email addresses, dates,
phone-number patterns, and other predictable text structures.

A regex does not understand language meaning. It identifies text
according to the pattern provided by the programmer.

### 6.2 Example pattern 1: Repeated whitespace

Pattern:

``` regex
\s+
```

**Meaning:**

-   `\s` matches a whitespace character, such as a space, tab, or line
    break.
-   `+` means one or more occurrences.

This pattern matches one or more consecutive whitespace characters.

Example input:

``` text
The    student
opened     the file.
```

If the pattern is replaced with one ordinary space, the output becomes:

``` text
The student opened the file.
```

Python example:

``` python
import re

text = "The    student\nopened     the file."
cleaned = re.sub(r"\s+", " ", text).strip()

print(cleaned)
```

Output:

``` text
The student opened the file.
```

This is useful for standardizing accidental spacing in ordinary prose.
It should not be used without modification when paragraph boundaries,
indentation, or line structure are important.

### 6.3 Example pattern 2: Finding email addresses

A simplified email-matching pattern is:

``` regex
[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}
```

**Meaning:**

-   `[\w.+-]+` matches one or more letters, digits, underscores, dots,
    plus signs, or hyphens before the `@`.
-   `@` matches the literal at-sign.
-   `[\w.-]+` matches a simplified domain portion.
-   `\.` matches a literal dot.
-   `[A-Za-z]{2,}` matches at least two alphabetic characters at the
    end.

Example input:

``` text
Contact the library at library.support@example.org for assistance.
```

Matched text:

``` text
library.support@example.org
```

Python example:

``` python
import re

text = "Contact the library at library.support@example.org for assistance."

emails = re.findall(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    text
)

print(emails)
```

Output:

``` text
['library.support@example.org']
```

This is a simplified pattern for illustrative extraction, not a complete
implementation of every valid email-address rule.

### 6.4 How an overly broad regex can damage text

A regular expression can remove useful information if its pattern is too
general.

For example, consider this pattern:

``` regex
[^A-Za-z]
```

It matches every character that is not an English letter.

If a programmer removes every match, the following sentence:

``` text
The result is not 100% correct!
```

may become:

``` text
Theresultisnotcorrect
```

The process removes spaces, numbers, punctuation, and the percentage
symbol. This can damage sentence boundaries and remove information that
may be important to a classification or information-extraction task.

It can also damage multilingual text. A pattern that retains only `A-Z`
and `a-z` may delete Arabic, Somali text containing non-ASCII
characters, Chinese characters, and many accented letters.

**Best practice:** write narrow patterns for a clearly defined purpose,
test them on representative examples, and inspect the output before
applying them to an entire dataset.

------------------------------------------------------------------------

## 7. Case Study: How Preprocessing Choices Affect Text Classification

### 7.1 Research paper

A relevant experimental study is:

**HaCohen-Kerner, Y., Miller, D., and Yigal, Y. (2020). "The Influence
of Preprocessing on Text Classification Using a Bag-of-Words
Representation." PLOS ONE, 15(5), e0232525.**

The study investigates how combinations of basic preprocessing methods
affect text-classification accuracy. It is especially relevant because
it does not assume that one cleaning strategy is best for every dataset.

Reference: https://doi.org/10.1371/journal.pone.0232525

### 7.2 Research task

The task was **text classification**: automatically assigning documents
or messages to predefined categories.

The researchers used a Bag-of-Words representation, in which each
document is represented using the occurrences of selected words.

The main research question was whether different combinations of
preprocessing operations could improve classification accuracy.

### 7.3 Datasets

The study examined four benchmark text corpora:

  -----------------------------------------------------------------------
  Dataset                 Description             Classification purpose
  ----------------------- ----------------------- -----------------------
  WebKB                   University-related      Classifying webpage
                          webpages                categories

  R8                      Newswire documents      Classifying news topics
                          derived from            
                          Reuters-21578           

  SMS Spam Collection     5,574 English SMS       Distinguishing spam
                          messages                from legitimate
                                                  messages

  Sentiment Labelled      Sentences from          Identifying sentiment
  Sentences               review-related sources  categories
  -----------------------------------------------------------------------

The datasets were different in content and purpose. This was important
because a preprocessing method that helps identify webpage topics might
not have the same effect on sentiment analysis.

### 7.4 Preprocessing options compared

The researchers investigated six basic preprocessing methods:

  Method                         Description
  ------------------------------ ---------------------------------------------
  Spelling correction            Correcting selected common misspellings
  Lowercasing                    Converting uppercase letters into lowercase
  HTML removal                   Removing HTML objects and tags
  Punctuation removal            Removing punctuation marks
  Stopword removal               Removing selected frequent words
  Repeated-character reduction   Reducing repeated characters

Not every operation was equally relevant to every dataset. For example,
HTML removal was relevant to webpage data such as WebKB.

The researchers tested combinations of these operations rather than
evaluating only one fixed cleaning pipeline.

### 7.5 Experimental method

The researchers used:

-   Four benchmark datasets.
-   Three supervised Machine Learning methods: Bayes Networks,
    Sequential Minimal Optimization (SMO, an SVM-related method), and
    Random Forest.
-   Bag-of-Words representations using word unigrams.
-   The WEKA machine-learning platform.
-   A 67% training and 33% testing split.
-   Ten repetitions of each experiment.
-   Classification accuracy as the evaluation measure.

They tested 31 non-empty combinations of five preprocessing methods and
63 non-empty combinations of six methods, depending on the dataset and
applicable operations.

This systematic design allowed them to compare the effects of individual
methods and combinations.

### 7.6 Main findings

The results showed that preprocessing choices affected classification
differently across the four datasets.

**Finding 1: Stopword removal helped some datasets.**

Stopword removal was the strongest single preprocessing method in three
datasets: WebKB, R8, and SMS Spam Collection. It produced significant
improvements over the baseline in those cases.

**Finding 2: Stopword removal did not help the sentiment dataset.**

For the Sentiment Labelled Sentences dataset, stopword removal was not
beneficial. For two of the Machine Learning methods, it caused
significant declines. Lowercasing was the only single preprocessing
operation that significantly improved the baseline in that dataset.

This result is important because sentiment can depend on short, frequent
words, especially negation terms.

**Finding 3: Combinations mattered.**

The best-performing combinations were not identical across datasets.
Some combinations involving spelling correction, HTML removal,
lowercasing, punctuation handling, or stopword removal improved
performance, but the best combination depended on the dataset and
classifier.

**Finding 4: There was no universal preprocessing pipeline.**

The researchers concluded that preprocessing methods should be tested
systematically on the target corpus and task. A method that improves one
dataset may be ineffective or harmful for another.

### 7.7 Interpretation of the case study

The study provides practical evidence that preprocessing is a modeling
decision, not merely a routine cleaning stage.

For example, removing frequent words can reduce the number of features
in a traditional classifier. This may help a topic-classification task.
However, frequent words may carry useful information in sentiment
analysis, so removing them can reduce accuracy.

The findings also demonstrate why researchers should establish a
baseline and compare alternative preprocessing pipelines using the same
training and test conditions.

### 7.8 Case-study summary

  -----------------------------------------------------------------------
  Research element                    Description
  ----------------------------------- -----------------------------------
  Authors                             HaCohen-Kerner, Miller, and Yigal

  Year                                2020

  Research task                       Text classification

  Representation                      Bag-of-Words word unigrams

  Datasets                            WebKB, R8, SMS Spam Collection,
                                      Sentiment Labelled Sentences

  Preprocessing options               Spelling correction, lowercasing,
                                      HTML removal, punctuation removal,
                                      stopword removal,
                                      repeated-character reduction

  Models                              Bayes Networks, SMO, Random Forest

  Main finding                        Different preprocessing
                                      combinations worked best for
                                      different datasets

  Main lesson                         Preprocessing must be evaluated in
                                      relation to the task and corpus
  -----------------------------------------------------------------------

**Conclusion from the case study:** more preprocessing does not
automatically produce better results. The best approach is to compare
carefully selected alternatives and retain the operations that improve
performance without removing useful information.

------------------------------------------------------------------------

## Overall Conclusion

Text preprocessing is an important stage in NLP because raw text often
contains inconsistent formatting, spelling variation, encoding
differences, and other characteristics that make computational analysis
difficult. Operations such as normalization, tokenization, stopword
handling, stemming, lemmatization, and regular-expression matching can
make text more consistent and useful.

However, preprocessing must be guided by the objective of the NLP
application. Removing negation can damage sentiment analysis, deleting
numbers can harm price extraction, and aggressive character filtering
can destroy multilingual text. Tokenization also affects the vocabulary
and the model's ability to represent rare words.

The case study by HaCohen-Kerner et al. (2020) provides experimental
evidence that no single preprocessing combination works best for every
dataset. A reliable NLP project should therefore begin with a clear
baseline, test alternative preprocessing choices, evaluate the results
using appropriate metrics, and preserve information that is important to
the task.

------------------------------------------------------------------------

# References

1.  Chai, C. P. (2023). Comparison of text preprocessing methods.
    *Natural Language Engineering, 29*(3), 509--553.
    https://doi.org/10.1017/S1351324922000213

2.  HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020). The influence of
    preprocessing on text classification using a bag-of-words
    representation. *PLOS ONE, 15*(5), e0232525.
    https://doi.org/10.1371/journal.pone.0232525

3.  Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to
    Information Retrieval*. Cambridge University Press.
    https://nlp.stanford.edu/IR-book/

4.  Miller, G. A. (1995). WordNet: A lexical database for English.
    *Communications of the ACM, 38*(11), 39--41.
    https://doi.org/10.1145/219717.219748

5.  NLTK Project. (n.d.). *WordNetLemmatizer: NLTK source
    documentation*.
    https://www.nltk.org/\_modules/nltk/stem/wordnet.html

6.  Porter, M. F. (1980). An algorithm for suffix stripping. *Program,
    14*(3), 130--137. https://doi.org/10.1108/eb046814

7.  Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine
    translation of rare words with subword units. In *Proceedings of the
    54th Annual Meeting of the Association for Computational Linguistics
    (Volume 1: Long Papers)* (pp. 1715--1725). Association for
    Computational Linguistics. https://doi.org/10.18653/v1/P16-1162

8.  Unicode Consortium. (n.d.). *Unicode FAQ: Normalization*.
    https://www.unicode.org/faq/normalization.html

9.  Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode
    Normalization Forms*. https://www.unicode.org/reports/tr15/

10. Willett, P. (2006). The Porter stemming algorithm: Then and now.
    *Program: Electronic Library and Information Systems, 40*(3),
    219--223. https://doi.org/10.1108/00330330610681295

11. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A
    comparative evaluation of pre-processing techniques and their
    interactions for Twitter sentiment analysis. *Expert Systems with
    Applications, 110*, 298--310.
    https://doi.org/10.1016/j.eswa.2018.06.022

------------------------------------------------------------------------

## Key Points to Remember

1.  **Text preprocessing:** Preparing raw text for a specific NLP task.
2.  **Normalization:** Standardizing text forms, such as case,
    whitespace, and Unicode representation.
3.  **Tokenization:** Dividing text into word, subword, or character
    units.
4.  **Vocabulary size:** The number of distinct tokens a tokenizer can
    represent.
5.  **Stopwords:** Frequently occurring words that may or may not be
    useful for a particular task.
6.  **Stemming:** Reducing word forms through rules, sometimes producing
    non-dictionary stems.
7.  **Lemmatization:** Finding a linguistically meaningful dictionary
    form.
8.  **Regular expressions:** Patterns used to identify, extract,
    replace, or validate text.
9.  **Over-cleaning:** Removing information that the NLP task actually
    needs.
10. **Research lesson:** Preprocessing should be selected and evaluated
    for the particular dataset, language, and task.
