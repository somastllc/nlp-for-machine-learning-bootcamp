Assignment 2 – Preprocessing

Course: NLP for Machine Learning Bootcamp
Assignment: Assignment 2 – Preprocessing
Prepared by: Nasteho Abdi Aden

---

Part A – Research

1. What Is Text Preprocessing?

Text preprocessing is the process of preparing raw text data before it is used in Natural Language Processing (NLP) systems. It involves transforming unstructured text into a cleaner and more consistent format that computers can analyze.

Common preprocessing techniques include lowercasing, removing unnecessary spaces, tokenization, stopword removal, stemming, lemmatization, and handling punctuation.

NLP systems need preprocessing because raw text often contains inconsistencies, spelling variations, unnecessary symbols, and formatting problems. These issues can make it difficult for machine learning models to identify patterns and understand the information in text.

However, excessive cleaning can sometimes reduce model performance because some words and symbols carry important meaning. Removing useful information may change the original message.

Real-life example:

Consider the following customer review:

"I do NOT recommend this product!!!"

If a preprocessing system removes the word "NOT" and all punctuation, the sentence becomes:

"i do recommend this product"

The original negative opinion has now become positive. This demonstrates why preprocessing must be carefully selected according to the NLP task.

Therefore, effective preprocessing should improve text quality while preserving important information.

2. Text Normalization

Text normalization is the process of converting text into a consistent format so that different representations of the same information can be processed more easily.

Common normalization techniques include:

a. Lowercasing

Lowercasing converts all uppercase letters into lowercase letters.

Example:

- Before: "MACHINE Learning Is Powerful"
- After: "machine learning is powerful"

This helps a system treat words such as "Apple" and "apple" as the same token when capitalization is not important.

b. Whitespace Normalization

Whitespace normalization removes unnecessary spaces, tabs, and repeated line breaks.

Example:

- Before: "Natural   Language    Processing"
- After: "Natural Language Processing"

This makes text more consistent and prevents unnecessary differences between tokens.

c. Unicode Normalization

Unicode normalization ensures that characters with equivalent representations are stored in a consistent form. For example, an accented character may be represented as one combined character or as a letter followed by a combining accent.

Unicode provides four main normalization forms: NFC, NFD, NFKC, and NFKD.

NFC is commonly suitable for general text, while NFKC can be useful when matching different compatibility representations, such as certain full-width characters.

When normalization helps:

In a search engine, lowercasing allows a search for "COMPUTER" to match documents containing "computer". Whitespace normalization also helps prevent formatting differences from affecting text matching.

When normalization removes useful information:

Lowercasing can remove the distinction between a company name such as "Apple" and the ordinary word "apple". This distinction may be important in Named Entity Recognition (NER). Similarly, removing punctuation can damage meaning in messages such as "I'm happy" and "I'm not happy" if the cleaning process also removes important words or symbols.

Therefore, normalization should depend on the purpose of the NLP application.

3. Word-Level, Subword-Level, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. These units can be words, subwords, or individual characters.

There are three common tokenization approaches:

a. Word-Level Tokenization

Word-level tokenization divides a sentence into complete words.

Example:

Sentence: "Natural language processing is useful."

Tokens: ["Natural", "language", "processing", "is", "useful"]

Advantages:

- Easy to understand and interpret.
- Preserves complete words.
- Useful for traditional NLP tasks.

Disadvantages:

- Can produce a very large vocabulary.
- Rare or unknown words may not be recognized.
- Different word forms may be treated as separate tokens.

b. Subword-Level Tokenization

Subword tokenization divides words into smaller meaningful units, such as prefixes, suffixes, or frequently occurring character sequences.

Example:

"unhappiness" may be divided into:

["un", "happi", "ness"]

The exact result depends on the tokenizer.

Advantages:

- Handles rare and previously unseen words.
- Reduces the problem of unknown words.
- Can represent related word forms using shared units.

Disadvantages:

- Some words are divided into units that are difficult to interpret.
- Tokenization can be more complex than word-level tokenization.

Byte Pair Encoding (BPE) is a well-known subword tokenization approach. Sennrich, Haddow, and Birch (2016) demonstrated that subword units help neural machine translation systems process rare and unknown words.

c. Character-Level Tokenization

Character-level tokenization divides text into individual characters.

Example:

Word: "cat"

Tokens: ["c", "a", "t"]

Advantages:

- Can represent almost any word.
- Useful for handling spelling variations and unknown words.
- Does not require a large vocabulary of complete words.

Disadvantages:

- Produces longer sequences.
- Requires models to learn word structures from individual characters.
- May increase computational requirements.

What Is Vocabulary Size?

Vocabulary size refers to the total number of unique tokens that a tokenizer can recognize or use.

For example, if a dataset contains the tokens:

["cat", "dog", "cat", "bird"]

The vocabulary is:

["cat", "dog", "bird"]

Vocabulary size = 3.

The choice of tokenizer affects vocabulary size. Word-level tokenization may require many unique words, while character-level tokenization usually needs a smaller set of characters. Subword tokenization provides a balance between the two.

4. Stopwords

Stopwords are common words that appear frequently in a language and may contribute relatively little information to certain NLP tasks.

Examples in English include:

- the
- is
- am
- are
- and
- of
- to
- in

Stopword removal is a preprocessing technique that removes selected common words from text.

Benefits of Removing Stopwords

1. Reduces the number of tokens that a model needs to process.
2. Can reduce the size of text representations.
3. May improve information retrieval and some text classification tasks.
4. Can help highlight more informative words in certain document analysis tasks.

Risks of Removing Stopwords

Removing stopwords can also destroy important information. Some common words express negation, relationships, and grammatical meaning.

For example:

Original: "I do not like this product."

After removing "not":

"I like this product."

The meaning changes from negative to positive.

Why Stopword Lists Depend on the Task and Language

Not every NLP application needs the same stopword list. In sentiment analysis, words such as "not", "never", and "no" are important because they can change the meaning of an opinion.

In machine translation, grammatical words may be necessary to preserve sentence structure.

Different languages also have different grammatical structures and frequently used words. Therefore, an English stopword list should not automatically be used for Somali or other languages.

Stopword removal should always be based on the language, dataset, and purpose of the NLP task.

5. Stemming and Lemmatization

Stemming and lemmatization are two NLP techniques used to reduce different forms of words to a common representation. However, they use different methods.

a. Stemming

Stemming is a process that removes prefixes or suffixes from words to produce a shorter stem. It generally uses predefined rules rather than understanding the complete meaning of a word.

Examples:

- connected → connect
- connecting → connect
- organization → organ (depending on the stemming algorithm)

One well-known stemming algorithm is the Porter Stemmer, introduced by Martin Porter in 1980.

The Porter algorithm applies a sequence of suffix-removal rules to reduce related English words to common stems.

Other well-known algorithms include:

- Lancaster Stemmer: Uses more aggressive suffix-removal rules.
- Snowball Stemmer: A language-specific stemming framework that includes an improved English stemming algorithm.

Advantages of stemming:

- Fast and relatively simple.
- Reduces vocabulary variations.
- Useful in information retrieval and search applications.

Disadvantages:

- May produce stems that are not real English words.
- Can sometimes merge words with different meanings.
- Does not fully understand grammatical context.

b. Lemmatization

Lemmatization is the process of converting a word into its dictionary form, known as its lemma.

Unlike stemming, lemmatization considers vocabulary and grammatical information. It can use the part of speech (POS) of a word to determine its correct base form.

Examples:

- running → run
- wolves → wolf
- better → good (when used as an adjective)
- was → be

The WordNet Lemmatizer is a well-known approach available in the NLTK Python library. It uses WordNet, a lexical database of English, to find dictionary forms.

Advantages of lemmatization:

- Produces meaningful dictionary words.
- Considers grammatical information when supplied.
- Useful in sentiment analysis and language understanding tasks.

Disadvantages:

- Usually requires more processing than simple stemming.
- Depends on the quality of lexical resources.
- May produce incorrect results when the part of speech is not identified correctly.

Comparison Table

Feature| Stemming| Lemmatization
Method| Removes affixes using rules| Uses vocabulary and grammatical information
Output| Stem| Dictionary form
Speed| Usually faster| Usually slower
Real words| Not always| Usually
Context| Limited| Uses POS information
Example| studies → studi| studies → study

When Should Each Be Used?

Stemming is useful when speed and reducing vocabulary size are important, such as in basic search systems.

Lemmatization is preferable when preserving meaningful word forms is important, such as in sentiment analysis, text understanding, and linguistic analysis.

6. Regular Expressions in Text Cleaning

Regular expressions (Regex) are patterns used to search, identify, extract, and replace specific sequences of characters in text.

In NLP preprocessing, regular expressions are useful for cleaning text by identifying unwanted symbols, URLs, email addresses, phone numbers, and repeated punctuation.

Python provides the "re" module for working with regular expressions.

Example 1: Removing Non-Alphabetic Characters

Pattern:

"[^a-zA-Z\s]"

This pattern matches characters that are not uppercase letters, lowercase letters, or whitespace.

Example:

- Before: "Hello! Welcome to NLP 2026."
- After: "Hello Welcome to NLP "

It removes punctuation and numbers.

Example 2: Identifying Email Addresses

Pattern:

"[\w.-]+@[\w.-]+\.[A-Za-z]{2,}"

This pattern matches common email address formats.

Example:

- Text: "Contact us at student@example.com"
- Match: "student@example.com"

Risk of Overly Broad Regex Patterns

An overly broad pattern may remove important information.

For example, the pattern "[^a-z\s]" removes every character that is not a lowercase English letter or whitespace. If applied to a spam message, it may remove phone numbers, currency symbols, punctuation, and even characters from other languages.

This can destroy useful information needed by a classification system.

Therefore, regex rules should be designed carefully according to the purpose of the cleaning process.

7. Case Study: A Comparative Evaluation of Pre-processing Techniques for Twitter Sentiment Analysis

A useful research study is the paper by Symeonidis, Effrosynidis, and Arampatzis (2018), titled:

"A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis."

Research Task

The study investigated how different text preprocessing techniques affect sentiment analysis. Sentiment analysis is the task of identifying whether a text expresses a positive, negative, or neutral opinion.

The researchers used two Twitter datasets and four machine learning algorithms:

- Linear Support Vector Classifier (Linear SVC)
- Bernoulli Naive Bayes
- Logistic Regression
- Convolutional Neural Networks (CNN)

Preprocessing Techniques Compared

The researchers experimentally evaluated sixteen preprocessing techniques, including:

- Lowercasing.
- Removing numbers.
- Removing punctuation.
- Replacing contractions.
- Stemming.
- Lemmatization.
- Other text cleaning techniques.

They examined the effects of individual preprocessing methods and combinations of methods on classification accuracy and the number of features produced.

Main Findings

The study found that preprocessing choices can significantly affect sentiment classification.

Some important findings included:

1. Lemmatization improved classification accuracy in the experiments.
2. Removing numbers and replacing contractions also improved accuracy.
3. Removing punctuation did not consistently improve results.
4. Combining preprocessing methods could produce different results from applying them individually.
5. The interaction between preprocessing techniques was important when designing a complete pipeline.

Conclusion

This study demonstrates that there is no single preprocessing method that is best for every NLP task. Each technique should be tested using the actual dataset and classification objective.

For example, removing punctuation may be harmful when repeated exclamation marks express strong emotions, while lemmatization may help a model recognize different grammatical forms of the same word.

The findings support the idea that preprocessing should preserve meaningful information and be evaluated experimentally rather than applied automatically.

---

References

1. Porter, M. F. (1980). An algorithm for suffix stripping. Program, 14(3), 130–137.

2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, 1715–1725. https://doi.org/10.18653/v1/P16-1162

3. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. Expert Systems with Applications, 110, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022

4. Unicode Consortium. (2026). Unicode Standard Annex #15: Unicode Normalization Forms. https://www.unicode.org/reports/tr15/

5. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media. https://www.nltk.org/book/

6. NLTK Project. (n.d.). Natural Language Toolkit Documentation. https://www.nltk.org/


Part B – Practical Report

Task 1: Lowercasing

Lowercasing converts uppercase letters into lowercase letters to make text consistent during NLP processing.

Before:

URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!

After:

urgent!!! you have won a £1,000 gift card. call 0712345678 now!

Explanation:

Lowercasing makes words easier to compare because uppercase and lowercase versions become identical. However, a spam detector may lose capitalization information because words such as "URGENT" and "NOW" show emphasis and may indicate suspicious messages.

Task 2: Whitespace Normalization

Whitespace normalization removes unnecessary spaces between words.

Before:

'   The   delivery   was   fast,   and the  packaging   was   great.'

After:

'The delivery was fast, and the packaging was great.'

The result contains consistent spacing, making the text easier to process.

Task 3: Regex Cleaning

Regular expressions are used to identify, remove, or replace specific patterns in text.

Pattern: [^a-z\s]

This pattern matches characters that are not lowercase English letters or whitespace.

Before:

urgent!!! you have won a £1,000 gift card. call 0712345678 now!

After:

urgent you have won a  gift card call  now

Deleted characters:

- Exclamation marks (!)
- Currency symbol (£)
- Numbers (1,000)
- Comma (,)
- Full stop (.)
- Phone number (0712345678)

Custom Regex Replacement

The custom regex rules replaced:

- Phone number with "PHONE_NUMBER"
- Money amount with "MONEY"
- Three exclamation marks with one exclamation mark.

Result:

urgent! you have won a MONEY gift card. call PHONE_NUMBER now!

Explanation:

Deleting all numbers and symbols can remove important information from spam messages. Replacing phone numbers and money amounts with meaningful labels preserves useful information for machine learning.

Task 4: Tokenization

Tokenization divides text into smaller units called tokens.

Two methods were tested:

1. "split()" – separates text using whitespace.
2. "word_tokenize()" – uses NLTK to separate words and punctuation.

T2 Token Counts:

Method| Token Count
split()| 14
word_tokenize()| 16

T4 Tokenization:

- Total tokens: 11
- Vocabulary size: 10

The vocabulary size represents the number of unique tokens. Tokenization helps NLP systems understand and analyze individual words.

Task 5: Stopwords and Negation

Stopwords are common words that are sometimes removed during text preprocessing.

After default stopword removal:

["n't", "think", "jacket", "worth", "price", ".", "happy"]

After keeping negation words:

["n't", "think", "jacket", "worth", "price", ".", "Not", "happy"]

Explanation:

The default stopword list removes the word "Not", which can change the meaning of a negative customer review. The custom stopword list preserves negation words such as "not" and "n't". This is important for sentiment analysis because negative words help identify customer dissatisfaction.

Task 6: Stemming

Stemming reduces words to shorter root-like forms by removing prefixes or suffixes.

Three algorithms were compared: Porter, Lancaster, and Snowball.

Stemming Comparison Table

Word| Porter| Lancaster| Snowball
connection| connect| connect| connect
connected| connect| connect| connect
connecting| connect| connect| connect
organization| organ| org| organ
organized| organ| org| organ
university| univers| univers| univers
universal| univers| univers| univers
flies| fli| fli| fli
wolves| wolv| wolv| wolv
better| better| bet| better
ran| ran| ran| ran

Explanation:

The results show that different words can be reduced to the same stem. For example, connection, connected, and connecting all become "connect". Some generated stems, such as "univers", "fli", and "wolv", are not valid English dictionary words. Lancaster is more aggressive in some cases than Porter and Snowball.

Task 7: Lemmatization

Lemmatization converts words into their meaningful dictionary base forms.

The NLTK WordNetLemmatizer was tested using its default setting and appropriate part-of-speech hints.

Word| Default| With POS Hint
connection| connection| connection
connected| connected| connect
connecting| connecting| connect
organization| organization| organization
organized| organized| organize
university| university| university
universal| universal| universal
flies| fly| fly
wolves| wolf| wolf
better| better| good
ran| ran| run

Explanation:

The POS hints improved lemmatization by identifying the grammatical role of words. The words connected, connecting, organized, better, and ran changed when the appropriate hints were provided. For example, "ran" became "run", while "better" became "good". Lemmatization produces meaningful dictionary words rather than simply shortening words.

Task 8: Sentence Splitting

Sentence splitting separates text into individual sentences.

Original T3:

Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m.

Results:

1. Mr. Smith arrived in London on Friday.
2. He met the Apple team at 9 a.m.

Total sentences: 2

Explanation:

NLTK successfully identified two sentences. The abbreviations "Mr." and "a.m." did not cause incorrect sentence splitting in this example.

Task 9: POS Tagging and Named Entity Recognition (NER)

POS tagging identifies the grammatical role of words, while Named Entity Recognition identifies names of people, places, organizations, and other important entities.

Named Entity Table

Entity| NLTK Result| Expected Label| Correct?
Mr.| PERSON| Part of PERSON| Partial
Smith| PERSON| PERSON| Yes
London| GPE| GPE| Yes
Friday| Not recognized| DATE| No
Apple| Not recognized| ORGANIZATION| No

Explanation:

The NLTK model recognized Smith as a person and London as a geographical location. However, it did not recognize Friday as a date or Apple as an organization. It also separated "Mr." and "Smith" into two PERSON entities. This demonstrates that NER models can miss important entities or classify them incorrectly depending on context.

Task 10: Full Preprocessing Pipeline

The customer review in T2 was processed through five stages:

1. Lowercasing
2. Regex cleaning
3. Tokenization
4. Stopword removal
5. Lemmatization

Two pipelines were tested.

Pipeline A: Default NLTK Stopwords

Final result:

["n't", "think", "jacket", "worth", "price", "happy"]

Pipeline B: Keeping Negation Words

Final result:

["n't", "think", "jacket", "worth", "price", "not", "happy"]

Explanation:

The default stopword pipeline removes the word "not", which can weaken the negative meaning of the customer's review. The second pipeline keeps negation words and therefore preserves more of the original complaint. Pipeline B is more suitable for sentiment analysis because it retains important negative expressions.

Conclusion: Preprocessing Choices for a Spam Detector

For a spam detection system, I would use lowercasing, whitespace normalization, tokenization, and carefully designed regular expressions. I would replace phone numbers and money amounts with labels such as "PHONE_NUMBER" and "MONEY" instead of deleting them because these patterns can provide useful spam detection information. I would also preserve important punctuation and avoid blindly removing stopwords because words such as "not" and uppercase expressions such as "URGENT" can contain useful information. Based on my practical results, preprocessing should preserve important information while making text easier for machine learning models to analyze.