1. Definition and Importance of Text Preprocessing

Text preprocessing is the process of transforming raw textual data into a more organized and suitable format for Natural Language Processing (NLP) systems. It involves applying different techniques to prepare text for computational analysis. These techniques may include lowercasing, tokenization, whitespace normalization, punctuation handling, stopword removal, stemming, and lemmatization.

NLP systems need text preprocessing because human language is naturally inconsistent. People use different capitalization styles, punctuation marks, abbreviations, spelling variations, and sentence structures. These variations can make it difficult for some NLP models to identify similar words and patterns. Preprocessing helps reduce unnecessary differences and prepares textual data for feature extraction, classification, and other language-processing tasks.

However, applying too many cleaning operations can sometimes reduce model performance because some information that appears unnecessary may actually be important. For example, removing negation words such as "not" can completely change the meaning of a sentence.

A real-life example can be found in customer review analysis. Consider the sentence, "I am NOT happy with this jacket!" If a preprocessing system removes the word "NOT" and the exclamation mark, an NLP model may fail to recognize the customer's negative opinion. Therefore, preprocessing should be selected according to the purpose of the NLP application rather than removing as much text as possible.

In conclusion, effective text preprocessing aims to improve text representation while preserving information that is important for the intended NLP task.

2. Text Normalization

Text normalization is the process of converting text into a consistent format before processing it with an NLP system. Common normalization techniques include lowercasing, whitespace normalization, and Unicode normalization.

Lowercasing converts uppercase letters into lowercase letters, while whitespace normalization removes unnecessary spaces and makes spacing consistent. Unicode normalization helps represent equivalent characters in a standardized form.

Normalization can improve text consistency, but it may also remove useful information. For example, the words "US" and "us" have different meanings: "US" can refer to the United States, while "us" is a pronoun. Converting both into lowercase may cause an NLP system to lose this distinction. Therefore, normalization techniques should be selected according to the NLP task.

### 3. Word-Level, Subword, and Character-Level Tokenization

Tokenization is the process of dividing text into smaller units called tokens. The three common approaches are word-level, subword, and character-level tokenization.

**Word-level tokenization** divides text into individual words. For example, "I love NLP" becomes `["I", "love", "NLP"]`. It is easy to understand and produces meaningful tokens. However, it may require a large vocabulary and can struggle with new or unseen words.

**Subword tokenization** divides words into smaller meaningful or frequently occurring parts. For example, the word "unhappiness" might be divided into "un", "happi", and "ness". This approach helps NLP models handle unfamiliar words by recognizing their smaller parts. It also provides a balance between vocabulary size and the number of tokens.

**Character-level tokenization** divides text into individual characters. For example, "CAT" becomes `["C", "A", "T"]`. It has a relatively small vocabulary because it uses individual characters, but it produces more tokens and may require the model to learn word meanings from character sequences.

Vocabulary size refers to the total number of unique tokens a tokenizer can recognize. Word-level tokenization generally has a larger vocabulary, while character-level tokenization has a smaller vocabulary. Subword tokenization provides a balance between the two.

The choice of tokenizer affects NLP performance, vocabulary size, and the model's ability to process unfamiliar words. For languages with many word variations, such as Somali, subword tokenization can be useful because it allows models to process unfamiliar words through smaller units.

### 4. Stopwords and Their Removal

Stopwords are common words that occur frequently in a language but may contribute relatively little information to certain NLP tasks. Examples in English include "the", "is", "a", "an", and "and".

Removing stopwords can reduce the number of words in a text and help some NLP systems focus on more informative terms. For example, removing common words from documents can be useful in certain information retrieval and text classification tasks.

However, stopword removal can also cause information loss. For example, removing the word "not" from the sentence "This product is not good" changes its meaning and may cause a sentiment analysis model to classify the negative opinion incorrectly.

Stopword lists are task-dependent because words that are unimportant for one task may be essential for another. They are also language-dependent because different languages have different vocabulary and grammatical structures. Therefore, an English stopword list should not automatically be applied to Somali text.

In conclusion, stopword removal should be carefully selected according to the NLP task and language to avoid losing important information.

### 5. Stemming and Lemmatization

Stemming and lemmatization are text preprocessing techniques used to reduce words to their base forms. However, they use different approaches.

**Stemming** removes prefixes or suffixes from words using predefined rules to produce a stem. For example, "connected" and "connecting" may both be reduced to "connect". However, stemming does not always produce a valid dictionary word. A well-known algorithm is the Porter Stemmer, developed by Martin Porter in 1980. It applies a series of rules to remove common word endings.

**Lemmatization** converts words into their correct dictionary forms, known as lemmas. Unlike stemming, it considers linguistic information, including the grammatical role of a word. One well-known approach is WordNet Lemmatization, which uses the WordNet lexical database. For example, "better" can be converted to "good", while "wolves" becomes "wolf". In NLTK, providing a Part-of-Speech (POS) hint helps the lemmatizer identify the correct base form.

Stemming is generally faster and useful for search engines and information retrieval, where matching related words is important. Lemmatization is more suitable for tasks such as sentiment analysis and text understanding, where preserving meaningful and grammatically correct words is important.

In conclusion, the choice between stemming and lemmatization depends on the NLP task, the importance of linguistic accuracy, and the available computational resources.

### 6. Regular Expressions (Regex) in Text Cleaning

Regular Expressions (Regex) are patterns used to search, identify, extract, or replace specific parts of text. In Python, the `re` module provides functions for working with regular expressions.

For example, the pattern `\d+` matches one or more digits. It can identify numbers such as 25 or 100 in a text. Another pattern, `[^a-z\s]`, matches characters that are not lowercase English letters or whitespace. It can be used to remove punctuation, numbers, and special symbols.

Regex is useful in text cleaning because it allows developers to remove unwanted characters, identify phone numbers, and replace specific information with labels such as `PHONE_NUMBER` or `MONEY`.

However, overly broad Regex patterns can damage text by removing meaningful information. For example, removing all punctuation may delete exclamation marks that express strong emotions. Similarly, an incorrect pattern that removes the word "not" can reverse the meaning of a sentence. Therefore, Regex patterns should be designed carefully according to the NLP task.

### 7. Case Study: The Influence of Preprocessing on Text Classification

HaCohen-Kerner, Miller, and Yigal (2020) conducted a study titled "The Influence of Preprocessing on Text Classification Using a Bag-of-Words Representation." The study investigated how different text preprocessing techniques affect text classification performance.

The researchers tested four benchmark datasets: WebKB, R8, SMS Spam Collection, and Sentiment Labelled Sentences. They compared several preprocessing techniques, including lowercasing, stopword removal, punctuation removal, spelling correction, HTML tag removal, and reducing repeated characters. They evaluated the results using three machine learning methods: Bayes Networks, Support Vector Machines (SVM), and Random Forest.

The results showed that preprocessing techniques affected classification performance differently across datasets. Stopword removal improved performance in three datasets, but it negatively affected some results in the Sentiment Labelled Sentences dataset. Lowercasing was beneficial for the sentiment dataset. The researchers also found that no single preprocessing combination performed best across all datasets.

This case study demonstrates that preprocessing should not be applied automatically. Instead, techniques should be selected and tested according to the NLP task, dataset, and machine learning model.

**Reference:**

HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020). The influence of preprocessing on text classification using a bag-of-words representation. *PLOS ONE, 15*(5), e0232525. https://doi.org/10.1371/journal.pone.0232525


Removing stopwords can negatively affect sentiment analysis because some stopwords, such as "not" and "never," carry important meaning. Removing these words can reverse the meaning of a sentence and cause the model to produce incorrect predictions. Therefore, sentiment analysis systems should preserve negation words.

<!-- Task 5 -->
Stopword removal can reduce unnecessary words and improve text processing efficiency. However, stopwords are task-dependent. In sentiment analysis, removing negation words such as "not," "no," "never," and "n't" can change the meaning of a sentence and lead to incorrect predictions. Therefore, a custom stopword list should preserve these important words.


**Task 6: Stemming**

The Porter, Lancaster, and Snowball stemmers reduce words to their stems. In the experiment, different words such as "university" and "universal" produced the same stem, "univers." Lancaster was the most aggressive stemmer because it produced shorter stems, such as "better" becoming "bet." Some resulting stems, such as "fli" and "wolv," are not valid English words.

**Task 7: Lemmatization**

WordNetLemmatizer reduces words to their dictionary forms. By default, it treats words as nouns, so providing the correct part-of-speech (POS) hint can improve the results. For example, "ran" remains unchanged by default but becomes "run" when the verb POS hint is provided. Lemmatization generally produces meaningful dictionary words, unlike stemming, which may produce incomplete word forms.

**Task 9 – POS Tagging and Named Entity Recognition**

POS tagging assigns grammatical categories to words, such as nouns, verbs, and pronouns. Named Entity Recognition (NER) identifies real-world entities such as people, locations, organizations, and dates. NLTK correctly identified Smith as a person and London as a geographical location. However, it failed to recognize Friday as a date and Apple as an organization, showing that NER systems can make mistakes depending on context and training data.

**Task 10 – Full Preprocessing Pipeline**

The full preprocessing pipeline includes lowercasing, regex cleaning, tokenization, stopword removal, and lemmatization. The default stopword pipeline removed the word "not," which is important for understanding negative sentiment. In contrast, the custom stopword pipeline preserved negation words, maintaining the original complaint's meaning. Therefore, the custom pipeline is more suitable for sentiment analysis because it reduces unnecessary words without removing important sentiment information.