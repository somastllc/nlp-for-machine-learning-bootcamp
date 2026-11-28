# Assignment 2 – Preprocessing

## Part A – Research

A1. Text preprocessing is the process of cleaning, transforming, and organizing raw text data before using it in Natural Language Processing (NLP) tasks. It involves techniques such as lowercasing, removing unnecessary characters, handling whitespace, tokenization, stopword removal, stemming, and lemmatization.

Real Life Example : Consider a real estate company collecting customer reviews about its apartments:

        Review 1: "The apartment is GOOD!"
        Review 2: "The apartment is NOT good!"

        Preprocessing can help standardize these reviews by correcting unnecessary spaces and handling inconsistent formatting. However, if the system removes the word "NOT" during cleaning, it may misunderstand the customer's negative feedback as positive. Therefore, text preprocessing should be carefully selected according to the NLP task, rather than removing as much text as possible.
        Research References
        Waxaan ka duulay research cilmiyeedkaan:
        1. Chai, C. P. (2023). Comparison of text preprocessing methods. Natural Language Engineering, 29(3), 509–553.
        2. Palomino, M. A., & Aider, F. (2022). Evaluating the Effectiveness of Text Pre-Processing in Sentiment Analysis. Applied Sciences, 12(17), 8765.
        3. Al Sharou, K., Li, Z., & Specia, L. (2021). Towards a Better Understanding of Noise in Natural Language Processing. 

A2. Text normalization is the process of converting text into a consistent and standardized format before processing it 
    with NLP systems. It helps reduce unnecessary differences in text and makes it easier for computers to compare and analyze information.

    Lowercasing: Converting all uppercase letters into lowercase letters. For example, "APARTMENT" becomes "apartment".
    Whitespace Normalization: Removing unnecessary spaces and standardizing spacing. For example, "The   apartment   is good" becomes "The apartment is good".
    Unicode Normalization: Converting text characters that have equivalent Unicode representations into a consistent form. For example, the character "é" can be represented using different Unicode sequences. Unicode normalization helps make equivalent representations consistent.

    Situation Where Normalization Helps:

        In a real estate search system, customers may search for "APARTMENT", "Apartment", or "apartment". Lowercasing these words allows the system to treat them consistently and improves search matching.

        Situation Where Normalization Removes Useful Information:

        Lowercasing can remove important information from text. For example, "Apple" may refer to a company, while "apple" may refer to a fruit. In Named Entity Recognition (NER), capitalization can help identify proper names. Removing it may make entity recognition less accurate. Therefore, text normalization should be selected carefully depending on the NLP task, because some text differences contain useful information.

        Research References
        Unicode Consortium – Unicode Normalization FAQ. Read source. 
        Mayhew, S., Tsygankova, T., & Roth, D. (2019). NER and POS when nothing is capitalized. 

A3. Word-level, Subword and Character-level Tokenization

        Word-level Tokenization: This method splits text into complete words. For example, "I love apartments" becomes ["I", "love", "apartments"]. It is simple to understand, but it can create a large vocabulary and struggle with new or unseen words.
        Subword-level Tokenization: This method splits words into smaller meaningful units, especially when words are rare or unfamiliar. For example, "unhappiness" could become ["un", "happi", "ness"]. It helps reduce unknown words while maintaining a manageable vocabulary size.
        Character-level Tokenization: This method splits text into individual characters. For example, "House" becomes ["H", "o", "u", "s", "e"]. It has a small vocabulary and can represent unfamiliar words, but it produces longer sequences that require more processing.

        Vocabulary Size refers to the total number of unique tokens available in a tokenizer's vocabulary. 
        Word-level tokenization generally requires a larger vocabulary because different word forms are treated as separate tokens. 
        Character-level tokenization uses a much smaller vocabulary, while subword tokenization provides a balance between the two.

        The choice of tokenizer affects vocabulary size, processing efficiency, and how well an NLP model handles unfamiliar words.

        Research References
        Hugging Face. Tokenizers – NLP Course. 
        Hugging Face. Tokenization Algorithms. 
        Daðason, J., & Loftsson, H. (2026). Subword Tokenization for Low- and Medium-Resource Languages: A Systematic Evaluation. 

A4. Stopwords are common words that appear frequently in a language but often carry limited information for certain 
    NLP tasks. Examples in English include "the", "is", "a", "and", and "of".

    Benefits of Removing Stopwords:

    Removing stopwords can reduce the number of tokens, decrease data size, and help NLP systems focus on more meaningful words. This can be useful in tasks such as document classification and information retrieval.

    Risks of Removing Stopwords:
    Some stopwords carry important meaning and should not always be removed. For example, removing the word "not" from "The apartment is not good" changes the meaning and may cause a sentiment analysis model to classify negative feedback incorrectly.

    Why Stopword Lists Depend on the Task and Language:
    Different NLP tasks require different stopword lists. For example, sentiment analysis should preserve negation words such as "not" and "never", while some search applications may remove frequent words to improve search efficiency. Additionally, each language has its own vocabulary and grammatical structure, so an English stopword list may not work correctly for Somali.
    Therefore, stopword removal should be customized according to the language, dataset, and purpose of the NLP application.

    Research References
        Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python, Chapter 2. 
        Saeed et al. (2021). Stopwords in technical language processing. PLOS ONE.
        University of Sydney. Cleaning and Preparing Your Data.

A5. Stemming and Lemmatization: are NLP preprocessing techniques used to reduce different forms of words to a common base form.
    Stemming removes prefixes or suffixes from words using predefined rules. It does not always produce a meaningful dictionary word. For example, "connected", "connecting", and "connection" may be reduced to "connect".
    Lemmatization converts words into their correct dictionary base forms, called lemmas. Unlike stemming, it considers linguistic information such as vocabulary and part of speech. For example, "wolves" becomes "wolf", while "better" becomes "good" when identified as an adjective.

    Comparison and Use Cases:

    Stemming is generally faster and simpler, making it suitable for applications such as search and information retrieval. Lemmatization is more linguistically accurate and is preferred when preserving the correct meaning of words is important, such as in sentiment analysis and linguistic processing.

    The choice between them depends on the NLP task, required accuracy, and processing resources.

    Research References
        Porter, M. F. (1980). An Algorithm for Suffix Stripping. Program, 14(3), 130–137. Porter Stemmer documentation.
        Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. NLTK Book.
        Speech and Language Processing – Jurafsky & Martin.
    

A6. Regular Expressions (Regex): Regular Expressions (Regex) are patterns used to search, identify, extract, 
    or replace specific parts of text. In NLP preprocessing, they help clean raw text by removing unwanted characters, 
    identifying phone numbers, and standardizing text formats.

    Example 1: Finding Phone Numbers

        The pattern \d{10} matches exactly ten consecutive digits. For example, in the text "Call us at 0712345678", it identifies "0712345678" as a phone number.

    Example 2: Normalizing Whitespace

        The pattern \s+ matches one or more whitespace characters, including spaces, tabs, and newlines. It can replace multiple spaces with a single space, making text more consistent.

    How Overly Broad Regex Can Damage Text:

        An overly broad pattern may remove important information. For example, using [^a-z] to remove everything except lowercase letters can delete numbers, punctuation, currency symbols, and spaces. In a real estate dataset, this could remove important property prices such as "$500" and make the text difficult to understand.

        Therefore, regex patterns should be designed carefully to remove unwanted content without losing information that is important for the NLP task.

        Research Reference
        Python Software Foundation. Regular Expression Operations – Python Documentation. 

A7.  Case Study: Comparing Text Preprocessing Methods for Twitter Sentiment Analysis

        Research Paper: "A Comparative Evaluation of Pre-processing Techniques and Their Interactions for Twitter Sentiment Analysis" by Symeon Symeonidis, Dimitrios Effrosynidis, and Avi Arampatzis (2018).

        Task: The researchers aimed to investigate how different text preprocessing techniques affect the accuracy of Twitter sentiment analysis systems. The goal was to identify which preprocessing methods improve the classification of positive and negative tweets.

        Data: The study used two Twitter datasets and four machine learning algorithms: Linear SVC, Bernoulli Naive Bayes, Logistic Regression, and Convolutional Neural Networks (CNN).

        Method: The researchers experimentally compared 16 preprocessing techniques, including lowercasing, stopword removal, punctuation removal, number removal, lemmatization, and contractions replacement. They also examined how different preprocessing techniques interact when combined.

        Findings: The results showed that lemmatization, number removal, and contractions replacement improved classification accuracy. However, punctuation removal did not provide a significant improvement. The researchers also found that combining preprocessing techniques can affect model performance.

        Conclusion: This case study demonstrates that preprocessing choices directly influence NLP model performance. It also shows that removing more text does not always produce better results, so preprocessing techniques should be selected according to the specific NLP task.

## Part B – Practical
Task 1. Lowercasing converts all uppercase letters into lowercase letters. It helps standardize text, but a spam detector may lose useful capitalization patterns. For example, words such as "URGENT" and "WON" may indicate promotional or suspicious messages when written in uppercase.

Taks2. Whitespace Normalization removes unnecessary spaces and standardizes spacing in text. In this task, Python's split() and join() methods are used to replace multiple spaces with a single space and remove leading and trailing spaces. The repr() function is used to make whitespace visible when printing the original and normalized text.

Task 3: Regex Cleaningc, The pattern [^a-z\s] removes all characters except lowercase English letters and whitespace. When applied to lowercased T1, it deletes exclamation marks, the pound symbol, numbers, commas, and periods.

In the second step, custom regex rules replace the phone number with PHONE_NUMBER, the money amount with MONEY, and !!! with a single !. This preserves important information instead of deleting it completely.

Task4. Tokenization: The split() method separates text using whitespace, while NLTK's word_tokenize() provides more detailed tokenization by separating punctuation into individual tokens. As a result, the two methods may produce different token counts. For T4, the token count represents the total number of tokens, while the vocabulary size represents the number of unique tokens.


Task5. Stopwords and Negation Removing NLTK's default English stopwords removes common words, but it can also remove important negation words such as "not" and "don't". I created a custom stopword list that preserves negation words. This helps retain the negative meaning of the customer's complaint, which is important for sentiment analysis.

Task 6: Stemming Stemming reduces words to their root-like forms by removing prefixes or suffixes. I compared Porter, Lancaster, and Snowball stemmers using the given word list. Different stemmers may produce different results, and some stems may not be valid English words. Stemming can also merge different words into the same stem.


Task 9: POS Tagging and Named Entity Recognition 
POS tagging assigns grammatical categories, such as nouns and verbs, to words. Named Entity Recognition identifies entities such as people, locations, and organizations. I used NLTK's pos_tag() and ne_chunk() to analyze T3 and identify the recognized entities. The results also help reveal entities that the model may have missed or incorrectly classified.

Task 10: Full Preprocessing Pipeline

I applied lowercase conversion, regex cleaning, tokenization, stopword removal, and lemmatization to T2. I ran the pipeline twice using NLTK's default stopwords and a custom stopword list that preserves negation words. The custom pipeline better preserves the customer's negative sentiment because it keeps important words such as "not" and "n't", while default stopword removal may remove important negation information.