
## Part A – Research

1. Text preprocessing
Text preprocessing is the set of operations that prepares raw text for a particular NLP task. It may include normalization, tokenization, filtering, replacing patterns, and reducing inflected words to a common form. These steps can make text more consistent, reduce noise, and make features easier for a model to learn. However, cleaning is not automatically beneficial: deleting words or punctuation may remove evidence the model needs. For example, in a customer review, “I am not happy” could become “happy” if a stopword list removes not, reversing the sentiment signal.

2. Text normalization
Normalization converts different surface forms into a consistent representation. Lowercasing can combine “Gift” and “gift”; whitespace normalization can turn repeated spaces, tabs, or line breaks into a single space. Unicode normalization (such as NFC) can make visually identical characters encoded in different ways consistent; compatibility normalization (NFKC) may additionally fold some formatting distinctions. Normalization can help match text despite inconsistent formatting. It can also remove useful clues: lowercasing “Apple” may make it harder to distinguish the company from the fruit, while normalizing away emoji or unusual symbols can remove emotion or fraud/spam cues.

3. Tokenization and vocabulary
Tokenization	What it creates	Strengths	Limitations
Word-level	Whole words/punctuation tokens	Readable; captures words directly	Large vocabulary; unseen words and spelling variants become out-of-vocabulary
Subword	Pieces of words (e.g., common segments)	Balances vocabulary size and coverage; can represent unseen words using known pieces	A word may be split into multiple pieces; token boundaries may be less intuitive
Character-level	Individual characters or character units	Handles misspellings, morphology, and arbitrary new words	Long sequences; word-level meaning is less explicit and modeling can be harder
Vocabulary size is the number of distinct token types in the tokenizer's vocabulary or, in a corpus measurement, the number of unique tokens observed. Word tokenizers usually have larger vocabularies than character tokenizers; subword methods are designed to balance size and coverage. In this practical, T4's vocabulary size is the count of unique output tokens, not the number of tokens overall.

4. Stopwords
Stopwords are very frequent words that some tasks treat as having limited discriminative value, such as articles, auxiliary verbs, or prepositions. Removing them can reduce feature count and focus a model on content terms, particularly in some bag-of-words retrieval or classification settings. But removal is risky: “not,” “never,” and “no” change meaning; function words can also matter for syntax, authorship, translation, and question answering. Stopword lists are task-dependent and language-dependent. A list designed for English cannot safely be applied to Somali or another language without linguistic review and evaluation.

5. Stemming and lemmatization
Stemming uses heuristic rules to strip affixes and may produce a non-dictionary stem. Lemmatization uses vocabulary and morphological knowledge (often with part-of-speech information) to return a dictionary lemma. Stemming is typically faster and useful when approximate grouping is acceptable, such as search indexing; lemmatization is preferable when interpretable lexical forms or grammatical distinctions matter. The Porter Stemmer is a well-known suffix-stripping algorithm that applies ordered rules. Lancaster stemming is more aggressive, while Snowball (Porter2) is a revised stemming algorithm for multiple languages. NLTK's WordNetLemmatizer uses WordNet and defaults to noun POS; supplying POS can improve results (e.g., a verb form can be mapped to its verb lemma).

6. Regular expressions
Regular expressions describe patterns in text and can be used to find, remove, or replace matching substrings. For example, \\d{4} matches a sequence of four digits, while \\s+ matches one or more whitespace characters. In this practical, [^a-z\\s] matches any character that is not a lowercase English letter or whitespace, so it deletes punctuation, digits, currency symbols, uppercase characters (if not lowercased first), and emoji. An overly broad rule such as deleting all non-letter characters can damage information by removing numbers, apostrophes, emoji, or punctuation that signal price, negation contractions, emotion, or spam intent.

7. Case study: preprocessing for sentiment analysis
Symeonidis, Effrosynidis, and Arampatzis (2018) experimentally compared 16 preprocessing techniques for Twitter sentiment analysis using two Twitter datasets and four classifiers (Linear SVC, Bernoulli Naïve Bayes, Logistic Regression, and CNN). They measured classification accuracy and feature counts and also studied combinations/interaction effects. Their reported findings indicate that lemmatization, number removal, and contraction replacement could improve accuracy, while punctuation removal did not consistently help; interactions between methods mattered, including replacing numbers and repeated punctuation. This illustrates why preprocessing should be selected and tested for a specific dataset/task rather than applied as a universal checklist.

Part B – Practical
Important: Run preprocessing_practice.py in your course environment to generate the actual results. Copy the output produced by your script into the practical report where your instructor requires it; do not fabricate results. This environment did not have the required NLTK data packages installed, so I have not claimed to execute the experiments here.

Practical observations to verify from your generated output
Lowercasing: Lowercasing T1 makes it easier to treat case variants consistently, but a spam detector could lose the all-caps urgency signal in “URGENT” and “NOW.” Case may be useful as a feature, so preserve the original text or a capitalization feature as well as a normalized copy.

Regex cleaning: The specified [^a-z\\s] rule deletes punctuation, currency symbols, and digits from lowercased T1; it removes the currency marker and amount, the phone number, exclamation marks, and sentence punctuation. The replacement approach in the script preserves the semantic categories as MONEY and PHONE_NUMBER, and compresses repeated exclamation marks, retaining more useful information than deletion.

Stopwords and negation: Standard English stopword removal may remove negation words, causing “not happy” to be reduced to “happy.” The custom list removes common stopwords but retains negation tokens, so the negative sentiment remains represented; compare the two token lists from the script.

Stemming table: Paste the actual comparison printed by your run (Word / Porter / Lancaster / Snowball) here. Check the table for two distinct words merged to one stem and identify at least one stem that is not a dictionary word. Because stemmer versions and algorithms differ, use your printed results as evidence rather than guessing.

Lemmatization: Paste the default and POS-hinted results from the script. The default WordNet lemmatizer assumes noun POS; a suitable verb/adjective hint may change inflected forms such as verbs or comparative adjectives. State which entries changed only after checking your output.

Sentence splitting: NLTK's sentence tokenizer uses learned punctuation context and should normally treat abbreviations such as “Mr.” and “a.m.” carefully. Report the exact sentence count and whether your installed tokenizer split at those abbreviations based on the output.

POS and named entities: Use the POS-tag and named-entity tree printed by your run. NLTK's classic chunker may label person/place/organization names, but it is not guaranteed to recognize every entity, weekday, time, or organization. Fill the entity table from your output and assess each label against the sentence meaning; note any missed entity (for example, the weekday Friday or company Apple, depending on the model's predictions).

Full pipeline: The pipeline in the script prints every stage twice. The negation-preserving variant should retain “not” (and the contraction token, if tokenization yields it) while the default list may remove it. The version that retains negation better preserves the customer's complaint, though emoji removal also discards an emotional signal in this particular example.

Spam detector: preprocessing choices
For a spam detector, I would normalize whitespace and consider lowercasing, but I would preserve the original message or add features for capitalization, repeated punctuation, digits, money symbols, phone numbers, and emoji. I would replace sensitive or structured patterns with labels rather than delete them (such as PHONE_NUMBER and MONEY), and I would avoid aggressive stopword removal, stemming, and deletion of punctuation until experiments showed a benefit. The practical example T1 contains urgency, a prize claim, currency, a phone number, and strong punctuation—signals that overly aggressive cleaning could erase.

References
Chai, C. P. (2023). Comparison of text preprocessing methods. Natural Language Engineering, 29(3), 509–553. https://doi.org/10.1017/S1351324922000213
Camacho-Collados, J., & Pilehvar, M. T. (2018). On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis. Proceedings of the EMNLP Workshop BlackboxNLP, 40–46. https://doi.org/10.18653/v1/W18-5406
Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2018.06.022
Palomino, M. A., & Aider, F. (2022). Evaluating the Effectiveness of Text Pre-Processing in Sentiment Analysis. Applied Sciences, 12(17), 8765. https://doi.org/10.3390/app12178765
NLTK Project. (n.d.). NLTK documentation: Tokenizers, stemming, WordNet, POS tagging, and named entity chunking. https://www.nltk.org/
Context window full
Start a new chat with an automatic summary of this conversation, or upgrade to continue here with more context available in every chat.




