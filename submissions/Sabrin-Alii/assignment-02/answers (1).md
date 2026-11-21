# Assignment 2 – Preprocessing

## Part A – Research

### 1. What is text preprocessing?

Text preprocessing is the preparation of raw text before it is given to an NLP system. It can make writing more consistent and easier to analyze, for example by organizing spaces or separating words. However, removing too much can change the message: deleting **not** from “The service is not reliable” can make a negative opinion look positive. 

### 2. Text normalization

Text normalization makes different forms of writing more consistent. Lowercasing can help when a search system should treat “REPORT” and “Report” as the same term, while whitespace normalization can turn repeated spaces or tabs into one space. 

### 3. Word-level, subword and character-level tokenization

A word-level tokenizer treats words and punctuation as separate units, so a vocabulary may contain many entries because different word forms are stored separately. A character-level tokenizer represents text one character at a time; its vocabulary is usually smaller, but a sentence becomes a much longer sequence. A subword tokenizer divides uncommon or long words into learned pieces, giving it a vocabulary between character and word systems and helping it represent unfamiliar words. Vocabulary size means the number of unique token entries a tokenizer knows, not the number of tokens in one document [1, 4].

### 4. Stopwords

Stopwords are frequent words that some NLP systems remove to reduce the number of features or focus on more informative terms. This can help certain search or topic tasks, but it can also damage meaning when a common word expresses negation, such as “not” or “never.” The list must match both the task and the language: an English stopword list should not simply be applied to Somali text, and sentiment analysis may need to retain words that a topic-search system could remove [2, 5].

### 5. Stemming and lemmatization

Stemming reduces words by applying suffix rules; it is usually quick, but its result may not be a dictionary word. The Porter stemmer is a well-known English method that applies a sequence of suffix-stripping rules [6]. Lemmatization instead aims to return a valid dictionary form and can use the word's grammatical role; for example, NLTK's WordNet lemmatizer can return different results when it receives a verb or adjective hint [7]. Stemming may be useful for fast matching, while lemmatization is preferable when a meaningful base form is important.

### 6. Regular expressions in text cleaning

A regular expression (regex) describes a pattern that a program can find, replace or remove. For example, `\s{2,}` matches two or more whitespace characters and can replace them with one space; `\b\d{4}-\d{2}-\d{2}\b` matches a date written as YYYY-MM-DD. A broad rule such as `[^a-z\s]` can remove numbers, currency symbols, apostrophes, emoji and letters outside the English alphabet. This may delete evidence needed by the task, so regex rules should be tested on real examples [1, 2].

### 7. Case study: comparing preprocessing choices

HaCohen-Kerner, Miller and Yigal (2020) studied text classification using a bag-of-words representation. They compared combinations of spelling correction, lowercasing, HTML-tag removal, punctuation removal, stopword removal and repeated-character reduction across four benchmark datasets: WebKB, R8, SMS Spam Collection and Sentiment Labelled Sentences. They tested three supervised learning methods and evaluated classification accuracy. Their findings showed that the best preprocessing combination depended on the dataset: stopword removal helped as a single step on three datasets, but not on the fourth; lowercasing helped on all four, and the strongest choice for one dataset combined spelling correction with lowercasing. This study shows that preprocessing should be tested for each task rather than assuming that every cleaning step is always helpful [5].

## Part B – Practical

Run `preprocessing_practice.py` after completing the course NLTK setup. Save the real printed output as `output.txt`; the practical report below must be checked against that output before submission.

### Practical observations to complete from `output.txt`

**Task 1 – Lowercasing:** Lowercasing T1 makes differently capitalized versions of a word more consistent. It also removes the all-capital emphasis in the message, which could be a useful signal for a spam detector.

**Task 2 – Whitespace:** The script prints T4 before and after whitespace normalization using `repr()`, so the repeated spaces can be seen. The normalized text should have one space between each item.

**Task 3 – Regex cleaning:** The broad pattern replaces every character outside lowercase English letters and whitespace with spaces. In T1 this removes the currency sign, amount digits, comma, punctuation and phone digits; the task-specific rules instead preserve the information by replacing the amount and phone number with markers and reducing repeated exclamation marks.

**Task 4 – Tokenization:** Copy the `split()` and `word_tokenize()` token lists and counts from `output.txt`. Also report the T4 token count and vocabulary size. The vocabulary size is the number of unique tokens; the script reports both case-sensitive and lowercased counts to make the effect of capitalization clear.

**Task 5 – Stopwords and negation:** Compare the two token lists in `output.txt`. The custom list is designed to keep `not`, `n't`, `no` and `never`, because removing negation can change the opinion expressed by a sentence.

**Task 6 – Stemming table:** Copy the generated table from `output.txt`.

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

Porter merges the different words “connected” and “connecting” into the same stem, `connect`. The stem `fli` produced from “flies” is not a normal English dictionary word, which shows that stemming does not always return a real word. Stemmers can also merge words that have different meanings, so their results should be checked for the task.

**Task 7 – Lemmatization:** Copy the default and POS-hinted results from `output.txt`. In this word list, the verb and adjective hints are important for words such as “connected,” “connecting,” “organized,” “better” and “ran.” The default NLTK WordNet lemmatizer assumes a noun unless another part of speech is supplied [7].

**Task 8 – Sentence splitting:** Copy the number of sentences and the sentence list from `output.txt`. Check that “Mr.” remains part of the person's title and that “a.m.” does not cause an incorrect extra sentence boundary.

**Task 9 – POS tagging and NER:** Copy the actual NER tree and review table from `output.txt`. Compare each prediction with the context in T3; the intended entities are Mr. Smith (person), London (place), Friday (day) and Apple (organization). Do not claim the model recognized an entity unless it appears in the output.

| Text | Expected label | Predicted label from output.txt | Correct? |
|---|---|---|---|
| Mr. Smith | PERSON | Complete after running script | Complete after running script |
| London | GPE | Complete after running script | Complete after running script |
| Friday | DATE | Complete after running script | Complete after running script |
| Apple | ORGANIZATION | Complete after running script | Complete after running script |

**Task 10 – Full pipeline:** Compare both runs printed by the script. The version that keeps negation should preserve the words that show the customer's dissatisfaction; the default stopword list may remove “not” and make part of the complaint less clear. The regex stage also removes the emoji in T2, so that is another loss of information to notice.

### Preprocessing choices for a spam detector

For a spam detector, I would normalize unnecessary spaces and use tokenization consistently. I would avoid deleting all numbers, currency symbols, links or repeated punctuation because they may help identify suspicious messages; replacing phone numbers and money amounts with markers can keep their importance without storing every exact value. I would also avoid removing negation words automatically and would test whether lowercasing or stemming actually improves validation results before using them.

## References

[1] Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

[2] Chai, C. P. (2023). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

[3] Unicode Consortium. (2026). *Unicode Standard Annex #15: Unicode Normalization Forms*. https://www.unicode.org/reports/tr15/

[4] Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed., online manuscript). Chapter 2: Words and Tokens. https://web.stanford.edu/~jurafsky/slp3/

[5] HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020). The influence of preprocessing on text classification using a bag-of-words representation. *PLOS ONE, 15*(5), e0232525. https://doi.org/10.1371/journal.pone.0232525

[6] Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

[7] NLTK Project. (n.d.). *WordNetLemmatizer documentation*. https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html

