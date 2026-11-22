# Assignment Two: Preprocessing

**Author:** Ali Omar Abdi     
**Subject:** Natural Language Processing (NLP)
  
## Part A – Research

### 1. What is text preprocessing?

Text preprocessing means preparing raw text before an NLP system analyses it. Text can contain different letter cases, extra spaces, punctuation, symbols, and different forms of words. Preprocessing makes the text more suitable for a computer to process.

For example, a search system may treat `Book`, `BOOK`, and `book` as different words. Lowercasing can make them consistent.

However, more cleaning does not always give better results. Some text contains useful information. For example, removing `not` from “I am not satisfied” changes the meaning. Removing numbers and currency symbols from a spam message may also remove clues that help identify spam. The best preprocessing steps depend on the task (Chai, 2023).

### 2. Text normalization

Text normalization changes text into a consistent form.

| Method | Meaning | Example |
|---|---|---|
| Lowercasing | Changes capital letters to small letters | `WELCOME` → `welcome` |
| Whitespace normalization | Removes extra spaces | `good   morning` → `good morning` |
| Unicode normalization | Makes equivalent character forms consistent | Two Unicode forms of `é` become comparable |

Normalization helps when a website search needs to match `LAPTOP` and `laptop`.

However, lowercasing can remove useful information. For example, `US` can mean the United States, while `us` is a pronoun. Capital letters can also help identify names and abbreviations. Unicode normalization is useful for comparing equivalent characters, but compatibility normalization can remove some distinctions. The Unicode Consortium explains the differences between these normalization forms (Unicode Consortium, n.d.).

### 3. Word-level, subword, and character-level tokenization

Tokenization divides text into smaller units called tokens. A token can be a complete word, part of a word, or one character.

| Type | How it works | Example: `unhelpful` | Advantage | Limitation |
|---|---|---|---|---|
| Word-level | Uses complete words | `unhelpful` | Easy to understand | May not recognize unfamiliar words |
| Subword | Divides words into smaller pieces | `un`, `help`, `ful` | Can handle rare words | One word may become several tokens |
| Character-level | Uses individual characters | `u`, `n`, `h`, `e`, ... | Can represent unfamiliar words | Creates longer sequences |

**Vocabulary size** means the number of different tokens in a tokenizer's vocabulary. Word-level tokenizers often need larger vocabularies because they store many complete words. Subword tokenizers use reusable pieces. Character-level tokenizers usually have smaller vocabularies because they use individual characters.

Sennrich, Haddow, and Birch (2016) showed that subword units help machine translation process rare and unfamiliar words. This is one reason subword tokenization is widely used in modern NLP.

### 4. Stopwords

Stopwords are common words that are sometimes removed during text processing. English examples include `the`, `is`, `and`, and `of`.

Removing stopwords can reduce the number of words a system processes. This may help some search and text-classification tasks.

However, removing stopwords can change meaning:

- `The student passed.`
- `The student did not pass.`

If the system removes `not`, it loses the negative meaning.

Stopword lists should depend on the task and language. An English list may not work for Somali because the languages have different grammar and vocabulary. Saif et al. (2014) studied stopword removal in Twitter sentiment analysis and found that precompiled stopword lists could reduce classification performance in their experiments.

### 5. Stemming and lemmatization

Stemming and lemmatization reduce different forms of a word. They use different methods.

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Removes word endings using rules | Uses vocabulary and grammatical information |
| Result | May not be a real word | Usually a dictionary word |
| Example | `studies` → `studi` | `studies` → `study` |
| Processing | Usually faster | Often needs more resources |

**Stemming:** The Porter Stemmer is a well-known algorithm. It uses rules to remove word endings. For example, `connected` and `connecting` can both become `connect`. Stemming is useful when a search system needs to group related word forms quickly (Porter, 1980).

**Lemmatization:** Lemmatization tries to find the dictionary form of a word. For example, the verb `running` becomes `run`, and the noun `children` becomes `child`. It can use part-of-speech information to choose the correct form.

Stemming may be preferred when speed is important. Lemmatization may be preferred when meaningful base words and grammar matter.

### 6. Regular expressions in text cleaning

Regular expressions, also called regex, are patterns used to find, replace, or remove text. Python provides the `re` module for this work.

| Regex pattern | What it matches | Example |
|---|---|---|
| `\b\d{10}\b` | A separate sequence of ten digits | `0712345678` |
| `\b\d{4}-\d{2}-\d{2}\b` | A date in year-month-day format | `2026-11-22` |

Regex can help find phone numbers, dates, email addresses, and repeated punctuation.

A pattern that is too broad can remove useful information. For example, a rule that keeps only English letters and spaces may delete numbers, currency symbols, punctuation, and emoji. These may be useful clues in a spam detector. It can be safer to replace important information with labels instead of deleting it.

### 7. Case study: Preprocessing for Twitter sentiment analysis

Symeonidis, Effrosynidis, and Arampatzis (2018) studied how preprocessing affects Twitter sentiment analysis. Their task was to classify tweets according to their sentiment.

The researchers compared 16 preprocessing techniques using two Twitter datasets and four machine-learning methods: Linear SVC, Bernoulli Naive Bayes, Logistic Regression, and a Convolutional Neural Network.

Their results showed that some methods, including lemmatization, number removal, and replacing contractions, improved performance in their experiments. Removing punctuation did not provide the same benefit.

This study shows that preprocessing choices can affect model results. A method that helps one dataset may not help another. NLP developers should test preprocessing methods on the data and task they are working with (Symeonidis et al., 2018).

---

## References

1. Chai, C. P. (2023). Comparison of text preprocessing methods. *Natural Language Engineering, 29*(3), 509–553. https://doi.org/10.1017/S1351324922000213

2. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814

3. Saif, H., Fernandez, M., He, Y., & Alani, H. (2014). On stopwords, filtering and data sparsity for sentiment analysis of Twitter. *Proceedings of LREC*, 810–817. https://aclanthology.org/L14-1265/

4. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL*, 1715–1725. https://doi.org/10.18653/v1/P16-1162

5. Symeonidis, S., Effrosynidis, D., & Arampatzis, A. (2018). A comparative evaluation of pre-processing techniques and their interactions for Twitter sentiment analysis. *Expert Systems with Applications, 110*, 298–310. https://doi.org/10.1016/j.eswa.2018.06.022

6. Unicode Consortium. (n.d.). *Unicode Standard Annex #15: Unicode normalization forms*. https://www.unicode.org/reports/tr15/
