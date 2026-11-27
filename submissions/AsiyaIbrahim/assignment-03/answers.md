# Assignment 3 – Vectorization
# Part A – Research

## 1. What is text vectorization? What is one-hot encoding?

**Text vectorization** means changing words or sentences into numbers. Machine learning models work with numbers, so we must change text into numbers before giving it to a model.

A model usually needs every document to have the same number of features (columns). For example, a short SMS and a long email should both be changed into vectors with the same number of columns.

**One-hot encoding** gives each word its own list of numbers. The list contains `1` in the word's position and `0` in all other positions.

Example vocabulary: `[cash, class, free, lunch, notes, win]`

The word `free` becomes:

`[0, 0, 1, 0, 0, 0]`

**Why can one-hot encoding be a problem for full documents?**

It creates one vector for each word. A sentence with 3 words creates 3 vectors, while a sentence with 40 words creates 40 vectors. This means the number of vectors changes with the sentence length. Many traditional machine learning models need one fixed-size vector for each document, so we must use another method or combine the word vectors.

One-hot encoding also does not explain word meaning. It treats `free` and `cash` as two separate words, just as it treats `free` and `lunch` as separate words.

**Example:** A spam filter must turn both a short message such as “Win a free phone!” and a long email into a format the model can process.

## 2. What is Bag of Words (BoW)? What is a document-term matrix?

**Bag of Words (BoW)** changes a document into word counts. It does not keep the order of the words.

A **document-term matrix** is a table:

- Each row represents one document.
- Each column represents one word in the vocabulary.
- Each cell shows how many times that word appears in that document.

It is called a “bag” because the words are treated like words placed in a bag. We count them, but we do not remember their order.

**When can losing word order be acceptable?**

For spam detection, words such as `free`, `win`, and `prize` can be useful even if we do not know their exact order.

**When can losing word order cause a problem?**

It can cause problems when the order changes the meaning. For example, “not good” and “good” have different meanings. BoW may not understand this difference when it uses single words only. It can also confuse “dog bites man” with “man bites dog”.

## 3. Vocabulary size and sparse data

The **vocabulary** is the list of different words found in the documents. Each word becomes a column in the document-term matrix.

A large collection of documents can have thousands of columns. However, one document uses only a small number of those words. Most cells are therefore `0`. This is called **sparse data**.

A **sparse matrix** saves memory by storing the non-zero values and their positions instead of storing every zero.

These settings can change the vocabulary:

| Setting | Simple meaning |
|---|---|
| `min_df` | Remove words that appear in fewer than the chosen number of documents. |
| `max_df` | Remove words that appear in too many documents. |
| `max_features` | Keep only the chosen number of most common words. |
| Lowercasing | Change capital letters to small letters, so `Win` and `win` count as the same word. |
| Stop words | Remove selected common words, such as `the` or `to`. |
| Stemming | Change related word forms to a shorter common form, such as `prizes` to `prize`. |

## 4. What are n-grams? What are their benefits and costs?

An **n-gram** is a group of words that appear next to each other.

- **Unigram:** one word, such as `good`.
- **Bigram:** two words, such as `not good`.
- **Trigram:** three words, such as `the food was`.

N-grams help keep some word order. For example, the bigram `not good` can help a model understand that the opinion is negative.

**The disadvantage:** Adding bigrams and trigrams creates many more features. This makes the matrix larger and more full of zeros. Some word groups may appear only once, so the model may learn details that do not help it with new data. This is called **overfitting**.

In our practical work, the vocabulary grows from 32 words to 67 features with unigrams and bigrams, and to 98 features with unigrams, bigrams, and trigrams.

## 5. What is TF-IDF? Why do rare words get higher scores?

**TF-IDF** gives a score to each word. A word gets a higher score when it appears often in one document but does not appear in many other documents.

TF-IDF has two parts:

- **TF (Term Frequency):** how often a word appears in one document.
- **IDF (Inverse Document Frequency):** gives more weight to words that appear in fewer documents.

The basic formula is:

`TF-IDF = TF × IDF`

`IDF = log(N / df)`

Here, `N` is the total number of documents, and `df` is the number of documents that contain the word.

**Why do rare words get higher scores?**

If a word appears in only a few documents, its IDF score is higher. If it appears in every document, its IDF score is low (zero in this basic formula). This makes rare words more useful for telling documents apart.

**How is scikit-learn different?**

scikit-learn uses a smoothed IDF formula and then usually applies L2 normalization to each document vector. This is why its scores can be different from a hand calculation using the basic formula.

TF-IDF scores are **importance scores**, not probabilities. A score does not tell us the chance that a word will appear.

**Background:** Karen Spärck Jones explained the idea of inverse document frequency in 1972. Salton and Buckley later discussed term-weighting methods in information retrieval in 1988.

## 6. What are `fit`, `transform`, `fit_transform`, OOV words, and data leakage?

- **`fit`:** learns the vocabulary and, for TF-IDF, the word IDF scores from the training text.
- **`transform`:** changes text into numbers using the vocabulary already learned.
- **`fit_transform`:** learns the vocabulary and changes the text into numbers in one step.
- **OOV (out-of-vocabulary) words:** words that were not found when the vectorizer learned the vocabulary. They do not get a column and are usually ignored.

**What is data leakage?**

Data leakage happens when information from the test data is used while preparing or training the model. For example, if we learn the vocabulary from both training and test text, the model has already seen information from the test set. This can make the test result look better than it should.

**Correct steps:**

1. Split the data into training data and test data.
2. Use `fit` or `fit_transform` on the training text only.
3. Use `transform` on the training text and test text.
4. Train the model using the training data.
5. Test the model using the test data.

## 7. Case study: Somali news classification using TF-IDF

A useful Somali-language example is the 2026 study by **M. M. Ali**, titled *“Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus.”* It was published in **Scientific Reports**.

**What problem did the study work on?**

The study aimed to automatically place Somali news articles into the correct topic group. This is called news topic classification.

**What data did the researchers use?**

They prepared a collection of **7,240 Somali news articles** from BBC Somali and VOA Somali. The articles were placed into eight topic groups:

1. Caafimaad
2. Ciyaaro
3. Dagaal
4. Diini
5. Ganacsi
6. Madadaalo
7. Siyaasad
8. Tiknoolajiyad

**What method did they use?**

They changed the news text into numbers using **word-level TF-IDF with single words (unigrams)**. They then used a machine learning model called **Linear Support Vector Machine (Linear SVM)** to predict the topic.

The researchers also compared different machine learning models and other SVM feature settings. Their best Linear SVM setup reached **94.20% accuracy** in an 80/20 train-test evaluation. In repeated testing and five-fold cross-validation, the average accuracy was about **93%**.

**What can we learn from this study?**

The study shows that TF-IDF can be useful for classifying Somali news. It also shows that a traditional machine learning model can give strong results without needing a very large or expensive deep-learning model.

However, this study mainly tests TF-IDF with different models and settings. It does **not** directly prove that TF-IDF is always better than every Bag of Words method or every word-embedding method. The result applies to the dataset and test setup used in this study.

This connects to our practical work: TF-IDF gives more weight to words that help separate documents. For example, words such as `cash` and `prizes` can be more useful for identifying a spam message when they appear in fewer documents.

**Reference for this case study:**

Ali, M. M. (2026). *Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus.* Scientific Reports. https://doi.org/10.1038/s41598-026-68159-5

## References

- Spärck Jones, K. (1972). “A statistical interpretation of term specificity and its application in retrieval.” *Journal of Documentation*, 28(1).
- Salton, G., & Buckley, C. (1988). “Term-weighting approaches in automatic text retrieval.” *Information Processing & Management*, 24(5).
- Ali, M. M. (2026). “Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus.” *Scientific Reports*. https://doi.org/10.1038/s41598-026-68159-5
- scikit-learn documentation. “Text feature extraction.” https://scikit-learn.org/stable/modules/feature_extraction.html

---

# Part B – Practical

The numbers below come from the Python program's `output.txt` file.

## Document-term matrix (Task 2)

The columns show the 32 vocabulary words.

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |

| Doc | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | **2** |
| D2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **2** | 0 | 0 | 1 | 0 | 1 | 0 |

Shape: 5 × 32.

## Count vs TF-IDF for every word in D2 (Task 7)

| Word | Count | TF-IDF |
|---|---|---|
| call | 1 | 0.3891 |
| cash | 1 | 0.3891 |
| entry | 1 | 0.3891 |
| prizes | 1 | 0.3891 |
| free | 1 | 0.3140 |
| now | 1 | 0.3140 |
| today | 1 | 0.3140 |
| win | 1 | 0.3140 |

## Hand calculation for D2 (Task 8)

D2 tokens (as the vectorizer counts them): `free, entry, win, cash, prizes, today, call, now` → length 8. N = 5 documents.

| Word | count | df | TF = count/8 | IDF = ln(5/df) | TF-IDF |
|---|---|---|---|---|---|
| cash | 1 | 1 | 0.1250 | ln(5/1) = 1.6094 | 0.2012 |
| today | 1 | 2 | 0.1250 | ln(5/2) = 0.9163 | 0.1145 |

## Short answers

**Task 1 (one-hot).** `free` = `[0,0,1,0,0,0]` and `cash` = `[1,0,0,0,0,0]`. `win free cash` gives 3 vectors, while `free cash` gives 2. The number of vectors equals the number of words, so documents of different lengths give different shapes, which a standard ML model cannot accept. One-hot also does not represent the sentence as a whole.

**Task 2 (Bag of Words).** The matrix is 5 × 32. The cell (D1, `win`) is 2 because "win" appears twice in D1 ("Win a FREE phone… Reply WIN…"); CountVectorizer lowercases, so `Win` and `WIN` are counted together. `win free phone` and `phone free win` give the **identical** vector, which proves that word order is lost.

**Task 4 (preprocessing).** The default vectorizer removed `a` because it keeps words with at least two characters. It also ignores punctuation such as `!`. It changed capital letters to small letters, so `WIN` and `Win` became the same word: `win`. With `lowercase=False`, `WIN`, `Win`, and `FREE` are treated as different words, so the vocabulary becomes larger. The English stop-word list removed `now` and `to`. With `stop_words="english"` on D1–D5, **`not` was removed**. D5 becomes `lecture good notes great`, which says the lecture was good and the notes great, the opposite of the real review ("not good"). So blindly removing stopwords can flip sentiment.

**Task 5 (limiting vocabulary).** Full vocabulary = 32; `min_df=2` → 7 words (`free, lecture, notes, now, the, today, win`); `max_features=5` → 5 words (`free, lecture, now, the, win`). The survivors are words that appear in several documents, which includes frequent function words like `the`. This reduces noise and size, but can also throw away rare yet informative words (for example `cash`, `prizes`).

**Task 6 (n-grams).** The bigram **`not good`** keeps the reviewer's opinion (the unigram `good` alone suggests a positive review). Vocabulary size on D1–D5 grows from 32 (unigrams) to 67 (1–2) to 98 (1–3), so it roughly doubles and then triples even on five tiny documents; on real data growth is much worse.

**Task 7 (TF-IDF).** In D2 every word appears once, but `call, cash, entry, prizes` (each appearing in only 1 document) get **0.3891**, while `free, now, today, win` (each appearing in 2 documents) get **0.3140**. Words that appear in fewer documents get a higher IDF score (2.0986 instead of 1.6931). In D1, `win` has the highest weight (**0.5739**) because it appears **twice** in D1 (TF = 2), which more than compensates for its lower IDF (it also appears in D2).

**Task 8 (by hand).** `cash` = 0.2012 and `today` = 0.1145 by the textbook formula; scikit-learn gave 0.3891 and 0.3140. The values are different because scikit-learn uses a smoothed IDF formula and normalizes each document vector. However, the **order is the same**: `cash` has a higher score than `today` because `cash` appears in fewer documents (df = 1, while `today` has df = 2).

**Task 10 (fit/transform).** Fit on D1–D4 gives 26 words. N1 vector has 1s for `claim, free, now`; OOV words in N1: `your`, `prize`. N2 vector has 1s for `lunch, meeting, the, to`; OOV words in N2: `is`, `moved`, `tomorrow`. `prize` is unknown because the vocabulary has the exact string `prizes`, and CountVectorizer does no stemming or lemmatization. **Stemming or lemmatization** (Assignment 2) would map `prizes` and `prize` to the same form.

**Task 11 (data leakage).** The leaky vocabulary has 31 words versus 26. Words leaked from the test texts: `is, moved, prize, tomorrow, your`. If we fit the vectorizer using test text, it learns words from data that should be kept unseen until testing. With TF-IDF, it also learns document-frequency information from the test data. Correct order: split → fit on training only → transform both.

**Task 12 (no meaning).** All three pairs have cosine similarity **0.750**. `tasty` is **not** more similar to `delicious` than to `terrible`, because each is just a different column; the reviews share 3 of 4 words (`the, food, was`), and the one differing word counts the same no matter what it means. BoW sees whether words are the same; it does not understand what the words mean.

## Which vectorizer settings for a spam detector?

For spam detection I would use **TF-IDF (or counts) with unigrams, possibly adding bigrams**, lowercase on, and `min_df=2` to drop one-off noise, because spam is recognized by key words like `free`, `win`, `claim`, `prizes`, `cash`, and word order matters little for this task (Task 2: `win free phone` = `phone free win`). TF-IDF is helpful because it gives rare spammy words (e.g. `cash`, `prizes`, df = 1) more weight than common ones (`today`, `now`, df = 2), as in Task 7. I would **be careful with stopword removal**: `now` and `to` were removed in Task 4, but `now` and `call to action` words can be spam signals ("Call now"), and Task 4 also showed `not` is deleted, which can flip meaning. I would avoid very large n-grams (vocabulary jumped from 32 to 98 in Task 6) and too small `max_features` (Task 5 kept `the` but could drop `cash`). Finally, the vectorizer must be **fit on training messages only** (Task 11), and OOV spam words like `prize` (Task 10) show why stemming/lemmatization or character n-grams may help a real system.
