# Assignment 3 – Vectorization

**Goal:** Understand how text is turned into numbers that Machine Learning models can use, including one-hot encoding, Bag of Words, the document-term matrix, vocabulary, n-grams, TF-IDF, fitting versus transforming, data leakage, and why these methods do not capture meaning.

## Part A – Research

### Instructions
- **Do not use** the explanations or examples we provided earlier (e.g., classroom notes).
- You must conduct **deep research** using books, scholarly articles, and reliable sources.
- **You may use AI tools**, but **do not rely blindly**: do not submit answers you don’t fully understand or verify.
- The purpose is for you to **learn through research** — therefore, explain ideas clearly, restate key concepts in your own words, and reference the sources you used.
- Write in **English**, in a research-oriented style (with headings, paragraphs, and tables/figures where appropriate), and include **references** at the end.

1. Define **text vectorization**. Why do most Machine Learning models need every document described with the **same number of numerical features**? Explain **one-hot encoding** of words and why it is not enough to represent a whole document. Use a real-life example.

2. Explain **Bag of Words** and the **document-term matrix**: what the rows, columns and cells represent. Why is it called a "bag"? Give one task where losing word order does little harm and one where it causes real problems.

3. Explain **vocabulary size** and **sparsity**. Why do real document-term matrices become very large and mostly zero, and how are sparse matrices stored efficiently? Describe how settings such as `min_df`, `max_df` and `max_features`, and preprocessing choices (lowercasing, stopwords, stemming), change the vocabulary.

4. What are **n-grams**? Explain how unigrams, bigrams and trigrams keep part of the word order, give one example where a bigram changes the meaning of a text, and discuss the cost of using larger n-grams.

5. Explain **TF-IDF**. Research where the idea of **inverse document frequency** came from, write the textbook formula, and explain why rarer words get higher weights. Then research how scikit-learn's `TfidfVectorizer` differs from the textbook formula (for example smoothing and normalization) and explain why TF-IDF values are **weights, not probabilities**.

6. Explain the difference between **`fit`**, **`transform`** and **`fit_transform`**. What happens to **out-of-vocabulary** words? What is **data leakage** in vectorization, why does it make evaluation results look better than they really are, and what is the correct order of steps?

7. Find one **case study** (research paper or article) that compares text representations (for example Bag of Words, TF-IDF, n-grams or word embeddings) for an NLP task, preferably in Somali or another African language. Summarize the task, the data, the representations compared, and the results. Explain how the study shows (or discusses) the main limitation of Bag of Words and TF-IDF: they **do not capture meaning**.

## Part B – Practical

Now apply what you researched. Write your own Python script, `vectorization_practice.py`, that vectorizes the documents below and **prints** the result of every step (like the class teaching script: one heading per step, then the result).

### Setup
- Use **scikit-learn** (`CountVectorizer`, `TfidfVectorizer`) and **NumPy** (`pip install scikit-learn numpy`).
- Save the full printed output to `output.txt` (for example: `python vectorization_practice.py > output.txt`).
- **Do not type outputs by hand.** Every result in your report must come from your own script.

### Documents (training corpus)

```text
D1: Win a FREE phone now! Reply WIN to claim.
D2: Free entry: win cash prizes today. Call now.
D3: Are we still meeting for lunch today?
D4: Please send me the lecture notes before class.
D5: The lecture was not good, but the notes were great.
```

### New texts (test set)

```text
N1: Claim your free prize now
N2: Is the lunch meeting moved to tomorrow?
```

### Reviews

```text
R1: the food was tasty
R2: the food was delicious
R3: the food was terrible
```

### Tasks

1. **One-hot encoding:** build your own one-hot encoder (plain Python, no library) with the vocabulary `[cash, class, free, lunch, notes, win]`. Print the vector for each word in `free cash`. Then print the one-hot vectors for `win free cash` and explain why the number of vectors depends on the sentence length.

2. **Bag of Words:** fit a `CountVectorizer` on **D1–D5**. Print the vocabulary, the shape and the full document-term matrix. Which cell in D1 has the value `2`, and why? Then transform `win free phone` and `phone free win` and print both vectors. Are they the same?

3. **Tokens vs vocabulary size:** print the total number of tokens (word occurrences) in D1–D5 and the vocabulary size. List the words that appear in **more than one** document.

4. **Preprocessing inside the vectorizer:** fit three vectorizers on **D1 only**: the default, `lowercase=False` and `stop_words="english"`. Print each vocabulary. What happened to `a`, `!` and the two forms of `win`? Which words did the stopword list remove? Then fit `stop_words="english"` on **D1–D5** and check: was `not` removed? Explain what that means for D5.

5. **Limiting the vocabulary:** fit vectorizers on D1–D5 with `min_df=2` and with `max_features=5`. Print both vocabularies and their sizes next to the full vocabulary size. Which kind of words survived?

6. **N-grams:** fit `ngram_range=(1, 1)` and `ngram_range=(1, 2)` on **D5** only and print both vocabularies. Which bigram keeps the reviewer's opinion? Then fit `(1, 1)`, `(1, 2)` and `(1, 3)` on D1–D5 and print the vocabulary size for each. How fast does it grow?

7. **TF-IDF:** fit a `TfidfVectorizer` on D1–D5 and print the TF-IDF matrix (rounded to 2 decimals). For **D1** and **D2**, print each word with its count and its TF-IDF weight. In D2 every word appears once: why do some words get a higher weight than others? In D1, why does `win` get the highest weight?

8. **TF-IDF by hand:** for **D2**, compute the textbook TF, IDF = `log(N / df)` and TF-IDF for `cash` and `today` (count the words in D2 the way the vectorizer does, and find each word's df from your matrix). Print every number. Do your results rank the two words the same way as scikit-learn did in task 7?

9. **Sparse matrix:** for the count matrix from task 2, print the total number of cells, the non-zero cells, the zero cells and the **percentage** of zeros.

10. **Fit and transform:** fit a new `CountVectorizer` on **D1–D4** (training text). Transform **N1** and **N2** and print their vectors. List every out-of-vocabulary word in N1 and N2. Why is `prize` unknown when D2 contains `prizes`? Which preprocessing step from Assignment 2 could fix this?

11. **Data leakage:** fit one vectorizer on **D1–D4 only** and another on **D1–D4 + N1 + N2**. Print both vocabularies and the words that **leaked** from the test set into the vocabulary.

12. **No meaning:** fit a `CountVectorizer` on **R1–R3** and print the vocabulary and the three vectors. Then compute the **cosine similarity** between every pair of reviews with `sklearn.metrics.pairwise.cosine_similarity` and print the matrix. Is R1 (`tasty`) more similar to R2 (`delicious`) than to R3 (`terrible`)? Explain why.

### Report

Add a **Practical** section to `answers.md` with:
- the document-term matrix from task 2 as a table (one row per document, one column per word),
- a table from task 7 comparing count and TF-IDF weight for every word in D2,
- your hand calculations from task 8,
- a short answer (2–4 sentences) for each question in tasks 1, 2, 4, 5, 6, 7, 8, 10, 11 and 12,
- one paragraph: **which vectorizer settings (counts or TF-IDF, stopwords, n-grams, vocabulary limits) would you use for a spam detector, and which would you avoid?** Use your own outputs as evidence.

## Submission

Submit everything in `submissions/<github-username>/assignment-03/` when submissions open:

- `answers.md` – Part A research answers and the Part B practical report
- `vectorization_practice.py` – your script
- `output.txt` – the full printed output of your script
