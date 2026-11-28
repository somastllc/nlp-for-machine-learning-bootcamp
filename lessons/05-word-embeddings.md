# Word Embeddings

## Goal

Learn how Word2Vec turns words into dense vectors and use those vectors to classify SMS spam.

## 1. From sparse TF-IDF to dense word vectors

In Lesson 3, `good` and `great` occupied separate TF-IDF columns, even though they expressed a similar opinion. A **word embedding** gives each word in its vocabulary a vector of learned numbers. Words used in similar contexts can have similar vectors.

Each number is a **coordinate**. A vector with 50 coordinates has 50 dimensions. Relationships between words come from the full vector. A single coordinate generally has no standalone label such as “positive.”

**Example from the SMS project**

| Representation | Example | Dimensions | Nonzero values |
|---|---|---:|---:|
| Sparse TF-IDF document row | `free prize` | 27,438 | 2 |
| Dense Word2Vec word vector | `free` | 50 | 50 |

The first five coordinates of `free` were `[-1.600, -1.200, -1.797, 0.559, 0.079]`. The TF-IDF row has one position per vocabulary term. The Word2Vec vector has 50 learned positions for each word, regardless of vocabulary size.

## 2. How Word2Vec learns from context

Word2Vec learns by predicting words from the text it sees. A **context window** sets how far to look on each side of a target word. If `prize` and `reward` often appear near `claim`, `win`, and `free`, their vectors can become similar. This is the **distributional idea**.

Word prediction is **self-supervised** because the words in the corpus supply the prediction targets. The project learns from the 3,598 training messages. The spam and ham labels are used to stratify the split. The embedding model receives only tokens.

**Example.** With a window of two, the target `free` in `claim your free prize now` has the context `claim`, `your`, `prize`, and `now`. The script lowercases the text and uses Lesson 2's NLTK tokenizer to separate punctuation from words.

| Step | Output for `Claim your FREE prize now!` |
|---|---|
| Lowercase and split on spaces | `['claim', 'your', 'free', 'prize', 'now!']` |
| Lowercase and use `word_tokenize` | `['claim', 'your', 'free', 'prize', 'now', '!']` |
| Keep tokens that start with a word character | `['claim', 'your', 'free', 'prize', 'now']` |

## 3. CBOW and skip-gram

The two Word2Vec methods use the same kind of examples with the prediction direction reversed.

| Method | Input for the example above | Prediction target | Gensim setting |
|---|---|---|---|
| Continuous bag of words (CBOW) | `claim`, `your`, `prize`, `now` | `free` | `sg=0` |
| Skip-gram | `free` | Each nearby word, such as `prize` | `sg=1` |

The project uses CBOW. To see how the training objective changes the word vectors, set `sg=1` in the Word2Vec call and rerun the project. On the same split, the cosine similarity for `good` and `great` was **0.94** with CBOW and **0.77** with skip-gram. Step 3 also shows how the nearest words change.

## 4. Cosine similarity

**Cosine similarity** is the cosine of the angle between two nonzero vectors. A score of 1 means they point in the same direction, 0 means they form a right angle, and −1 means they point in opposite directions.

```text
cosine(a, b) = dot product(a, b) / (length(a) × length(b))
```

**Worked example.** For `a = (1, 2)` and `b = (2, 4)`, the dot product is `1×2 + 2×4 = 10`. Their lengths are √5 and √20, whose product is 10. The cosine similarity is **1**. For `c = (2, −1)`, the dot product with `a` is `1×2 + 2×(−1) = 0`, so the cosine similarity is **0**.

**Example from the SMS project**

| Word pair | CBOW cosine similarity |
|---|---:|
| `good` and `great` | 0.94 |
| `good` and `free` | 0.04 |
| `free` and `win` | 0.91 |
| `home` and `house` | 0.68 |

Try the Lesson 3 example again with the individual terms. When we transform `good` and `great` as one-word inputs, their TF-IDF vectors occupy separate columns and have cosine similarity **0.00**. The two full movie reviews also share other words. The **0.94** from Word2Vec shows what the learned representation adds for the two opinion words themselves.

## 5. Nearest words

Gensim's `most_similar` method ranks other words by cosine similarity to a chosen word. The results show which words the model learned to associate in its training corpus.

In the CBOW run, neighbors of `good` included `morning`, `great`, and `day`. A phrase such as `good morning` illustrates how neighbors can be related through common expressions. For `free`, the neighbors included `tone`, `rental`, and `nokia`. These associations reflect the SMS corpus and its promotional messages.

Word2Vec learns a single vector for each word. Uses of `free` in `free prize` and `are you free tonight?` contribute to the same vector.

## 6. Settings and vocabulary coverage

| Setting | Project value | What it controls |
|---|---:|---|
| `vector_size` | 50 | Number of coordinates per word |
| `window` | 3 | Maximum number of positions examined on either side |
| `min_count` | 2 | Minimum number of training occurrences for a word to get a vector |
| `epochs` | 20 | Number of passes over the training text |

The project learned vectors for **3,030** words. A word that is absent or too rare in the training text is **out of vocabulary** and has no vector. In `the delivery was superb`, the script marks `superb` as unknown.

A small SMS corpus gives rare words few contexts to learn from, and its best-known words reflect SMS usage. A larger corpus can give a model more examples of how a word is used. Check coverage and inspect neighbors when moving to a new kind of text.

The example values came from one run on the cleaned SMS data. Coordinates, neighbors, and scores can vary with the training setup.

## Lesson recap

- TF-IDF gives each term a column. Word2Vec gives each known word a dense vector.
- CBOW predicts a word from context. Skip-gram predicts nearby words from a word.
- Cosine similarity compares vector direction. Nearest words reflect how the training corpus uses them.
- The training corpus and `min_count` determine which words have vectors.

## Project 2 — SMS Spam Classification with Word Embeddings

**Problem.** An SMS spam filter needs to catch junk messages while keeping legitimate messages in the inbox.

**Goal.** Classify incoming messages and compare spam precision, recall, F1, and false alarms with Project 1.

**What we are building.** A Random Forest that takes one SMS message, averages its known Word2Vec vectors into one message vector, and predicts `spam` or `ham`. We reuse Lesson 4's cleaned data and stratified split.

## Let's start

- [`code/sms-data-processing.py`](../code/sms-data-processing.py): cleaning the raw data.
- [`code/sms-embedding-classification.py`](../code/sms-embedding-classification.py): training Word2Vec and building, evaluating and using the spam classifier.
