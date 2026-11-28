# Assignment 5 – Word Embeddings

## Part A – Research

### 1. Define a word embedding. Compare a dense embedding with a sparse TF-IDF document vector.

A **word embedding** is a learned numerical vector that represents a word. Instead of treating a word as only a text string, a model maps it to a fixed-length list of numbers. The values are learned from patterns in a training corpus, especially the contexts in which words appear. Words used in similar contexts may receive vectors with similar directions.

A **dense embedding** has a relatively small number of coordinates and most coordinates are non-zero. For example, a 50-dimensional Word2Vec vector has 50 learned values. A coordinate does not normally have a simple human label such as “royalty”; it contributes to a distributed representation learned by the model.

A **sparse TF-IDF document vector** has one coordinate for each term in the fitted vocabulary, and most coordinates are zero because a document contains only a small fraction of all vocabulary terms. For example, the coordinate for `prize` stores the TF-IDF weight of that term in the document. TF-IDF weights terms using their frequency in a document and how common they are across the document collection.

| Property | Dense word embedding | Sparse TF-IDF document vector |
|---|---|---|
| Represents | Usually one word | A complete document/message |
| Dimensions | Fixed, e.g. 50 | Vocabulary-dependent, possibly thousands |
| Values | Usually mostly non-zero | Mostly zero |
| Coordinate meaning | Learned distributed feature | Weight of a particular vocabulary term |
| Main strength | Captures contextual/semantic associations | Preserves explicit word importance |

### 2. Explain how Word2Vec learns from surrounding words. Show CBOW and skip-gram.

**Word2Vec** learns word vectors by training a model to predict words from their contexts or contexts from words. During training, the model adjusts its weights so that words appearing in related contexts develop useful vector representations.

Use this sentence: **“The student drinks hot coffee every morning.”** Suppose the target word is `coffee`.

- **CBOW (Continuous Bag of Words):** the surrounding words are the input, and the central word is the prediction target. Input: `student`, `drinks`, `hot`, `every`, `morning`. Target: `coffee`.
- **Skip-gram:** the central word is the input, and the surrounding words are prediction targets. Input: `coffee`. Targets: `student`, `drinks`, `hot`, `every`, `morning`.

| Architecture | Input | Prediction target |
|---|---|---|
| CBOW | Context words | Central word |
| Skip-gram | Central word | Context words |

CBOW predicts a word from its context; skip-gram predicts context words from a selected word. In this assignment, `sg=0` selects CBOW and `sg=1` selects skip-gram in Gensim.

### 3. Define cosine similarity and calculate an example.

**Cosine similarity** measures the angle-based similarity between two vectors. It compares their directions rather than simply comparing their magnitudes. It is often used to compare word embeddings.

\[
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
\]

Let \(A=[1,2]\) and \(B=[2,1]\).

1. Dot product: \(A\cdot B=(1\times2)+(2\times1)=4\).
2. Length of A: \(\|A\|=\sqrt{1^2+2^2}=\sqrt5\).
3. Length of B: \(\|B\|=\sqrt{2^2+1^2}=\sqrt5\).
4. Similarity: \(4/(\sqrt5\times\sqrt5)=4/5=0.8\).

The cosine similarity is **0.8**, indicating that the vectors point in similar directions. A value near 1 means similar directions, 0 means perpendicular directions, and -1 means opposite directions.

Words with different dictionary meanings can still become nearest neighbors because Word2Vec learns contextual relationships, not dictionary synonymy. For example, `doctor` and `hospital` may occur in similar contexts. They are related in use, although they are not synonyms.

### 4. Explain `vector_size`, `window`, `min_count`, and `epochs`.

| Parameter | What it controls | Assignment setting |
|---|---|---:|
| `vector_size` | Number of dimensions in each learned word vector | 50 |
| `window` | Maximum context distance around a target word | 3 |
| `min_count` | Minimum frequency needed for a word to enter the vocabulary | 2 |
| `epochs` | Number of training passes over the corpus | 20 |

A larger `vector_size` gives the model more dimensions in which to represent patterns, but it also increases the number of parameters and the amount of data needed. A small `window` focuses more on nearby context, while a larger window can capture broader relationships. `min_count=2` excludes words seen only once, because a single occurrence is usually insufficient evidence for a reliable vector. More epochs give the model more opportunities to learn, although excessive training can waste time or overfit the corpus.

Corpus characteristics matter. A larger corpus usually provides more varied contexts. Frequent words receive more training examples and tend to have better-estimated vectors; rare words may be excluded by `min_count` or have unreliable representations. Language also matters because grammar, morphology, spelling, and word order differ. An English model is not automatically useful for Somali. Domain matters too: vectors trained on SMS messages may capture promotional and mobile-phone vocabulary better than vectors trained on formal scientific writing.

### 5. Explain how to average word vectors into a message vector. What is lost?

Word2Vec gives a vector to each known word, while a classifier needs one fixed-size vector for each SMS. A simple approach is to average all known word vectors in the message coordinate by coordinate.

For example, suppose:

\[
hello=[1,2,3],\quad world=[3,4,5]
\]

Then:

\[
message=\frac{[1,2,3]+[3,4,5]}{2}=[2,3,4]
\]

This produces one three-dimensional message vector. In this assignment, each message vector has 50 dimensions. If a message has no tokens known to the fitted Word2Vec vocabulary, the script returns a 50-dimensional zero vector so every message has the same shape.

Averaging loses word order and some grammatical information. For example, `free prize call now` and `now call prize free` contain the same words, so their mean vectors are identical. Averaging also makes it difficult to represent the effect of negation or distinguish different sentence structures. It is a simple representation, not a complete representation of sentence meaning.

### 6. Why fit TF-IDF and Word2Vec on training text only? How can the comparison be fair?

The test set is meant to represent unseen data. If TF-IDF is fitted on all messages, its vocabulary and document-frequency statistics use information from the test messages. If Word2Vec is trained on all messages, the test corpus also influences its vocabulary and learned word relationships. This is a form of **data leakage** because the representation has already learned from the evaluation data, even if it has not used the test labels.

The correct procedure is to split the data first, fit Word2Vec and TF-IDF using training text only, and then use the fitted models to transform the test messages. Classifiers are fitted only on training representations and evaluated on the test representations.

A fair comparison changes the representation while holding the classifier and split fixed. For example, train Logistic Regression on mean CBOW vectors and train another Logistic Regression on TF-IDF vectors. Both use the same training rows, test rows, labels, `class_weight`, and `random_state`. Comparing the same metrics then helps isolate the effect of the representation.

### 7. Case study using word embeddings for text classification

Joulin, Grave, Bojanowski, and Mikolov (2017) introduced **fastText** in “Bag of Tricks for Efficient Text Classification.” The paper evaluates text classification on sentiment datasets and a large-scale tag-prediction task. For tag prediction, it uses the YFCC100M dataset, containing nearly 100 million images with titles, captions, and tags. The experiment predicts tags from the text (titles and captions), not from the images. After filtering infrequent words and tags, the training set contains 91,188,648 examples, with 930,497 validation examples and 543,424 test examples. There are 312,116 unique tags.

fastText learns word-feature representations and averages them into a text representation, then uses a linear classifier. The authors compare it with a frequency baseline and Tagspace. On YFCC100M, the frequency baseline reaches 2.2 precision-at-one. Tagspace with 50 hidden units reaches 30.1, while fastText with 50 hidden units reaches 31.2. Adding bigrams raises fastText precision-at-one to 36.7. With 200 hidden units and bigrams, it reaches 46.1. The paper also evaluates sentiment datasets and reports that adding bigram information improves accuracy by approximately 1–4% on those datasets.

One limitation is that the result is not evidence that pretrained Word2Vec alone caused the improvement. fastText learns its own word features as part of a supervised text-classification model and uses n-gram information; the paper also compares several architectural and training choices. The results therefore demonstrate the usefulness of learned word representations in an efficient classifier, but they do not isolate the effect of changing only Word2Vec versus TF-IDF.

---

## Part B – Practical

### Reproducibility and setup

The cleaned SMS file used here is `clean_sms_spam_practice.csv`, containing 5,140 messages with columns `label` and `message`. The labels are 4,498 ham and 642 spam. The script uses one stratified split with `test_size=0.2`, `random_state=7`, and reuses that split for all models. Word2Vec is fitted only on training tokens; TF-IDF is fitted only on training messages.

Install the packages:

```bash
pip install pandas numpy scikit-learn gensim nltk
```

Put the cleaned CSV in `dataset/clean_sms_spam_practice.csv`, then run:

```bash
python word_embeddings_practice.py > output.txt
```

The script uses NLTK `word_tokenize()` with `preserve_line=True`, so it applies NLTK's word tokenizer without requiring the separate Punkt sentence-boundary model. It lowercases tokens and removes punctuation-only tokens, while retaining numbers and alphanumeric tokens because they can be useful spam clues.

### Task 1 – Explore and split

The script loaded **5,140** cleaned messages: **4,498 ham** and **642 spam**. The training set contains **4,112** messages, including **514 spam** (12.5000%). The test set contains **1,028** messages, including **128 spam** (12.4514%). Stratification preserves the class proportions approximately in both subsets. This is important because spam is the minority class; an unstratified split could accidentally produce a test set with too few spam examples.

### Task 2 – Tokenize

The script prints one real training message before and after tokenization. In this run:

- Before: `K:)all the best:)congrats...`
- After: `['k', 'all', 'the', 'best', 'congrats']`

The tokenizer separates punctuation, lowercase normalization makes tokens consistent regardless of capitalization, and punctuation-only tokens are removed. Numeric and alphanumeric tokens are retained because phone numbers, prices, and codes can be useful spam signals. The exact example is data-dependent but is printed by the script.

### Task 3 – Train CBOW and skip-gram

Both models use the same training tokens and settings: `vector_size=50`, `window=3`, `min_count=2`, `epochs=20`, `workers=1`, and `seed=7`. The only architecture setting changed is `sg`: 0 for CBOW and 1 for skip-gram.

In this run, both models learned a vocabulary of **3,549 words**. The selected inspection word was `free`. The first five coordinates printed by the script were:

| Model | First five coordinates for `free` |
|---|---|
| CBOW | -0.30755, 0.72604, -0.36695, -1.32621, 0.42016 |
| Skip-gram | 0.00811, 0.09383, 0.38584, -0.33184, -0.24466 |

The coordinate values differ because CBOW and skip-gram learn from different prediction tasks. Vector coordinates do not have a simple independent human interpretation.

### Task 4 – Inspect the embeddings

The five nearest CBOW neighbors for `free` were `bluetooth`, `camcorder`, `tone`, `08000839402`, and `reply`. The five nearest skip-gram neighbors were `camcorder`, `100`, `unlimited`, `bluetooth`, and `1st`.

These associations make sense for this SMS corpus because `free` often appears in promotional messages about mobile phones, ringtones, services, and offers. These words are related by context in this dataset; they are not necessarily synonyms. The neighbors are corpus-specific and should not be treated as universal English word relationships.

The two selected cosine-similarity pairs were:

| Word pair | CBOW cosine | Skip-gram cosine |
|---|---:|---:|
| `call` – `free` | 0.738506 | 0.365337 |
| `win` – `prize` | 0.900341 | 0.673639 |

**Manual cosine calculation for `call` and `free` using CBOW:**

The script produced a dot product of 34.00595093, vector lengths of 6.76599503 and 6.80564308, and:

\[
\frac{34.00595093}{6.76599503\times6.80564308}
=0.73850608
\]

Gensim's similarity result was 0.73850602. The values match within floating-point precision (`np.isclose` returned `True`).

### Task 5 – Build mean CBOW message vectors

The training matrix shape is **(4112, 50)** and the test matrix shape is **(1028, 50)**. Five test messages required zero vectors because all their tokens were unknown to the CBOW vocabulary. Each row represents one SMS, and each of the 50 columns represents one averaged embedding coordinate.

### Task 6 – Train three classifiers

The same split was reused for all three models:

1. Mean CBOW vectors + Random Forest (`n_estimators=200`, `class_weight="balanced"`, `random_state=7`).
2. Mean CBOW vectors + Logistic Regression (`max_iter=1000`, `class_weight="balanced"`, `random_state=7`).
3. TF-IDF (`stop_words="english"`, `ngram_range=(1, 2)`) + Logistic Regression (`max_iter=1000`, `class_weight="balanced"`, `random_state=7`).

The TF-IDF vocabulary contained **30,945 features**. TF-IDF was fitted on training messages only and then used to transform test messages.

### Task 7 – Evaluate and compare

The following results were printed by the script on the fixed test set:

| Model | Accuracy | Spam precision | Spam recall | Spam F1 | False alarms |
|---|---:|---:|---:|---:|---:|
| Mean CBOW + Random Forest | 96.89% | 90.68% | 83.59% | 86.99% | 11 |
| Mean CBOW + Logistic Regression | 93.68% | 68.42% | 91.41% | 78.26% | 54 |
| TF-IDF + Logistic Regression | 97.86% | 91.41% | 91.41% | 91.41% | 11 |

**Confusion matrix – Model 2: Mean CBOW + Logistic Regression**

Rows are actual labels; columns are predicted labels. Label order is ham, spam.

| | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 846 | 54 |
| Actual spam | 11 | 117 |

TN = 846, FP = 54, FN = 11, TP = 117.

**Confusion matrix – Model 3: TF-IDF + Logistic Regression**

| | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 889 | 11 |
| Actual spam | 11 | 117 |

TN = 889, FP = 11, FN = 11, TP = 117.

**What changed when Logistic Regression kept the same settings but the representation changed?** Moving from mean CBOW vectors to TF-IDF increased accuracy from 93.68% to 97.86%, precision from 68.42% to 91.41%, and F1 from 78.26% to 91.41%. Recall stayed at 91.41%. False alarms fell from 54 to 11. On this dataset and split, TF-IDF gave Logistic Regression a much more reliable separation between spam and ham. One likely reason is that TF-IDF preserves explicit terms and bigrams, whereas averaging embeddings loses word order and can blur distinctive SMS phrases.

### Task 8 – Analyze Model 1 mistakes

Model 1 missed spam examples involving adult chat/premium-rate services, a ringtone password, a paid chat service, a flower-order promotion, and paid text content. These examples suggest that spam is not always marked by obvious words such as `free` or `win`; some messages resemble ordinary notifications or conversational messages.

The false alarms included a personal message asking whether a phone still belonged to someone, a smiley, a short conversational message, a message about a video game, and a short technical-looking string. Some are extremely short or ambiguous, so the averaged vector may not contain enough context to distinguish them reliably. The word `phone` or promotional-looking vocabulary may also appear in legitimate contexts.

### Task 9 – Inspect the effect of averaging

The two messages were:

- `free prize call now`
- `now call prize free`

The printed 50-dimensional mean vectors were identical, and `np.allclose()` returned **True**. This demonstrates that mean word vectors ignore word order when the messages contain the same known words with the same multiplicities. Therefore, a classifier using only these mean vectors cannot distinguish the two messages based on their word order.

### Task 10 – Predict new messages

The script predicted:

| New message | Model 1 prediction |
|---|---|
| Congratulations you have won a free cash prize call now | Spam |
| URGENT claim your reward by texting WIN to 80000 | Spam |
| You have been selected for a free holiday offer reply YES | Spam |
| Hey are we still meeting for lunch today | Ham |
| Thanks for your help I will call you when I arrive | Ham |

All five predictions matched their intended categories. The third message was interesting because it uses an offer and a reply request without an explicit phone number or price, but the model still classified it as spam. The results are examples of model behavior, not proof that every similar real-world message will be classified correctly.

### Final recommendation for a phone spam filter

I would choose **TF-IDF + Logistic Regression** for this experiment. It achieved the highest accuracy (97.86%) and spam F1 (91.41%), with 91.41% precision and 91.41% recall. It also produced only 11 false alarms and missed 11 spam messages. Mean CBOW + Logistic Regression caught the same number of spam messages but produced 54 false alarms, while Mean CBOW + Random Forest had 83.59% recall. On a phone, both missed spam and false alarms matter: missed spam reaches the user, but a false alarm can hide a legitimate message such as a delivery update, bank alert, or personal text. TF-IDF + Logistic Regression provides the best balance on this test split. In a real phone, suspected spam should be moved to a recoverable spam folder rather than permanently deleted, and the decision threshold should be selected using validation data and the actual cost of each type of mistake.

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. International Conference on Learning Representations (ICLR) Workshop. https://arxiv.org/abs/1301.3781
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. Advances in Neural Information Processing Systems 26. https://arxiv.org/abs/1310.4546
3. Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, 427–431. https://aclanthology.org/E17-2068/
4. Gensim documentation. *Word2Vec*. https://radimrehurek.com/gensim/models/word2vec.html
5. NLTK documentation. *NLTK Tokenize*. https://www.nltk.org/api/nltk.tokenize.html
6. scikit-learn documentation. *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
7. scikit-learn documentation. *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
8. scikit-learn documentation. *RandomForestClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
