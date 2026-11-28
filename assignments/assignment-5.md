# Assignment 5 – Word Embeddings

**Goal.** Understand how Word2Vec learns word vectors and evaluate whether averaged word vectors help classify SMS spam.

## Part A – Research

### Instructions

- Research the topics using books, scholarly articles, and reliable sources. Write your own explanations and examples.
- You may use AI tools. Verify their claims against your sources and submit explanations you can defend in your own words.
- Write in **English**. Use headings, paragraphs, and tables or figures where appropriate. Include **references** at the end.

1. Define a **word embedding**. Compare a dense word embedding with a sparse TF-IDF document vector. Give one example of what a coordinate represents in each.

2. Explain how **Word2Vec** learns from surrounding words. Use a sentence of your own to show the input and prediction target for **CBOW** and **skip-gram**.

3. Define **cosine similarity**. Show a calculation with two short vectors and explain how words with different meanings can become nearest neighbors.

4. Explain what `vector_size`, `window`, `min_count`, and `epochs` control. Discuss how corpus size, word frequency, and language affect which words get useful vectors.

5. Explain how to average several **word vectors** into one **message vector**. Use your own example to show what averaging loses. Describe how you would handle a message with no known words.

6. Explain why Word2Vec and TF-IDF should both be fitted on the training text. How can you design a comparison that measures the effect of changing the text representation while keeping the classifier fixed?

7. Find one **case study** that uses word embeddings for text classification. Summarize the dataset, the embedding method, the baseline, and the results. Explain one limitation of the comparison.

## Part B – Practical

Write your own Python script, `word_embeddings_practice.py`, to build and evaluate spam classifiers. Follow the format of the class script by printing a numbered heading and the results for each step. [Lesson 5 project script](../code/sms-embedding-classification.py)

### Setup

- Install the course requirements.
- Run the cleaning script to generate `dataset/clean_sms_spam.csv`. [SMS data-processing script](../code/sms-data-processing.py)
- Make one stratified train/test split with `test_size=0.2` and `random_state=7`, then reuse it throughout. Fit each representation and classifier on training data.
- Save the full printed output with `python word_embeddings_practice.py > output.txt`. Generate every result in your report from your own script.

### Tasks

1. **Explore and split.** Print the number of cleaned messages, label counts, and the size and spam share of each split. Explain what stratification preserves.

2. **Tokenize.** Use NLTK's `word_tokenize()` to tokenize the messages. Show one message before and after your preprocessing, and explain what your rules do to punctuation and case.

3. **Train CBOW and skip-gram.** Fit both on the same training tokens with `vector_size=50`, `window=3`, `min_count=2`, and `epochs=20`. Use `workers=1` and `seed=7` in both runs. Change `sg` between the two models. Print each vocabulary size and the first five coordinates for a word you choose.

4. **Inspect the embeddings.** Choose a word both models know and print its five nearest neighbors from each model. Choose two pairs of known words and print their cosine similarities under both models. For one pair, calculate cosine similarity using the dot product and vector lengths and check it against the model's result. Explain one association you found.

5. **Build message vectors.** Use CBOW to average the known word vectors for each SMS. Print the shape of both matrices and count test messages that need a zero vector because all their tokens are unknown.

6. **Train three models.** Use the same training and test messages for all three. Set `class_weight="balanced"` and `random_state=7` for each classifier.
   1. Mean CBOW vectors + Random Forest with 200 trees
   2. Mean CBOW vectors + Logistic Regression with `max_iter=1000`
   3. TF-IDF + Logistic Regression with `max_iter=1000`. Use `stop_words="english"` and `ngram_range=(1, 2)` for TF-IDF.

7. **Evaluate and compare.** Print a table with accuracy, spam precision, spam recall, spam F1, and false alarms for each model. Print labeled confusion matrices for models 2 and 3. What changed when you switched representations while keeping Logistic Regression fixed?

8. **Analyze mistakes.** Print up to five missed spam messages and five false alarms from model 1. Describe a pattern you see in each group or explain what the examples suggest if there are too few errors to see a pattern.

9. **Inspect the effect of averaging.** Write two short messages with the same known words in different orders. Print both mean vectors and use `np.allclose()` to check whether they match. Explain what this shows about the information available to the classifier.

10. **Predict new messages.** Write five of your own messages, including at least two spam and two ham. Print model 1's prediction for each using the same tokenizer and fitted word vectors. Which prediction surprised you most, and why?

### Report

Include the following in the **Practical** section of `answers.md`.

- the cosine calculation from task 4 and the comparison table and confusion matrices from task 7
- a short answer for each task that asks you to explain or interpret a result
- one paragraph choosing a model for a phone's spam filter. Use your scores and examples to explain which mistakes matter most for that use.

## Submission

Submit these files in `submissions/<github-username>/assignment-05/` when submissions open.

- `answers.md` – Part A research answers and the Part B practical report
- `word_embeddings_practice.py` – your script
- `output.txt` – the full printed output of your script
