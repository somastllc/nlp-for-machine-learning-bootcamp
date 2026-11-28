# Assignment 4 – Classical Classification

**Goal:** Understand how a supervised text classifier is built and evaluated: train/test splits, pipelines, Naive Bayes, Logistic Regression, class imbalance, precision, recall, F1, the confusion matrix and error analysis.

## Part A – Research

### Instructions
- **Do not use** the explanations or examples we provided earlier (e.g., classroom notes).
- You must conduct **deep research** using books, scholarly articles, and reliable sources.
- **You may use AI tools**, but **do not rely blindly**: do not submit answers you don’t fully understand or verify.
- The purpose is for you to **learn through research** — therefore, explain ideas clearly, restate key concepts in your own words, and reference the sources you used.
- Write in **English**, in a research-oriented style (with headings, paragraphs, and tables/figures where appropriate), and include **references** at the end.

1. Define **supervised text classification**. Explain binary, multiclass and multilabel classification with one real-life example of each.

2. Explain the **train/test split** and **stratification**. Why must duplicates be removed before splitting, and why should the test set never influence model choices?

3. Explain how **Multinomial Naive Bayes** classifies text. What is the “naive” assumption, and why does it still work well for text?

4. Explain how **Logistic Regression** classifies text. What do its feature weights mean, and how can they be used to inspect what the model learned?

5. Define **precision**, **recall**, **F1** and **accuracy** using the confusion matrix. Why can accuracy be misleading on an **imbalanced** dataset?

6. Research methods for handling **class imbalance** (for example, class weights, resampling or changing the decision threshold). Explain the trade-off each one makes.

7. Find one **case study** (research paper or article) on spam detection or another text classification task. Summarize the data, the features, the models compared and the results.

## Part B – Practical

Write two Python scripts. Use `data_processing_practice.py` to explore and clean the SMS dataset, then use `classification_practice.py` to load the cleaned data and build and evaluate the classifiers. Both scripts should print a heading and the results for each step.

### Setup

- Install the course requirements.
- Use `dataset/sms_spam.csv` as the input to the processing script. Save the processed messages to `dataset/clean_sms_spam_practice.csv` and load that file in the classifier script.
- Use **your own split**: `test_size=0.2`, `random_state=7`, `stratify` on the label. Your numbers will therefore differ from the class results.
- Save the full printed output of both scripts in `output.txt`. Generate every result in your report from your own scripts.

```bash
python data_processing_practice.py > output.txt
python classification_practice.py >> output.txt
```

### Tasks

1. **Explore.** In the processing script, print the number of rows, missing text, duplicate text and the label counts. What share of messages is spam?

2. **Clean and split.** Remove rows with missing text and duplicate messages in the processing script, then save the cleaned CSV. In the classifier script, load the cleaned CSV and split it with the settings above. Print the size and spam share of the training and test sets. Explain why the spam shares are close.

3. **Five pipelines:** train these five models on the same split and print a classification report for each:
   1. `CountVectorizer` + `MultinomialNB`
   2. `TfidfVectorizer` + `MultinomialNB`
   3. `TfidfVectorizer` + `LogisticRegression(max_iter=1000)`
   4. `TfidfVectorizer` + `LogisticRegression(max_iter=1000, class_weight="balanced")`
   5. `TfidfVectorizer(stop_words="english")` + `LogisticRegression(max_iter=1000, class_weight="balanced")`

4. **Comparison table:** make a table with each model's accuracy, spam precision, spam recall and spam F1. Which model has the best spam recall? Did counts or TF-IDF work better with Naive Bayes on this split?

5. **Confusion matrices:** print the confusion matrix of models 3 and 4. For each, write down TP, FP, FN and TN.

6. **Metrics by hand:** using the confusion matrix of model 4, calculate precision, recall and F1 yourself and print the calculation. Check that it matches the classification report.

7. **Class imbalance:** compare models 3 and 4. How many more spam messages did `class_weight="balanced"` catch, and how many extra false alarms did it cause?

8. **What the model learned:** print the 10 words with the largest positive weights and the 10 with the most negative weights in model 4. Do they make sense? Is any of them surprising?

9. **Error analysis:** print 5 missed spam messages and 5 false alarms from model 4. For each group, explain one pattern you notice.

10. **New messages:** write **five** new messages of your own (at least two spam and two ham). Print model 4's prediction and spam probability for each. Was any prediction wrong or uncertain?

### Report

Add a **Practical** section to `answers.md` with:
- the comparison table from task 4 and your hand calculation from task 6,
- a short answer (2–4 sentences) for each question in tasks 1–10,
- one paragraph: **which model would you choose for a phone's spam filter, and why?** Use your own outputs as evidence, and say which mistake (false alarm or missed spam) you think is worse for this use.

## Submission

Submit everything in `submissions/<github-username>/assignment-04/` when submissions open:

- `answers.md` – Part A research answers and the Part B practical report
- `data_processing_practice.py` – your data exploration and cleaning script
- `classification_practice.py` – your model training and evaluation script
- `output.txt` – the full printed output of both scripts
