# Classical Classification — Project 1: SMS Spam Classifier

## Goal

Build your first complete NLP project: a spam filter that uses everything from Lessons 1–3 to clean raw SMS data, turn it into TF-IDF features, train classical classifiers and evaluate them honestly.

## What we are building

A model that reads one SMS message and predicts **`spam`** or **`ham`** (a normal message).

| | |
|---|---|
| Input | The text of one SMS message |
| Output | `spam` or `ham` |
| Data | `dataset/sms_spam_raw.csv`, the real [Kaggle SMS Spam Collection](../dataset/README.md) with a few problems added on purpose for cleaning practice |
| Models | TF-IDF + Naive Bayes and TF-IDF + Logistic Regression |

Along the way we clean the raw data, split it fairly, compare the two models, fix a class-imbalance problem, test whether preprocessing helps, look inside the model and predict new messages.

## Let's start

- [`code/sms-data-processing.py`](../code/sms-data-processing.py): cleaning the raw data.
- [`code/sms-classical-classification.py`](../code/sms-classical-classification.py): building, evaluating and using the spam classifier.