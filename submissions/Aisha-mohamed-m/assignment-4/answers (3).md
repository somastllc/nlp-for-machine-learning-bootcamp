# Assignment 4 – Classical Classification

## Part A – Research

### 1. Supervised Text Classification

Supervised text classification is a machine learning task in which a model learns from text examples that already have known labels. The model learns patterns connecting the input text to its class and then uses those patterns to classify new, unseen text.

There are three common forms:

- **Binary classification:** two possible classes. Example: SMS messages classified as `spam` or `ham`.
- **Multiclass classification:** one of more than two classes. Example: news articles classified as sports, politics, business, or technology.
- **Multilabel classification:** one text can receive multiple labels at the same time. Example: a document can be labeled as both `machine learning` and `NLP`.

This assignment uses binary supervised text classification because each SMS message is labeled as either spam or ham.

### 2. Train/Test Split and Stratification

A train/test split separates the dataset into two parts. The training set is used to learn the model, while the test set is used only to evaluate how well the trained model generalizes to unseen messages.

This assignment uses an 80/20 split with `random_state=7` and `stratify=y`. Stratification keeps approximately the same class proportions in the training and test sets. This is particularly useful when the classes are not equally represented.

Duplicate messages should be removed before splitting. If duplicates are allowed across both sets, the model may see the same or nearly identical message during training and testing, which can produce overly optimistic evaluation results.

The test set should not influence model selection, preprocessing choices, or hyperparameter decisions. It should remain an independent evaluation set.

### 3. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier that is especially suitable for discrete word-count features. It estimates how likely a document belongs to each class based on the words appearing in the document.

The naive assumption is that features are conditionally independent given the class. In natural language this assumption is not literally true because words can depend on one another, but the method can still work well in text classification.

It works well for spam detection because spam messages often contain characteristic words and phrases such as `free`, `claim`, `prize`, `urgent`, and promotional terms. Count-based representations provide useful evidence for the classifier.

### 4. Logistic Regression for Text Classification

Logistic Regression is a linear classification model that estimates the probability that a message belongs to a class. In text classification, the message is first converted into numerical features using methods such as CountVectorizer or TF-IDF.

Each feature has a learned coefficient (weight). A positive coefficient pushes the prediction toward spam, while a negative coefficient pushes it toward ham. Inspecting these weights helps explain which words or phrases are strongly associated with each class.

In this practical, Model 4 uses TF-IDF features and balanced Logistic Regression. Its learned feature weights include strongly positive features such as `call`, `txt`, `free`, `claim`, and `mobile`, while features such as `my`, `me`, `it`, `ok`, and `come` have negative weights and therefore push predictions toward ham.

### 5. Precision, Recall, F1, and Accuracy

Classification performance can be described using a confusion matrix:

- **True Positive (TP):** spam correctly classified as spam.
- **False Positive (FP):** ham incorrectly classified as spam.
- **False Negative (FN):** spam incorrectly classified as ham.
- **True Negative (TN):** ham correctly classified as ham.

Precision measures how many messages predicted as spam were actually spam:

`Precision = TP / (TP + FP)`

Recall measures how many actual spam messages were detected:

`Recall = TP / (TP + FN)`

F1 combines precision and recall:

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

Accuracy measures the proportion of all predictions that are correct:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Accuracy can be misleading when classes are imbalanced. For example, if ham messages greatly outnumber spam messages, a model could achieve high accuracy by predicting most messages as ham while missing many spam messages. For spam detection, precision and recall are therefore important.

### 6. Class Imbalance

Class imbalance occurs when one class has substantially more examples than another. Several approaches can be used:

1. **Class weights:** give more importance to the minority class during training. This is the approach used by Model 4 through `class_weight="balanced"`.
2. **Resampling:** oversample the minority class or undersample the majority class.
3. **Threshold adjustment:** change the probability threshold used to convert predicted probabilities into class labels.

These approaches involve trade-offs. Increasing attention to spam can improve recall and catch more spam, but it can also increase false alarms where legitimate messages are incorrectly classified as spam.

### 7. Case Study: SMS Spam Classification

SMS spam classification is a practical example of supervised text classification. Messages are transformed into numerical representations using CountVectorizer or TF-IDF, and classifiers such as Multinomial Naive Bayes and Logistic Regression are trained on labeled examples.

In this practical, five pipelines were compared. Model 4, TF-IDF with balanced Logistic Regression, achieved the strongest overall balance in the current run, with accuracy of 0.977, spam precision of 0.924, spam recall of 0.886, and spam F1 of 0.905. The balanced model substantially improved spam recall compared with the unbalanced Logistic Regression model.

---

# Part B – Practical

## 1. Dataset Exploration

The preprocessing notebook loads `sms_spam_raw.csv` and performs the required cleaning steps:

- removes rows with missing text;
- normalizes labels to lowercase and removes trailing periods;
- decodes HTML entities;
- normalizes extra whitespace;
- removes duplicate messages;
- saves the cleaned dataset.

The classification notebook then loads the cleaned dataset and checks the number of rows, columns, missing text, duplicate text, median words per message, label counts, spam share, and example messages.

Removing missing text and duplicates before model training helps prevent invalid or repeated examples from affecting the evaluation.

## 2. Train/Test Split

The assignment requires:

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=7,
    stratify=df["label"]
)
```

The `stratify` argument keeps the spam/ham class distribution approximately consistent between the training and test sets. Therefore, the model receives a representative training set and is evaluated on a representative test set.

The current model run used the required 80/20 split with `random_state=7`.

## 3. Model Comparison

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Model 1 - CountVectorizer + NB | 0.977 | 0.970 | 0.845 | 0.903 |
| Model 2 - TF-IDF + NB | 0.930 | 1.000 | 0.440 | 0.612 |
| Model 3 - TF-IDF + Logistic Regression | 0.946 | 0.991 | 0.570 | 0.724 |
| Model 4 - TF-IDF + Balanced Logistic Regression | 0.977 | 0.924 | 0.886 | 0.905 |
| Model 5 - TF-IDF + Stopwords + Balanced Logistic Regression | 0.975 | 0.923 | 0.870 | 0.896 |

Model 4 achieved the highest spam recall (0.886) and the highest F1 score (0.905). Model 2 achieved perfect spam precision (1.000), but its recall was only 0.440, meaning it missed many spam messages.

For the Naive Bayes comparison, CountVectorizer performed much better than TF-IDF in this run. Model 1 achieved F1 = 0.903, while Model 2 achieved F1 = 0.612. This shows that the representation of the text can strongly affect Naive Bayes performance.

## 4. Confusion Matrices

### Model 3 – TF-IDF + Logistic Regression

Confusion matrix:

```text
[[1348    1]
 [  83  110]]
```

Therefore:

- TN = 1348
- FP = 1
- FN = 83
- TP = 110

The model generated very few false alarms, but it missed 83 spam messages.

### Model 4 – TF-IDF + Balanced Logistic Regression

Confusion matrix:

```text
[[1335   14]
 [  22  171]]
```

Therefore:

- TN = 1335
- FP = 14
- FN = 22
- TP = 171

Model 4 catches substantially more spam than Model 3, although it creates more false alarms.

## 5. Hand Calculation for Model 4

Using:

- TP = 171
- FP = 14
- FN = 22

Precision:

`171 / (171 + 14) = 0.924`

Recall:

`171 / (171 + 22) = 0.886`

F1:

`2 × (0.924 × 0.886) / (0.924 + 0.886) = 0.905`

The hand-calculated values match the values reported by scikit-learn after rounding.

## 6. Model 3 vs Model 4

Model 4 caught:

`171 - 110 = 61`

additional spam messages compared with Model 3.

It also produced:

`14 - 1 = 13`

additional false alarms.

Therefore, balancing the Logistic Regression model increased spam recall substantially. The trade-off was 13 additional false alarms.

## 7. Feature Weights of Model 4

### Top positive features

| Feature | Weight |
|---|---:|
| call | 5.649560 |
| txt | 4.468106 |
| free | 4.378961 |
| to | 3.972732 |
| text | 3.963546 |
| stop | 3.503293 |
| your | 3.264882 |
| claim | 3.145137 |
| mobile | 3.052044 |
| or | 3.012214 |

Positive coefficients push the prediction toward spam. Words such as `free`, `claim`, `call`, and `mobile` are therefore strong indicators of spam in this model.

### Top negative features

| Feature | Weight |
|---|---:|
| my | -2.438815 |
| me | -2.294871 |
| it | -2.140741 |
| that | -1.726157 |
| ok | -1.708764 |
| but | -1.523638 |
| come | -1.481953 |
| how | -1.385083 |
| can | -1.377699 |
| ll | -1.312271 |

Negative coefficients push predictions toward ham. These features are more associated with ordinary personal conversations in the learned model.

## 8. Error Analysis

The balanced Logistic Regression model produced 25 missed spam messages and 14 false alarms.

Examples of missed spam included promotional Christmas offers, financial promotions, unusual messages, and messages containing phone numbers. These messages may be difficult to classify because they do not always use the most common spam vocabulary.

Examples of false alarms were short personal messages such as:

- `Send me the new number`
- `On the road so cant txt`
- `Text her. If she doesnt reply let me know so i can have her log in`
- `Now am free call me pa.`
- `Then get some cash together and I'll text jason`

These messages contain words such as `number`, `txt`, `call`, and `free`, which can also appear in spam messages. The model therefore sometimes treats normal personal communication as spam.

## 9. Does Preprocessing Help?

The preprocessing comparison produced:

| Preprocessing | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Raw text | 0.975 | 0.923 | 0.870 | 0.896 |
| Letters only | 0.971 | 0.898 | 0.870 | 0.884 |
| Markers | 0.981 | 0.941 | 0.907 | 0.923 |

Marker-based preprocessing performed best on all four reported metrics. Replacing phone numbers, money amounts, and URLs with markers such as `phonenumber`, `money`, and `url` preserves useful spam-related information while making the text easier for the classifier to learn from.

The letters-only version did not improve the model and produced a lower F1 score. Therefore, the marker-based preprocessing is the strongest of the three approaches in this experiment.

## 10. Error Analysis Examples

The balanced Logistic Regression model had:

- **25 missed spam messages**
- **14 false alarms**

Five missed-spam examples were:

1. `INTERFLORA - åÒIt's not too late to order Interflora flowers for christmas call 0800 505060 to place your order before Midnight tomorrow.`
2. `Did you hear about the new \Divorce Barbie\"? It comes with all of Ken's stuff!`
3. `How come it takes so little time for a child who is afraid of the dark to become a teenager who wants to stay out all night?`
4. `Santa Calling! Would your little ones like a call from Santa Xmas eve? Call 09058094583 to book your time.`
5. `Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per..`

Five false-alarm examples were:

1. `Send me the new number`
2. `On the road so cant txt`
3. `Text her. If she doesnt reply let me know so i can have her log in`
4. `Now am free call me pa.`
5. `Then get some cash together and I'll text jason`

These examples show that the model can struggle with unusual promotional spam and with short personal messages that contain spam-associated words.

## 11. Single-Message Check and New Message Predictions

The selected test message was:

`Bought one ringtone and now getting texts costing 3 pound offering more tones etc`

Its actual label was **spam**.

| Model | Prediction |
|---|---|
| Model 3 - Logistic Regression | ham |
| Model 1 - Naive Bayes | spam |
| Model 4 - Balanced Logistic Regression | ham |

Model 1 correctly classified the selected message, while Models 3 and 4 missed it.

For the five new messages, Model 4 produced:

| New message | Prediction | Spam probability |
|---|---|---:|
| URGENT! You have won a free reward, claim now | spam | 0.96 |
| Are we still meeting for lunch tomorrow? | ham | 0.12 |
| Call this number to get your prize today | spam | 0.79 |
| Can you send me the notes from class? | ham | 0.16 |
| Congratulations! You have won a cash prize. Call now to claim. | spam | 0.98 |

The model correctly identified the clearly promotional examples as spam and the normal personal messages as ham in this small manual test.

## 12. Final Model Choice

The recommended model is **Model 4 – TF-IDF + Balanced Logistic Regression**.

The main reason is its strong balance between precision and recall. It achieved **0.977 accuracy, 0.924 spam precision, 0.886 spam recall, and 0.905 spam F1**. Compared with Model 3, it caught **61 additional spam messages**, although it generated **13 additional false alarms**.

For an SMS spam filter, missing spam can be more harmful than sending a legitimate message for manual review. Therefore, the higher recall of Model 4 is valuable. However, false alarms should still be monitored because incorrectly classifying legitimate messages as spam can inconvenience users.
