# Assignment 4 – Classical Classification

## Part A: Research Questions

### 1. What is supervised text classification?

Supervised text classification is a machine learning task where a model learns from text examples that already have labels. It studies the relationship between the words in a message and its correct category, then uses what it learned to classify new messages.

There are three common types:

- **Binary classification:** Assigns one of two labels, such as spam or ham.
- **Multiclass classification:** Assigns one label from more than two possible classes, such as classifying a news article as sports, business, politics, or technology.
- **Multilabel classification:** Assigns multiple labels to one text when it belongs to more than one category, such as an article about both technology and education.

### 2. What is a train/test split, and why is stratification important?

A train/test split separates a dataset into two parts. The training data teaches the model, while the testing data evaluates how well it predicts examples it did not train on.

Stratification preserves approximately the same class proportions in both sets. This is important when a dataset is imbalanced because the smaller class should still be represented in the test set.

Duplicates should be removed before splitting because identical messages in both sets can make evaluation misleading. The model may appear to perform well because it has already encountered the same messages during training.

The test set should not influence model selection or parameter tuning. Otherwise, information from the test set can leak into the development process, making the final performance estimate less reliable.

### 3. Explain Multinomial Naive Bayes.

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification. It calculates the probability that a message belongs to a class based on the words it contains.

It is called "naive" because it assumes that features (words) are conditionally independent given the class. In real language, words are not completely independent, but this assumption makes the algorithm simple and efficient.

It works well for text because messages can be represented as word-count vectors. It can also work with TF-IDF features. Laplace smoothing helps prevent zero probabilities for words that were not observed in a particular class during training.

### 4. Explain Logistic Regression for text classification.

Logistic Regression is a supervised classification algorithm that estimates the probability that an example belongs to a class.

For text classification, the message is converted into numerical features using methods such as TF-IDF. Logistic Regression learns a coefficient (weight) for each feature.

A positive coefficient in a binary spam-versus-ham model can push the prediction toward spam, while a negative coefficient can push it toward ham, depending on the class ordering.

Inspecting these weights helps explain which words influence the model's predictions. However, a word's weight is an association learned from the training data, not proof that the word always indicates spam.

### 5. Explain accuracy, precision, recall, and F1-score.

A confusion matrix compares actual labels with predicted labels. In binary classification, it contains:

- **True Positive (TP):** Spam correctly predicted as spam.
- **True Negative (TN):** Ham correctly predicted as ham.
- **False Positive (FP):** Ham incorrectly predicted as spam.
- **False Negative (FN):** Spam incorrectly predicted as ham.

The main metrics are:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** Of all messages predicted as spam, the proportion that really are spam.
- **Recall:** Of all actual spam messages, the proportion correctly detected.
- **F1-score:** The harmonic mean of precision and recall.

Accuracy can be misleading when classes are imbalanced. A model that predicts nearly every message as ham could achieve high accuracy because ham messages are much more common, while missing many spam messages.

### 6. How can class imbalance be handled?

Class imbalance happens when one class has many more examples than another. Several methods can help:

- **Class weights:** Give more importance to mistakes involving the minority class. This can improve its recall, but may increase false positives.
- **Resampling:** Oversampling increases minority-class examples, while undersampling reduces majority-class examples. Oversampling can increase overfitting, and undersampling can discard useful information.
- **Decision threshold adjustment:** Changes the probability threshold used to assign a class. Lowering the spam threshold may catch more spam but can also flag more legitimate messages.

The best method depends on the cost of each type of mistake.

### 7. Case study: SMS Spam Collection

Almeida, Hidalgo, and Yamakami (2011) presented the SMS Spam Collection and studied machine learning methods for filtering SMS spam. The collection contains real, labeled SMS messages categorized as ham or spam. It was created from several sources and made available for research.

Their paper compared established classification methods and reported that Support Vector Machines performed best among the methods they evaluated. The study also highlighted the difficulty of filtering short messages, because SMS messages often contain limited textual information.

This research is relevant to this assignment because it uses the same general task: learning from labeled SMS messages to distinguish spam from legitimate messages.

**References**

1. Scikit-learn. (2026). *Multinomial Naive Bayes*. https://scikit-learn.org/stable/modules/naive_bayes.html
2. Scikit-learn. (2026). *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
3. Scikit-learn. (2026). *Train-test split*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
4. AWS. *Splits and data leakage*. https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-operations-planning/splits-leakage.html
5. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. Proceedings of the 2011 ACM Symposium on Document Engineering. https://doi.org/10.1145/2034691.2034742
6. UCI Machine Learning Repository. *SMS Spam Collection*. https://doi.org/10.24432/C5CC84

---

## Part B: Practical Work

### Dataset preparation

The original dataset contained 5,582 rows. It had 20 missing SMS messages and duplicate messages. After removing missing and empty messages, normalizing the labels, and removing duplicate text messages, the cleaned dataset contained 5,140 rows.

The cleaned label distribution was:

| Label | Messages |
|---|---:|
| Ham | 4,498 |
| Spam | 642 |
| Total | 5,140 |

Spam represented approximately 12.49% of the cleaned dataset.

### 1. Separate features and labels

The `text` column was used as the feature (X), while the `label` column was used as the target (y). The dataset contained 5,140 messages and 5,140 corresponding labels.

### 2. Split the dataset

I used `train_test_split()` with `test_size=0.2`, `random_state=7`, and `stratify=y`.

| Set | Total | Ham | Spam |
|---|---:|---:|---:|
| Training | 4,112 | 3,598 | 514 |
| Testing | 1,028 | 900 | 128 |

Stratification preserved the class proportions in both sets.

### 3. Train and compare five models

I trained five pipelines using CountVectorizer, TF-IDF, Multinomial Naive Bayes, and Logistic Regression.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Count + NB | 0.9854 | 0.9748 | 0.9062 | 0.9393 |
| TF-IDF + NB | 0.9562 | 1.0000 | 0.6484 | 0.7867 |
| TF-IDF + LR | 0.9650 | 0.9894 | 0.7266 | 0.8378 |
| Balanced LR | 0.9815 | 0.9225 | 0.9297 | 0.9261 |
| Balanced LR + Stop Words | 0.9805 | 0.9219 | 0.9219 | 0.9219 |

CountVectorizer with Multinomial Naive Bayes achieved the highest accuracy and spam F1-score in this experiment. Balanced Logistic Regression achieved the highest spam recall.

### 4. Compare CountVectorizer and TF-IDF with Naive Bayes

CountVectorizer performed better on this test split. Its spam recall was 0.9062 compared with 0.6484 for TF-IDF, and its spam F1-score was 0.9393 compared with 0.7867.

TF-IDF achieved perfect spam precision (1.0000), but it missed more actual spam messages.

### 5. Confusion matrices

The confusion matrices used the label order `["ham", "spam"]`.

**Model 3: TF-IDF + Logistic Regression**

```text
[[899   1]
 [ 35  93]]
```

- TN = 899
- FP = 1
- FN = 35
- TP = 93

**Model 4: Balanced Logistic Regression**

```text
[[890  10]
 [  9 119]]
```

- TN = 890
- FP = 10
- FN = 9
- TP = 119

### 6. Manually calculate Model 4 metrics

Using TP = 119, FP = 10, and FN = 9:

Precision = 119 / (119 + 10) = **0.9225**

Recall = 119 / (119 + 9) = **0.9297**

F1-score = 2 × (0.9225 × 0.9297) / (0.9225 + 0.9297) = **0.9261**

These calculations match the values reported by the classification program.

### 7. Compare Model 3 and Model 4

Balanced Logistic Regression detected 119 spam messages, compared with 93 for ordinary Logistic Regression. It caught 26 additional spam messages and reduced missed spam from 35 to 9.

However, false positives increased from 1 to 10. This shows the trade-off between catching more spam and incorrectly flagging legitimate messages.

### 8. Inspect the learned feature weights

The ten strongest positive feature weights included:

`call`, `txt`, `text`, `uk`, `reply`, `www`, `free`, `stop`, `150p`, and `claim`.

The ten strongest negative feature weights included:

`me`, `my`, `ok`, `that`, `it`, `lt`, `gt`, `am`, `come`, and `but`.

The positive words are often associated with promotions, links, and instructions to reply. The negative words are more common in ordinary personal conversations. These are learned associations, not rules that always apply.

### 9. Error analysis

Model 4 produced 9 false negatives and 10 false positives.

Examples of missed spam included a football update, ringtone messages, a quiz asking for a reply, and a mobile billing complaint. These messages did not always contain the obvious promotional words the model had learned.

Examples of false positives included ordinary messages about missed calls, lunch, mobile phones, and a technical topic. Some contained words such as `call` that the model strongly associated with spam.

This shows that the model can make mistakes when legitimate messages use words commonly found in spam, or when spam messages use ordinary conversational language.

### 10. Test five original messages

I selected three spam messages and two ham messages from the cleaned dataset and tested them with Model 4.

| Actual label | Predicted label | Spam probability |
|---|---|---:|
| Spam | Spam | 0.9847 |
| Spam | Spam | 0.8998 |
| Spam | Spam | 0.9371 |
| Ham | Ham | 0.0289 |
| Ham | Ham | 0.0452 |

All five predictions were correct. However, these messages were selected from the same cleaned dataset used to train the model, so this test is a demonstration rather than an independent evaluation.

### Final reflection: Choosing a phone spam filter

For a phone spam filter, I would consider Balanced Logistic Regression because it detected more spam and missed fewer spam messages. Missing a dangerous or misleading spam message can be more harmful than temporarily flagging a legitimate message. However, false positives still matter because users should not lose important personal messages. A practical system could place suspicious messages in a spam folder instead of permanently deleting them, allowing users to review them.

The best model depends on the intended use and the cost of each type of mistake. In this experiment, CountVectorizer with Naive Bayes also performed very well and achieved the highest spam F1-score, so it should not be ignored.