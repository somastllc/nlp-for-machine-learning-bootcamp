# Assignment 4 – Classical Classification

## Part A – Research

### 1. Supervised text classification

Supervised text classification is a machine-learning task in which a model learns to assign categories to text using examples whose correct labels are already known. Each training example contains an input document (such as an SMS, email, review, or news story) and a target label. A text-processing method converts the words into numerical features, and a classifier learns a relationship between those features and the labels. After training, the model can predict labels for previously unseen messages.

- **Binary classification** assigns one of two mutually exclusive labels. Example: classifying an SMS as *spam* or *ham* (legitimate).
- **Multiclass classification** assigns exactly one label from three or more possible classes. Example: routing a customer-support message to *billing*, *technical support*, *delivery*, or *returns*.
- **Multilabel classification** allows one text to receive several labels simultaneously. Example: a news article may be tagged *politics*, *economy*, and *international* at the same time.

The important distinction is whether a document can receive one label only or several labels together.

### 2. Train/test split, stratification, duplicates, and leakage

A train/test split separates the available labeled examples into two non-overlapping groups. The training set is used to fit the model; the test set is held back to estimate how well the finished modeling procedure generalizes to unseen data. In this assignment, 80% is used for training and 20% for testing.

Stratification means that the split approximately preserves the original class proportions in both subsets. For example, if roughly 13% of all messages are spam, stratification aims to keep the spam proportion close to 13% in both training and test sets. This is especially useful when the minority class is relatively small.

Exact duplicate messages should be removed before splitting because identical or near-identical copies can land in both sets. The model may then appear to perform well by recognizing text it has effectively already seen, rather than generalizing to new messages. This is a form of data leakage. The test set must not guide vocabulary selection, feature selection, hyperparameter tuning, threshold selection, or model choice: repeatedly making decisions based on test performance turns the test set into part of the development process and makes its final score optimistically biased. Use training data and (ideally) cross-validation for model selection, and reserve the test set for the final evaluation.

### 3. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier commonly used with word-count features. It estimates how likely each word is in each class and combines these likelihoods with the prior probability of each class. For a new message, it estimates the posterior probability of each class and predicts the class with the larger posterior probability.

The “naive” assumption is conditional independence: given the class, the model treats the occurrence/count of each word as independent of the other words. Natural language does not satisfy this assumption—word order, grammar, and phrases matter—but the model can still work well because many classification tasks contain strong individual-word signals. Words such as “claim”, “prize”, or “winner” may be more common in spam than in ordinary messages. The model is simple, fast, and often a strong baseline, although it may miss meaning expressed through word order, context, sarcasm, or combinations of words.

### 4. Logistic Regression for text

Logistic Regression is a linear classifier that combines numerical text features using learned coefficients (weights) and converts the resulting score into a probability. In binary classification, the probability is commonly modeled using the logistic (sigmoid) function:

\[
P(y=1\mid x)=\frac{1}{1+\exp[-(b+\sum_j w_jx_j)]}
\]

Here, \(x_j\) is a feature value, \(w_j\) is its learned coefficient, and \(b\) is the intercept. With a spam label coded as the positive class, a positive coefficient pushes the prediction toward spam, while a negative coefficient pushes it toward ham, holding other features constant. The size of a coefficient indicates the strength of that feature's contribution on the model's feature scale; it is not automatically a causal effect or a standalone probability.

For a bag-of-words or TF-IDF model, the feature names can be retrieved from the fitted vectorizer and paired with the coefficients. Sorting the coefficients reveals words with the strongest positive and negative associations. This is useful for inspecting what the model has learned, identifying suspicious shortcuts, and checking whether the influential words make domain sense. Correlated words and preprocessing choices can affect individual coefficients, so weights should be interpreted as model associations rather than proof of causation.

### 5. Accuracy, precision, recall, F1, and the confusion matrix

For a binary spam classifier, define spam as the positive class:

| | Predicted spam | Predicted ham |
|---|---:|---:|
| **Actually spam** | True positive (TP) | False negative (FN) |
| **Actually ham** | False positive (FP) | True negative (TN) |

- **Precision** measures how many messages predicted as spam really are spam: \(TP/(TP+FP)\).
- **Recall** measures how many actual spam messages were successfully caught: \(TP/(TP+FN)\).
- **F1 score** is the harmonic mean of precision and recall: \(2PR/(P+R)\). It is high only when both precision and recall are reasonably high.
- **Accuracy** measures the proportion of all predictions that are correct: \((TP+TN)/(TP+TN+FP+FN)\).

Accuracy can be misleading when classes are imbalanced. If only 5% of messages are spam, a classifier that predicts every message as ham achieves 95% accuracy while catching zero spam. Therefore, spam precision, spam recall, F1, and the confusion matrix provide essential information that accuracy alone hides.

### 6. Methods for handling class imbalance

**Class weighting.** A classifier can assign a larger penalty to mistakes on the minority class. In scikit-learn, `class_weight="balanced"` computes weights inversely related to class frequencies. This encourages the model to pay more attention to spam without physically changing the training examples. The common trade-off is higher spam recall but potentially lower precision, because more legitimate messages may be flagged.

**Random oversampling.** This duplicates minority-class training examples until the class distribution is less uneven. It gives the learner more exposure to spam, but repeated copies can increase overfitting and do not add genuinely new linguistic variety.

**Random undersampling.** This removes some majority-class training examples. It reduces the dominance of ham and can make training faster, but useful examples are discarded and the model may learn a less complete picture of legitimate messages.

**Synthetic oversampling.** Methods such as SMOTE create synthetic minority examples in feature space rather than simply copying rows. This can help some numeric-feature problems, but synthetic points in sparse, high-dimensional text representations may not correspond to meaningful natural-language messages. It should be validated carefully and applied only to training folds.

**Decision-threshold adjustment.** Many probabilistic classifiers predict spam when its estimated probability is at least 0.5. Lowering this threshold generally catches more spam (higher recall) but also increases false alarms (lower precision). Raising it generally reduces false alarms but misses more spam. Choose a threshold using validation data and the real costs of each error, not the final test set.

No balancing method is universally best. The choice depends on the minority-class performance required, the cost of false positives and false negatives, and validation results.

### 7. Case study: SMS spam filtering

Almeida, Gómez Hidalgo, and Yamakami (2011) introduced a publicly available, non-encoded SMS spam collection and compared established machine-learning approaches for SMS spam filtering. Their motivation was that public SMS datasets were scarce and that short messages, abbreviations, and informal language make the task challenging. The dataset is commonly known as the SMS Spam Collection and contains legitimate messages and spam messages gathered from real sources.

The study evaluated several text-classification approaches using message-derived textual features and tokenization/feature representations. The authors reported that Support Vector Machines outperformed the other evaluated classifiers in their experiments and recommended SVM as a useful baseline for comparison. A later study by Almeida, Gómez Hidalgo, and Silva (2013) further investigated SMS spam filtering and reported strong results for combinations including Logistic Regression and SVM. The reported figures depend on the dataset version, tokenization, split, and evaluation protocol; they should not be treated as expected results for this assignment's independently created 80/20 split.

This case study illustrates two practical lessons. First, a public benchmark enables reproducible comparisons. Second, preprocessing and tokenization choices matter, and a model that performs well in one experiment is not guaranteed to be best on another split or deployment population.

---

## Part B – Practical

### Reproducibility and dataset

The original file contained 5,582 rows and two columns (`label` and `text`). The processing script removed missing/blank text and labels, standardized the labels, and removed duplicate message text. It saved the resulting 5,140 messages to `dataset/clean_sms_spam_practice.csv`. The classifier used the required stratified split: `test_size=0.2`, `random_state=7`, and `stratification on y`.

### Task 1 – Explore

The original dataset had 5,582 rows, 20 missing text values, and 430 duplicate-text entries as reported by pandas. The original labels were 4,832 `ham`, 747 `spam`, and 3 `ham.`; after normalizing `ham.` to `ham`, the original spam share was 13.3823% (747/5,582). The original dataset also contained 20 missing messages, so the cleaned spam share differs after removing missing and duplicate rows.

### Task 2 – Clean and split

After removing missing/blank rows and duplicate messages, 5,140 unique messages remained: 4,498 ham and 642 spam. The training set contained 4,112 messages with 514 spam (12.5000%), while the test set contained 1,028 messages with 128 spam (12.4514%). These shares are close because stratification preserves the class proportions approximately while assigning whole messages to each subset.

### Task 3 – Five pipelines

All five requested pipelines were trained on the same training set and evaluated on the same test set. CountVectorizer with Multinomial Naive Bayes performed strongly, while TF-IDF with ordinary Logistic Regression missed more spam; using balanced class weights improved the spam recall substantially. The full classification reports for all five models are recorded in `output.txt`.

### Task 4 – Comparison table

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|
| CountVectorizer + MultinomialNB | 98.54% | 97.48% | 90.62% | 93.93% |
| TF-IDF + MultinomialNB | 95.62% | 100.00% | 64.84% | 78.67% |
| TF-IDF + LogisticRegression | 96.50% | 98.94% | 72.66% | 83.78% |
| TF-IDF + balanced LogisticRegression | 98.15% | 92.25% | 92.97% | 92.61% |
| TF-IDF English stop words + balanced LogisticRegression | 98.05% | 92.19% | 92.19% | 92.19% |

The best spam recall was achieved by Model 4, balanced Logistic Regression, at 92.97%. CountVectorizer worked better than TF-IDF with Multinomial Naive Bayes on this split: its spam F1 was 93.93%, compared with 78.67% for TF-IDF Naive Bayes. TF-IDF Naive Bayes had perfect spam precision, but its recall was only 64.84%, meaning it missed many spam messages.

### Task 5 – Confusion matrices

The confusion matrices use rows for actual labels and columns for predicted labels, in the order `[ham, spam]`.

- **Model 3 (TF-IDF + ordinary Logistic Regression):** TN = 899, FP = 1, FN = 35, TP = 93.
- **Model 4 (TF-IDF + balanced Logistic Regression):** TN = 890, FP = 10, FN = 9, TP = 119.

Model 4 detected 26 more spam messages than Model 3, but it also incorrectly flagged 9 additional legitimate messages.

### Task 6 – Hand calculation

For Model 4, TP = 119, FP = 10, FN = 9, and TN = 890.

\[
Precision=\frac{TP}{TP+FP}=\frac{119}{119+10}
=\frac{119}{129}=0.922481
\]

\[
Recall=\frac{TP}{TP+FN}=\frac{119}{119+9}
=\frac{119}{128}=0.929688
\]

\[
F1=\frac{2PR}{P+R}
=\frac{2(0.922481)(0.929688)}{0.922481+0.929688}
=0.926070
\]

The calculated precision (92.2481%), recall (92.9688%), and F1 (92.6070%) match Model 4's classification report, apart from rounding.

### Task 7 – Class imbalance

Compared with ordinary Logistic Regression (Model 3), the balanced model (Model 4) caught 26 additional spam messages: TP increased from 93 to 119, and FN decreased from 35 to 9. It caused 9 additional false alarms: FP increased from 1 to 10. This demonstrates the trade-off from class weighting: the model becomes more sensitive to the minority spam class, but more ham messages are incorrectly sent to the spam class.

### Task 8 – What the model learned

The ten largest positive weights were `call`, `txt`, `text`, `uk`, `reply`, `www`, `free`, `stop`, `150p`, and `claim`. These words make sense because the messages in this dataset often use calls to action, web links, premium-rate SMS prices, and prize/claim language in spam. The ten most negative weights were `me`, `my`, `ok`, `that`, `it`, `lt`, `gt`, `am`, `come`, and `but`; conversational words are plausible ham indicators. The `lt` and `gt` features are surprising because they appear to be HTML-escaped fragments (`&lt;` and `&gt;`) left in some message text, showing that text encoding artifacts can become predictive features.

### Task 9 – Error analysis

The missed spam examples included a sports score message, a ringtone download message, a quiz asking for a multiple-choice reply, a premium-rate/customer-service complaint, and a ringtone notification. A pattern is that not every spam message uses obvious words such as “FREE” or “WIN”; some look like ordinary alerts, trivia, or conversational texts. The false alarms included ordinary questions and personal messages, a technical phrase (“Host-based IDPS for linux systems”), a question about a missed call, and a message mentioning a mobile phone. A likely pattern is that short messages with words such as “call” or “mobile” can resemble promotional messages even when their actual context is legitimate.

### Task 10 – New messages

The three newly written promotional/prize messages were all predicted as spam: their spam probabilities were 0.9960, 0.8591, and 0.9749. The two ordinary personal messages were predicted as ham, with spam probabilities of 0.0854 and 0.1472. All five predictions matched their intended labels; the second promotional example had the lowest spam probability among the three spam examples, but 0.8591 was still a confident spam prediction under the model's learned probabilities.

### Final model recommendation

For a phone spam filter, I would choose CountVectorizer with Multinomial Naive Bayes as the cautious default for this experiment. It achieved the highest accuracy (98.54%) and spam F1 (93.93%), with high spam precision (97.48%) and 90.62% spam recall; the balanced Logistic Regression model caught more spam (92.97% recall) but had lower precision (92.25%) and produced 10 false alarms compared with only 1 for ordinary Logistic Regression. I consider a false alarm especially harmful on a phone because it can hide an important legitimate message, such as a bank alert, delivery update, or personal text. A practical deployment could place suspected spam in a recoverable spam folder rather than deleting it, and then adjust the decision threshold using validation data if the user prefers catching more spam.

## References

1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. Proceedings of the 11th ACM Symposium on Document Engineering, 259–262. https://doi.org/10.1145/2034691.2034742
2. Almeida, T. A., Gómez Hidalgo, J. M., & Silva, T. P. (2013). *Towards SMS Spam Filtering: Results under a New Dataset*. International Journal of Information Security Science, 2(1). https://dergipark.org.tr/en/pub/ijiss/article/167834
3. McCallum, A., & Nigam, K. (1998). *A Comparison of Event Models for Naive Bayes Text Classification*. AAAI-98 Workshop on Learning for Text Categorization. https://www.cse.iitb.ac.in/~soumen/readings/papers/McCallumN1998event.pdf
4. Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers*. Proceedings of the 20th International Conference on Machine Learning.
5. scikit-learn developers. *Working with Text Data*. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
6. scikit-learn developers. *CountVectorizer documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
7. scikit-learn developers. *Confusion Matrix documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
8. Carvalho, M., Pinho, A. J., & Brás, S. (2025). *Resampling approaches to handle class imbalance: a review from a data perspective*. Journal of Big Data, 12, 71. https://doi.org/10.1186/s40537-025-01119-4
9. *Dealing with Data Imbalance in Text Classification*. Procedia Computer Science, 159 (2019), 736–745. https://doi.org/10.1016/j.procs.2019.09.229
