### 1. Supervised Text Classification

Supervised text classification is a machine learning task in which a model learns to assign predefined categories or labels to text documents using labeled training data. During training, the model analyzes examples containing text and their corresponding correct labels to identify patterns. It then uses the learned patterns to classify new, previously unseen text.

There are three main types of text classification:

**1. Binary Classification**

Binary classification involves assigning each text example to one of two possible classes. For example, an SMS spam detection system classifies messages as either *Spam* or *Ham (Not Spam)*.

**2. Multiclass Classification**

Multiclass classification involves assigning each text example to exactly one class from three or more possible classes. For example, a news classification system can categorize articles into Politics, Education, Sports, or Technology.

**3. Multilabel Classification**

Multilabel classification allows one text example to be assigned multiple labels simultaneously. For example, a movie description may be classified under Action, Comedy, and Romance at the same time.

**In summary**, binary classification has two possible classes, multiclass classification has more than two possible classes but assigns one class to each example, while multilabel classification allows multiple labels for a single example.

Research references
Kowsari et al. (2019), Text Classification Algorithms: A Survey. Information, 10(4), 150. https://doi.org/10.3390/info10040150

### 2. Train/Test Split and Stratification

A train/test split is a technique used to divide a dataset into separate subsets for model training and evaluation. The training set is used to teach the model patterns from labeled examples, while the testing set evaluates how well the trained model performs on previously unseen data.

Stratification is a sampling technique that maintains approximately the same class proportions in the training and testing sets as in the original dataset. For example, if an SMS dataset contains 80% Ham messages and 20% Spam messages, stratification attempts to preserve these proportions in both subsets.

Duplicates should be removed before splitting to prevent data leakage. If identical messages appear in both the training and testing sets, the model may recognize messages it has already encountered during training. This can produce artificially high evaluation results that do not reflect its ability to classify new messages.

Furthermore, the test set should never influence model selection or parameter tuning because repeatedly using test results to make decisions can lead to overfitting to the test data. Model selection should instead use training data and validation techniques, while the test set should be reserved for final evaluation.

**References**

- Scikit-learn Developers. (n.d.). *train_test_split*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.

### 3. Multinomial Naive Bayes Classification

Multinomial Naive Bayes is a supervised machine learning algorithm commonly used for text classification tasks, such as SMS spam detection. It is based on Bayes' Theorem and uses probabilities learned from labeled training data to predict the most likely class of a new text message.

Before classification, the text must be converted into numerical features using techniques such as CountVectorizer or TF-IDF. The model then learns two important types of probabilities:

1. **Class probabilities:** These represent the probability of each class occurring in the training dataset. For example, if 20% of the messages are Spam and 80% are Ham, then P(Spam) = 0.20 and P(Ham) = 0.80.
2. **Word probabilities:** These represent the probability of a word appearing in a particular class. For example, the word "prize" may have a higher probability of appearing in Spam messages than in Ham messages.

When a new message is received, the model calculates the probability of each possible class based on the words in that message. It then predicts the class with the highest probability.

The term **Multinomial** refers to the model's use of word counts or frequencies as features. It considers how frequently words appear in text documents when learning patterns for classification.

**The Naive Assumption**

The term "Naive" refers to the assumption that text features (words) are conditionally independent of one another given the class. In other words, the model treats each word as contributing independently to the classification decision once the class is known.

For example, when classifying the message "Win a free prize," the model considers the probabilities of the individual words given the Spam class, rather than modeling all relationships between the words.

**Why Multinomial Naive Bayes Works Well for Text Classification**

Although the independence assumption is not always realistic in natural language, Multinomial Naive Bayes performs well in many text classification tasks because it is computationally efficient, handles high-dimensional and sparse text data, and can learn useful patterns from word frequencies. It is also relatively simple to implement and provides a strong baseline for text classification problems.

**References**

- McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 41–48.
- Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of Naive Bayes text classifiers. *Proceedings of the 20th International Conference on Machine Learning (ICML)*, 616–623.
- Scikit-learn Developers. (n.d.). *MultinomialNB*. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

### 4. Logistic Regression Classification

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification tasks, such as SMS spam detection. It learns patterns from labeled training data and uses a mathematical function called the sigmoid function to estimate the probability that a given message belongs to a particular class.

For example, an SMS classification system can use two labels:

- **Spam (1):** Unwanted or promotional messages.
- **Ham (0):** Normal or legitimate messages.

After converting text into numerical features using techniques such as TF-IDF, Logistic Regression calculates a probability for each new message. If the predicted probability is above the classification threshold, commonly 0.5, the message is classified as Spam; otherwise, it is classified as Ham.

**Feature Weights**

Feature weights are numerical values learned by the Logistic Regression model during training. They indicate how individual features influence the model's predictions.

- **Positive weights:** Increase the model's tendency to predict the positive class (Spam, if Spam is encoded as 1).
- **Negative weights:** Decrease the model's tendency to predict Spam and favor the negative class (Ham).
- **Weights close to zero:** Indicate a relatively small influence on the model's decision.

For example, if the word "free" has a positive weight of +2.5, it contributes toward predicting Spam. If the word "meeting" has a negative weight of -2.0, it contributes toward predicting Ham. These are illustrative values, not actual model results.

**How to Inspect Learned Weights**

In Python, the learned feature weights can be inspected using the `coef_` attribute of a trained Logistic Regression model. The feature names can be retrieved from the vectorizer using `get_feature_names_out()`.

Example:

```python
model = pipeline.named_steps["logisticregression"]
vectorizer = pipeline.named_steps["tfidfvectorizer"]

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]
```

The feature names and their corresponding weights can then be combined and sorted to identify the words with the highest positive and lowest negative weights. The highest positive weights indicate features that push predictions toward Spam, while the most negative weights indicate features that push predictions toward Ham.

**In conclusion**, Logistic Regression is an effective classification algorithm that estimates class probabilities using learned feature weights. Inspecting these weights helps researchers understand which words contribute most strongly to the model's predictions.

**References**

- Scikit-learn Developers. (n.d.). *LogisticRegression*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

### 5. Classification Evaluation Metrics

Classification evaluation metrics are used to measure how well a machine learning model predicts the correct classes. In SMS spam detection, these metrics help evaluate how accurately a model distinguishes between Spam and Ham messages.

**Confusion Matrix**

A confusion matrix is a table that compares the actual class labels with the model's predicted labels. In binary classification, it contains four important outcomes:

- **True Positive (TP):** The model correctly identifies a Spam message as Spam.
- **True Negative (TN):** The model correctly identifies a Ham message as Ham.
- **False Positive (FP):** The model incorrectly classifies a Ham message as Spam. This is also called a false alarm.
- **False Negative (FN):** The model incorrectly classifies a Spam message as Ham. This means the model misses a Spam message.

**Four Main Evaluation Metrics**

**1. Accuracy**

Accuracy measures the proportion of all predictions that the model classifies correctly.

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

**2. Precision**

Precision measures how many messages predicted as Spam are actually Spam. High precision means fewer legitimate Ham messages are incorrectly classified as Spam.

Formula:

Precision = TP / (TP + FP)

**3. Recall**

Recall measures how many actual Spam messages the model successfully identifies. High recall means the model misses fewer Spam messages.

Formula:

Recall = TP / (TP + FN)

**4. F1-Score**

F1-score is the harmonic mean of Precision and Recall. It provides a balance between these two metrics and is useful when both false alarms and missed Spam messages are important.

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Why Accuracy Can Be Misleading on Imbalanced Datasets**

Accuracy can be misleading when a dataset contains significantly different numbers of examples in each class. For example, consider an SMS dataset containing 950 Ham messages and 50 Spam messages. If a model predicts every message as Ham, it achieves 95% accuracy because it correctly classifies all 950 Ham messages. However, it fails to identify any of the 50 Spam messages.

This example demonstrates that a high accuracy score does not always indicate a good classification model. Therefore, Precision, Recall, and F1-score should also be considered, particularly when working with imbalanced datasets.

**Conclusion**

In SMS spam detection, the choice of evaluation metric depends on the application's objectives. Precision helps reduce false alarms, Recall helps reduce missed Spam messages, and F1-score balances both. Evaluating multiple metrics provides a more reliable understanding of model performance than using accuracy alone.

**References**

- Scikit-learn Developers. (n.d.). *Metrics and scoring: Quantifying the quality of predictions*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn Developers. (n.d.). *confusion_matrix*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Scikit-learn Developers. (n.d.). *f1_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


### 6. Handling Class Imbalance

Class imbalance occurs when a dataset contains significantly different numbers of examples in different classes. For example, an SMS spam dataset may contain 900 Ham messages and only 100 Spam messages. In this situation, a classification model may become biased toward the majority class and perform poorly when identifying the minority class.

Several techniques can help address class imbalance:

**1. Class Weights**

Class weighting assigns different weights to classes during model training. The minority class receives a higher weight so that classification errors involving that class have a greater influence on the learning process. In Scikit-learn, `class_weight="balanced"` automatically calculates weights inversely proportional to class frequencies.

A major advantage is that class weighting can address imbalance without changing the original number of training samples. However, it may increase false positives and reduce precision.

**2. Resampling**

Resampling changes the class distribution in the training data. Two common approaches are:

- **Oversampling:** Increases the number of minority-class examples. It can help the model learn minority-class patterns, but duplicating existing samples may increase the risk of overfitting.
- **Undersampling:** Reduces the number of majority-class examples. It can make training faster, but useful information may be lost when samples are removed.

Resampling should be applied only to the training data. Applying it before splitting the dataset can cause data leakage and produce overly optimistic evaluation results.

**3. Classification Threshold Changes**

A classification threshold is the probability boundary used to convert predicted probabilities into class labels. For example, a model using a threshold of 0.50 predicts Spam when its estimated Spam probability is at least 0.50.

Reducing the threshold, for example from 0.50 to 0.30, can help identify more Spam messages and increase Recall. However, it may also increase false positives. Increasing the threshold can reduce false alarms but may cause the model to miss more actual Spam messages.

**Trade-offs and Conclusion**

Each technique has advantages and disadvantages. Class weighting changes the importance of classification errors, oversampling increases minority-class representation, undersampling removes majority-class examples, and threshold adjustment changes the decision boundary. The appropriate technique depends on the application and the cost of false positives and false negatives. For an SMS spam detection system, these methods should be evaluated using Precision, Recall, and F1-score rather than relying on accuracy alone. Model selection and threshold tuning should use validation data, while the test set should remain reserved for final evaluation.

**References**

- Scikit-learn Developers. (n.d.). *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn Developers. (n.d.). *Tuning the decision threshold for class prediction*. https://scikit-learn.org/stable/modules/classification_threshold.html
- Imbalanced-learn Developers. (n.d.). *Common pitfalls and recommended practices*. https://imbalanced-learn.org/stable/common_pitfalls.html
- Imbalanced-learn Developers. (n.d.). *Over-sampling*. https://imbalanced-learn.org/stable/over_sampling.html
- Imbalanced-learn Developers. (n.d.). *Under-sampling*. https://imbalanced-learn.org/stable/under_sampling.html

### 7. Case Study: SMS Spam Detection Using Classical Machine Learning

**Research Paper**

Soysaldı Şahin, M., Şahin, D. Ö., and Salah, A. F. (2026). *Revisiting SMS Spam Detection: The Impact of Feature Representation on Classical Machine Learning Models*. Electronics, 15(4), 894. https://doi.org/10.3390/electronics15040894

**1. Dataset**

The researchers used the SMS Spam Collection dataset, which contains 5,574 SMS messages. It consists of 4,827 Ham (legitimate) messages and 747 Spam messages. The dataset is imbalanced because Ham messages represent a much larger proportion of the data than Spam messages.

**2. Feature Representation**

The researchers investigated seven text feature representation techniques to convert SMS messages into numerical data:

- Count Vectorization
- Binary Bag of Words (BoW)
- TF-IDF
- Word bigrams
- Character 3-grams
- Enhanced TF-IDF
- Hybrid word and character representation

These techniques represent different aspects of text, including word frequency, word importance, word sequences, and character patterns. Character 3-grams are particularly useful for capturing spelling variations and unusual character patterns in SMS messages.

**3. Classification Models**

Six classical machine learning classifiers were evaluated:

- Naive Bayes (NB)
- Linear Support Vector Machine (Linear SVM)
- Radial Basis Function Support Vector Machine (RBF SVM)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)

The researchers combined the seven feature representation techniques with the six classifiers, producing 42 experimental configurations. They used 10-fold cross-validation to evaluate the configurations.

**4. Results**

The best-performing combination was Character 3-grams with Logistic Regression. It achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.55% |
| Precision | 98.55% |
| Recall | 90.50% |
| F1-score | 94.32% |
| AUC | 0.9893 |

These results indicate that the combination achieved strong overall classification performance and identified a high proportion of actual Spam messages. The researchers also found that Linear SVM achieved comparable performance.

**5. Main Findings and Limitations**

The study demonstrated that feature representation has a significant impact on SMS spam detection performance. In particular, character-level features combined with classical machine learning classifiers can achieve excellent results without requiring complex deep learning architectures.

However, the study focused on the English SMS Spam Collection benchmark dataset, which reflects SMS messages collected around 2012. Therefore, its results may not represent the performance of spam detection systems on modern messages or other languages, including Somali.

**Conclusion**

This case study demonstrates that classical machine learning algorithms can achieve strong results in SMS spam detection when combined with suitable text feature representations. It also highlights the importance of evaluating Precision, Recall, and F1-score alongside Accuracy, especially when working with imbalanced datasets.


---

# Part B: Practical Tasks

### Task 1: Dataset Loading and Exploration

I successfully loaded the SMS Spam dataset using Pandas. The original dataset contained 5,572 rows and 5 columns. I examined the dataset structure, column names, missing values, and data types. I identified that only the `v1` and `v2` columns contained the main classification information, representing message labels and message text. The three unnecessary columns contained many missing values and were removed during cleaning.

### Task 2: Data Cleaning and Duplicate Removal

I cleaned the dataset by renaming the original columns `v1` and `v2` to `label` and `text`. During data exploration, I identified 403 duplicate messages. I removed these duplicates using Pandas `drop_duplicates()` to prevent repeated messages from affecting model training and evaluation. After cleaning, the dataset contained 5,169 rows and 2 columns, with zero remaining duplicates and no missing values in the `label` and `text` columns.
### Task 3: Label Distribution and Class Imbalance

After cleaning the dataset, I examined the distribution of the target labels. The dataset contained 4,516 Ham messages (87.37%) and 653 Spam messages (12.63%). This shows that the dataset is imbalanced because Ham messages represent the majority class, while Spam messages represent the minority class. Class imbalance is important because a model may become biased toward predicting the majority class.

### Task 4: Train/Test Split and Stratification

I divided the cleaned dataset into training and testing sets using Scikit-learn's `train_test_split()` function. I used `test_size=0.2` to allocate 80% of the data for training and 20% for testing. I also used `random_state=7` to ensure reproducibility and `stratify=y` to maintain approximately the same class distribution in both sets. The training set contained 4,135 messages, while the testing set contained 1,034 messages.

### Task 5: Training Classification Models

I trained five different text classification pipelines using Scikit-learn. These pipelines combined CountVectorizer or TfidfVectorizer with Multinomial Naive Bayes and Logistic Regression classifiers. I also tested Logistic Regression with `class_weight="balanced"` to address class imbalance and another pipeline using English stop-word removal. All five models completed training successfully.

### Task 6: Model Evaluation and Comparison

I evaluated the five classification models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix. I focused particularly on the Spam class because detecting spam messages is important. The CountVectorizer with Multinomial Naive Bayes achieved a Spam Recall of 0.89, while the balanced Logistic Regression model also achieved a Spam Recall of 0.88. The balanced

### Task 7: Feature Importance

I examined the feature weights of the balanced Logistic Regression model to understand which words influenced its predictions. Words with positive weights, such as `call`, `free`, and `claim`, pushed predictions toward the Spam class. Words with negative weights, such as `me`, `my`, and `ok`, pushed predictions toward the Ham class. This helped me understand how Logistic Regression uses word features to distinguish between Spam and Ham messages. However, a single word does not determine the final prediction because the model considers multiple features.

### Task 8: Testing the Model with New SMS Messages

I tested the balanced Logistic Regression model using five new SMS messages that were not included in the training dataset. The model correctly classified all five example messages as either Ham or Spam. It also provided probability scores for its predictions. This practical test demonstrated how the trained model can classify new, unseen SMS messages. However, five examples are not enough to prove that the model will always perform accurately in real-world situations.