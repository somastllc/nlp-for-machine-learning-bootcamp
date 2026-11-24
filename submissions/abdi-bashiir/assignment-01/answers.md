# Assignment 1 – NLP Foundations

## 1. Definition of Natural Language Processing (NLP)

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence (AI) that enables computers to understand, interpret, process, and generate human language. AI is the broader field of making machines perform tasks that require human intelligence, while Machine Learning (ML) is a subset of AI that allows computers to learn patterns from data and improve their performance.

**Real-life example: Google Translate**

When a user enters a sentence in Somali and translates it into English, NLP helps the system process and understand the language, ML learns patterns from large amounts of language data to improve translation accuracy, and AI brings these technologies together to perform the translation automatically.

## 2. Core Terms in NLP

1. **Corpus:** A collection of text documents used for NLP tasks.
2. **Document:** A single piece of text within a corpus, such as one football news article.
3. **Token:** A word, part of a word, or punctuation mark obtained by splitting a document into smaller units.
4. **Vocabulary:** The collection of all unique tokens found in a corpus.
5. **Feature:** A measurable piece of information extracted from text and used as input by a Machine Learning model to make predictions.
6. **Label:** The correct category or target assigned to a document.

**Example Dataset: Football News Classification**

- **Corpus:** A collection of 1,000 football news articles.
- **Document:** One individual article about a football match.
- **Token:** Words such as "Manchester", "City", and "Victory".
- **Vocabulary:** All unique tokens found in the 1,000 articles.
- **Feature:** The number of times the word "Football" appears in an article.
- **Label:** "Sports", "Politics", or "Business", indicating the category of the article.

## 3. Four Common NLP Tasks

**1. Text Classification:** Assigning an entire text to predefined categories.

- **Input:** A football news article.
- **Output:** A category such as Sports or Politics.

**2. Sequence Labeling:** Assigning labels to individual words or parts of a sentence to identify specific information.

- **Input:** "Mohamed Salah plays for Liverpool."
- **Output:** Identifying "Mohamed Salah" as a Person and "Liverpool" as an Organization.

**3. Search and Similarity:** Finding relevant or similar documents based on a search query.

- **Input:** A search query such as "Manchester City results".
- **Output:** Relevant football news articles related to the search.

**4. Text Generation:** Producing new text based on a given input or prompt.

- **Input:** A prompt such as "Write a summary of a football match."
- **Output:** A generated summary of the football match.

## 4. Supervised vs. Unsupervised Learning in NLP

**1. Supervised Learning:** A type of learning where a model is trained using labeled data to predict specific categories or outcomes.

- **Example:** Classifying football news articles as Sports or Politics.
- **Target Labels:** Required because the model learns from examples that already have known correct answers.

**2. Unsupervised Learning:** A type of learning where a model works with unlabeled data to discover hidden patterns or groups.

- **Example:** Grouping football news articles based on similar topics without providing predefined categories.
- **Target Labels:** Not required because the model discovers patterns and groups from the data without predefined answers.

**Main Difference:** Supervised learning uses labeled data to learn what to predict, while unsupervised learning uses unlabeled data to discover patterns and relationships independently.

## 5. Numerical Representations in NLP

Text must be converted into numerical representations because most Machine Learning models cannot directly understand or process human language. They require numbers to perform calculations, identify patterns, and make predictions.

**Example: Bag of Words (BoW)**

Bag of Words is a simple text representation method that converts text into numerical values by counting how many times each word appears in a document.

For example:

- **Input:** "Football is great."
- **Vocabulary:** Football, is, great.
- **Output:** [1, 1, 1]

Each number represents the frequency of a word in the document.

**Main Limitation:** Bag of Words ignores word order and context. Therefore, it cannot fully capture the meaning of sentences or understand how words relate to each other.

## 6. NLP Challenges for Low-Resource Languages (Somali)

Low-resource languages such as Somali face several challenges in NLP because of limited digital data and language complexity.

**1. Limited Datasets:** There are not enough Somali text datasets available for training NLP models. This makes models more difficult to train and can reduce their accuracy when processing Somali text.

**2. Spelling Variation:** The same Somali word may be written in different ways. For example, "Soomaaliya" and "Somalia" may be used to refer to the same country. A basic NLP system may treat them as different words, reducing its ability to recognize related text.

**3. Word Forms:** Somali words can have different grammatical forms because of changes in word endings. For example, "nin" means "man", while "ninka" means "the man". A simple word-based NLP system may treat these as completely different tokens, making it difficult to recognize their relationship.

**4. Mixed-Language Text:** People sometimes mix Somali and English in the same sentence. For example, someone may write "meeting-ka". This can make it difficult for an NLP system to identify and process the correct language and word structure.

**Conclusion:** These challenges make it difficult to develop accurate and reliable NLP systems for the Somali language. More Somali datasets and suitable language-processing methods are needed to improve their performance.

## 7. Case Study: Somali Sentiment Analysis

**Research Paper:** _Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification (2023)_ by Kadar Bahar and Nehad T. A. Ramaha.

**1. Task:** The study focuses on identifying people's opinions and emotions in Somali text, such as positive and negative sentiments.

**2. Data:** The researchers created a small-scale Somali sentiment-annotated dataset using crowdsourcing, where text examples were collected and labeled with sentiment information.

**3. Method:** The researchers used lexicon-based methods and feature engineering techniques to extract sentiment information from Somali text and train a sentiment classification model.

**4. Findings:** The study achieved competitive results compared with existing sentiment analysis approaches for under-resourced languages. It demonstrated that sentiment analysis is possible in Somali even with a small-scale dataset and limited language resources.

\***\*NLP Task:** Text Classification, because the system classifies Somali text according to its sentiment, such as positive or negative.

## References

1. Bahar, K., & Ramaha, N. T. (2023). _Exploring Somali Sentiment Analysis: A Resource-Light Approach for Small-scale Text Classification_. International Conference on Applied Engineering and Natural Sciences, 1(1), 620–628. https://doi.org/10.59287/icaens.1069
2. Jurafsky, D., & Martin, J. H. (2025). _Speech and Language Processing_ (3rd ed. draft). Stanford University.
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_. Cambridge University Press.
