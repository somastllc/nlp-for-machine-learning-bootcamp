ASSIGNMENT 1 – NLP FOUNDATIONS

Question 1. Define Natural Language Processing (NLP). What is its relationship to Artificial Intelligence and Machine Learning? Use a real-life example to illustrate how they work together.
Answer:

Natural Language Processing (NLP) is a branch of computer science and Artificial Intelligence that enables computers to process, analyze, understand, and generate human languages such as English, Somali, and Arabic.

NLP helps computers perform language-related tasks, including translation, text classification, sentiment analysis, speech recognition, and text summarization.
Relationship between AI, ML, and NLP
- Artificial Intelligence (AI): A broad field of computer science that focuses on developing systems capable of performing tasks associated with human intelligence.
- Machine Learning (ML): A branch of AI that allows computers to learn patterns from data and make predictions without being explicitly programmed for every decision.
- Natural Language Processing (NLP): A field that focuses on enabling computers to work with human language.
Real-life example: Email Spam Detection
An email application can use AI, ML, and NLP together to identify unwanted emails.
1. AI provides the overall intelligent system.
2. NLP analyzes the words and sentences contained in emails.
3. ML learns patterns from previously classified emails and predicts whether a new email is spam or legitimate.
For example, when an email contains suspicious advertising language, the system analyzes its text and may classify it as spam.

Therefore, AI provides the broader intelligence, NLP handles language processing, and ML helps the system learn from data.
Question 2. Explain the terms corpus, document, token, vocabulary, feature and label. Use one example dataset of your choice to show how each term applies.
Answer:

These six terms describe the basic elements used when preparing and processing text data in NLP.

For this question, consider a dataset containing customer reviews from an online shopping application.
Example dataset
Document ID	Customer review	Label
D1	The delivery was very fast	Positive
D2	The product quality was poor	Negative
D3	I received my order quickly	Positive
D4	The service was disappointing	Negative

Explanation of the terms
Term	Definition	Example
Corpus	A collection of text documents used for analysis.	All four customer reviews.
Document	One individual text within a corpus.	Review D1.
Token	A unit of text produced by a tokenizer.	The, delivery, was, very, fast.
Vocabulary	A collection of unique tokens in a dataset.	delivery, fast, product, quality.
Feature	A numerical value used as input for a machine learning model.	Frequency of the word delivery.
Label	The expected category assigned to an example.	Positive or Negative.

How they work together
First, customer reviews are collected into a corpus. Each review becomes a separate document. The documents are divided into tokens, and unique tokens form the vocabulary.

Next, numerical features are extracted from the documents. Finally, labels are assigned to the reviews to train a supervised machine learning model.

The trained model can then predict whether a new customer review is positive or negative.
Question 3. Describe four common NLP tasks, such as text classification, sequence labeling, search and similarity, and text generation. For each, state what the input is and what the output is.
Answer:

NLP supports several tasks that allow computers to process and generate human language. Four common NLP tasks are explained below.
1. Text Classification
Text classification is the process of assigning a category to an entire text document.

Example: Identifying whether a customer review is positive or negative.
- Input: A customer review.
- Output: Positive, Negative, or Neutral.
2. Sequence Labeling
Sequence labeling assigns labels to individual words or tokens within a sentence.

Example: Identifying names of people and places in a sentence.

Input: "Ahmed studies at Somali National University."
Word	Output label
Ahmed	PERSON
studies	OTHER
at	OTHER
Somali National University	ORGANIZATION

- Input: A sentence or sequence of tokens.
- Output: Labels assigned to relevant words.
3. Search and Similarity
Search and similarity identify documents that are relevant to a user's search query or similar to another document.

Example: A student searches for cybersecurity training opportunities.
- Input: A search query and a collection of documents.
- Output: A ranked list of relevant documents.
4. Text Generation
Text generation is the process of automatically producing new text based on an instruction or existing information.

Example: An AI application generates a summary of a research article.
- Input: A research article and a summarization instruction.
- Output: A newly generated summary.
These tasks are useful in search engines, email applications, translation systems, chatbots, and document management systems.
Question 4. Compare supervised and unsupervised learning in NLP, giving an example of each. Explain why one task needs target labels and the other does not.
Answer:

Supervised and unsupervised learning are two important machine learning approaches used in NLP.
1. Supervised Learning
Supervised learning is a method in which a model learns from training data that contains examples with correct target labels.

Example: Customer Sentiment Classification
Customer review	Label
The product is excellent	Positive
The service is terrible	Negative
I love this product	Positive
I am disappointed	Negative

The model studies the relationship between the text and its assigned labels. After training, it can predict the sentiment of new customer reviews.

Why are labels required?

Labels provide the correct answers that guide the model during training. The model uses them to learn how to predict the required categories.
2. Unsupervised Learning
Unsupervised learning is a method in which a model identifies patterns and relationships in data without receiving target labels for the task.

Example: Customer Review Grouping

An online company collects thousands of customer reviews without assigning positive or negative labels.

The system can group similar reviews into categories based on their content, such as:
- Product quality.
- Delivery problems.
- Payment issues.
- Customer service.
Why are labels not required?

The objective is to discover hidden patterns and similarities in the data rather than predict predefined categories.
Comparison
Supervised learning	Unsupervised learning
Uses labeled data.	Uses data without target labels.
Requires target answers.	Does not require target answers.
Learns to predict known categories.	Discovers hidden patterns.
Example: Sentiment classification.	Example: Customer review clustering.

The main difference is that supervised learning learns from known answers, while unsupervised learning discovers patterns without predefined answers.
Question 5. Explain why text must be converted into numerical representations before most Machine Learning models can use it. Describe one simple representation method and its main limitation.
Answer:

Most traditional Machine Learning algorithms operate on numerical data rather than directly processing raw sentences.

Therefore, NLP uses numerical representation techniques to convert written language into numbers that computers can process.

This process is commonly called text vectorization.

Numerical representations allow machine learning models to identify patterns, compare documents, measure word frequencies, and make predictions.
Bag of Words (BoW)
Bag of Words is a simple technique that converts text into numerical vectors based on the occurrence or frequency of words.

For example, consider these two sentences:
- Document 1: "Students learn Python."
- Document 2: "Students learn Data Science."
First, we create a vocabulary:

["Students", "learn", "Python", "Data", "Science"]

Then, each document is converted into a numerical vector.
Document	Students	learn	Python	Data	Science
Document 1	1	1	1	0	0
Document 2	1	1	0	1	1

The resulting vectors are:
- Document 1: [1, 1, 1, 0, 0]
- Document 2: [1, 1, 0, 1, 1]
Main limitation of Bag of Words
The main limitation of Bag of Words is that it does not preserve word order or fully understand the meaning of sentences.

For example:
- The manager approved the employee.
- The employee approved the manager.
Both sentences contain the same words, but their meanings are different.

Bag of Words may represent both sentences using the same vector because it ignores word order.

Despite this limitation, it is useful for basic text classification and document analysis.
Question 6. Research the challenges of NLP for low-resource languages such as Somali. Discuss at least two challenges (for example, limited datasets, spelling variation, word forms or mixed-language text) and explain how each affects an NLP system.
Answer:

Low-resource languages are languages that have limited digital resources available for developing and evaluating language technologies.

Somali is an example of a language that faces challenges in NLP development compared with languages that have large digital datasets and advanced language-processing tools.

The following are four important challenges.
1. Limited Datasets
One major challenge is the shortage of large and accurately labeled Somali text datasets.

Machine learning models require sufficient examples to learn vocabulary, grammatical structures, and language patterns.

For example, a Somali sentiment analysis system trained using only a small number of customer reviews may not recognize different expressions of positive and negative opinions.

Effects on NLP systems:
- Reduced prediction accuracy.
- Poor performance on unfamiliar sentences.
- Difficulty learning different writing styles.
- Limited ability to generalize to new data.
2. Spelling Variation
Somali speakers may use different spellings when writing online, especially in informal communication.

For example, the same geographical name might appear as "Soomaaliya" or "Somalia."

A basic NLP system may recognize these as two separate vocabulary entries.

Effects on NLP systems:
- Increased vocabulary size.
- Difficulty recognizing similar expressions.
- Reduced search accuracy.
- Incorrect text classification.
Text normalization can help reduce spelling inconsistencies.
3. Morphological Complexity
Somali words can change their forms depending on grammatical structures and word endings.

For example, related forms such as "nin" and "ninka" may be treated as separate tokens by a basic word-based system.

Effects on NLP systems:
- Related words may not be recognized as connected.
- Vocabulary size may increase.
- Search systems may miss relevant documents.
- Models may need more training data.
Stemming and lemmatization can help address this challenge.
4. Mixed-Language Text
Many Somali speakers combine Somali and English when communicating through social media and text messages.

For example:

"Waxaan rabaa inaan computer science barto."

A system designed only for English may struggle to process this sentence correctly.

Effects on NLP systems:
- Incorrect tokenization.
- Difficulty identifying sentence structures.
- Missing vocabulary entries.
- Reduced translation and classification accuracy.
Multilingual NLP models and mixed-language training datasets can help address this problem.
Summary
Challenge	Effect on NLP	Possible solution
Limited datasets	Poor model performance	Collect more training data
Spelling variation	Vocabulary inconsistency	Text normalization
Morphological complexity	Difficulty recognizing word forms	Stemming and lemmatization
Mixed-language text	Incorrect language processing	Multilingual models

These challenges demonstrate why developing effective Somali NLP applications requires language-specific datasets, appropriate preprocessing techniques, and further research.
Question 7. Find one case study (research paper or article) that applies NLP to a real problem, preferably in Somali or another African language. Summarize its task, data, method and findings, and identify which NLP task from question 3 it covers.
Answer:
Case Study: Somali Information Retrieval Corpus
Research title: Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

Authors: Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

Publication: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).
1. Research Task
The main objective of the study was to address the shortage of dedicated Somali-language resources for information retrieval.

Information retrieval is the process of finding relevant documents from a collection based on a user's search query.

The researchers aimed to develop a Somali information retrieval corpus that could support searching for information directly in Somali.
2. Data Used
The researchers developed a dataset containing 2,335 Somali documents.

The documents were collected from different sources, including:
- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.
These documents were used to develop a dedicated resource for Somali information retrieval.
3. Method Used
The study used a technique called Pseudo-Relevance Feedback (PRF) Query Expansion.

This method improves a search query using information obtained from documents initially retrieved by the search system.

The process involves:
1. The user enters a search query.
2. The system retrieves an initial group of documents.
3. Information from those documents is used to expand the original query.
4. The improved query is used to retrieve potentially more relevant documents.
5. Main Findings
The research introduced a dedicated Somali information retrieval corpus containing 2,335 documents.

It demonstrated the importance of developing language-specific resources to improve information retrieval for Somali speakers.

The study also provided a useful foundation for future Somali NLP research and applications.
5. NLP Task Covered
The study mainly covers Search and Similarity (Information Retrieval).
Component	Description
Input	Somali search query
Dataset	2,335 Somali documents
Method	Pseudo-Relevance Feedback Query Expansion
Output	Relevant retrieved documents
NLP task	Search and Similarity

6. Importance of the Study
This research is important because it addresses the limited availability of Somali digital language resources.

It demonstrates how developing dedicated datasets and appropriate retrieval techniques can improve access to information for Somali speakers.

It can also support future research in other Somali NLP applications.

Research source: Badel et al. (2023), Association for Computational Linguistics.
Read the original research paper – ACL Anthology
References
1. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. Proceedings of EMNLP 2023. Association for Computational Linguistics.

2. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit. O'Reilly Media. https://www.nltk.org/book/

3. Jurafsky, D., & Martin, J. H. (2025). Speech and Language Processing (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

4. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-based Word Lemmatization Approach for the Somali Language. arXiv. https://doi.org/10.48550/arXiv.2308.01785

5. Stanford Human-Centered Artificial Intelligence. (n.d.). What is Natural Language Processing (NLP)? https://hai.stanford.edu/ai-definitions/what-is-nlp

6. Stryker, C., & Holdsworth, J. (2024). What is NLP (Natural Language Processing)? IBM. https://www.ibm.com/think/topics/natural-language-processing

7. Bag-of-Words Technique in Natural Language Processing: A Primer for Radiologists. (2021). National Library of Medicine. https://pmc.ncbi.nlm.nih.gov/articles/PMC8415041/