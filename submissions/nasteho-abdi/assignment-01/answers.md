Assignment 1 – NLP Foundations

Course: NLP for Machine Learning Bootcamp
Assignment: Assignment 1 – NLP Foundations
Prepared by: Nasteho Abdi Aden

---

1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) that enables computers to understand, interpret, analyze, and generate human language. It combines computer science, linguistics, and Machine Learning to help computers process languages such as English, Somali, Arabic, and French.

Relationship Between NLP, AI, and Machine Learning

- Artificial Intelligence (AI): A broad field of computer science focused on developing machines that can perform tasks requiring human-like intelligence.
- Machine Learning (ML): A subset of AI that allows computers to learn patterns from data and improve their performance without being explicitly programmed for every situation.
- Natural Language Processing (NLP): A field of AI that focuses on enabling computers to work with human language. Machine Learning is one of the main techniques used to develop modern NLP systems.

Real-Life Example: Email Spam Detection

An email spam detection system demonstrates how AI, ML, and NLP work together.

1. AI provides the overall goal of creating an intelligent system that can identify unwanted emails.
2. NLP processes the email content by analyzing words, sentences, and language patterns.
3. Machine Learning uses previously labeled emails to learn the differences between spam and legitimate messages.
4. The system predicts whether a new email is spam or not spam.

Output: The email is automatically classified as either Spam or Not Spam.

Therefore, AI provides the broader intelligence, NLP handles human language, and ML helps the system learn from examples.

---

2. Explain the Terms: Corpus, Document, Token, Vocabulary, Feature, and Label

These terms are important in NLP because they describe how text data is organized and prepared for Machine Learning.

For this explanation, consider a small dataset containing customer reviews about a mobile application.

Term| Definition| Example
Corpus| A collection of text documents used for NLP research or model training.| A collection of 1,000 customer reviews.
Document| A single piece of text within a corpus.| "This application is very useful."
Token| A unit of text, usually a word or punctuation mark, produced during tokenization.| "This", "application", "is", "very", "useful".
Vocabulary| The set of unique tokens found in a corpus.| {This, application, is, very, useful, bad}.
Feature| A measurable or numerical property extracted from text and used by a Machine Learning model.| The frequency of the word "useful".
Label| The target category or correct answer assigned to a document in supervised learning.| Positive, Negative, or Neutral.

Example Dataset

Consider the following customer reviews:

Document| Review| Label
D1| This application is very useful.| Positive
D2| The application is slow and bad.| Negative
D3| This application is easy to use.| Positive

In this example:

- Corpus: All three customer reviews together.
- Document: Each individual review, such as D1.
- Tokens: Individual words in each review.
- Vocabulary: The unique words appearing across all three reviews.
- Features: Numerical values representing the words, such as word counts.
- Labels: Positive or Negative categories assigned to each review.

These elements help organize text data before training an NLP model.

---

3. Four Common NLP Tasks

NLP supports different tasks that allow computers to process and understand human language. Four common tasks are text classification, sequence labeling, search and similarity, and text generation.

3.1 Text Classification

Text classification is the process of assigning a predefined category to a text document.

Input: A text document or sentence.

Output: A category or label.

Example: An email filtering system receives an email and classifies it as Spam or Not Spam.

Other applications include sentiment analysis, news categorization, and language identification.

3.2 Sequence Labeling

Sequence labeling is an NLP task in which each word or token in a sentence receives a specific label.

Input: A sentence containing multiple words.

Output: A label for each token.

Example:

Sentence: "Ahmed lives in Mogadishu."

Token| Label
Ahmed| PERSON
lives| O
in| O
Mogadishu| LOCATION

This task is commonly used in Named Entity Recognition (NER), Part-of-Speech tagging, and information extraction.

3.3 Search and Similarity

Search and similarity involve finding documents that are relevant or similar to a user's query.

Input: A search query and a collection of documents.

Output: A ranked list of relevant documents.

Example: A student searches for "Machine Learning tutorials" on a search engine. The system identifies and ranks documents that are related to the query.

This task is useful in search engines, digital libraries, and information retrieval systems.

3.4 Text Generation

Text generation is the process of producing new human-readable text based on an input or learned language patterns.

Input: A prompt, question, or starting sentence.

Output: A generated text response.

Example: A user asks an AI chatbot, "What is Artificial Intelligence?" The system generates an explanation in natural language.

Applications include chatbots, automatic translation, text summarization, and writing assistants.

Summary of NLP Tasks

NLP Task| Input| Output
Text Classification| Text document| Category or label
Sequence Labeling| Sentence or sequence of tokens| Label for each token
Search and Similarity| Query and document collection| Ranked relevant documents
Text Generation| Prompt or input text| New text

---

4. Supervised and Unsupervised Learning in NLP

Machine Learning approaches in NLP can be divided into supervised and unsupervised learning.

4.1 Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from training data that contains both input examples and their correct target labels.

The model studies the relationship between the text and its labels so that it can predict labels for new, unseen text.

Example: Sentiment Analysis

A model is trained using customer reviews that have already been classified as Positive or Negative.

Text| Label
I love this product.| Positive
This product is terrible.| Negative
The service is excellent.| Positive

The model learns from these examples and predicts the sentiment of new reviews.

Why are labels needed?

Labels are necessary because they provide the correct answers during training. They allow the model to compare its predictions with the expected results and learn how to make better predictions.

4.2 Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model works with data that does not contain predefined target labels.

Instead of learning from correct answers, the model discovers patterns, relationships, and similarities within the data.

Example: Customer Review Clustering

A company collects thousands of customer reviews without assigning categories.

An unsupervised learning algorithm can group similar reviews into clusters, such as:

- Cluster 1: Reviews about application performance.
- Cluster 2: Reviews about customer service.
- Cluster 3: Reviews about pricing.

Why are labels not needed?

Labels are not required because the main objective is to discover hidden patterns or structures in the data rather than predict predefined categories.

Comparison

Supervised Learning| Unsupervised Learning
Uses labeled data.| Uses unlabeled data.
Learns from correct examples.| Discovers hidden patterns.
Predicts predefined categories or values.| Groups or organizes similar data.
Example: Sentiment classification.| Example: Document clustering.

In conclusion, supervised learning requires target labels because it learns to predict known outcomes, while unsupervised learning does not require labels because it searches for patterns independently.

---

5. Why Must Text Be Converted into Numerical Representations?

Most Machine Learning algorithms operate on mathematical values rather than directly understanding written words and sentences.

Human language consists of characters, words, and sentences. However, Machine Learning models require numerical inputs to perform mathematical operations, identify patterns, calculate relationships, and make predictions.

Therefore, text must be converted into numerical representations before it can be processed by most traditional Machine Learning models.

Bag-of-Words Representation

One simple method of converting text into numerical data is called Bag of Words (BoW).

Bag of Words represents each document by counting how many times each vocabulary word appears. It creates a numerical vector for every document.

Example Dataset:

- D1: "I love machine learning."
- D2: "I love natural language."

Vocabulary:

[I, love, machine, learning, natural, language]

Numerical representation:

Document| I| love| machine| learning| natural| language
D1| 1| 1| 1| 1| 0| 0
D2| 1| 1| 0| 0| 1| 1

The numbers represent how many times each word appears in each document.

Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and context.

For example:

- "The dog bites the man."
- "The man bites the dog."

These sentences contain almost the same words, but their meanings are different. Bag of Words may represent them with identical vectors because it only counts word occurrences.

Other limitations include large vocabulary sizes and difficulty handling words that were not present in the training vocabulary.

Despite these limitations, Bag of Words remains a simple and useful method for basic text classification and other NLP applications.

---

6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited digital text, annotated datasets, language-processing tools, and computational resources available for NLP development.

Somali is an example of a language that has historically received less computational research attention compared with languages such as English, French, and Chinese.

Several challenges affect the development of effective NLP systems for Somali.

6.1 Limited Datasets

One major challenge is the shortage of large, high-quality, and properly labeled Somali text datasets.

Many modern NLP models require large amounts of text to learn language patterns effectively. However, Somali has fewer publicly available annotated datasets for tasks such as sentiment analysis, text classification, and Named Entity Recognition.

Effect on NLP systems:

- Models may not learn enough Somali language patterns.
- Predictions may become less accurate.
- Systems may perform poorly when processing unfamiliar text.
- Researchers may struggle to evaluate their models properly.

For example, a sentiment analysis model trained using only a small number of Somali reviews may fail to understand many expressions used in everyday communication.

6.2 Spelling Variation and Non-Standard Writing

Somali users may write the same word or expression differently, especially in informal social media messages and text conversations.

People may also use abbreviations, missing vowels, or inconsistent spelling.

Effect on NLP systems:

- The same word may be treated as several different tokens.
- Vocabulary size may increase unnecessarily.
- Search systems may fail to find relevant documents.
- Text classification models may struggle to identify similar meanings.

For example, a system may fail to recognize two differently written forms as the same intended word.

6.3 Morphological Complexity and Word Forms

Somali has rich morphology, meaning that words can take different forms through grammatical changes and derivation.

A single root may produce multiple related word forms.

Effect on NLP systems:

- A model may treat related forms as unrelated words.
- Vocabulary size may become larger.
- Search results may miss documents containing alternative word forms.
- Accurate tokenization and lemmatization become more challenging.

Lemmatization can help by reducing related word forms to their root or dictionary form.

6.4 Code-Switching and Mixed-Language Text

Somali speakers sometimes combine Somali with English or Arabic, particularly in social media posts and informal digital communication.

For example:

"Maanta assignment-ka Machine Learning ayaan dhammeeyay."

This sentence combines Somali and English terminology.

Effect on NLP systems:

- Language identification becomes more difficult.
- A model trained only on standard Somali may misunderstand English terms.
- Tokenization and language processing become less reliable.
- Sentiment and meaning may be lost when words from different languages are mixed.

Possible Solutions

Researchers can improve Somali NLP by:

1. Collecting more Somali digital text from reliable sources.
2. Creating high-quality datasets annotated by native Somali speakers.
3. Developing Somali tokenizers, stemmers, and lemmatizers.
4. Including different Somali dialects in training datasets.
5. Developing multilingual models that can handle mixed-language text.
6. Encouraging collaboration between Somali linguists and Machine Learning researchers.

These improvements can make NLP systems more accurate and useful for Somali-speaking communities.

---

7. Case Study: Somali Information Retrieval Corpus

Research Paper

Title: Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

Authors: Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

Publication: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).

Source: https://aclanthology.org/2023.emnlp-main.462/

7.1 Research Problem and Task

The research focuses on improving information retrieval for the Somali language.

Information retrieval is the process of finding relevant documents from a collection based on a user's search query.

Before this research, Somali information retrieval research was limited and often depended on translating Somali queries into other languages because dedicated Somali text resources were insufficient.

The researchers aimed to address this problem by developing a dedicated Somali information retrieval corpus.

7.2 Data Used

The researchers created an annotated corpus containing 2,335 Somali documents.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The dataset provided Somali-language resources that could be used to develop and evaluate information retrieval systems.

7.3 Method Used

The researchers developed a Somali information retrieval system using a technique called Pseudo-Relevance Feedback (PRF) for query expansion.

The method works by using initially retrieved documents to identify additional useful terms related to the original search query.

These terms are then used to expand the query and improve the retrieval process.

In simple terms:

1. A user enters a search query.
2. The system retrieves an initial set of documents.
3. The system identifies useful related terms from the retrieved documents.
4. The query is expanded using those terms.
5. The system performs an improved search.

7.4 Findings and Contributions

The study contributed a dedicated Somali information retrieval corpus containing 2,335 documents.

It demonstrated the importance of creating language-specific resources rather than depending entirely on query translation.

The research also provided a foundation for further Somali NLP applications, including search systems, question answering, and text classification.

The availability of such datasets can help researchers develop more effective tools for processing Somali text.

7.5 Which NLP Task Does It Cover?

This research covers Search and Similarity (Information Retrieval), which is one of the four NLP tasks discussed in Question 3.

The input is a user's search query and a collection of Somali documents.

The output is a ranked collection of documents that are relevant to the query.

7.6 Conclusion of the Case Study

This case study demonstrates that NLP can solve practical problems affecting Somali-speaking communities.

By developing a dedicated Somali information retrieval corpus, the researchers addressed an important resource gap and provided a foundation for future research.

It also shows that collecting and organizing language data is an essential step toward developing effective NLP systems for low-resource languages.

---

Conclusion

Natural Language Processing is an important field of Artificial Intelligence that enables computers to work with human language.

This assignment explored the relationship between AI, Machine Learning, and NLP, introduced important text-processing concepts, and explained four common NLP tasks.

It also compared supervised and unsupervised learning and demonstrated why numerical text representations are necessary for Machine Learning models.

Finally, the assignment discussed the challenges of developing NLP systems for low-resource languages such as Somali and examined a real research study focused on Somali information retrieval.

Overall, developing better datasets, language-processing tools, and research methods can help improve NLP technologies and make them more accessible to Somali-speaking communities.

---

References

1. Jurafsky, D., & Martin, J. H. (2025). Speech and Language Processing (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
2. Eisenstein, J. (2019). Introduction to Natural Language Processing. MIT Press. https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing/
3. Eisenstein, J. (2018). Natural Language Processing. Georgia Institute of Technology. https://github.com/jacobeisenstein/gt-nlp-class
4. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. Proceedings of EMNLP 2023, 7463–7469. https://aclanthology.org/2023.emnlp-main.462/
5. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-based Word Lemmatization Approach for the Somali Language. International Conference on Learning Representations Workshop. https://arxiv.org/abs/2308.01785
6. Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. Proceedings of EMNLP 2023. https://aclanthology.org/2023.emnlp-main.862/