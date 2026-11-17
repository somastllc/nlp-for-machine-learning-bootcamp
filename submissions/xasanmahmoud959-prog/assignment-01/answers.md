
Assignment 1 – NLP Foundations

1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that helps computers understand, process, and generate human languages such as English and Somali.

NLP is part of AI, while Machine Learning (ML) is one of the methods used to develop NLP systems.

- Artificial Intelligence (AI): Makes computers perform tasks that normally require human intelligence.
- Machine Learning (ML): Allows computers to learn patterns from data.
- Natural Language Processing (NLP): Helps computers work with human language.

Real-life example:

Consider a language translation application.

1. AI provides the overall intelligent system.
2. Machine Learning helps the system learn from language examples.
3. NLP processes the words and sentences.
4. The application produces a translation in another language.

Therefore, AI, ML, and NLP work together to make language applications possible.

2. Explain the Main NLP Terms

The following terms are important when working with text data.

Term| Meaning
Corpus| A collection of text documents used for language research or training.
Document| One individual text inside a corpus.
Token| A single unit of text, such as a word or punctuation mark.
Vocabulary| The collection of unique tokens in a dataset.
Feature| Information represented as input for a Machine Learning model.
Label| The correct category or answer assigned to a document.

Example dataset: Movie Reviews

Consider these three reviews:

1. "This movie is excellent."
2. "This movie is boring."
3. "The movie is interesting."

- Corpus: All three reviews together.
- Document: One individual movie review.
- Token: Words such as "movie", "excellent", and "boring".
- Vocabulary: The unique words found in all three reviews.
- Feature: Numerical information representing the words in each review.
- Label: The category assigned to each review, such as Positive or Negative.

Example:

Document| Text| Label
D1| This movie is excellent.| Positive
D2| This movie is boring.| Negative
D3| The movie is interesting.| Positive

This dataset can help a model learn how to classify movie reviews.

3. Four Common NLP Tasks

A. Text Classification

Text classification assigns a category or label to an entire text.

- Input: A sentence, message, or document.
- Output: A category or label.

Example: Classifying an email as Spam or Not Spam.

B. Sequence Labeling

Sequence labeling assigns a label to each word or token in a sentence.

- Input: A sentence containing several words.
- Output: A label for each token.

Example:

Sentence: "Hassan lives in Somalia."

Word| Label
Hassan| Person
lives| Other
in| Other
Somalia| Location

C. Search and Similarity

This task helps computers find documents that are related to a user's search or question.

- Input: A search query or text.
- Output: Relevant documents or similar texts.

Example: Searching for information about Artificial Intelligence and receiving related articles.

D. Text Generation

Text generation allows a computer to produce new human-language text.

- Input: A prompt or instruction.
- Output: Newly generated text.

Example: ChatGPT receives a question and generates an answer.

4. Supervised and Unsupervised Learning

Machine Learning has different learning methods. Two common types are supervised and unsupervised learning.

Supervised Learning| Unsupervised Learning
Uses labeled data.| Uses unlabeled data.
Learns from examples with correct answers.| Finds patterns in data without given answers.
Predicts known categories or values.| Discovers groups or hidden structures.

Supervised Learning Example:

Email classification.

The training dataset contains emails labeled as Spam or Not Spam.

The model learns from these examples and predicts the category of new emails.

Unsupervised Learning Example:

Grouping news articles.

The model receives many articles without labels and groups similar articles together based on their content.

Why are labels needed in one but not the other?

Supervised learning needs target labels because the model must learn the relationship between input data and correct answers.

Unsupervised learning does not need target labels because its purpose is to discover patterns and relationships directly from the data.

5. Why Must Text Be Converted into Numbers?

Most Machine Learning models cannot directly understand normal human text.

Computers process numerical values, so text must first be converted into numerical representations.

This process allows a model to identify patterns and relationships between words.

Bag of Words (BoW)

Bag of Words is a simple method that represents text by counting how many times each vocabulary word appears in a document.

Example:

Document 1: "I like AI"

Document 2: "I like Python"

Vocabulary: [I, like, AI, Python]

Document| I| like| AI| Python
D1| 1| 1| 1| 0
D2| 1| 1| 0| 1

The numbers represent the frequency of each word.

Main limitation:

Bag of Words does not understand the meaning or order of words. It only counts word occurrences.

For example, two sentences with the same words in different orders may receive the same representation.

6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited digital text, datasets, and language-processing tools.

Somali faces several challenges in NLP.

A. Limited Datasets

There are fewer large, high-quality, labeled Somali datasets compared with English datasets.

This makes it difficult to train and evaluate accurate NLP models.

Effect on NLP systems:

Models may produce incorrect results because they have not learned enough Somali language examples.

B. Spelling Variations

People may write the same Somali word in different ways, especially on social media and in informal messages.

Effect on NLP systems:

The model may treat different spellings as separate words, making it difficult to recognize their similar meanings.

C. Word Forms and Morphology

Somali words can have different forms because of grammatical changes and word formation.

Effect on NLP systems:

A model may fail to recognize that different word forms are related to the same root word.

D. Mixed-Language Text

Some Somali speakers mix Somali and English in the same sentence.

For example:

"Waxaan rabaa inaan learn programming."

Effect on NLP systems:

A model trained mainly on standard Somali may have difficulty understanding sentences containing both languages.

These challenges show why Somali needs more datasets, language resources, and specialized NLP research.

7. NLP Case Study: Somali Information Retrieval

Research title: Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

Authors: Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

Year: 2023.

A. Task

The research focuses on building a Somali Information Retrieval system.

Information Retrieval helps users find relevant documents based on their search queries.

B. Data

The researchers created a Somali dataset containing 2,335 documents.

The documents were collected from different sources, including Somali news websites and poetry books.

C. Method

The researchers worked with language experts to create an annotated corpus.

They developed an information retrieval system using a technique called Pseudo-Relevance Feedback (PRF).

This technique expands a search query using information from initially retrieved documents.

D. Findings

The research provided a dedicated Somali information retrieval dataset and system.

It also helped address the lack of digital Somali language resources.

The dataset can support future research and the development of other Somali NLP applications.

E. Which NLP Task Does It Cover?

This research covers Search and Similarity (Information Retrieval) because its main purpose is to find relevant Somali documents based on user queries.

Conclusion

Natural Language Processing helps computers work with human language.

It uses Machine Learning techniques to process text, identify patterns, classify information, search documents, and generate new text.

However, low-resource languages such as Somali still need more datasets and research to improve NLP systems.

Understanding these foundations is important for developing better language technologies.

References

1. Jurafsky, D., & Martin, J. H. (2026). Speech and Language Processing (3rd ed. draft).

2. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. Proceedings of EMNLP 2023.

3. Duh, K., McNamee, P., Post, M., & Thompson, B. (2020). Benchmarking Neural and Statistical Machine Translation on Low-Resource African Languages. Proceedings of LREC 2020.

4. Nekoto, W., et al. (2020). Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages. Findings of EMNLP 2020.

5. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-based Word Lemmatization Approach for the Somali Language.



Note: I used ChatGPT to help me conduct extensive research for this assignment, understand the concepts, and find relevant academic sources.

