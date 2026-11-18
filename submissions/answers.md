# Assignment 1 – NLP Foundations

**Student Name:** Idman Ahmed Ali  
**Course:** Natural Language Processing (NLP)  
**Assignment Title:** NLP Foundations

---

## 1. Definition of Natural Language Processing (NLP)

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that enables computers to understand, interpret, analyze, and generate human languages such as English, Somali, and Arabic.

NLP combines computer science, linguistics, mathematics, and Machine Learning to develop systems that can process human communication in written or spoken form.

### Relationship Between NLP, AI, and Machine Learning

Artificial Intelligence, Machine Learning, and NLP are closely related, but they have different purposes.

- **Artificial Intelligence (AI):** A broad field of computer science that focuses on developing machines capable of performing tasks that normally require human intelligence.
- **Machine Learning (ML):** A branch of AI that enables computers to learn patterns from data and make predictions without being explicitly programmed for every situation.
- **Natural Language Processing (NLP):** A field of AI that focuses on enabling computers to understand and process human languages. Machine Learning provides techniques that are commonly used to develop NLP applications.

Therefore, AI is the broader concept, Machine Learning provides learning methods, and NLP applies computational techniques to human language.

### Real-Life Example: Email Spam Detection

An email spam detection system demonstrates how AI, ML, and NLP work together.

1. NLP processes the text of incoming emails and identifies language patterns.
2. Machine Learning learns from previously classified emails labeled as spam or not spam.
3. The AI application uses the trained model to classify new emails automatically.

For example, an email containing suspicious offers, misleading messages, and unusual links may be classified as spam.

This demonstrates how NLP processes language, Machine Learning learns from examples, and AI uses the learned capability to perform an intelligent task.

---

## 2. Explanation of Core NLP Terms

Understanding the basic terminology of NLP is important because these concepts describe how text data is organized and prepared for Machine Learning.

For this question, consider a small email classification dataset containing three email messages.

### Example Dataset

| Document | Email Text | Label |
|---|---|---|
| D1 | Win a free phone today | Spam |
| D2 | Your university examination starts tomorrow | Not Spam |
| D3 | Claim your free reward now | Spam |

The following terms explain the different components of this dataset.

### 2.1 Corpus

A corpus is a collection of written or spoken language data organized for linguistic analysis or for developing and evaluating NLP systems.

In the example, all three email messages together form an email corpus.

### 2.2 Document

A document is one individual piece of text within a corpus. It can be an email, article, paragraph, or social media post.

For example, "Win a free phone today" is one document in the email corpus.

### 2.3 Token

A token is an individual unit produced when text is divided into smaller parts during tokenization. Tokens can be words, punctuation marks, or subword units.

For example:

Original sentence: "Win a free phone today"

Tokens: `["Win", "a", "free", "phone", "today"]`

This sentence contains five word tokens.

### 2.4 Vocabulary

Vocabulary is the collection of unique tokens found in a corpus after text preprocessing.

Using lowercase normalization, the vocabulary of the example dataset is:

`["win", "a", "free", "phone", "today", "your", "university", "examination", "starts", "tomorrow", "claim", "reward", "now"]`

The vocabulary contains 13 unique words.

### 2.5 Feature

A feature is a measurable property or numerical representation of a document that a Machine Learning model uses to identify patterns and make predictions.

For example, the presence of the word "free" can be represented as a binary feature:

- 1 means the word is present.
- 0 means the word is absent.

### 2.6 Label

A label is the target category assigned to a training example in supervised Machine Learning.

In the example dataset:

- D1 has the label Spam.
- D2 has the label Not Spam.
- D3 has the label Spam.

The labels provide the correct answers that help the model learn how to classify new emails.

---

## 3. Four Common NLP Tasks

Natural Language Processing includes different tasks that enable computers to perform specific operations on human language.

Four important NLP tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is an NLP task that assigns a predefined category to an entire text document.

It is commonly used in spam detection, sentiment analysis, news categorization, and harmful-content detection.

**Input:** A text document or sentence.

**Output:** A category or class assigned to the text.

**Example:**

Input: "I really enjoyed this course."

Output: Positive sentiment.

Another example is an email classification system that identifies whether a message is Spam or Not Spam.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which a system assigns a label to each token in a sentence rather than assigning one label to the entire document.

It is commonly used in Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.

**Input:** A sentence divided into tokens.

**Output:** A corresponding label for each token.

**Example:**

Input: "Idman studies at Jamhuriya University."

| Token | Label |
|---|---|
| Idman | B-PER |
| studies | O |
| at | O |
| Jamhuriya | B-ORG |
| University | I-ORG |

Here, B-PER identifies the beginning of a person's name, B-ORG identifies the beginning of an organization's name, I-ORG continues the organization name, and O means that the token does not belong to a named entity.

This task helps computers identify people, organizations, locations, and other important information in text.

### 3.3 Search and Similarity

Search and similarity is an NLP task that helps systems find documents relevant to a user's query or determine how similar two pieces of text are in meaning.

It is useful in search engines, digital libraries, document retrieval, and question-answering applications.

**Input:** A search query and a collection of documents.

**Output:** A ranked list of relevant documents or similarity scores.

**Example:**

Query: "How can students improve their programming skills?"

A search system may return documents about programming tutorials, coding exercises, and software development courses.

The system attempts to identify relevant information even when the exact words used in the documents are different from the search query.

### 3.4 Text Generation

Text generation is an NLP task that enables computers to produce new human-readable text based on an instruction, question, or previous context.

It is commonly used in chatbots, AI assistants, automatic report writing, and language generation applications.

**Input:** A prompt, question, or previous conversation.

**Output:** Newly generated text.

**Example:**

Input: "Explain the importance of education."

Output: "Education plays an important role in developing knowledge, improving skills, and creating better opportunities for individuals and communities."

Modern language models use neural network architectures to generate contextually relevant responses.

### Comparison of the Four NLP Tasks

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | Text document | Category or class |
| Sequence Labeling | Sequence of tokens | Label for each token |
| Search and Similarity | Query and documents | Ranked documents or similarity scores |
| Text Generation | Prompt or context | Newly generated text |

---

## 4. Supervised and Unsupervised Learning in NLP

Machine Learning approaches can be divided into different learning types depending on whether the training data contains predefined target labels.

### 4.1 Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from training data containing both input examples and their correct target labels.

During training, the model compares its predictions with the correct answers and adjusts its parameters to reduce prediction errors.

**NLP Example: Sentiment Analysis**

Consider a dataset containing customer reviews with sentiment labels.

| Text | Label |
|---|---|
| This product is excellent | Positive |
| I hate this service | Negative |
| The experience was wonderful | Positive |

The model learns the relationship between words and their sentiment categories.

When a new review is submitted, the model predicts whether the review expresses positive or negative sentiment.

**Why are target labels needed?**

Target labels are needed because they provide the correct answers that guide the model during training. The model uses these answers to learn how to predict the appropriate category for new examples.

### 4.2 Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model receives data without predefined target labels and attempts to discover patterns, structures, or relationships within the data.

**NLP Example: Document Clustering**

Imagine a collection of 1,000 news articles without category labels.

An unsupervised learning algorithm can analyze the words and similarities between the articles and group related documents together.

For example:

- Group 1: Articles about education.
- Group 2: Articles about sports.
- Group 3: Articles about technology.

The model discovers these groups from the data rather than receiving predefined categories.

**Why are target labels not needed?**

The purpose of unsupervised learning is to discover hidden structures in the available data rather than predict previously supplied correct answers. Therefore, predefined target labels are not required.

### Comparison Between Supervised and Unsupervised Learning

| Feature | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Target labels | Required | Not required |
| Main purpose | Predict known outcomes | Discover hidden patterns |
| NLP example | Spam classification | Document clustering |
| Learning approach | Learns from correct examples | Learns from data structures |

In conclusion, supervised learning is appropriate when labeled examples are available and the goal is to predict specific categories. Unsupervised learning is useful when the objective is to explore data and discover patterns without predefined answers.

---

## 5. Why Text Must Be Converted into Numerical Representations

Most Machine Learning algorithms cannot directly process ordinary human-readable text because their mathematical operations require numerical inputs.

Text contains words, sentences, and symbols, whereas Machine Learning algorithms generally perform calculations using numbers, vectors, and matrices.

Therefore, NLP systems convert text into numerical representations before most Machine Learning models can process it.

This process is commonly known as text vectorization or text representation.

### Bag-of-Words Representation

One simple method of converting text into numbers is called Bag of Words (BoW).

Bag of Words represents each document by counting how many times each vocabulary word appears. It creates a numerical vector for every document.

Consider these two sentences:

- Document 1: "I love machine learning."
- Document 2: "I love natural language processing."

First, create a vocabulary:

`["I", "love", "machine", "learning", "natural", "language", "processing"]`

Next, count the occurrences of each word.

| Document | I | love | machine | learning | natural | language | processing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| Document 2 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

The original sentences have now been converted into numerical vectors that Machine Learning algorithms can process.

### Main Limitation of Bag of Words

The main limitation of Bag of Words is that it ignores word order and much of the contextual meaning of a sentence.

For example:

- "The student teaches the teacher."
- "The teacher teaches the student."

These sentences contain the same words but have different meanings.

Bag of Words produces identical word-count representations for both sentences because it does not preserve the order of words.

Other limitations include:

- Large vocabulary sizes can produce high-dimensional vectors.
- Unseen words may not be represented correctly.
- The method does not capture deeper semantic relationships between words.

Despite these limitations, Bag of Words remains a useful basic method for understanding how text can be represented numerically.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited digital text collections, annotated datasets, linguistic tools, and computational resources available for developing NLP applications.

Somali is an example of a language that faces several challenges in modern NLP research.

Although Somali is spoken by millions of people, the availability of high-quality digital language resources is limited compared with languages such as English.

### 6.1 Limited Datasets and Annotated Language Resources

One major challenge is the lack of large, diverse, and accurately annotated Somali datasets.

Many NLP models require substantial amounts of training data to learn language patterns and produce reliable results.

For example, developing a Somali sentiment analysis system requires many Somali sentences labeled as positive, negative, or neutral.

However, collecting and manually labeling such data requires time, language expertise, and financial resources.

**Effects on NLP systems:**

- Models may not learn enough Somali language patterns.
- Predictions may become inaccurate when processing unfamiliar text.
- Models may perform poorly on new topics.
- Systems may become biased toward the limited data available during training.

### 6.2 Spelling Variation and Informal Writing

Another challenge is inconsistent spelling and informal writing, especially on social media platforms.

People may write the same Somali word differently because of typing habits, regional differences, abbreviations, or inconsistent digital writing practices.

For example, the standard spelling "macallin" may appear in informal writing as "macalin".

These differences can make it difficult for NLP systems to recognize that different written forms may communicate the same meaning.

**Effects on NLP systems:**

- The system may treat similar words as different tokens.
- Vocabulary size may increase unnecessarily.
- Search systems may fail to retrieve relevant documents.
- Classification models may produce incorrect predictions when encountering unfamiliar spellings.

### 6.3 Complex Word Forms and Morphology

Somali has a rich morphological structure. Words can have different grammatical forms and derivational changes.

For example:

- "Macallin" refers to one teacher.
- "Macallimiin" refers to multiple teachers.

Both words are related, but their written forms are different.

An NLP system that does not understand these relationships may treat them as unrelated words.

**Effects on NLP systems:**

- Models may need more training examples to learn different word forms.
- Search engines may miss documents containing related forms.
- Text classification and language analysis may become less accurate.

### 6.4 Mixed-Language Text and Code-Switching

Somali speakers sometimes combine Somali with English or Arabic in the same sentence, particularly in online communication.

For example:

"Waxaan maanta sameeyay programming assignment."

This sentence combines Somali and English.

A language-processing system trained only on formal Somali may struggle to understand such mixed-language content.

**Effects on NLP systems:**

- Language identification becomes more difficult.
- Tokenization and text normalization may become less reliable.
- Models may misunderstand the meaning of sentences.
- Sentiment analysis and text classification may produce incorrect results.

### Summary of Somali NLP Challenges

| Challenge | Main Problem | Effect on NLP |
|---|---|---|
| Limited datasets | Insufficient annotated Somali text | Weak model performance |
| Spelling variation | Inconsistent written forms | Tokenization and classification errors |
| Complex morphology | Multiple forms of related words | Vocabulary and search difficulties |
| Code-switching | Mixing Somali with other languages | Language identification and understanding errors |

These challenges demonstrate why developing Somali NLP requires more local datasets, improved language resources, linguistic expertise, and models trained specifically for Somali language characteristics.

---

## 7. Case Study: Somali Information Retrieval Corpus

### Research Paper Information

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

**Authors:** Abdisalam Mahamed Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

**Year:** 2023.

**DOI:** https://doi.org/10.18653/v1/2023.emnlp-main.462

### 7.1 Research Problem

The researchers identified that Somali information retrieval research was limited because there were not enough dedicated Somali-language digital datasets.

Many existing approaches relied on translating search queries into other languages instead of searching directly within Somali-language resources.

This created difficulties for users who wanted to find relevant information written in Somali.

### 7.2 Task

The main objective was to develop a Somali Information Retrieval Corpus and investigate methods for retrieving relevant documents using Somali search queries.

The researchers aimed to provide a language-specific resource that could support the development of better Somali search systems.

### 7.3 Data Used

The researchers created an annotated corpus containing **2,335 Somali documents**.

The documents were collected from different online sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The dataset provided Somali-language documents that could be used to investigate information retrieval techniques.

### 7.4 Method Used

The researchers developed a Somali information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF) Query Expansion**.

This technique works by using initially retrieved documents to identify additional useful terms related to the original search query.

The system then expands the original query with these terms to improve document retrieval.

In simple terms, the system attempts to improve search results by identifying additional relevant words from the documents it initially retrieves.

### 7.5 Findings and Contributions

The research contributed a dedicated Somali information retrieval corpus containing 2,335 documents.

It demonstrated the importance of developing language-specific datasets instead of depending entirely on translated search queries.

The researchers explained that collecting and organizing Somali-language data can support the development of additional NLP applications, including information retrieval, question answering, and text classification.

The study provides a useful foundation for further research into Somali-language search technologies.

### 7.6 Which NLP Task Does This Case Study Cover?

This research mainly covers **Search and Similarity (Information Retrieval)**, which is one of the four NLP tasks discussed in Question 3.

The system receives a Somali search query and attempts to retrieve documents that are relevant to that query.

### 7.7 Importance of This Research

This research is important because it addresses a real problem affecting Somali-language digital information access.

By developing a dedicated Somali corpus, researchers provide a foundation for building search engines and other language technologies that can serve Somali-speaking communities.

It also demonstrates how academic research can contribute to improving NLP resources for low-resource languages.

---

## Conclusion

Natural Language Processing is an important field of Artificial Intelligence that allows computers to understand, analyze, and generate human language.

This assignment has explored the relationship between NLP, AI, and Machine Learning, together with the fundamental concepts of text data, common NLP tasks, and different learning approaches.

It has also explained why numerical text representations are necessary for Machine Learning and examined the challenges associated with developing NLP systems for low-resource languages such as Somali.

Finally, the Somali Information Retrieval Corpus research demonstrates that developing local language datasets is essential for improving language technologies and making digital information more accessible.

Through these concepts and examples, it is clear that NLP has significant potential to solve real-world communication and information-access problems, particularly in languages that remain underrepresented in modern Artificial Intelligence research.

---

## References

1. Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

2. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press. https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing/

3. Badel, A. M., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462

4. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and Rule-based Word Lemmatization Approach for the Somali Language. *ICLR Workshop*. https://arxiv.org/abs/2308.01785

5. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. *Proceedings of the 7th Workshop on African Natural Language Processing*, 179–189. https://doi.org/10.18653/v1/2026.africanlp-main.17
