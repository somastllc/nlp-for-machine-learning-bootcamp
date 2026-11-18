# Assignment 1 – NLP Foundations

**Student:** Atika Isse Ali  
**GitHub:** [atikamTech](https://github.com/atikamTech)  
**Course:** NLP for Machine Learning Bootcamp

---

## 1. Definition of Natural Language Processing (NLP)

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that focuses on enabling computers to work with human language. It allows computer systems to process, analyze, understand and generate text or speech in languages that people use to communicate.

Human language is complex because words can have different meanings depending on their context. People also express the same ideas in different ways, using different words, sentence structures and writing styles. NLP provides methods that help computers handle these challenges and perform language-related tasks.

### Relationship between AI, Machine Learning and NLP

Artificial Intelligence, Machine Learning and NLP are closely connected, but they have different meanings.

- **Artificial Intelligence (AI):** A broad field of computer science concerned with building systems that can perform tasks normally associated with human intelligence, such as reasoning, problem-solving and language understanding.
- **Machine Learning (ML):** A branch of AI that enables computers to learn patterns from data and use those patterns to make predictions or decisions without requiring programmers to write a separate rule for every situation.
- **Natural Language Processing (NLP):** A field of AI concerned with processing and working with human language. It uses different approaches, including machine learning, linguistic rules and statistical methods.

Machine learning is commonly used in NLP because it allows systems to learn language patterns from examples. However, not every NLP system requires machine learning. Some systems use dictionaries, predefined linguistic rules or other traditional methods.

### Real-life example: Email spam detection

An email spam filter demonstrates how AI, ML and NLP can work together.

Imagine receiving hundreds of emails, including advertisements, unwanted messages and important personal communication. An email filtering system can automatically identify messages that are likely to be spam.

The process can involve the following:

1. **NLP:** The system processes the email text and examines language-related information, such as words, phrases and message content.
2. **Machine Learning:** The model learns patterns from previously labeled spam and legitimate emails. For example, it may learn that certain combinations of words are frequently associated with unwanted messages.
3. **Artificial Intelligence:** The complete system uses its processing and learned patterns to make decisions and classify incoming messages.

For example, an email containing suspicious promotional language and misleading links may be classified as spam, while a normal message from a colleague may be classified as legitimate.

This example demonstrates how NLP helps a computer work with human language, machine learning helps it recognize patterns from data, and AI provides the broader field in which these technologies are developed.

---

## 2. Core NLP Terminology

Before developing an NLP system, it is important to understand how text data is organized and represented. The terms corpus, document, token, vocabulary, feature and label describe different parts of the data used in NLP.

For this question, I will use a small example dataset containing customer reviews of an online shopping service.

### Example dataset

| ID | Customer review | Label |
|---|---|---|
| 1 | The delivery was fast and excellent. | Positive |
| 2 | The product quality was poor. | Negative |
| 3 | I received my order on time. | Positive |
| 4 | The delivery was late and disappointing. | Negative |

### Explanation of the terms

**1. Corpus**

A corpus is a collection of text documents gathered for language analysis or NLP research. It provides the data researchers use to study language, develop language-processing methods or train and evaluate NLP systems.

In this example, all four customer reviews together form the corpus.

**2. Document**

A document is one individual unit of text within a corpus. Depending on the dataset, a document can be a sentence, customer review, article, email or complete book.

In the example, each customer review represents one document. Therefore, the dataset contains four documents.

**3. Token**

A token is an individual unit produced when text is divided into smaller pieces through a process called tokenization. Tokens can be complete words, punctuation marks or smaller subword units, depending on the tokenization method.

For example, the sentence:

"The delivery was fast."

Can be divided into these tokens:

`["The", "delivery", "was", "fast", "."]`

This sentence contains five tokens, including the punctuation mark.

**4. Vocabulary**

Vocabulary refers to the collection of unique tokens found in a corpus after applying a chosen tokenization and normalization method.

For example, if we consider the customer reviews and convert words to lowercase while excluding punctuation, the vocabulary is:

`{the, delivery, was, fast, and, excellent, product, quality, poor, i, received, my, order, on, time, late, disappointing}`

Repeated words are included only once in the vocabulary. This means that even if the word "delivery" appears in multiple reviews, it is represented once in the vocabulary.

**5. Feature**

A feature is a measurable or numerical representation of information that a machine learning model uses to identify patterns and make predictions.

For example, the number of times a particular word appears in a customer review can be used as a feature.

The following table shows the word-count features for four reviews:

| Review | fast | excellent | poor | late |
|---|---:|---:|---:|---:|
| Review 1 | 1 | 1 | 0 | 0 |
| Review 2 | 0 | 0 | 1 | 0 |
| Review 3 | 0 | 0 | 0 | 0 |
| Review 4 | 0 | 0 | 0 | 1 |

The numbers indicate whether and how frequently each selected word appears in a review. These values can be used as numerical input features for a machine learning model.

**6. Label**

A label is the expected category or correct output associated with a data example. In supervised learning, labels help a model learn the relationship between input data and the desired output.

In this dataset, the labels are Positive and Negative. They indicate the sentiment expressed in each customer review.

### Summary of the terminology

| Term | Meaning | Example |
|---|---|---|
| Corpus | Collection of documents | All four reviews |
| Document | One individual text | Review 1 |
| Token | A unit of text | "delivery" |
| Vocabulary | Unique tokens in the corpus | Unique review words |
| Feature | Numerical information used by a model | Count of "excellent" |
| Label | Expected category or output | Positive |

These concepts are connected. A corpus contains documents, documents are divided into tokens, and tokens contribute to a vocabulary. Features represent information numerically, while labels provide the expected outputs when training a supervised learning model.

---

## 3. Four Common NLP Tasks

NLP supports many applications, and each task has a different purpose. Some tasks identify categories in text, while others recognize individual words, retrieve information or generate new sentences.

The following are four common NLP tasks.

### 3.1 Text Classification

Text classification is the process of assigning a predefined category to a piece of text based on its content.

It is commonly used in spam detection, sentiment analysis, news categorization and customer feedback analysis.

- **Input:** A text document, sentence or message.
- **Output:** A category or class assigned to the input.

**Example:**

Input: "I really enjoyed using this application."

Output: Positive sentiment.

A sentiment analysis system can use this task to identify whether customer feedback expresses a positive, negative or neutral opinion.

### 3.2 Sequence Labeling

Sequence labeling is an NLP task in which a label is assigned to each token in a sequence. Instead of classifying an entire document with one category, the system identifies the role or category of individual words.

One common application is Named Entity Recognition (NER), which identifies names of people, organizations and locations in text.

- **Input:** A sentence divided into tokens.
- **Output:** A corresponding label for each token.

**Example:**

Input: "Atika studies at Jamhuriya University."

| Token | Label |
|---|---|
| Atika | PERSON |
| studies | O |
| at | O |
| Jamhuriya | ORGANIZATION |
| University | ORGANIZATION |

Here, `O` means that the token does not belong to one of the named entity categories being identified.

Sequence labeling is useful for extracting important information from documents, news articles and other written content.

### 3.3 Search and Similarity

Search and similarity are NLP applications that help users find documents or text that are relevant to a particular query.

A search system compares a user's query with available documents and ranks the results according to their relevance. Similarity methods can also compare two pieces of text to estimate how closely their content or meanings are related.

- **Input:** A search query and a collection of documents.
- **Output:** A ranked list of relevant documents or similarity scores.

**Example:**

A user searches for:

"Universities offering computer science courses."

A search system may return documents containing information about university programs, even when they use slightly different wording, such as "computer science degrees" or "computing education."

This task is useful in search engines, digital libraries, information retrieval and question-answering systems.

### 3.4 Text Generation

Text generation is the process of producing new human-readable text based on an input, instruction or previous context.

Modern language models learn patterns from large amounts of text and use those patterns to generate sequences of words that form responses.

- **Input:** A prompt, instruction or previous text.
- **Output:** Newly generated text.

**Example:**

Input: "Write a short introduction about Artificial Intelligence."

Output: A generated paragraph explaining what Artificial Intelligence is and how it is used.

Text generation is used in chatbots, writing assistants, automatic summarization and conversational systems.

### Comparison of the four tasks

| NLP task | Input | Output | Example application |
|---|---|---|---|
| Text classification | Text document | Category | Spam detection |
| Sequence labeling | Token sequence | Label for each token | Named Entity Recognition |
| Search and similarity | Query and documents | Ranked results or scores | Search engines |
| Text generation | Prompt or context | New text | Chatbots |

Although these tasks work with human language, their objectives are different. Classification assigns categories, sequence labeling identifies information within text, search retrieves relevant information, and generation creates new content.

---

## 4. Supervised and Unsupervised Learning in NLP

Machine learning systems can learn from data in different ways. Two important approaches are supervised learning and unsupervised learning.

### 4.1 Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using examples that contain both input data and their correct target labels.

The model examines the relationship between the input and the known answers. Through training, it learns patterns that can help it predict labels for new, previously unseen data.

**NLP example: Customer sentiment classification**

Imagine creating a system that identifies whether customer reviews are positive or negative.

| Input text | Target label |
|---|---|
| The service was excellent. | Positive |
| The delivery was disappointing. | Negative |
| I am happy with my purchase. | Positive |

The model learns from these labeled examples. After training, it can receive a new review and predict its sentiment.

**Why are labels needed?**

Labels are needed because they provide the correct target answers during training. The model compares its predictions with these answers and uses the differences to improve its performance.

### 4.2 Unsupervised Learning

Unsupervised learning is a machine learning approach in which a model works with data that does not contain predefined target labels.

Instead of learning from known answers, the model identifies patterns, similarities, relationships or groups within the available data.

**NLP example: Grouping customer reviews**

Imagine collecting 1,000 customer reviews without assigning positive or negative labels.

An unsupervised learning algorithm, such as K-means clustering, could group similar reviews based on their numerical text representations.

For example, the algorithm might identify groups containing:

- Reviews discussing delivery speed.
- Reviews discussing product quality.
- Reviews discussing customer service.

These groups are discovered from patterns in the data rather than from predefined categories. The resulting clusters do not automatically have meaningful names; a person may examine them and interpret what each group represents.

**Why are labels not required?**

The objective is to discover structures or relationships in the data, not to predict a previously supplied correct answer. Therefore, the algorithm can learn without target labels.

### Comparison

| Comparison | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Target labels | Required | Not required |
| Main purpose | Predict known categories or values | Discover hidden patterns |
| NLP example | Sentiment classification | Customer review clustering |
| Common algorithm | Naive Bayes | K-means clustering |

The main difference is the availability of target labels. Supervised learning uses known answers to guide the learning process, whereas unsupervised learning searches for patterns without being given those answers.

---

## 5. Why Text Must Be Converted into Numerical Representations

Computers store and process information using numerical values. However, human language consists of words, sentences and symbols that cannot be directly used as input by most traditional machine learning algorithms.

For this reason, text must be transformed into numerical representations before most machine learning models can process it.

This process is commonly called text vectorization or text representation.

Numerical representations allow machine learning algorithms to calculate word frequencies, compare documents, measure relationships and identify patterns in text data.

### Example: Bag of Words (BoW)

Bag of Words is a simple and widely used text representation method. It converts documents into numerical vectors by counting how frequently each vocabulary word appears.

The method first creates a vocabulary from the available documents. Each vocabulary word becomes a feature, and its value represents the number of times that word occurs in a particular document.

Consider these two sentences:

- Document 1: "I love machine learning."
- Document 2: "I love artificial intelligence."

First, create a vocabulary:

`["I", "love", "machine", "learning", "artificial", "intelligence"]`

Then represent each sentence using word counts.

| Document | I | love | machine | learning | artificial | intelligence |
|---|---:|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| Document 2 | 1 | 1 | 0 | 0 | 1 | 1 |

The sentences have now been transformed into numerical vectors that a machine learning algorithm can process.

### Main limitation of Bag of Words

One major limitation of Bag of Words is that it does not preserve word order or understand the meaning of words in context.

For example:

- "The student teaches the teacher."
- "The teacher teaches the student."

These sentences contain almost the same words, but their meanings are different. A basic Bag of Words representation gives both sentences the same word-count vector.

Another limitation is that the representation can become very large and sparse when the vocabulary contains thousands of unique words.

Therefore, Bag of Words is useful for simple text analysis and classification, but it cannot fully represent the meaning and context of natural language.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages are languages that have limited digital resources available for developing and evaluating language technologies. These resources include text collections, dictionaries, annotated datasets, speech recordings and language-processing tools.

Somali is an important example of a language that has received less NLP research attention and has fewer dedicated language-processing resources than major high-resource languages such as English.

This creates several challenges for researchers and developers who want to build reliable NLP systems for Somali.

### 6.1 Limited Digital Datasets

One major challenge is the availability of sufficiently large and diverse Somali-language datasets.

Machine learning systems often depend on training data to learn language patterns. For example, a sentiment analysis system needs examples of Somali sentences labeled as positive, negative or neutral.

When there are not enough examples, the model may struggle to recognize different expressions, writing styles and meanings.

This can lead to inaccurate predictions and poor performance when the system encounters text that is different from its training data.

The problem is not only the quantity of available text. High-quality annotated data is also important because incorrect or inconsistent labels can affect the learning process.

Muhammad et al. (2023), through their AfriSenti research, demonstrate the importance of creating annotated datasets for African languages. Their benchmark contains more than 110,000 tweets across 14 African languages. Somali is not among those 14 languages, but the study illustrates the wider need for language-specific datasets in African NLP research.

Similarly, Badel et al. (2023) identified the lack of dedicated Somali information retrieval resources as an important research barrier. Their work addressed this issue by creating a Somali information retrieval corpus containing 2,335 documents.

### 6.2 Spelling Variation and Written Differences

Another challenge is that the same Somali word may appear in different written forms.

For example, Badel et al. (2023) identify the Somali word for prime minister, which can appear as:

- `ra'iisulwasaare`
- `raysulwasaare`

Although both forms refer to the same concept, a basic text-processing system may treat them as different words.

This can create problems for search engines and classification systems because documents containing one spelling may not be matched with queries containing the other spelling.

To address this issue, NLP systems need suitable text normalization methods that recognize common spelling variations without incorrectly changing the meaning of words.

### 6.3 Morphological Complexity and Word Forms

Somali is an agglutinative language, meaning that grammatical information can be expressed through changes and additions to word forms.

This creates challenges for tokenization, stemming and lemmatization.

Different grammatical forms of a word may be related to the same root but appear differently in written text. A simple NLP system may treat them as unrelated vocabulary items.

When this happens, the vocabulary can become larger, and the system may fail to recognize that different word forms are connected.

Lemmatization and morphological analysis can help by identifying the base form of a word and understanding its grammatical structure.

Mohamed and Mohamed (2023) investigated this issue by developing an initial lexicon and rule-based lemmatization approach for Somali. Their work demonstrates the importance of language-specific tools for handling Somali word forms.

### 6.4 Mixed-Language Text

In online communication, people sometimes combine two or more languages within the same sentence. This is known as code-switching.

For example, a Somali speaker might write:

"Maanta assignment-ka ayaan submit gareeyay."

This sentence combines Somali and English expressions.

A language-processing system trained mainly on formal Somali text may have difficulty identifying the correct language, separating the words and interpreting their meaning.

This can reduce the accuracy of language identification, sentiment analysis and other NLP applications.

A system designed for real-world Somali communication therefore needs to consider how people actually write online, including informal expressions and the use of multiple languages.

### Summary of the challenges

| Challenge | Effect on NLP systems | Possible direction |
|---|---|---|
| Limited datasets | Models have fewer examples to learn from | Collect and annotate more Somali text |
| Spelling variation | The same word may be treated as different words | Text normalization |
| Morphological complexity | Related word forms may be treated separately | Lemmatization and morphological analysis |
| Mixed-language text | Systems may misunderstand combined languages | Multilingual and code-switching-aware models |

These challenges demonstrate that developing NLP systems for Somali requires more than simply applying existing English-language tools. Researchers need to consider Somali writing patterns, grammar, vocabulary and real communication practices.

Improving digital language resources and involving native speakers in dataset development can help make future Somali NLP applications more accurate and useful.

---

## 7. Case Study: Somali Information Retrieval Corpus

### Research paper

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources

**Authors:** Abdisalam Mahamed Badel, Ting Zhong, Wenxin Tai and Fan Zhou.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), pages 7463–7469.

**Research paper:** https://aclanthology.org/2023.emnlp-main.462/

### 7.1 Research Problem

The researchers identified a lack of dedicated digital resources for Somali information retrieval.

Information retrieval is the process of finding relevant information from a collection of documents based on a user's search query.

Many existing approaches to Somali information retrieval relied on translating queries into another language. However, translation-based systems can fail when the meaning of a word or expression is not accurately preserved.

The researchers wanted to address this problem by creating a dedicated Somali-language document collection and developing a retrieval system that could search those documents.

### 7.2 Data Used

The researchers created a corpus containing 2,335 Somali documents collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Publicly available Somali cultural and history books, including poetry.

The corpus contained the following:

| Data measurement | Quantity |
|---|---:|
| Documents | 2,335 |
| Sentences | 16,450 |
| Words | 578,216 |
| Search queries | 16 |
| Relevant document judgments | 144 |

The documents were collected, cleaned and annotated with the support of researchers and language experts.

The researchers also prepared search queries and relevance judgments to evaluate how well the retrieval system could identify useful documents.

### 7.3 Method Used

The researchers developed a Somali information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF)**.

PRF is a query expansion method that attempts to improve search results by using information from the documents retrieved during an initial search.

The method works in several steps:

1. A user enters a search query.
2. The system retrieves an initial set of documents.
3. It examines the highest-ranked documents and selects important terms.
4. The selected terms are added to the original query to create an expanded query.
5. The system searches again using the improved query and returns the ranked results.

In this study, the researchers selected the top five terms for query expansion.

They also developed a Python-based indexing system that tokenized the documents, removed identified Somali stop words and created an inverted index. They used BM25 to rank documents according to their relevance to a query.

### 7.4 Findings

The researchers compared the Pseudo-Relevance Feedback approach with a traditional TF-IDF retrieval method.

Their reported results showed that PRF performed better than TF-IDF on the evaluation corpus.

| Evaluation metric | TF-IDF | PRF |
|---|---:|---:|
| F-measure at top 10 (F@10) | 53.7% | 72.1% |
| F-measure at top 20 (F@20) | 43.6% | 57.2% |
| Recall at top 20 (R@20) | 68.9% | 91.7% |

These results suggest that query expansion helped the system retrieve more relevant Somali documents than the baseline TF-IDF approach in their experiment.

However, the researchers also identified limitations. For example, the same Somali word can have different spellings, and the lack of a fully functional Somali stemmer affected the study.

The researchers recommended developing better Somali stemming tools and expanding the corpus in future work.

### 7.5 Which NLP Task Does This Case Study Cover?

This research mainly covers **Search and Similarity**, specifically information retrieval.

The system receives a user's search query and a collection of Somali documents as input. It then retrieves and ranks documents according to their relevance to the query.

This connects directly to Question 3 because search and similarity systems help users find information that matches their information needs.

### 7.6 My Understanding of the Research

I found this study particularly relevant because it addresses a real challenge in developing language technology for Somali.

It demonstrates that NLP research does not always require building a chatbot or a large language model. Developing a reliable search system for a language with limited digital resources is also an important contribution.

The creation of a Somali document corpus provides a foundation for further research and can support future applications such as digital libraries, question-answering systems and other information retrieval tools.

For me, the most important lesson from this research is that language technology should consider the actual language and its speakers. A system developed for English cannot automatically be expected to work accurately for Somali without appropriate data and language-specific processing.

---

## Conclusion

Natural Language Processing provides methods for computers to work with human language and supports applications such as text classification, information retrieval, sequence labeling and text generation.

Understanding the relationship between AI, machine learning and NLP is important because it explains how language-based systems are developed and how they learn from data.

This assignment also shows why text must be represented numerically before most machine learning algorithms can process it, and why different learning approaches are suitable for different NLP tasks.

The challenges of low-resource languages demonstrate that successful NLP development depends not only on algorithms but also on the availability and quality of language resources.

The Somali information retrieval case study provides a practical example of how research can address these challenges by developing language-specific datasets and systems.

Overall, improving NLP for Somali and other underrepresented languages requires continued research, high-quality datasets, appropriate language-processing methods and collaboration with native speakers.

---

## References

Badel, A. M., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information retrieval corpus: Bridging the gap between query translation and dedicated language resources. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 7463–7469). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.462

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models* (3rd ed., online manuscript released August 19, 2026). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and rule-based word lemmatization approach for the Somali language*. arXiv. https://arxiv.org/abs/2308.01785

Muhammad, S. H., Abdulmumin, I., Ayele, A. A., Ousidhoum, N., Adelani, D. I., Yimam, S. M., Ahmad, I. S., Beloucif, M., Mohammad, S. M., Ruder, S., Hourrane, O., Brazdil, P., Jorge, A., Ali, F. D. M. A., David, D., Osei, S., Bello, B. S., Ibrahim, F., Gwadabe, T., Rutunda, S., Belay, T., Messelle, W. B., Balcha, H. B., Chala, S. A., Gebremichael, H. T., Opoku, B., & Arthur, S. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 13968–13981). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.862
