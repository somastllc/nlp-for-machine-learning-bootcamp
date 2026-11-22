# Assignment 1 – NLP Foundations

**Student Name:** Idman Ahmed Ali  
**Course:** Natural Language Processing  
**Assignment:** NLP Foundations

---

## Introduction

Natural Language Processing (NLP) is an important field of Artificial Intelligence (AI) that enables computers to understand, process, analyze, and generate human language. It combines linguistics, computer science, and machine learning to develop systems that can work with text and speech.

NLP is used in many real-world applications, including search engines, chatbots, machine translation, sentiment analysis, spam detection, and voice assistants. This assignment explains the fundamental concepts of NLP, common NLP tasks, machine learning approaches, text representation, challenges in low-resource languages, and a research case study related to Somali language information retrieval.

## Question 1: What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a branch of Artificial Intelligence that focuses on enabling computers to understand, interpret, analyze, and generate human language in written or spoken form.

NLP is closely connected to AI and Machine Learning (ML):

- **Artificial Intelligence (AI):** The broader field of developing computer systems that perform tasks requiring human-like intelligence.
- **Machine Learning (ML):** A branch of AI that allows computers to learn patterns from data and improve their performance without being explicitly programmed for every situation.
- **Natural Language Processing (NLP):** A field of AI that uses linguistic knowledge and computational methods, including machine learning, to process human language.

For example, an email spam filter uses NLP and machine learning to examine the words and patterns in incoming emails. It learns from previously labeled messages and predicts whether a new email is spam or legitimate.

Another example is Google Translate, which processes text in one language and generates an equivalent translation in another language.

Therefore, NLP helps computers communicate with people using natural language.

## Question 2: Explain the following NLP terms: Corpus, Document, Token, Vocabulary, Feature, and Label.

These terms are important when preparing text data for NLP and machine learning.

| Term | Definition | Example |
|---|---|---|
| Corpus | A collection of text documents used for linguistic analysis or machine learning. | A collection of 1,000 customer reviews. |
| Document | A single text item within a corpus. | One customer review. |
| Token | A basic unit of text, such as a word or punctuation mark, produced through tokenization. | "NLP is useful" becomes "NLP", "is", "useful". |
| Vocabulary | The set of unique tokens found in a corpus after preprocessing. | {NLP, useful, good, bad}. |
| Feature | A measurable or numerical representation of information used by a machine learning model. | The frequency of the word "good". |
| Label | The target category or value assigned to an example in supervised learning. | Positive or negative sentiment. |

**Practical example:**

Consider a small dataset containing customer reviews:

| Document | Text | Label |
|---|---|---|
| D1 | This product is very good. | Positive |
| D2 | The product is bad. | Negative |
| D3 | I really like this product. | Positive |

In this example:

- The **corpus** is the collection of all three reviews.
- Each review is a **document**.
- Words such as "product", "good", and "bad" are **tokens**.
- The unique words form the **vocabulary**.
- Word frequencies can be converted into numerical **features**.
- Positive and Negative are the **labels**.

These concepts help transform raw text into structured information that machine learning algorithms can process.

## Question 3: Describe Four Common NLP Tasks and Their Inputs and Outputs.

NLP includes several tasks that allow computers to process and understand human language. Four common tasks are classification, sequence labeling, information retrieval and similarity, and text generation.

### 1. Text Classification

Text classification assigns a predefined category to a given text.

**Input:** A text document, sentence, or message.

**Output:** A category or class label.

**Example:** An email classification system receives an email and predicts whether it is Spam or Not Spam.

Other applications include sentiment analysis, news categorization, and topic detection.

### 2. Sequence Labeling

Sequence labeling assigns a label to each token in a sentence. It is commonly used to identify named entities, parts of speech, and other linguistic structures.

**Input:** A sentence divided into tokens.

**Output:** A label for each token.

**Example:**

Sentence: "Idman studies Computer Science."

| Token | Label |
|---|---|
| Idman | PERSON |
| studies | O |
| Computer | FIELD |
| Science | FIELD |

Sequence labeling helps computers identify important information within sentences.

### 3. Information Retrieval and Similarity

Information retrieval is the process of finding relevant documents from a collection based on a user's query. Similarity methods compare texts to determine how closely they are related.

**Input:** A search query and a collection of documents.

**Output:** A ranked list of relevant documents or similarity scores.

**Example:** A user searches for "machine learning courses". A search engine retrieves and ranks documents related to machine learning courses.

This task is used in search engines, digital libraries, and document recommendation systems.

### 4. Text Generation

Text generation involves producing new human-readable text based on an input, instruction, or context.

**Input:** A prompt, question, or previous text.

**Output:** Newly generated text.

**Example:** A user asks an AI chatbot, "Explain machine learning." The system generates an explanation in natural language.

Applications include chatbots, automatic summarization, report generation, and writing assistants.

### Summary of the Four Tasks

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | Text document | Category or label |
| Sequence Labeling | Tokenized sentence | Label for each token |
| Information Retrieval | Search query and documents | Ranked relevant documents |
| Text Generation | Prompt or context | Generated text |

## Question 4: Compare Supervised and Unsupervised Learning in NLP.

Machine learning is widely used in NLP to discover patterns in language data. Two major approaches are supervised learning and unsupervised learning.

### Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using labeled data. Each training example contains an input and its correct output.

The model learns the relationship between the input features and their labels. After training, it can predict labels for new, unseen examples.

**Example:** Sentiment classification.

A model is trained using customer reviews labeled as Positive or Negative. It learns patterns associated with each category and predicts the sentiment of new reviews.

**Why are labels necessary?**

Labels provide the correct answers during training. They allow the model to compare its predictions with the expected results and adjust its parameters to reduce prediction errors.

### Unsupervised Learning

Unsupervised learning uses data without predefined labels. Instead of learning from correct answers, the model discovers hidden structures, similarities, or patterns within the data.

**Example:** Topic clustering.

A system receives thousands of news articles without topic labels. It groups similar articles into clusters based on their content. One cluster may contain sports articles, while another contains political articles.

**Why are labels not required?**

The objective is to discover patterns or structures directly from the data rather than predict predefined categories.

### Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Training data | Labeled data | Unlabeled data |
| Main objective | Predict known outputs | Discover hidden patterns |
| Labels | Required | Not required |
| Example | Spam detection | Document clustering |
| Evaluation | Often compared with known labels | Uses clustering or other structure-based evaluation methods |
| Common applications | Sentiment analysis, text classification | Topic modeling, clustering |

Both approaches are useful in NLP. Supervised learning is appropriate when labeled examples are available, while unsupervised learning is useful when the goal is to explore large collections of unlabeled text.

## Question 5: How Can Text Be Represented Numerically for Machine Learning? Explain One Method and Its Limitation.

Machine learning algorithms generally operate on numerical data rather than raw text. Therefore, NLP systems must convert words, sentences, or documents into numerical representations. This process is called text vectorization or text representation.

One common method is **Bag of Words (BoW)**.

### Bag of Words (BoW)

Bag of Words represents a document by counting the occurrences of words from a predefined vocabulary. It creates a numerical vector for each document.

The method focuses on word frequency and generally ignores grammar and word order.

**Example:**

Consider the following documents:

- D1: "I like machine learning."
- D2: "I like natural language processing."

First, create a vocabulary:

Vocabulary = [I, like, machine, learning, natural, language, processing]

Then, represent each document using word counts.

| Document | I | like | machine | learning | natural | language | processing |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| D2 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

Each row is a numerical vector that can be used as input to a machine learning model.

### Advantages of Bag of Words

- It is simple to understand and implement.
- It converts text into numerical features.
- It can work well for basic text classification tasks.
- It does not require complex language models.

### Limitations of Bag of Words

- It ignores word order and grammar.
- It does not understand the meaning or context of words.
- It can create large and sparse vectors when the vocabulary is extensive.
- It treats words as separate units, even when they have related meanings.

For example, the sentences "The student helps the teacher" and "The teacher helps the student" contain similar words, but their meanings are different. Bag of Words may represent them similarly because it does not preserve word order.

Other text representation methods include TF-IDF, Word2Vec, GloVe, and contextual embeddings from transformer-based models.

## Question 6: What Are the Main Challenges of Developing NLP Systems for Low-Resource Languages Such as Somali?

A low-resource language is a language that has limited digital data, linguistic tools, annotated datasets, and computational resources available for developing language technologies.

Somali is an important example of a language that faces several challenges in NLP development.

### 1. Limited Datasets

One major challenge is the shortage of large, high-quality Somali text datasets. Many modern NLP models require extensive training data. However, there are fewer publicly available Somali datasets compared with languages such as English.

This makes it difficult to train reliable models for tasks such as sentiment analysis, machine translation, and question answering.

### 2. Spelling and Writing Variations

Somali users may write words differently because of spelling mistakes, informal writing, dialect differences, and variations in online communication.

For example, social media users may use different spellings or abbreviations for the same word. Such variations can make text preprocessing and model training more difficult.

### 3. Morphological Complexity

Somali contains grammatical structures in which words can change depending on grammatical relationships and context. Word forms may include prefixes, suffixes, and other grammatical elements.

A system that does not properly process these variations may fail to recognize that different word forms are related.

### 4. Code-Switching

Code-switching occurs when speakers use two or more languages within the same conversation or sentence.

For example:

"Waxaan maanta leeyahay programming assignment."

This sentence combines Somali and English. A language processing system may struggle to identify the language of each word and understand the complete meaning.

### 5. Limited NLP Tools and Resources

Somali has fewer publicly available language-processing tools, annotated corpora, benchmark datasets, and pretrained models than many high-resource languages.

This creates additional difficulties for researchers who want to build and evaluate Somali NLP applications.

### Possible Solutions

Researchers and developers can help address these challenges by:

- Collecting high-quality Somali text from reliable sources.
- Creating publicly available annotated datasets.
- Developing Somali tokenization and text normalization tools.
- Building models that support Somali and code-switched text.
- Collaborating with Somali language experts and local communities.
- Evaluating NLP systems using real Somali language data.

Improving these resources can support the development of useful applications such as Somali chatbots, search engines, educational systems, and translation tools.

## Question 7: Present a Case Study of a Research Paper Related to NLP.

### Case Study: Somali Information Retrieval Corpus

**Research Paper Title:** *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*

**Authors:** Abdisalam Mahamed Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).

**Research Link:** https://aclanthology.org/2023.emnlp-main.462/

### 1. Research Problem

The researchers identified that Somali information retrieval systems were limited because of the shortage of dedicated Somali-language datasets.

Many existing approaches relied on translating Somali search queries into other languages. However, translation-based approaches can introduce errors and may fail to preserve the original meaning.

The researchers aimed to address this problem by creating a dedicated Somali information retrieval corpus.

### 2. Dataset Used

The researchers developed an annotated corpus containing **2,335 Somali documents**.

The documents were collected from several sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The corpus was designed to support the development and evaluation of information retrieval systems for Somali.

### 3. Methodology

The researchers used a technique called **Pseudo-Relevance Feedback (PRF) Query Expansion**.

The method works by:

1. Receiving an initial search query.
2. Retrieving an initial group of documents.
3. Selecting frequent terms from the top-ranked documents.
4. Adding selected terms to the original query.
5. Submitting the expanded query to retrieve potentially more relevant documents.

In this research, the top five terms were used for query expansion.

This method helps improve the search process by adding useful terms that may not have appeared in the original query.

### 4. Findings and Contributions

The research introduced a dedicated annotated corpus for Somali information retrieval.

Its main contributions include:

- Providing 2,335 Somali documents for research.
- Supporting the development of Somali-language search systems.
- Demonstrating the use of pseudo-relevance feedback for Somali information retrieval.
- Creating a resource that can support future research in other NLP tasks, including text classification and question answering.

The study is important because it helps address the lack of digital language resources for Somali and provides a foundation for further NLP development.

### 5. NLP Task Category

The primary NLP task in this research is **Information Retrieval (IR)**.

Information retrieval involves finding relevant documents from a collection based on a user's search query.

The paper also contributes to low-resource language NLP because it creates a dedicated Somali dataset that may support other language-processing applications.

### 6. Why This Research Is Important

This research is valuable because it focuses on a real challenge affecting Somali language technology.

By creating a dedicated Somali corpus, the researchers provide a resource that can help developers and researchers build better search engines and other NLP applications.

It also demonstrates that developing language-specific datasets is an important step toward improving NLP systems for low-resource languages.

## Conclusion

Natural Language Processing is an important field of Artificial Intelligence that enables computers to work with human language. This assignment discussed the basic concepts of NLP, including corpus, document, token, vocabulary, feature, and label. It also explained four common NLP tasks and compared supervised and unsupervised learning.

Furthermore, the assignment examined how text can be converted into numerical representations using Bag of Words and discussed its limitations. It highlighted the challenges of developing NLP systems for low-resource languages such as Somali, including limited datasets, spelling variations, morphology, and code-switching.

Finally, the Somali Information Retrieval Corpus research demonstrated how dedicated language resources can support NLP development. Overall, improving datasets, algorithms, and language-processing tools is essential for making NLP more accessible to languages with limited digital resources.

## References

1. Badel, A. M., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469. Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.462

2. Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

