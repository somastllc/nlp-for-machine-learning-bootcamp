# Assignment 1 – NLP Foundations

---

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that focuses on enabling computers to process, analyze, understand, and generate human languages. These languages include Somali, English, Arabic, and many others.

NLP allows computers to perform tasks such as translating languages, identifying important information in documents, detecting spam messages, analyzing opinions, and answering questions.

### Relationship between AI, Machine Learning, and NLP

Artificial Intelligence, Machine Learning, and NLP are closely connected, but they have different roles.

- **Artificial Intelligence (AI):** A broad area of computer science that focuses on developing systems capable of performing tasks that normally require human intelligence.
- **Machine Learning (ML):** A part of AI that enables computers to learn patterns from data and make predictions without requiring programmers to write every decision as a fixed rule.
- **Natural Language Processing (NLP):** A field concerned with enabling computers to work with human language. It can use Machine Learning, deep learning, or rule-based methods.

### Real-life example: Somali customer support chatbot

Imagine a customer sends a Somali message to a company's support system:

*"Dalabkaygii xagee marayaa?"* (Where is my order?)

The system can use these technologies together:

1. **AI:** Provides the overall intelligent customer-support application.
2. **NLP:** Processes the Somali sentence and identifies that the customer is asking about an order.
3. **Machine Learning:** Uses patterns learned from previous customer messages to predict the correct intent, such as `order_status`.
4. **Backend system:** Retrieves the order information and returns an appropriate response.

This example demonstrates how NLP enables language processing, Machine Learning helps identify patterns, and AI brings the different components together into an intelligent application.

However, not every NLP system requires Machine Learning. Some systems use predefined rules, while others learn from data (Jurafsky & Martin, 2026).

---

## 2. Explain the terms corpus, document, token, vocabulary, feature, and label.

These terms describe the main components used when preparing text data for Natural Language Processing and Machine Learning.

To understand them, consider a small dataset containing Somali customer messages collected by an online shopping company.

### Example dataset

| Customer message | Label |
|---|---|
| "Waxaan rabaa inaan dalab sameeyo" | Order |
| "Dalabkaygii wali ima soo gaarin" | Delivery |
| "Adeeggiinnu aad buu u fiican yahay" | Positive Feedback |
| "Lacagtaydii waa la iga jaray" | Payment Issue |
| "Alaabtii aan helay way jabtay" | Complaint |

### Explanation of the terms

**1. Corpus**

A corpus is a complete collection of text documents used for language research or NLP applications.

In this example, all five customer messages together form one corpus.

**2. Document**

A document is one individual piece of text within a corpus.

For example:

"Dalabkaygii wali ima soo gaarin"

This single customer message is one document.

**3. Token**

A token is a unit of text produced by a tokenizer. It can be a complete word, part of a word, or punctuation, depending on the tokenization method.

For example, a simple word-level tokenizer could divide the sentence:

"Waxaan rabaa caawimaad"

Into:

`["Waxaan", "rabaa", "caawimaad"]`

These are three token occurrences.

**4. Vocabulary**

Vocabulary is the collection of unique tokens found in a text dataset after tokenization.

For example, if the messages contain:

`["dalab", "cusub", "dalab", "lacag"]`

The vocabulary is:

`["dalab", "cusub", "lacag"]`

The vocabulary size is 3 because there are three unique tokens.

**5. Feature**

A feature is a measurable numerical value that represents some information about a document and can be used as input to a Machine Learning model.

For example, the number of times the word `dalab` appears in a customer message can be a feature.

Other features may include word frequency, TF-IDF scores, or numerical word representations.

**6. Label**

A label is the correct category or target answer assigned to an individual training example in a supervised learning task.

For example, the message:

"Dalabkaygii wali ima soo gaarin"

May have the label `Delivery`.

### Summary

| Term | Example |
|---|---|
| Corpus | All five customer messages |
| Document | One individual customer message |
| Token | `dalab` |
| Vocabulary | All unique tokens |
| Feature | Numerical count of `dalab` |
| Label | `Delivery` |

These components help organize text data before using it to train or evaluate an NLP model.

---

## 3. Describe four common NLP tasks.

NLP systems perform different tasks depending on the problem they are designed to solve. Four common tasks are text classification, sequence labeling, search and similarity, and text generation.

### 3.1 Text Classification

Text classification is an NLP task that assigns a predefined category to an entire text document.

For example, an online shopping company may classify customer messages into categories such as complaints, payment issues, delivery questions, and positive feedback.

- **Input:** A customer message.
- **Output:** A predicted category or class label.

Example:

Input: "My payment was deducted twice."

Output: `Payment Issue`

### 3.2 Sequence Labeling

Sequence labeling identifies and assigns labels to individual words or tokens within a sentence.

One common application is Named Entity Recognition (NER), which identifies names of people, organizations, and locations.

Example:

Sentence: "Amina works at Somali National University in Mogadishu."

Possible output:

| Word | Label |
|---|---|
| Amina | PERSON |
| works | OTHER |
| at | OTHER |
| Somali National University | ORGANIZATION |
| in | OTHER |
| Mogadishu | LOCATION |

- **Input:** A sentence or sequence of tokens.
- **Output:** A label for each relevant token or word.

### 3.3 Search and Similarity

Search and similarity is an NLP task that finds documents related to a user's query or compares the similarity between different texts.

For example, a customer may search for "problems with mobile payments" in a collection of customer complaints.

The system can retrieve documents discussing failed transactions, payment errors, and money deductions.

- **Input:** A search query and a collection of documents.
- **Output:** A ranked list of relevant or similar documents.

This task is useful in search engines, document retrieval, and information management systems.

### 3.4 Text Generation

Text generation is an NLP task in which a computer produces new human-readable text based on an input or a learned language pattern.

Examples include automatic translation, summarization, and chatbot responses.

For example:

Input: "What are the benefits of learning Python?"

Output: "Python is useful because it is easy to learn and widely used in software development, data science, and artificial intelligence."

- **Input:** A prompt, question, or existing text.
- **Output:** Newly generated text.

### Comparison of the four tasks

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | Complete text | Category or class |
| Sequence Labeling | Sentence or tokens | Labels assigned to tokens |
| Search and Similarity | Query and documents | Ranked relevant documents |
| Text Generation | Prompt or text | New text |

Each task solves a different language-processing problem, and the appropriate task depends on the intended application.

---

## 4. Compare supervised and unsupervised learning in NLP.

Supervised and unsupervised learning are two important Machine Learning approaches used in NLP. Their main difference is whether the training data contains target labels.

### Supervised Learning

Supervised learning is a Machine Learning approach in which a model learns from examples that contain both input data and correct target labels.

For example, a company may collect 5,000 Somali customer messages and ask human annotators to classify them as:

- Complaint
- Payment Issue
- Delivery Question
- Positive Feedback

The model uses these labeled examples to learn patterns that help it classify new customer messages.

**Example:**

Input: "Alaabtii aan dalbaday wali ima soo gaarin."

Correct label: `Delivery Question`

The model learns from the message and its assigned label.

**Why are target labels needed?**

Target labels are needed because the model must learn the relationship between each input message and its correct category. During training, its predictions can be compared with the correct answers.

### Unsupervised Learning

Unsupervised learning is a Machine Learning approach in which a model works with data without receiving predefined target labels for the task.

For example, a company may collect thousands of customer reviews without classifying them as positive or negative.

An unsupervised method can group similar reviews based on their text representations.

Possible groups might contain:

- Reviews about delivery.
- Reviews about product quality.
- Reviews about customer service.

**Example:**

Input: A collection of customer reviews without categories.

Output: Groups of similar customer reviews.

**Why are target labels not needed?**

The objective is to discover patterns, relationships, or similarities within the data rather than predict a predefined correct category.

### Comparison

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labeled training data | Uses data without target labels |
| Learns from known answers | Discovers patterns in data |
| Can predict predefined categories | Can identify groups or similarities |
| Example: Spam classification | Example: Customer review clustering |

In conclusion, supervised learning is appropriate when correct answers are available, while unsupervised learning is useful when the goal is to explore and discover patterns without predefined answers.

---

## 5. Why must text be converted into numerical representations before most Machine Learning models can use it?

Human language is made up of words, sentences, and symbols. However, most traditional Machine Learning algorithms operate on numerical data rather than raw text.

For this reason, NLP systems convert text into numerical representations called **feature vectors**.

These numerical representations allow Machine Learning algorithms to identify patterns, calculate relationships, compare documents, and make predictions.

### Example: Bag of Words (BoW)

Bag of Words is a simple text representation technique that converts documents into numerical vectors based on the words they contain.

Suppose we have three customer messages:

- Document 1: "Somali food is good"
- Document 2: "Somali food is delicious"
- Document 3: "Technology is useful"

First, we create a vocabulary:

`["Somali", "food", "is", "good", "delicious", "Technology", "useful"]`

Next, we count the occurrence of each vocabulary word in every document.

| Document | Somali | food | is | good | delicious | Technology | useful |
|---|---:|---:|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| Document 2 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| Document 3 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |

The resulting numerical vectors are:

- Document 1: `[1, 1, 1, 1, 0, 0, 0]`
- Document 2: `[1, 1, 1, 0, 1, 0, 0]`
- Document 3: `[0, 0, 1, 0, 0, 1, 1]`

Now, a Machine Learning model can use these vectors as numerical inputs.

### Main limitation of Bag of Words

The main limitation of Bag of Words is that it does not understand the meaning or order of words.

For example:

- "The student helped the teacher."
- "The teacher helped the student."

Both sentences contain almost the same words, but their meanings are different.

Bag of Words may produce similar representations because it focuses on word occurrence rather than word order.

It also creates large, sparse feature vectors when a corpus contains many unique words.

Other techniques, such as TF-IDF and word embeddings, can provide alternative numerical representations. TF-IDF gives more importance to words that are distinctive across documents, while embeddings can represent relationships between words.

According to the scikit-learn documentation, text vectorization transforms documents into numerical feature vectors that can be used by Machine Learning algorithms (scikit-learn developers, n.d.).

---

## 6. Research the challenges of NLP for low-resource languages such as Somali.

Low-resource languages are languages that have limited digital language resources, such as text collections, labeled datasets, dictionaries, and language-processing tools.

Somali is an important example because it is widely spoken, but many NLP applications still face limitations related to available language resources and linguistic characteristics.

The following are four important challenges.

### 6.1 Limited Training Data and Labeled Datasets

One major challenge is the limited availability of high-quality Somali datasets, especially datasets that have been manually labeled for specific NLP tasks.

Machine Learning models require relevant examples to learn language patterns. When only a small number of labeled examples are available, it becomes difficult to train reliable models.

For example, a Somali spam detection system may have thousands of English spam examples available but very few accurately labeled Somali spam messages.

**Effect on NLP systems:**

- Models may fail to recognize unfamiliar Somali expressions.
- Predictions may become less accurate.
- Models may perform poorly on new messages.
- Developing and evaluating Somali NLP applications becomes more difficult.

Badel et al. (2023) identify the lack of dedicated Somali information retrieval resources as an important research problem and develop a corpus to address this gap.

### 6.2 Spelling Variation and Orthographic Differences

Another challenge is spelling variation.

Somali text may contain different spellings of the same intended word, particularly in informal online communication.

For example:

- `Soomaaliya`
- `Somalia`

A simple word-based tokenizer may treat these as two separate vocabulary entries.

This can cause problems because the model may learn one spelling during training but encounter another spelling when processing new messages.

**Effect on NLP systems:**

- Vocabulary size may increase unnecessarily.
- Similar words may be treated as unrelated.
- Search systems may fail to retrieve relevant documents.
- Text classification accuracy may decrease.

A suitable text-normalization process can help reduce some of these problems, but it must preserve important linguistic information.

### 6.3 Morphological Complexity and Word Forms

Somali has morphological features that allow grammatical information to be expressed through changes to word forms.

For example:

- `nin` – man
- `ninka` – the man

Although these forms are related, a basic word-level representation may treat them as separate tokens.

This can make it difficult for a model to recognize that different word forms are connected.

**Effect on NLP systems:**

- The vocabulary may contain many related word forms.
- Search systems may miss documents containing different forms of a search term.
- Models may need more training examples to learn related expressions.

Gedi et al. (2026) specifically investigate this issue through a Somali lemmatization corpus. Their research presents a lexicon containing more than 5,584 root words and 78,629 derivative forms.

This demonstrates the importance of developing language-specific tools for Somali morphology.

### 6.4 Mixed-Language Text and Language Varieties

Somali speakers sometimes combine Somali and English within the same sentence, especially in social media and informal communication.

For example:

"Meeting-ka maanta waa la baajiyay."

This sentence combines the English word `Meeting` with the Somali ending `-ka`.

A tokenizer designed mainly for English may not process this combination correctly.

Somali also has language varieties, including Standard Somali and Maay. A dataset containing mostly one variety may not represent the language patterns of speakers using another variety.

**Effect on NLP systems:**

- Tokenization may become inconsistent.
- Models may misunderstand mixed-language expressions.
- Performance may decrease when processing unfamiliar language varieties.
- Systems may produce unequal results for different groups of speakers.

### Possible solutions

Some approaches that can help address these challenges include:

1. Collecting more high-quality Somali text datasets.
2. Working with native Somali speakers during data annotation.
3. Developing Somali-specific tokenizers and normalization tools.
4. Building morphological resources and lemmatization systems.
5. Including different Somali language varieties in training and evaluation datasets.

In conclusion, developing effective Somali NLP applications requires more than choosing a powerful Machine Learning algorithm. It also requires understanding the language, collecting suitable data, and building appropriate preprocessing tools.

---

## 7. Case Study: Somali Information Retrieval Corpus

### Research Paper

**Title:** Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.

**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

**Publication:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).

**Link:** https://aclanthology.org/2023.emnlp-main.462/

### A. Problem and Task

The research focuses on the limited availability of resources for Somali Information Retrieval.

Information Retrieval is the process of finding relevant documents from a collection based on a user's search query.

Before this research, Somali information retrieval research was limited and often depended on translating search queries into other languages because dedicated Somali resources were not sufficiently available.

The researchers aimed to address this problem by creating an annotated Somali document corpus and developing a Somali information retrieval system.

### B. Data Used

The researchers created a corpus containing **2,335 Somali documents**.

The documents were collected from different sources, including:

- Hiiraan Online.
- Dhacdo.net.
- Somali poetry books.

The research involved collaboration between language experts and NLP researchers to develop the dataset.

The corpus provided a dedicated collection of Somali documents that could be used for information retrieval research.

### C. Method Used

The researchers developed a Somali information retrieval system using a technique called **Pseudo-Relevance Feedback (PRF) query expansion**.

Query expansion is a method that improves a search query by adding related terms.

Pseudo-Relevance Feedback uses initially retrieved documents to identify additional terms that may help improve the search.

For example, a user searching for information about education may receive relevant documents containing related terms that were not included in the original query.

The system can use these additional terms to expand the search and retrieve other potentially relevant documents.

The researchers applied this technique to the Somali document corpus.

### D. Findings and Contribution

The research produced two important contributions:

1. It introduced an annotated Somali information retrieval corpus containing 2,335 documents.
2. It developed a Somali information retrieval system using pseudo-relevance feedback and query expansion.

The study highlights the importance of building dedicated language resources for Somali instead of depending mainly on translated queries.

The corpus can also support future research in other Somali NLP applications, including question answering and text classification.

The published abstract does not provide a numerical performance score, so no specific accuracy improvement is claimed here.

### E. Which NLP Task Does This Case Study Cover?

This research mainly covers **Search and Similarity (Information Retrieval)**.

The reason is that its primary objective is to retrieve relevant Somali documents based on a user's search query.

The input is a search query and a collection of Somali documents, while the output is a set of relevant documents returned by the information retrieval system.

### Conclusion

This case study demonstrates how NLP research can address real problems affecting low-resource languages.

By creating a dedicated Somali corpus and developing an information retrieval system, the researchers contributed to improving the availability of language resources for Somali.

It also demonstrates that collecting quality datasets is an important step toward developing more advanced NLP applications.

---

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed., online manuscript). Stanford University.  
   https://web.stanford.edu/~jurafsky/slp3/

2. scikit-learn developers. (n.d.). *Feature extraction: Working with text data*. scikit-learn documentation.  
   https://scikit-learn.org/stable/modules/feature_extraction.html

3. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469.  
   https://doi.org/10.18653/v1/2023.emnlp-main.462

4. Duh, K., McNamee, P., Post, M., & Thompson, B. (2020). Benchmarking Neural and Statistical Machine Translation on Low-Resource African Languages. *Proceedings of the Twelfth Language Resources and Evaluation Conference*, 2667–2675.  
   https://aclanthology.org/2020.lrec-1.325/

5. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)*, 179–189.  
   https://doi.org/10.18653/v1/2026.africanlp-main.17

---

