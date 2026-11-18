# Assignment 1 

**Topic:** Natural Language Processing Foundations

## 1. Definition of Natural Language Processing (NLP)

Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that enables computers to understand, process, analyze, and generate human language, such as English and Somali.

NLP is related to AI and Machine Learning (ML). AI is the broader field of creating intelligent computer systems. Machine Learning is a part of AI that allows computers to learn patterns from data. NLP uses these techniques to work with human language.

**Real-life example: Google Translate**

Google Translate uses NLP to understand sentences in one language and translate them into another. Machine Learning helps the system learn language patterns from large amounts of translated text, while AI provides the broader technology that makes the translation system work.

## 2. Basic NLP Terminologies

The following terms are important when working with text data:

| Term | Definition |
|---|---|
| Corpus | A large collection of text documents used for language research. |
| Document | A single piece of text, such as a message or article. |
| Token | A small unit of text, usually a word or punctuation mark. |
| Vocabulary | The collection of unique tokens found in a dataset. |
| Feature | A measurable property of text used by a Machine Learning model. |
| Label | The correct category or answer assigned to a document. |

**Example dataset: Customer Reviews**

Consider these two customer reviews:

- "The service is excellent."
- "The service is terrible."

Their terms can be understood as follows:

- **Corpus:** The collection of both customer reviews.
- **Document:** Each individual customer review.
- **Token:** Words such as "service", "excellent", and "terrible".
- **Vocabulary:** The unique words: "the", "service", "is", "excellent", and "terrible".
- **Feature:** Word counts or the presence of positive and negative words.
- **Label:** "Positive" for the first review and "Negative" for the second.

## 3. Four Common NLP Tasks

NLP performs different tasks to solve language-related problems.

| NLP Task | Input | Output |
|---|---|---|
| Text Classification | A text message or document. | A category, such as spam or not spam. |
| Sequence Labeling | A sentence containing several words. | A label for each word, such as a name, location, or person. |
| Search and Similarity | A search query and a collection of documents. | Relevant documents ranked by similarity. |
| Text Generation | A question, instruction, or text prompt. | New text, such as an answer, summary, or story. |

**Examples:**

- Text classification can identify whether an email is spam.
- Sequence labeling can identify people's names and locations in a sentence.
- Search and similarity can find articles related to a user's search.
- Text generation can produce answers using an AI chatbot.

## 4. Supervised and Unsupervised Learning in NLP

Supervised and unsupervised learning are two important Machine Learning approaches.

| Supervised Learning | Unsupervised Learning |
|---|---|
| Uses labeled training data. | Uses data without labels. |
| Learns from examples with correct answers. | Finds patterns and groups in data. |
| Example: Spam detection. | Example: Grouping similar customer reviews. |

**Supervised Learning:** A model receives emails labeled as "spam" or "not spam". It learns from these examples and predicts the category of new emails.

**Unsupervised Learning:** A model receives many customer reviews without categories. It groups similar reviews based on their content.

Supervised learning needs target labels because the model learns to predict known answers. Unsupervised learning does not need labels because its purpose is to discover hidden patterns in the data.

## 5. Text Representation in Machine Learning

Most Machine Learning models cannot directly understand ordinary human text. Therefore, text must be converted into numerical representations that computers can process and analyze.

One simple method is **Bag of Words (BoW)**.

Bag of Words represents text by counting how many times each word appears in a document. It creates numerical values for each word in the vocabulary.

**Example:**

- Document 1: "I love learning English."
- Document 2: "I love learning Python."

The vocabulary is: [I, love, learning, English, Python].

| Document | I | love | learning | English | Python |
|---|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 |
| Document 2 | 1 | 1 | 1 | 0 | 1 |

**Main limitation:** Bag of Words ignores word order and context. For example, it may treat "I like this" and "I do not like this" as similar because it does not properly understand the meaning of the sentence.

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Low-resource languages have fewer digital text collections, language tools, and research resources compared with widely studied languages.

Somali faces several challenges:

**1. Limited Datasets**

There are fewer high-quality Somali text datasets available for training Machine Learning models. This makes it difficult for NLP systems to learn Somali language patterns accurately. As a result, models may perform poorly when processing Somali text.

**2. Spelling Variations**

People sometimes write the same Somali word in different ways, especially on social media. This creates difficulties for NLP systems because they may recognize different spellings as separate words.

**3. Complex Word Forms**

Somali words can have different forms depending on grammatical structure. A system may fail to recognize that different word forms are related to the same root word. This affects tasks such as searching and text analysis.

**4. Mixed-Language Text**

Many Somali speakers mix Somali and English in messages and social media posts. This is called code-switching. It makes language identification and text processing more difficult for NLP systems.

These challenges show why more Somali datasets, language tools, and research are needed.

## 7. NLP Case Study: Somali Information Retrieval

**Research Paper:** *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources* (Badel et al., 2023).

This research focuses on improving information retrieval for the Somali language.

**Task:**

The researchers developed a system that helps users find relevant Somali documents using search queries.

**Data:**

They created a Somali text collection containing 2,335 documents gathered from online sources, including Somali news websites and poetry books.

**Method:**

The researchers worked with language experts to create an annotated corpus. They developed an information retrieval system using a technique called Pseudo-Relevance Feedback (PRF), which expands search queries using information from initially retrieved documents.

**Findings:**

The research provided a dedicated Somali information retrieval corpus and a search system. It also addressed the lack of available Somali language resources, which is an important challenge in NLP research.

**Related NLP Task:**

This case study covers **Search and Similarity**, because its main purpose is to find relevant documents based on users' search queries.


