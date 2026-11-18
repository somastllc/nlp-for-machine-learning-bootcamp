# Assignment 1 – NLP Foundations

## Introduction

Natural Language Processing (NLP) helps computers work with human language. It is used in search engines, translation apps, chatbots, voice assistants, and spam filters.

This assignment explains basic NLP terms, common tasks, machine learning methods, text representation, and challenges in processing Somali.

## 1. What is NLP? How is it related to AI and ML?

**Natural Language Processing (NLP)** is a field of computer science and Artificial Intelligence (AI). It helps computers read, understand, analyze, and create human language.

For example, when a person asks, “Where can I find a hospital?”, an NLP system can understand that the person is looking for a hospital.

| Term | Simple meaning |
|---|---|
| Artificial Intelligence (AI) | Making computers do tasks that normally need human intelligence. |
| Machine Learning (ML) | A part of AI that learns patterns from data. |
| Natural Language Processing (NLP) | Technology that helps computers work with human language. |

**Example – Spam detection:**

1. NLP reads and processes the email.
2. ML uses patterns learned from earlier emails.
3. The system decides if the email is spam.
4. The email is moved to the spam folder if it is classified as spam.

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

These are basic terms used when preparing text for NLP.

| Term | Meaning | Example |
|---|---|---|
| Corpus | A collection of texts. | 10,000 customer reviews. |
| Document | One text in a corpus. | “The app is useful.” |
| Token | A word or punctuation mark. | `The`, `app`, `useful`, `.` |
| Vocabulary | All unique tokens in a corpus. | `["The", "app", "useful"]` |
| Feature | Information a model uses to learn. | Number of times a word appears. |
| Label | The correct category for a text. | Positive or Negative. |

**Example dataset:**

| Review | Label |
|---|---|
| The application is excellent. | Positive |
| The service is very poor. | Negative |
| The application works well. | Positive |
| The service is not good. | Negative |

These terms help organize text so that a machine learning model can learn from it.

## 3. Four Common NLP Tasks

| NLP task | What it does | Example |
|---|---|---|
| Text classification | Gives a text one category. | Marks an email as spam or not spam. |
| Sequence labeling | Gives a label to each word or token. | Finds names and places in a sentence. |
| Search and similarity | Finds documents related to a query. | Finds password-reset instructions. |
| Text generation | Creates new text. | A chatbot answers a question. |

**Sequence labeling example:**

Sentence: “Ahmed works at Eurosom Cloud in Mogadishu.”

| Word | Label |
|---|---|
| Ahmed | PERSON |
| works | O |
| Eurosom Cloud | ORGANIZATION |
| Mogadishu | LOCATION |

`O` means the word is not one of the entity types being identified.

## 4. Supervised and Unsupervised Learning

| Comparison | Supervised learning | Unsupervised learning |
|---|---|---|
| Data | Labeled data | Unlabeled data |
| Correct answers | Provided during training | Not provided |
| Main goal | Predict known categories | Find patterns or groups |
| NLP example | Positive/negative review classification | Grouping similar reviews |

**Supervised example:** A model learns from reviews already marked Positive or Negative. It then predicts the label of a new review.

**Unsupervised example:** A model receives reviews without labels and groups similar reviews, such as payment problems or delivery issues.

Supervised learning is useful when labeled examples are available. Unsupervised learning is useful when labels are not available or are expensive to create.

## 5. Why Convert Text into Numbers?

Most machine learning models work with numbers, not ordinary sentences. NLP therefore changes text into numerical data. This is called **text representation** or **vectorization**.

### Example: Bag of Words

Bag of Words (BoW) counts how often each word appears in a document.

Sentences:

- Document 1: “I like machine learning.”
- Document 2: “I like natural language.”

Vocabulary: `I, like, machine, learning, natural, language`

| Word | Document 1 | Document 2 |
|---|---:|---:|
| I | 1 | 1 |
| like | 1 | 1 |
| machine | 1 | 0 |
| learning | 1 | 0 |
| natural | 0 | 1 |
| language | 0 | 1 |

The sentences are now represented by numbers that a model can process.

**Main limitation:** BoW ignores word order and context. For example, “The student helped the teacher” and “The teacher helped the student” contain the same words but have different meanings.

Other methods include:

- **TF-IDF:** Gives more weight to words that are important in a document.
- **Word embeddings:** Represent words as number patterns that can capture relationships.
- **Contextual embeddings:** Represent a word based on the words around it.

## 6. Challenges of NLP for Somali

A **low-resource language** has fewer digital texts, datasets, dictionaries, and language tools available for NLP. Somali has many speakers, but its digital NLP resources are more limited than English resources.

| Challenge | What it means | Possible solution |
|---|---|---|
| Limited datasets | There are not enough Somali texts with correct labels. | Build and share more datasets. |
| Spelling variation | People may write the same word in different ways. | Use text cleaning and spelling tools. |
| Different word forms | A word can change form, such as `macalin` and `macalimiin`. | Use lemmatization and language rules. |
| Dialects | Somali speakers may use different regional words. | Include different dialects in training data. |
| Mixed languages | People may mix Somali with English or Arabic. | Train models with mixed-language examples. |

**Why these problems matter:** They can make search results, translation, and text classification less accurate.

**Ways to improve Somali NLP:**

1. Collect more Somali text from different sources.
2. Include different dialects and writing styles.
3. Build Somali dictionaries and language tools.
4. Create spelling and text-normalization tools.
5. Ask Somali speakers to check and label the data.
6. Test models using different types of Somali text.

## 7. Case Study: Somali Information Retrieval Corpus

**Research paper:** *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources* (Badel et al., 2023).

| Item | Details |
|---|---|
| Authors | Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou |
| Published | EMNLP 2023, pages 7463–7469 |
| Main task | Information retrieval (search) |
| Dataset | 2,335 Somali documents |
| Sources | Hiiraan Online, Dhacdo.net, Somali poetry books, and other content |

### Problem

Somali search systems often depended on translating Somali queries into English. Translation can change or lose some of the original meaning.

### Method

The researchers used **Pseudo-Relevance Feedback (PRF)**:

1. The user enters a search query.
2. The system finds documents that may be relevant.
3. It uses the top results to find related terms.
4. It improves the original query.
5. It searches again and ranks the results.

### Reported results

| Measure | TF-IDF | PRF |
|---|---:|---:|
| Precision at 10 | 0.512 | 0.687 |
| Recall at 20 | 0.689 | 0.917 |
| F-measure at 10 | 0.537 | 0.721 |

The paper reports better results with PRF than with the TF-IDF baseline on the tested dataset. For example, Recall at 20 increased from 0.689 to 0.917.

### Importance and limitations

The study provides a Somali document collection for search research and reduces the need to translate every query into English. However, the collection is still limited compared with large English datasets. Spelling differences and word forms can also affect search.

Future work could add more documents, dialects, spelling forms, and better Somali language tools.

**NLP task:** This case study is an example of **search and similarity**, because it finds and ranks documents related to a user's query.

## References

1. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed., draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
3. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). “Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.” *Proceedings of EMNLP 2023*, 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462
4. Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and Rule-based Word Lemmatization Approach for the Somali Language*. https://arxiv.org/abs/2308.01785
5. Gedi, A. A., et al. (2026). “Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform.” *AfricaNLP 2026*, 179–189. https://doi.org/10.18653/v1/2026.africanlp-main.17
6. Adelani, D. I., et al. (2023). “AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages.” *Proceedings of EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.862/

## Overall Conclusion

NLP helps computers work with human language. It uses text data, machine learning, and numerical representations to perform tasks such as classification, search, and text generation.

Somali NLP still faces challenges, including limited datasets, spelling differences, word forms, dialects, and mixed languages. The Somali Information Retrieval Corpus is one example of research that helps improve Somali-language technology.
