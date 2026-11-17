# Assignment 1 – NLP Foundations

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is an area of computing that allows computers to process and work with human languages, including written text and speech. It is used in applications such as translation, text search, and automatic question answering (Jurafsky & Martin, 2026).

NLP is connected to Artificial Intelligence (AI) and Machine Learning (ML), but the terms have different meanings. AI is the wider field of creating computer systems that perform tasks associated with human intelligence. ML is one approach within AI: a model learns patterns from examples instead of depending only on instructions written by a programmer. NLP focuses on language-related tasks and can use ML, written rules, or both.

For example, a translation application can translate a Somali sentence into English. NLP methods process the language, while an ML model can learn translation patterns from many examples. The full application is an AI system.

## 2. Basic Terms for Text Data

Consider a small dataset of customer reviews:

| Document | Review | Label |
|---|---|---|
| D1 | “The food tasted fresh.” | Positive |
| D2 | “The order arrived late.” | Negative |
| D3 | “The staff were helpful.” | Positive |

| Term | Meaning | Example from the dataset |
|---|---|---|
| **Corpus** | A collection of text documents. | All three reviews together. |
| **Document** | One text item in the collection. | Review D1. |
| **Token** | A unit produced when text is divided, often a word or part of a word. | `food` is a token in D1. |
| **Vocabulary** | All the different tokens found in the corpus. | For this example: `the, food, tasted, fresh, order, arrived, late, staff, were, helpful`. |
| **Feature** | A measurable value from the text that a model can use. | Whether the word `fresh` appears in a review. |
| **Label** | The category assigned to a document. | `Positive` or `Negative`. |

A document may contain many token occurrences, while the vocabulary keeps only the distinct tokens. Features describe the text used as model input; labels are the target categories a supervised model learns to predict (Jurafsky & Martin, 2026; Manning et al., 2008).

## 3. Four Common NLP Tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A complete text. | One category for that text. | Classifying a review as positive or negative. |
| **Sequence labeling** | A sentence divided into tokens. | A label for each token. | Identifying a person’s name and a place name in a sentence. |
| **Search and similarity** | A query and a collection of texts. | Relevant texts ranked by their match to the query. | Finding articles related to a user's search. |
| **Text generation** | A prompt or some starting text. | New text produced by the system. | Generating a short answer to a question. |

The tasks differ mainly in their outputs: a category for a whole text, labels for parts of a text, a ranked set of results, or newly produced language (Jurafsky & Martin, 2026; Manning et al., 2008).

## 4. Supervised and Unsupervised Learning in NLP

**Supervised learning** uses examples that already have target labels. For example, a model can learn from news articles labelled as sports, business, or politics, then predict the category of a new article. The labels are needed because they show the model the correct category during training.

**Unsupervised learning** uses examples without target labels. For example, a system can group news articles by similarities in their words and topics. It does not need pre-assigned categories because its purpose is to discover patterns or groups in the data, rather than predict a known answer (Jurafsky & Martin, 2026).

## 5. Why Text Must Be Converted into Numbers

Most Machine Learning models perform calculations on numbers, so raw words and sentences usually need to be changed into numerical features first. One simple method is **Bag of Words**. It creates a list of vocabulary terms and represents each text by counting how often each term appears.

For example, with the vocabulary `[tea, coffee, sugar]`, the text “tea with sugar” could be represented as `[1, 0, 1]` if we ignore words outside the vocabulary.

A limitation of Bag of Words is that it does not preserve word order or understand context. Therefore, two sentences containing the same words in a different order may receive the same representation even when their meanings differ (Manning et al., 2008).

## 6. Challenges of NLP for Somali

Some Somali varieties, especially written MAAY, are low-resource in NLP because there are fewer large, carefully labelled digital datasets and language tools available than for widely researched languages. This can make it difficult to train models that work well on new text.

Another challenge is variation in dialects and spelling. A model trained mostly on one variety may not recognise words or forms used in another variety. Informal writing can also use inconsistent spellings, which may cause the same or related words to be treated as different forms. Gedi et al. (2026) report that shared vocabulary between MAXAA TIRI and MAAY, spelling differences, and mixed-dialect writing can lead to classification errors.

Code-switching is also difficult. People may mix Somali and English in one message. A system trained mainly on single-language examples may miss the meaning of mixed expressions or process some words incorrectly (Gedi et al., 2026).

## 7. Case Study: Identifying Somali Dialects

Gedi et al. (2026) studied how to identify whether a text is written in **MAXAA TIRI** or **MAAY**. This is a **text classification** task because the system assigns one dialect category to each text.

**Data:** The researchers created a dataset of 8,947 Somali text samples from sources such as social media, news, blogs, books, and formal documents. Somali linguists manually labelled the texts as MAXAA TIRI or MAAY.

**Method:** They cleaned and prepared the text, converted it into numerical features using TF-IDF for traditional Machine Learning models, and tested classifiers including Naive Bayes and Support Vector Machine. They also tested deep-learning methods such as BiLSTM.

**Findings:** The study reported strong results for some models, especially Naive Bayes and BiLSTM. However, the authors warned that part of the MAAY data had been translated with expert assistance because naturally occurring written MAAY data were limited. Therefore, the high scores should not be assumed to represent performance on every type of real-world Somali text. The study shows both the potential of NLP for Somali and the importance of collecting more natural, varied data (Gedi et al., 2026).

## References

Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., & Ebdiyow, A. U. (2026). Somali dialect identification in low-resource settings using machine learning and deep learning. *Discover Data, 4*, Article 10. https://doi.org/10.1007/s44248-026-00109-y

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing* (3rd ed., online manuscript). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
