# Assignment 1 – NLP Foundations

## 1. What is Natural Language Processing?

Natural Language Processing (NLP) is a field of computing concerned with developing methods that allow computers to process, analyse, and produce human language. Human language is complex because meaning depends not only on individual words but also on grammar, context, and the situation in which an expression is used. NLP therefore combines computational methods with ideas from linguistics and statistics to perform tasks such as information retrieval, translation, text classification, and language generation (Bird, Klein, & Loper, 2009).

NLP is closely connected to Artificial Intelligence (AI) and Machine Learning (ML), but the three terms do not mean the same thing. AI is the broad area concerned with building systems that perform tasks associated with intelligent behaviour. ML is one approach within AI: instead of programming every decision as a fixed rule, developers train algorithms to identify patterns in examples. NLP is the language-focused area. It can use rules, machine learning, or a combination of both.

**Real-life example: an email spam filter.** An email application may use NLP to examine the words and structure of incoming messages. The wider AI system uses this capability to help decide how messages should be handled. A machine-learning classifier can be trained on many previously labelled emails, where each example is marked as "spam" or "not spam." During training, the algorithm learns statistical patterns associated with each category. When a new email arrives, the system processes its text and predicts a category. The NLP component makes language data usable, the ML component learns from examples, and the AI application uses the prediction to support an intelligent email-management function. The prediction is not guaranteed to be correct, so users may still need to report mistakes.

## 2. Core terms: corpus, document, token, vocabulary, feature, and label

The following example uses a small, fictional dataset of customer-service messages. It illustrates how the terms relate to one another.

| Message ID | Text (document) | Label |
|---|---|---|
| M1 | "The delivery arrived early." | Positive |
| M2 | "My package is late." | Negative |
| M3 | "The delivery was quick." | Positive |
| M4 | "The package arrived damaged." | Negative |

- **Corpus:** The complete collection of text documents being studied. In this example, all four messages together form the corpus.
- **Document:** One individual text item within the corpus. For example, M2 ("My package is late.") is one document.
- **Token:** A unit produced when text is divided by a tokenization method. With a simple word-level tokenizer, M2 could become `["My", "package", "is", "late"]`. Other tokenizers may split text into subword units or treat punctuation as separate tokens.
- **Vocabulary:** The set of distinct tokens identified across a corpus after applying a chosen tokenization and normalization procedure. For the four messages, a simplified lowercase vocabulary could include: the, delivery, arrived, early, my, package, is, late, was, quick, and damaged. Repeated occurrences do not create new vocabulary entries.
- **Feature:** A measurable input representation used by a model. For example, a binary feature named `damaged` can be 1 when that token occurs in a message and 0 when it does not. A feature can also be a word count or a TF-IDF weight.
- **Label:** The target category assigned to a document for a supervised task. Here, "Positive" and "Negative" are the labels. Each message has one label in this example.

The distinction is important: a corpus contains documents; tokenization produces tokens; a vocabulary defines the available token features; and labels provide the target answers when the task is supervised. These are standard terms in NLP textbooks (see Jurafsky & Martin, 2026).

## 3. Four common NLP tasks

NLP tasks differ partly in the kind of input they receive and the output they are expected to produce.

| NLP task | Input | Output | Example application |
|---|---|---|---|
| Text classification | A complete text, such as a review or email | One category or a set of categories for the text | Predicting whether a customer review is positive, neutral, or negative |
| Sequence labeling | A sequence of tokens, usually a sentence | A label attached to each token or text span | Identifying "Amina" as a person and "Hargeisa" as a location in a sentence |
| Search and similarity | A search query and a collection of documents, or two text items to compare | A ranked list of relevant documents or a similarity score | Finding Somali news articles related to a user's query about flooding |
| Text generation | A prompt, instruction, or source text | Newly produced text | Generating a short summary of a long report |

In text classification, the prediction concerns the document as a whole. In sequence labeling, the system must locate and classify particular parts of the text. Search systems rank existing material rather than necessarily creating new material. Text-generation systems produce a new sequence of language. These tasks can use different algorithms and evaluation methods, even though all work with language.

## 4. Supervised and unsupervised learning in NLP

**Supervised learning** uses training examples that include both input data and target labels. For example, a sentiment classifier can be trained on customer reviews labelled as positive, neutral, or negative. The model compares its predictions with the supplied labels during training and adjusts its parameters to reduce prediction errors. Labels are needed because they tell the learning algorithm what answer it is expected to learn to predict.

**Unsupervised learning** works with data that does not have target labels for the task. For example, a system could group thousands of unlabelled product reviews according to similarities in their language. A clustering algorithm may discover groups discussing delivery, product quality, or customer service. The groups are inferred from patterns in the data rather than learned from pre-assigned category answers.

The central difference is the source of the learning signal. Supervised learning receives an intended answer for each training example. Unsupervised learning is not given those target answers and instead searches for structure, patterns, or relationships in the input data. Unsupervised methods may still be evaluated later by people, but that does not mean target labels were used to train the original clustering process.

## 5. Why text must be represented numerically

Most conventional machine-learning algorithms operate on mathematical values rather than directly interpreting raw sentences. A computer can store text as characters, but a classifier generally needs a consistent numerical representation so that it can calculate weights, distances, and statistical relationships. Text representation converts documents into numerical features that an algorithm can process.

One simple method is **Bag of Words (BoW)**. First, a vocabulary is created from the training documents. Each document is then represented by a vector whose positions correspond to vocabulary terms. The values can indicate whether a word occurs or how many times it occurs (Juluru, Shih, Murthy, & Elnajjar, 2021).

For example, suppose the vocabulary is `[delivery, late, quick]`:

| Text | delivery | late | quick | BoW vector |
|---|---|---|---|---|
| "delivery quick" | 1 | 0 | 1 | [1, 0, 1] |
| "delivery late" | 1 | 1 | 0 | [1, 1, 0] |

Both sentences now have vectors of the same length, so a conventional classifier can use them as input.

The main limitation of basic BoW is that it largely ignores word order and context. For example, "the service was good" and "the service was not good" contain many of the same words, but their meanings differ. A basic unigram BoW representation may fail to capture the effect of "not." It also treats words with related meanings as separate features unless additional processing is introduced. N-grams can preserve some local word order, but they increase the number of possible features and do not solve every problem of meaning or context.

## 6. Challenges of NLP for low-resource languages such as Somali

A low-resource language is one for which the digital resources needed to build and evaluate language technologies (such as large text collections, labelled datasets, dictionaries, and language-processing tools) are limited compared with those available for widely researched languages. Somali illustrates several of these challenges.

### 6.1 Limited digital and labelled datasets

Many supervised NLP systems depend on examples that have been labelled by people. If only a small amount of Somali text is available, or if it covers only a narrow set of topics and writing styles, a model may not learn patterns that generalise to other situations. Collecting and annotating reliable data also requires time, funding, language expertise, and clear annotation guidelines.

Badel et al. (2023) describe the lack of a dedicated Somali information-retrieval corpus as a barrier to research. Their work created a collection of 2,335 documents and associated relevance judgements. This demonstrates both the shortage of resources and the value of developing datasets specifically for Somali.

### 6.2 Morphological variation and word forms

Somali is described as an agglutinative language in terms of its morphological formations (Badel et al., 2023), and it also has derivational features: grammatical information and word formation can produce related forms of a word. A simple system may treat each surface form as a different token, even when the forms share a root or meaning. This can make the vocabulary larger and cause a model to encounter some forms frequently while seeing other related forms rarely.

For example, a search system may fail to connect a query with a relevant document if the query and document use different forms of a related word. Lemmatization and stemming can help by mapping related forms towards a common base, but these processes must be designed and evaluated carefully so that they do not remove meaningful distinctions. Mohamed and Mohamed (2023) developed a Somali lexicon- and rule-based lemmatization approach, illustrating the need for language-specific processing resources.

### 6.3 Spelling variation and informal writing

The same Somali word may appear in different written forms, particularly in informal online text. A system that relies on exact token matching may treat these spellings as unrelated vocabulary items. As a result, a search engine may miss relevant documents, while a classifier may fail to recognise a word form that was uncommon in its training data.

Badel et al. (2023) identify spelling variation as a limitation in their Somali information-retrieval study. They give the example of "ra'iisulwasaare" and "raysulwasaare," which refer to the same concept but differ in spelling. Normalisation can reduce this problem, although an overly aggressive normalisation rule may incorrectly merge words that should remain distinct.

### 6.4 Mixed-language text and language varieties

Somali speakers may combine Somali with English or Arabic words, especially in social media, messaging, and professional communication. This practice is often called code-switching. A system trained only on formal Somali may not handle mixed-language expressions, borrowed terms, abbreviations, or informal spellings well. In addition, differences among written varieties and dialects can mean that a dataset does not represent all speakers equally. A practical system should therefore be tested on data from the communities and settings where it will be used.

These challenges show why simply applying an English-focused pipeline to Somali may produce weak results. Better performance requires language-aware data collection, annotation, tokenization, normalization, and evaluation.

## 7. Case study: Building a Somali information-retrieval corpus

A relevant case study is Badel, Zhong, Tai, and Zhou's (2023) research paper, "Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources." The study addresses the difficulty of searching Somali-language documents when dedicated resources for Somali information retrieval are limited.

**Task.** The research focuses on information retrieval: given a user's Somali query, the system aims to find and rank relevant documents from a collection. This corresponds to the search and similarity task described in Question 3.

**Data.** The authors created a Somali information-retrieval corpus containing 2,335 documents, 16,450 sentences, and 578,216 words. The documents came from sources including Somali news websites such as Hiiraan Online and Dhacdo.net, as well as cultural and historical materials. Native speakers helped formulate information needs as queries, and language experts and NLP researchers prepared relevance judgements. The paper reports 16 queries and 144 judged relevant documents.

**Method.** The researchers cleaned and normalised the documents, tokenised the text, removed identified Somali stop words, and built an inverted index. They used term-frequency and document-frequency information and ranked documents with BM25. They also tested pseudo-relevance feedback (PRF), a query-expansion method. In PRF, the system first retrieves highly ranked documents, selects useful terms from those documents, adds selected terms to the original query, and searches again. The study compared PRF with a TF-IDF-based approach.

**Findings.** The authors report that PRF performed better than TF-IDF on their test collection. At the top 10 results, the reported F-measure was 0.721 for PRF and 0.537 for TF-IDF. At the top 20 results, the reported recall was 0.917 for PRF and 0.689 for TF-IDF. The paper therefore provides evidence that query expansion can improve retrieval in this particular Somali corpus. These results should be interpreted within the limits of the dataset and evaluation setup; they do not prove that the same performance will occur for every Somali search application.

**Importance.** The study is valuable because it contributes a dedicated Somali resource rather than depending entirely on translating queries into another language. It also shows that corpus construction, language expertise, tokenisation, stop-word handling, indexing, and evaluation are all important parts of an NLP application. The authors note remaining challenges, including spelling variation and the absence of a fully functional Somali stemmer.

## Conclusion

NLP enables computers to work with human language, while AI provides the broader goal of intelligent applications and ML supplies methods for learning patterns from data. Text must usually be transformed into numerical features before conventional ML models can process it. The choice of representation, learning method, and evaluation depends on the task and the quality of the data. Somali NLP makes these issues especially visible: limited labelled resources, morphological variation, spelling differences, and mixed-language writing can all affect system performance. The Somali information-retrieval case study demonstrates how creating language-specific datasets and adapting retrieval methods can help address these challenges.

## References

Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information retrieval corpus: Bridging the gap between query translation and dedicated language resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462

Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python: Analyzing text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/

Juluru, K., Shih, H.-H., Murthy, K. N. K., & Elnajjar, P. (2021). Bag-of-words technique in natural language processing: A primer for radiologists. *RadioGraphics, 41*(5), 1420–1426. https://doi.org/10.1148/rg.2021210025

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models* (3rd ed., online manuscript released August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/

Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word lemmatization approach for Somali language. *ICLR 2023 Workshops: AfricaNLP*. https://mlanthology.org/iclrw/2023/mohamed2023iclrw-lexicon/
