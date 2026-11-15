# Assignment 1 – NLP Foundations


1. What is Natural Language Processing (NLP), and how does it relate to AI and ML?
Natural Language Processing (NLP) is the area of computing concerned with analyzing, understanding, processing, and generating human language. It includes work on text and speech, such as translation, sentiment analysis, information extraction, search, question answering, and dialogue systems. NLP draws on computer science, linguistics, statistics, and machine learning (Jurafsky & Martin, 2026).

Artificial Intelligence (AI) is the wider discipline of building systems that perform tasks associated with intelligent behavior. NLP is one branch/application area of AI because human language is a major way people communicate information and intentions. Machine Learning (ML) is a set of methods that enables a system to learn patterns from examples or data. ML is widely used in NLP, but NLP is not identical to ML: language systems can also use dictionaries, grammar rules, and other explicitly programmed methods.

The relationship can be summarized as: AI is the broad field; NLP is an AI area focused on language; ML is a family of learning methods often used to build NLP systems. These are overlapping concepts rather than three synonyms.

Real-life example—customer-support assistant. A customer writes, “My order has not arrived.” An NLP component processes the words and helps identify the user's intent as a delivery problem. A supervised ML model may have learned this intent from many examples labeled “delivery issue.” The broader AI application can then check an order database, apply company rules, and select an action. Finally, a language-generation component produces a response such as “I can check the delivery status for you.” Language processing, learned prediction, and the wider intelligent workflow cooperate.

2. Explain corpus, document, token, vocabulary, feature, and label
These terms refer to different parts of preparing text data for a computational task.

Term	Meaning	Example in a review dataset
Corpus	A collection of texts assembled for a purpose (plural: corpora).	All three reviews in the table below
Document	One individual text item in the corpus.	Review D1
Token	A unit produced by a tokenizer; it may be a word, punctuation mark, or subword.	excellent or phone
Vocabulary	The set of distinct tokens represented by a tokenizer/dataset.	{this, phone, is, excellent, the, battery, bad, camera, and}
Feature	A measurable or encoded input property used by a model.	A count indicating how often excellent occurs
Label	The target answer/category for an example in supervised learning.	Positive or Negative
Example dataset: product reviews
ID	Review	Sentiment label
D1	This phone is excellent	Positive
D2	The battery is bad	Negative
D3	Excellent camera and battery	Positive
The three review texts together form a corpus; each row's review is a document. With a simple word tokenizer, D1 might be split into This, phone, is, excellent. A vocabulary is created by collecting unique tokens from the documents (after any chosen normalization). In this example, lowercasing makes This and this the same token.

A model cannot use the raw word “excellent” as a numerical input in most conventional ML algorithms. A feature extractor could produce a vector of token counts. With vocabulary order [excellent, bad, phone], D1 has vector [1, 0, 1]. The sentiment column is the label: the answer a supervised classifier is trained to predict.

Important distinction: A token is a unit of text; a feature is the numerical representation supplied to a model. A token does not automatically equal one feature—one feature can be a token count, a binary presence value, or part of a more complex representation (Jurafsky & Martin, 2026).

3. Four common NLP tasks: input and output
NLP task	Typical input	Output	Concrete example
Text classification	One document, message, or text segment	One category or a set of categories	Label a Somali social-media post as positive, negative, or neutral
Sequence labeling	An ordered sequence of tokens	One label per token	Tag names in “Amina lives in Mogadishu”: Amina=PERSON, Mogadishu=LOCATION
Search and similarity	A query and candidate texts, or two texts	Ranked documents or similarity score(s)	Retrieve Somali news stories relevant to a query
Text generation	A prompt, context, or source text	A generated sequence of words/tokens	Produce a summary, translation, or chatbot reply
Text classification makes a decision about a whole text (or a defined text segment). Sequence labeling assigns decisions at token level and therefore keeps the output aligned with the input sequence; named-entity recognition is a common example. Search and similarity estimates which documents best match a query or how close two representations are. Text generation produces new text, often one token at a time based on context. The tasks can share language representations, but their outputs and evaluation methods differ (Jurafsky & Martin, 2026).

4. Supervised versus unsupervised learning
Supervised learning trains on examples with inputs and target labels. The algorithm compares its predictions with the provided correct answers and adjusts its parameters to reduce errors. For example, a Somali sentiment model can learn from messages manually labeled positive, negative, or neutral. The label is necessary because it tells the system what sentiment category each training message represents.

Unsupervised learning receives inputs without target labels and attempts to discover organization or patterns in them. For example, a clustering algorithm can group Somali news articles by similarities in their text representations. The algorithm does not receive a correct topic label for each article during training; a person may inspect the clusters and interpret them afterward.

Supervised	Unsupervised
Data supplied	Inputs + target labels	Inputs without target labels
Learning goal	Predict known targets for new examples	Find structure, groups, or patterns
NLP example	Sentiment classification	Document clustering
Why labels?	They define the answer the model should learn	No predefined answer is supplied
Unsupervised methods still produce outputs—such as cluster assignments or learned representations. The difference is that these outputs are not trained against human-provided target labels in the same way as supervised prediction.

5. Why numerical representations are needed; Bag of Words
Text is meaningful to people, but most traditional ML algorithms calculate with numerical values. To train a model, text is transformed into numbers so the algorithm can measure patterns, calculate similarities, and update parameters. Numerical representations can be simple counts or more sophisticated vectors such as embeddings.

Bag of Words (BoW)
Bag of Words represents a document using the presence or frequency of vocabulary terms, without preserving their order.

Assume the vocabulary is [good, bad, phone]:

Text	good	bad	phone
“good phone”	1	0	1
“bad phone”	0	1	1
These vectors can be passed to a classifier such as logistic regression or a support-vector machine.

Main limitation: BoW loses word order and much of the context. For instance, “the phone is good” and “the phone is not good” may have almost identical word-count vectors even though their meanings differ. It can also produce a very large, sparse matrix when the vocabulary is large. TF-IDF improves the weighting of terms but does not restore full word order or meaning. Embeddings and contextual models represent relationships more richly, though they also have limitations (Jurafsky & Martin, 2026).

6. Challenges for low-resource languages such as Somali
A low-resource language in NLP is one for which digital resources—large text corpora, labeled datasets, dictionaries, tokenizers, benchmarks, and other tools—are limited relative to the needs of a task. The term describes resource availability, not a language's value, complexity, or speakers. Research surveys show that African languages are substantially underrepresented in many NLP and language-model systems (Alabi et al., 2025; Hussen et al., 2026).

Challenge A: Limited text and annotated data
Many supervised models require labeled examples, and even methods without labels need enough representative text to learn useful patterns. For Somali, task-specific datasets and high-quality annotations may be less available than for English. Building a dataset requires collecting appropriate text, recruiting knowledgeable annotators, defining labels consistently, and checking annotation quality.

Impact: A model trained on few or unrepresentative examples may overfit, perform poorly on unfamiliar topics or communities, and have unreliable evaluation results. If a dataset mostly contains news, for example, it may not work well on informal messages. The 2025 African NLP survey examines the growth of resources and research while highlighting unequal coverage across languages (Alabi et al., 2025).

Challenge B: Tokenization and spelling variation
Somali digital text may include informal spellings, abbreviations, repeated characters, inconsistent punctuation, and English–Somali code-switching. Tokenizers learned mainly from high-resource languages may allocate too few vocabulary units to Somali words or split them into inefficient pieces. African-language LLM research identifies tokenization bias and data scarcity as major obstacles (Hussen et al., 2026).

Impact: Different spellings of the same expression may be treated as unrelated features; a tokenizer may break words into many pieces; and a system can miss meaning in mixed-language messages. This reduces data efficiency and can hurt classification, search, and generation.

Challenge C: Word forms and linguistic structure
Somali has grammatical word forms and language-specific patterns that generic preprocessing may not handle well. Related words can appear in different surface forms; however, indiscriminate stemming or normalization can also remove meaningful distinctions.

Impact: A search system may fail to match related forms, or a classifier may not generalize to forms missing from training. Language-aware preprocessing, carefully designed tokenization, and evaluation by fluent speakers are useful, but each normalization decision should be empirically tested.

Overall lesson: Data scarcity, orthographic variation, tokenization, and morphology interact. Better systems require representative data, documented preprocessing, language expertise, and evaluation on real Somali use—not merely translating English tools and assuming they will work.

7. Case study: Somali news topic classification (Ali, 2026)
Research problem and task
Mohamed M. Ali's 2026 Scientific Reports paper, “Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus,” studies automatic topic categorization of Somali news. This is text classification: the input is a news article and the output is one topic category.

Data
The study reports a manually curated corpus of 7,240 articles collected from BBC Somali and VOA Somali. Articles are organized into eight mutually exclusive topics:

Caafimaad (health)
Ciyaaro (sports)
Dagaal (war/conflict)
Diini (religion)
Ganacsi (business)
Madadaalo (entertainment)
Siyaasad (politics)
Tiknoolajiyad (technology)
The dataset is divided using a stratified 80/20 holdout split, so the split aims to preserve class proportions.

Method
The pipeline uses word-level TF-IDF unigram features with a Linear Support Vector Machine (SVM) classifier. TF-IDF converts documents into numerical vectors, giving terms more weight when they are informative for a document and less weight when they occur widely. A linear SVM learns boundaries between the eight topic classes from labeled articles.

This is a supervised approach: each training article has a topic label, and the classifier learns to predict those labels.

Findings
The paper reports that its best Linear SVM configuration achieved 94.20% accuracy on the holdout evaluation. It also reports mean accuracies of 93.11% across repeated stratified holdout testing and 93.20% in stratified five-fold cross-validation. These additional evaluations are useful because they test whether performance is stable across data partitions rather than relying on one split alone (Ali, 2026).

Interpretation and limitations
The results show that a comparatively lightweight model can perform strongly on a curated Somali news dataset. This is valuable because it offers a reproducible baseline without requiring a very large generative language model.

However, accuracy on this dataset does not prove that the classifier will work equally well on informal Somali social-media posts, dialect variation, or new sources. The articles come from two news outlets and eight defined categories; performance depends on the data distribution and label quality. A practical deployment would need testing on articles from other sources and periods, and examination of class-specific errors.

Connection to Assignment Question 3: The paper covers text classification. Input = a Somali news article represented by TF-IDF features; output = one of eight news-topic labels.

Conclusion
NLP is the AI area that works with human language, and ML supplies many methods for learning language patterns. A corpus contains documents, documents are tokenized, vocabularies define token sets, features encode text numerically, and labels provide target answers for supervised training. Classification, sequence labeling, search/similarity, and generation have different inputs and outputs. Supervised learning learns from target labels; unsupervised learning discovers structure without them. Numerical text representations make ML possible, but simple Bag of Words ignores context and order. Somali NLP also faces limited data, spelling and tokenization variation, and linguistic complexity. The Somali news case study demonstrates that a carefully curated dataset and a lightweight TF-IDF/SVM pipeline can establish a useful classification baseline, while also showing why evaluation must match the intended real-world use.

References
Ali, M. M. (2026). Resource-efficient Somali news classification using TF-IDF and linear SVM: A supervised baseline on a curated news corpus. Scientific Reports. https://doi.org/10.1038/s41598-026-68159-5

Alabi, J. O., Hedderich, M. A., Adelani, D. I., & Klakow, D. (2025). Charting the landscape of African NLP: Mapping progress and shaping the road ahead. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (pp. 27807–27841). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1414

Bahar, K., & Ramaha, N. T. (2023). Exploring Somali sentiment analysis: A resource-light approach for small-scale text classification. International Conference on Applied Engineering and Natural Sciences, 1(1), 620–628. https://doi.org/10.59287/icaens.1069

Hussen, K., Sewunetie, W., Ayele, A., Imam, S., Alemu, E., Muhammad, S., & Yimam, S. (2026). The state of large language models for African languages: Progress and challenges. Proceedings of Machine Learning Research, 302, 1–27. https://proceedings.mlr.press/v302/hussen26a.html

Jurafsky, D., & Martin, J. H. (2026). Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models (3rd ed., draft, August 19, 2026). Stanford University. https://web.stanford.edu/~jurafsky/slp3/





