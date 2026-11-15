# Assignment 1 – NLP Foundations

## 1. Natural Language Processing and Its Relationship to AI and ML

**Natural Language Processing (NLP)** is an area of computer science and artificial intelligence that focuses on enabling computers to process human language. It includes working with written text and spoken language. NLP systems can perform tasks such as translating languages, classifying messages, finding information, recognizing names and places, answering questions, and generating text.

NLP is part of **Artificial Intelligence (AI)** because it helps computers perform tasks that normally require human language abilities. **Machine Learning (ML)** is one important approach used to build NLP systems. Instead of writing a separate rule for every possible sentence, developers can train a model on examples so that it learns patterns from data. However, not every NLP system must use machine learning; some systems also use dictionaries, linguistic rules, or a combination of methods.

| Concept | Meaning | Role in language technology |
|---|---|---|
| Artificial Intelligence (AI) | The broad field of building systems that perform tasks associated with intelligence | Provides the wider goal of creating intelligent computer systems |
| Machine Learning (ML) | Methods that allow computers to learn patterns from data | Helps a system learn how to make predictions from language examples |
| Natural Language Processing (NLP) | The field concerned with processing human language | Focuses on understanding, analyzing, searching, and generating language |

**Real-life example: email spam detection.** An email service may receive a message saying, “Congratulations! Claim your prize now.” A spam-detection model can be trained using many emails that people have already labeled as *spam* or *not spam*. The NLP part processes the words and other language patterns in the message. The ML model learns which patterns are often associated with spam and uses what it has learned to classify a new email. AI is the broader field that includes this intelligent application.

## 2. Core Terms Used with Text Data

Consider a small example dataset containing Somali customer reviews. The purpose is to classify each review as positive or negative.

| Review ID | Review text | Sentiment label |
|---|---|---|
| 1 | Adeegyadu way fiican yihiin. | Positive |
| 2 | Adeegyadu way xun yihiin maanta. | Negative |
| 3 | Waxaan ku faraxsanahay adeegaaga. | Positive |

The main terms apply to this dataset as follows:

- **Corpus:** A collection of text documents gathered for language analysis. In this example, the corpus consists of all three customer reviews.
- **Document:** One individual text item within a corpus. Each review in the table is one document.
- **Token:** A unit created when text is divided into smaller pieces. Tokens are often words or subwords. For example, the review “Adeegyadu way fiican yihiin” can be split into the word tokens `Adeegyadu`, `way`, `fiican`, and `yihiin`.
- **Vocabulary:** The set of distinct tokens found in a corpus after the chosen text-processing steps. In this small example, after lowercasing and removing punctuation, the vocabulary is `adeegyadu`, `way`, `fiican`, `yihiin`, `xun`, `maanta`, `waxaan`, `ku`, `faraxsanahay`, and `adeegaaga`. The vocabulary size is the number of unique tokens, which is 10 here.
- **Feature:** A measurable input used by a machine-learning model. With a bag-of-words representation, each vocabulary term can become a feature. For example, the presence or count of the word `xun` can be one feature.
- **Label:** The expected answer that a supervised model is trained to predict. In this dataset, the label is the sentiment category, either `Positive` or `Negative`.

The review text provides the input, while the sentiment column provides the target label. A model can learn from the labeled examples and then predict the sentiment of a new review

## 3. Four Common NLP Tasks

NLP is used for different tasks depending on what information a system receives and what result it must produce.

| NLP task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A document, sentence, or message | One category or label for the whole text | Classifying a Somali news article as politics, sports, health, or business |
| **Sequence labeling** | A sequence of tokens, usually a sentence | A label assigned to each token | In “Mogadishu is a city,” identifying `Mogadishu` as a location |
| **Search and similarity** | A search query and a collection of documents | Relevant documents ranked by relevance, or similarity scores | Finding Somali news articles related to a user's query |
| **Text generation** | A prompt, conversation context, or source text | New text produced by the system | Generating a Somali answer in a chatbot or summarizing an article |

**Text classification** assigns one or more predefined categories to a complete text. Examples include spam detection and sentiment analysis.

**Sequence labeling** assigns a label to each token in a sequence. Part-of-speech tagging and named-entity recognition are common examples. For instance, a system may identify a person's name, a location, or an organization in a sentence.

**Search and similarity** helps users find documents related to a query. A search system compares the query with documents and returns the most relevant results. Similarity can be based on shared words or on representations that capture meaning.

**Text generation** produces new language based on an instruction or context. Examples include chatbot responses, summaries, and machine-generated translations.

## 4. Supervised and Unsupervised Learning in NLP

**Supervised learning** is a machine-learning approach in which the training examples contain inputs together with target labels. The model learns a relationship between the input and the correct output. For example, a Somali news classifier can be trained using articles labeled `politics`, `sports`, or `health`. After training, it can predict the category of an article it has not seen before.

Supervised learning needs target labels because the model is shown the expected answer during training. These labels provide the feedback used to learn how to make predictions.

**Unsupervised learning** uses data without target labels. Its purpose is to discover patterns, structures, or groups in the data. For example, a clustering algorithm can group Somali news articles according to similarities in their words without being given topic labels in advance.

Unsupervised learning does not require target labels because it is not learning to predict a supplied correct category. Instead, it identifies patterns in the data itself. A person may inspect the resulting groups and decide what topics they appear to represent.

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Text with target labels | Text without target labels |
| Main goal | Learn to predict a known target | Discover patterns or groups |
| Example in NLP | Classifying reviews as positive or negative | Grouping similar Somali news articles |
| Are target labels required? | Yes | No |

## 5. Why Text Must Be Converted into Numerical Representations

Most traditional machine-learning algorithms work with numbers rather than raw words. They perform mathematical operations on numerical inputs to identify patterns, calculate relationships, and make predictions. Therefore, text is usually converted into numerical vectors before it is given to these models. This process is called **text vectorization** or **numerical text representation**.

One simple method is **Bag of Words (BoW)**. First, the text is divided into tokens and a vocabulary is created. Then each document is represented by counting how often each vocabulary word appears. The word order is not preserved.

For example, using the first six vocabulary words from the review dataset, suppose the vocabulary is:

`[adeegyadu, way, fiican, yihiin, xun, maanta]`

Reviews 1 and 2 can be represented as:

| Document | BoW vector |
|---|---|
| “adeegyadu way fiican yihiin” | `[1, 1, 1, 1, 0, 0]` |
| “adeegyadu way xun yihiin maanta” | `[1, 1, 0, 1, 1, 1]` |

Each position in the vector is the count of the vocabulary word at the same position: the first number is the count of `adeegyadu`, the second is `way`, the third is `fiican`, the fourth is `yihiin`, the fifth is `xun`, and the sixth is `maanta`.

**Main limitation of Bag of Words:** It ignores word order and much of the context or meaning of a sentence. For example, “the service is good” and “the service is not good” may have very similar word-count representations even though their meanings are different. BoW can also create large, sparse vectors when a corpus contains many unique words. As a result, it may not represent meaning and relationships between words very well.

## 6. Challenges of Natural Language Processing for Somali

Somali is often described as a **low-resource language in NLP** because, compared with high-resource languages such as English, it has historically had fewer publicly available annotated datasets, language-processing tools, and widely used evaluation benchmarks. This does not mean that Somali has no NLP resources. Researchers and developers are building new datasets and tools, but important gaps remain.

### 6.1 Limited annotated datasets

Many NLP tasks require examples that have been labeled by people. For example, a sentiment-analysis model needs reviews labeled as positive or negative, while a part-of-speech tagger needs words labeled with grammatical categories.

For Somali, annotated resources for tasks such as sentiment analysis, named-entity recognition, part-of-speech tagging, and lemmatization are more limited than comparable English resources. This can make it difficult to train and evaluate reliable models. A model trained on a small collection of formal news articles may not perform well on social-media posts, regional vocabulary, or informal conversations.

The situation is improving. For example, Badel et al. (2023) created a Somali information-retrieval corpus containing 2,335 documents. This is a useful research resource, although one corpus cannot provide all the labeled data required for every NLP task.

### 6.2 Limited language-specific preprocessing resources

Many popular NLP libraries provide ready-made tools and language resources for English and several other languages. However, these resources do not always include Somali.

For example, the standard NLTK stopwords corpus does not list Somali among its available stopword languages. Therefore, a developer who wants to remove Somali stopwords may need to collect, review, and maintain a Somali stopword list instead of directly using `stopwords.words('somali')`.

This matters because an unsuitable stopword list may remove meaningful words or leave frequent words that are not useful for a particular task. Stopword removal should also depend on the task: a word that seems unimportant for one task may be important for sentiment, meaning, or grammar in another.

### 6.3 Stemming and lemmatization difficulties

Somali has rich word formation and uses inflectional and derivational forms. A word can appear in different forms depending on grammatical information or how it is formed. This can make it difficult for a system to recognize that related forms belong to the same root or lemma. In the review dataset above, for example, `adeegyadu` and `adeegaaga` both come from the same root `adeeg` (service), but without stemming or lemmatization they are counted as two different tokens.

**Stemming** attempts to reduce a word to a stem, often by applying rules that remove or change word endings. **Lemmatization** attempts to identify a word's dictionary or base form using linguistic information. These processes are not identical, and simply deleting a few final letters can produce an incorrect result.

Tools designed for English, such as the English stemmers and WordNet-based lemmatization resources commonly used with NLTK, should not be assumed to work correctly for Somali. A Somali system needs rules and language resources developed or evaluated for Somali.

Recent research shows that work is being done to address this gap:

- Badel et al. (2026) introduced **Somstem**, a hybrid stemmer for Somali. The authors note that no robust stemming solution existed for the language, which shows how much Somali's rich affixation complicates tasks such as machine translation and information retrieval.
- Gedi et al. (2026) developed a Somali lemmatization lexicon containing 5,584 root words and 78,629 derived forms, each with part-of-speech information. Their pilot lemmatizer, tested on 294 documents, reached 51.27% accuracy on full articles, 44.14% on excerpts, and 59.51% on short texts, showing that the task remains challenging.

These studies demonstrate both the difficulty of Somali morphology and the progress researchers are making.

### 6.4 Spelling variation and non-standard writing

People may write Somali words differently in online messages, social media, and formal publications. Informal writing can include shortened words, typing errors, inconsistent spelling, or different ways of representing the same expression. For example, `maanta` (today) may also be typed as `manta`.

This affects NLP because a computer may treat two spelling variants as separate tokens even when people understand them as the same word. The vocabulary becomes larger, useful examples are divided across variants, and search or classification systems may miss relevant text.

Text normalization can help, but normalization rules must be designed carefully so that they do not change the intended meaning.

### 6.5 Dialect variation and mixed-language text

Somali users may use regional vocabulary and different forms of expression. Online writing may also mix Somali with English or Arabic in the same sentence. This is often called **code-switching**.

For example: “Waxaan rabaa inaan machine learning barto.”

A system trained mainly on formal Somali may not recognize every dialect form or mixed-language expression. It may tokenize the sentence incorrectly, misunderstand some words, or assign an inaccurate category. Systems therefore need training and evaluation data that represent different writing styles and language combinations.

### Summary of Somali NLP challenges

| Challenge | How it affects an NLP system |
|---|---|
| Limited annotated datasets | Makes it harder to train and evaluate accurate models |
| Limited preprocessing resources | Requires developers to create and validate language-specific tools |
| Stemming and lemmatization | Makes it difficult to connect different forms of a word to a common base |
| Spelling variation | Causes one word or expression to appear as several different tokens |
| Dialect and code-switching | Reduces performance when the system has not seen those language varieties |

## 7. Case Study: Somali Information Retrieval

**Research paper:** Badel, A., Zhong, T., Tai, W., and Zhou, F. (2023). *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.* Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

**Research problem:** Somali information retrieval research was limited and often depended on translating a user's query into another language. The researchers addressed the lack of a dedicated Somali information-retrieval corpus.

| Aspect | Description |
|---|---|
| **Task** | Somali information retrieval: finding documents that are relevant to a user's search query |
| **Data** | An annotated corpus of 2,335 Somali documents collected from sources including Hiiraan Online, Dhacdo Net, and Somali poetry books |
| **Method** | The researchers constructed the corpus and developed a Somali information-retrieval system using pseudo-relevance feedback (PRF), a query-expansion technique |
| **Findings** | The work provided a dedicated resource for Somali information retrieval and demonstrated an approach to retrieving Somali documents without relying entirely on translating queries into another language |
| **NLP task from Question 3** | **Search and similarity** |

**Why this case study is important:** People who search for information in Somali should be able to find relevant Somali documents directly. The research addresses this need by creating a Somali-specific corpus and testing a retrieval approach. It also provides a resource that other researchers can use to develop and evaluate future Somali NLP applications.

## References

1. Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/

3. NLTK Project. (n.d.). *Sample usage for corpus: Word lists and stopwords*. https://www.nltk.org/howto/corpus.html

4. scikit-learn developers. (n.d.). *Feature extraction: Text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html

5. scikit-learn developers. (n.d.). *CountVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

6. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 7463–7469. https://aclanthology.org/2023.emnlp-main.462/

7. Badel, A. M., Tai, W., Xu, X., & Siraad, A. M. (2026). Somstem: A Hybrid Stemmer for the Somali Language. *2026 8th International Conference on Natural Language Processing (ICNLP)*, 271–275. IEEE. https://doi.org/10.1109/ICNLP69856.2026.11527795

8. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assoweh, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)*, 179–189. https://aclanthology.org/2026.africanlp-main.17/
