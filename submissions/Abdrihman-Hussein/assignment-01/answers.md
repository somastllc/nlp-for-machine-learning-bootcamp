# Assignment 1 – NLP Foundations

**Student:** Abdrihman Hussein

## 1. Natural Language Processing and Its Relationship to AI and ML

**Natural Language Processing (NLP)** is the branch of computer science and artificial intelligence that studies how computers can process, analyse, and generate human language. It covers both written text and speech, and it includes tasks such as classifying documents, extracting names and places, translating between languages, retrieving information, and producing new text (Jurafsky & Martin, 2025). NLP is therefore not only about "understanding" language; it is also about producing language that people can use.

The relationship between NLP, Artificial Intelligence (AI), and Machine Learning (ML) is often drawn as three nested circles, but that picture can be misleading. A clearer way to describe it is:

| Field | What it is | Role in a language application |
|---|---|---|
| **Artificial Intelligence (AI)** | The broad field that builds systems able to perform tasks we associate with intelligence | Sets the overall goal: a machine that can handle a customer request without a human |
| **Natural Language Processing (NLP)** | The part of AI that deals specifically with human language | Provides the language interface: reading the message, splitting it into words, recognising the intent |
| **Machine Learning (ML)** | A family of methods that learn patterns from data instead of relying only on hand-written rules | Supplies the classifier that learns from past examples which messages mean which thing |

NLP is a **subfield of AI**, not a subfield of ML. ML is one set of techniques that NLP systems may use. Early NLP systems relied heavily on grammars, dictionaries, and hand-written rules, and some still do. Modern systems usually combine linguistic processing with statistical or neural ML models, but the two are not the same thing (Hirschberg & Manning, 2015).

**Real-life example: a Somali mobile-money support line.** A customer writes:

> "Xawaaladda lacagta ayaa ii xirmay shalay, maanta waxba ma soo gaarin."

- **NLP** normalises the spelling, splits the sentence into tokens, and identifies the important expressions (`xawaaladda`, `lacagta`, `xirmay`, `maanta`). It may also detect that the message is in Somali rather than English.
- **ML** uses a classifier trained on hundreds of previous support tickets. Each ticket was labelled by an agent as, for example, *transfer delayed*, *wrong number*, or *agent complaint*. From these examples the model learns that words such as `xirmay` and `ma soo gaarin` are strong signals of a delayed transfer.
- **AI** is the whole system: it reads the message, predicts the intent, checks the transaction record, and decides whether to send troubleshooting steps or open a ticket.

The important point is that the three work together. NLP turns the language into a form the system can handle, ML learns the decision from data, and AI describes the overall intelligent behaviour.

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

These terms describe how a text collection is organised and then converted into model input.

| Term | Meaning |
|---|---|
| **Corpus** | A structured collection of texts gathered for analysis, training, or evaluation. The plural is *corpora*. |
| **Document** | One individual text item inside a corpus, such as one review, one article, or one message. |
| **Token** | The smallest unit produced when text is divided by a tokenizer. A token may be a word, a subword, a punctuation mark, or a character. |
| **Vocabulary** | The set of distinct token types that a system knows. Its size is usually written as \|V\|. |
| **Feature** | A measurable property of a document that is converted into a number and given to a model. In a bag-of-words model, each vocabulary item is one feature. |
| **Label** | The target answer attached to an example in supervised learning, such as a sentiment class or a topic. |

### Example dataset: Somali restaurant reviews

Consider a tiny corpus of three reviews collected to train a sentiment classifier.

| ID | Review | English meaning | Label |
|---|---|---|---|
| 1 | Cuntadu aad bay u macaanayd | The food was very delicious | Positive |
| 2 | Adeegga aad buu u gaabis ahaa | The service was very slow | Negative |
| 3 | Cuntadu macaanayd laakiin adeegga gaabis ahaa | The food was delicious but the service was slow | Negative |

The terms apply as follows:

- **Corpus:** the three reviews together. It is small, but it is still a corpus because it is a deliberate collection used for one purpose.
- **Document:** review 2 is one document. It is treated as a single example with a single sentiment label.
- **Token:** review 1 is split into `cuntadu`, `aad`, `bay`, `u`, `macaanayd`. If punctuation were present, the tokenizer would normally produce punctuation tokens as well. Different tokenizers can produce different tokens from the same sentence (Bird et al., 2009).
- **Vocabulary:** the distinct token types across all three reviews are `cuntadu`, `aad`, `bay`, `u`, `macaanayd`, `adeegga`, `buu`, `gaabis`, `ahaa`, and `laakiin`. The vocabulary size is 10.
- **Feature:** a numeric property derived from the text. For example, the count of the word `gaabis` is a feature; it is 0 in review 1, 1 in review 2, and 1 in review 3. A model does not see the words themselves, only these numbers.
- **Label:** the sentiment column, `Positive` or `Negative`. The label is the answer the model should learn to predict; the review text is the input.

A useful distinction is **token versus type**. The three reviews contain 17 word tokens in total but only 10 distinct types, which is why the vocabulary is smaller than the total token count. This difference matters when estimating how much data a model really has: a corpus can be long in tokens yet still have a small vocabulary.

| ID | Text | Feature: count of `gaabis` | Feature: count of `macaanayd` | Label |
|---|---|---:|---:|---|
| 1 | Cuntadu aad bay u macaanayd | 0 | 1 | Positive |
| 2 | Adeegga aad buu u gaabis ahaa | 1 | 0 | Negative |
| 3 | Cuntadu macaanayd laakiin adeegga gaabis ahaa | 1 | 1 | Negative |

## 3. Four Common NLP Tasks

NLP tasks are often distinguished by the **unit** they operate on and the **shape of their output**.

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | One document, sentence, or message | One label (or a set of labels) for the whole text | Deciding whether a Somali customer review is positive or negative |
| **Sequence labelling** | A sequence of tokens, usually a sentence | One label per token | Marking `Muqdisho` as a location and `Cali` as a person |
| **Search and similarity** | A query plus a collection of documents | A ranked list of documents, or similarity scores | A Somali news search engine returning articles about a drought |
| **Text generation** | A prompt, a source text, or a conversation history | New text | Translating an English sentence into Somali, or summarising a long article |

**Text classification** produces a decision about the whole input. Spam detection, sentiment analysis, and news topic classification are all examples. The output space is small and fixed: in the review dataset above, it is `{Positive, Negative}`.

**Sequence labelling** produces a decision for every position in the input. Part-of-speech tagging, named-entity recognition, and chunking belong here. The output is aligned with the input tokens, so the model must learn local context and sometimes long-distance dependencies. In the sentence `Cali wuxuu joogaa Muqdisho`, a named-entity recogniser would output something like `B-PERSON`, `O`, `O`, `B-LOCATION`.

**Search and similarity** produces a ranking rather than a class. The system compares a query with every document in a collection and orders the documents by estimated relevance. Unlike classification, there is no fixed set of labels; relevance depends on the query and on the user's information need.

**Text generation** produces a new sequence. Machine translation, summarisation, and chatbot responses are examples. Evaluation is harder than for classification because many different outputs can be correct, so researchers often combine automatic metrics with human judgement.

The four tasks are related. A search engine may use a classifier to filter spam, a sequence labeller to extract entities, and a generator to write a short answer. This is why modern NLP systems are usually pipelines or combinations of several tasks rather than a single model.

## 4. Supervised and Unsupervised Learning in NLP

**Supervised learning** uses training examples that contain both an input and the correct output. In NLP the input is usually a text and the output is a label, a tag sequence, or another text. The model learns a mapping from input to output by comparing its predictions with the given answers and reducing the difference (the loss).

A supervised example is the sentiment classifier from Question 2. Each review already has a `Positive` or `Negative` label, so the model can be trained to reproduce those labels on new reviews. Another example is named-entity recognition trained on sentences where every token has been tagged by a human annotator.

**Unsupervised learning** uses texts without target labels. The model looks for structure in the data: groups of similar documents, recurring topics, or a lower-dimensional representation. A classic example is topic modelling with Latent Dirichlet Allocation, which discovers groups of words that tend to occur together without being told the topic names in advance (Blei et al., 2003). Another example is clustering Somali news articles by their word distributions; the clusters may later be inspected and named by a human.

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Texts plus target labels | Texts only |
| Objective | Predict a specified target | Discover structure or groups |
| Example | Sentiment classification, named-entity recognition | Topic modelling, document clustering |
| Evaluation | Compare predictions with gold labels | Inspect the usefulness of the discovered structure |
| Main cost | Producing accurate labels | Interpreting the results and judging their value |

**Why one task needs labels and the other does not.** In supervised learning the label is the teaching signal. The loss function compares the model's output with the known target, and the model updates its parameters to reduce that error. Without labels there is no target to compare against, so the model cannot learn that specific mapping. In unsupervised learning the objective is internal to the data: for example, minimise the distance between documents in the same cluster while maximising the distance between clusters. No external answer is required because the method is not trying to reproduce a human-provided category.

The boundary is not perfectly sharp. **Self-supervised learning** creates labels automatically from the text itself. In masked language modelling, some tokens are hidden and the model is trained to predict them from the surrounding context (Devlin et al., 2019). This is how large language models are pre-trained on raw text without human annotation, which is especially important for languages such as Somali where labelled data are scarce.

## 5. Why Text Must Be Converted into Numerical Representations

Most machine-learning algorithms operate on vectors and matrices. They compute dot products, distances, weighted sums, and gradients. Raw text is a sequence of discrete symbols, and symbols do not support these operations directly. A computer can check whether two strings are identical, but it cannot naturally compute that `macaan` is closer to `fiican` than to `gaabis` unless the words have been placed in a numeric space. Converting text into numbers gives every document a fixed-length representation that a model can compare, average, and update during training (Jurafsky & Martin, 2025).

### A simple method: Bag of Words

**Bag of Words (BoW)** is one of the simplest representations, and it is an application of the vector-space idea in which documents are compared through the words they contain (Salton et al., 1975). The procedure is:

1. Tokenise every document.
2. Build a vocabulary from the training corpus.
3. Represent each document by the count of every vocabulary item in it.

Using the three reviews from Question 2, with the vocabulary ordered as `[cuntadu, aad, bay, u, macaanayd, adeegga, buu, gaabis, ahaa, laakiin]`, the documents become:

| Document | Vector |
|---|---|
| 1: Cuntadu aad bay u macaanayd | `[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]` |
| 2: Adeegga aad buu u gaabis ahaa | `[0, 1, 0, 1, 0, 1, 1, 1, 1, 0]` |
| 3: Cuntadu macaanayd laakiin adeegga gaabis ahaa | `[1, 0, 0, 0, 1, 1, 0, 1, 1, 1]` |

Each position corresponds to one vocabulary item, and the value is how many times that item occurs. The result is a numeric matrix that a classifier can use. The same idea is implemented by tools such as scikit-learn's `CountVectorizer` (scikit-learn developers, n.d.).

### Main limitation: loss of word order and context

The biggest weakness of BoW is that it discards word order. The Somali sentences

> "Nin baa libaax dilay" (A man killed a lion)

> "Libaax baa nin dilay" (A lion killed a man)

contain exactly the same words and therefore receive exactly the same BoW vector, even though they describe opposite events. Because the model only sees counts, it cannot distinguish who did what to whom. This is not a small detail: negation, questions, and many meaning-changing constructions depend on order and structure.

BoW has several related limitations:

- **Sparsity and size.** As the vocabulary grows, most values in each vector are zero. A corpus with 50,000 distinct words produces 50,000-dimensional vectors, which are expensive to store and can make learning harder.
- **No meaning or similarity.** `cunto` and `cuntada` are treated as unrelated features, and synonyms share no information. This is a serious problem for a morphologically rich language such as Somali.
- **Out-of-vocabulary words.** A word that was not in the training vocabulary has no feature at all, so the model cannot use it at test time.
- **Length bias.** Long documents have larger counts than short ones, which can dominate similarity calculations unless the vectors are normalised.

These problems are partly addressed by **TF-IDF**, which down-weights very common words, by **n-grams**, which keep some local order, and by **word embeddings**, which map words to dense vectors so that related words are close together. Even so, the BoW discussion is useful because it shows the central trade-off in text representation: the more information the representation keeps, the more complex and data-hungry the model becomes.

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A language is called **low-resource** in NLP when the digital text, annotated datasets, evaluation benchmarks, and processing tools available for it are limited relative to the task. The term describes the resource situation, not the number of speakers: Somali is spoken by more than 20 million people, yet its NLP resources are much smaller than those of English (Joshi et al., 2020). The problem is not only the amount of text but also the availability of annotated data, evaluation sets, and tools; Hedderich et al. (2020) survey the methods researchers use when these resources are missing.

Somali is written in the Latin script, which was officially adopted in 1972. That makes the writing system relatively accessible to existing tools, but it does not remove the deeper problems described below.

### 6.1 Limited annotated data and benchmarks

Supervised models need labelled examples, and labels are expensive because they require speakers who understand both the language and the task. For Somali, resources exist but are small and unevenly distributed across tasks. MasakhaNEWS, for example, provides news-topic classification data for 16 African languages, including Somali, but the Somali portion contains only about 1,500 articles across seven topics (Adelani et al., 2023). Information retrieval faced a similar gap: the CIRAL shared task was organised to support cross-lingual retrieval for African languages, and the Somali Information Retrieval Corpus was created because no dedicated Somali retrieval collection existed (Adeyemi et al., 2023; Badel et al., 2023).

**Effect on an NLP system.** A model trained on a small sample can memorise the training data and fail on new text. Evaluation is also less reliable, because a small test set produces noisy scores. A classifier trained mainly on formal news articles may perform poorly on social-media posts, customer messages, or regional vocabulary.

### 6.2 Rich morphology and word forms

Somali has extensive inflection and derivation, and grammatical information is often attached to the word rather than expressed by separate words. A single root can appear in many forms:

| Form | Meaning |
|---|---|
| `cunto` | food |
| `cuntada` | the food |
| `cuntadu` | the food (as subject) |
| `adeeg` | service |
| `adeegga` | the service |

A word-level model treats each form as a separate vocabulary item. The vocabulary becomes larger, each form is seen less often during training, and the model struggles with forms it has never seen. This is a form of **data sparsity** caused by morphology rather than by corpus size.

**Effect on an NLP system.** A sentiment classifier that learned the positive signal of `macaan` may not recognise `macaanayd`. A search engine may fail to match `adeeg` with `adeegga`. Possible responses include stemming or lemmatisation for Somali, character n-grams, and subword tokenisation such as Byte-Pair Encoding (Sennrich et al., 2016), which splits rare words into smaller units that the model can reuse.

### 6.3 Spelling variation and non-standard writing

Somali online text is not always edited. Writers shorten words, join or separate them differently, double vowels inconsistently, and substitute characters. For example, `maanta` ("today") may appear as `manta`, `waxaan` may be shortened to `waan`, and the same word may be written with or without a final vowel. Arabic-influenced spelling and keyboard limitations add further variants.

**Effect on an NLP system.** Two spellings of the same word become two different tokens. The vocabulary grows, the training signal is split between variants, and a search system may miss a relevant document simply because the query used a different spelling. Text normalisation can help, but the rules must be designed carefully so that they do not merge words that actually mean different things.

### 6.4 Code-switching and dialect variation

Somali speakers frequently mix Somali with English or Arabic in the same message, especially in technical, business, and educational contexts:

> "Waxaan rabaa inaan laptop cusub iibsado."

Somali also has regional varieties, including Maxaa tiri (the standard variety used in broadcasting and education) and Maay, which differs in pronunciation, vocabulary, and grammar. A model trained on one variety may not transfer well to another.

**Effect on an NLP system.** A language identifier may label a mixed message as English, a tokenizer may split English and Somali words inconsistently, and a classifier trained on standard Somali may misinterpret dialect words. Systems therefore need training and evaluation data that represent code-switching and regional variation, or at least a clear statement of the variety they support.

### 6.5 Gaps in basic language tools

Many standard NLP libraries ship stopword lists, stemmers, part-of-speech taggers, and named-entity recognisers for English and a few other languages, but not for Somali. Developers must build or adapt these resources themselves, and errors in preprocessing propagate to every later stage.

**Effect on an NLP system.** A borrowed English stopword list will not remove Somali function words, while a poorly designed Somali list may remove words that carry sentiment or grammatical meaning. A part-of-speech tagger trained on another language will produce unreliable tags, which then mislead any downstream parser or entity recogniser.

### Summary

| Challenge | Example | Effect on an NLP system | Possible response |
|---|---|---|---|
| Limited annotated data | Few labelled Somali sentiment or news datasets | Overfitting, unstable evaluation, weak transfer to new domains | Transfer learning, self-supervised pre-training, community annotation |
| Morphology and word forms | `cunto`, `cuntada`, `cuntadu` | Vocabulary explosion, sparsity, unseen forms at test time | Somali stemming/lemmatisation, subword tokenisation, character n-grams |
| Spelling variation | `maanta` vs `manta` | The same word is counted as different tokens; lower recall | Normalisation rules, spelling correction, fuzzy matching |
| Code-switching and dialects | Somali mixed with English; Maay vs Maxaa tiri | Language-ID errors, poor performance on unseen varieties | Multilingual and multi-dialect training data, language identification |
| Tool gaps | No standard Somali stopword list or POS tagger | Preprocessing errors propagate through the pipeline | Build and publish Somali-specific resources, evaluate carefully |

## 7. Case Study: Somali Information Retrieval

**Paper:** Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources.* Proceedings of EMNLP 2023, pages 7463–7469.

| Aspect | Description |
|---|---|
| **Task** | Somali information retrieval: given a Somali query, return the Somali documents that are relevant to it |
| **Data** | An annotated corpus of 2,335 Somali documents collected from Hiiraan Online, Dhacdo Net, and Somali poetry books |
| **Method** | The authors worked with language experts and NLP researchers to construct and annotate the corpus, then built a Somali retrieval system using pseudo-relevance feedback (PRF), a query-expansion technique |
| **Findings** | The work produced a dedicated Somali retrieval resource and showed that retrieval can be developed for Somali without depending entirely on translating queries into another language |
| **NLP task from Question 3** | **Search and similarity** |

**Why the paper matters.** Before this work, Somali retrieval research relied mainly on query translation: the user's Somali query was translated into a high-resource language, the search was performed there, and the results were mapped back. That approach introduces translation errors and depends on a good translation system, which is itself difficult for a low-resource language. A dedicated corpus allows researchers to evaluate Somali retrieval directly and to build and compare Somali-specific methods.

The paper is also useful as a case study of how resource creation and modelling go together. The corpus supports not only search but also other tasks, such as question answering and text classification, because it provides realistic Somali text with relevance judgements.

**Limitations.** The corpus contains 2,335 documents, which is small compared with standard English retrieval collections, and its sources are concentrated in news and poetry. The findings therefore may not transfer automatically to social media, legal text, or speech transcripts. The authors present the corpus as a foundation for further work rather than as a complete solution.

## References

1. Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). *MasakhaNEWS: News topic classification for African languages.* Proceedings of IJCNLP-AACL 2023. https://arxiv.org/abs/2304.09972
2. Adeyemi, M., Oladipo, A., Zhang, X., Alfonso-Hermelo, D., Rezagholizadeh, M., Chen, B., & Lin, J. (2023). *CIRAL at FIRE 2023: Cross-lingual information retrieval for African languages.* Proceedings of the 15th Annual Meeting of the Forum for Information Retrieval Evaluation. https://doi.org/10.1145/3632754.3633076
3. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). *Somali Information Retrieval Corpus: Bridging the gap between query translation and dedicated language resources.* Proceedings of EMNLP 2023, 7463–7469. https://aclanthology.org/2023.emnlp-main.462/
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing text with the Natural Language Toolkit.* O'Reilly Media. https://www.nltk.org/book/
5. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.
6. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019.* https://arxiv.org/abs/1810.04805
7. Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., & Klakow, D. (2020). *A survey on recent approaches for natural language processing in low-resource scenarios.* arXiv:2010.12309. https://arxiv.org/abs/2010.12309
8. Hirschberg, J., & Manning, C. D. (2015). Advances in natural language processing. *Science, 349*(6245), 261–266.
9. Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of ACL 2020.*
10. Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
11. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620.
12. scikit-learn developers. (n.d.). *Feature extraction: Text feature extraction.* https://scikit-learn.org/stable/modules/feature_extraction.html
13. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016.* https://arxiv.org/abs/1508.07909
