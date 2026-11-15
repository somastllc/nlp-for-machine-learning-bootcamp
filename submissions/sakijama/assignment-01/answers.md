# Assignment 1 – NLP Foundations

---

## Question 1. Natural Language Processing and Its Relationship to AI and ML

**Natural Language Processing (NLP)** is a field of computer science concerned with enabling computers to analyze, interpret, retrieve, and produce human language. It deals with language in forms such as written messages and speech transcripts. NLP is an application area within **Artificial Intelligence (AI)**, the wider field of creating computational systems that perform tasks requiring intelligent behavior. **Machine Learning (ML)** is a set of methods through which systems learn useful patterns from examples rather than depending entirely on manually specified rules. Many modern NLP applications use ML, although NLP can also use rule-based methods (Jurafsky & Martin, 2026).

**Real-life example:** A municipal water service receives written complaints through an online form. An NLP system reads each complaint and identifies whether it concerns a broken pipe, poor water pressure, or a billing problem. AI describes the broader objective of automating an intelligent service; NLP handles the language in the complaint; and ML can learn to categorize complaints from historical examples that staff have already classified. The resulting category helps route the complaint to the appropriate team. This is an illustrative application, not a reported research experiment.

## Question 2. Corpus, Document, Token, Vocabulary, Feature, and Label

The following terms describe the structure of a typical text dataset (Jurafsky & Martin, 2026; Manning et al., 2008).

| Term | Definition | Application in the example below |
|---|---|---|
| **Corpus** | A collection of texts assembled for analysis or model development. | All three library feedback entries together. |
| **Document** | One text item treated as an individual example. | One student's feedback statement. |
| **Token** | A unit produced by splitting text using a particular tokenization procedure; it may be a word, subword, or symbol. | `quiet` is a token in the first entry under word-level tokenization. |
| **Vocabulary** | The set of distinct token types used by the system. | The eight distinct lowercase words in this dataset. |
| **Feature** | A measurable, usually numerical property supplied to a model. | Whether the word `quiet` appears in a document (1 or 0). |
| **Label** | A target answer associated with an example in supervised learning. | `positive` or `negative` feedback. |

**Original illustrative dataset: university library feedback**

| Document ID | Text | Label | Binary feature: contains `quiet` |
|---|---|---|---:|
| D1 | Quiet room and helpful staff | Positive | 1 |
| D2 | Noisy room and slow service | Negative | 0 |
| D3 | Helpful staff and quiet room | Positive | 1 |

For a simple demonstration, lowercase the words and split on spaces, keeping no punctuation. The **corpus** consists of D1–D3; each row is a **document**. D1 yields five **tokens**: `quiet`, `room`, `and`, `helpful`, `staff`. Across the three documents, the **vocabulary** is `{quiet, room, and, helpful, staff, noisy, slow, service}`—**eight** distinct token types. The binary **feature** records the presence of `quiet`; the **label** is the human-assigned assessment of the feedback. This distinction matters: a feature is input information, while a label is the outcome the supervised model is trained to predict.

## Question 3. Four Common NLP Tasks

NLP tasks differ mainly in what they take as input and what they are expected to return (Jurafsky & Martin, 2026; Manning et al., 2008).

| Task | What it does | Input | Output | Original illustrative example |
|---|---|---|---|---|
| **Text classification** | Assigns one or more categories to a whole text. | A written customer request. | A category such as `refund`, `delivery`, or `account`. | An online shop routes “My parcel arrived damaged” to `delivery`. |
| **Sequence labeling** | Assigns a tag to each token or span of tokens, often identifying entities or grammatical roles. | “Dr. Nura visited Hargeisa on Monday.” | Tags such as `PERSON`, `LOCATION`, `DATE`, and `O` (other). | `Dr. Nura` → PERSON, `Hargeisa` → LOCATION, `Monday` → DATE. |
| **Search and similarity** | Ranks or finds documents related to a query or another document. | Query: “scholarship application deadlines” and a document collection. | A ranked list of relevant documents and possibly relevance scores. | A university portal returns the most relevant funding announcements. |
| **Text generation** | Produces a new sequence of words based on an instruction or source text. | A meeting transcript plus “Summarize the decisions.” | A newly written summary. | A tool writes a concise summary of approved project actions. |

The essential distinction is the **unit of output**: one document-level category, token-level tags, ranked documents, or generated language.

## Question 4. Supervised and Unsupervised Learning in NLP

**Supervised learning** trains a model using examples that include both input text and a target answer. The target labels tell the model what it should predict, so learning can be guided by the difference between its prediction and the known answer. **Unsupervised learning** searches for patterns or structure in input texts without task-specific target answers; the model may group similar documents or identify recurring themes instead of predicting a predefined category (Manning et al., 2008).

**Supervised example:** A university has emails marked by staff as `admissions`, `examinations`, or `fees`. A classifier learns from those labeled messages and predicts the category of a new email. Labels are required **during training** because the intended output categories must be taught through examples; a new email does not need a known label when the trained model makes its prediction.

**Unsupervised example:** A university wants to explore hundreds of unlabeled student suggestions. It represents each suggestion numerically and clusters similar suggestions, perhaps discovering groups about internet access, cafeteria conditions, and transport. No predefined theme labels are needed to fit the clusters, because similarity patterns—not correct category answers—determine the groups. Researchers may later name and interpret the clusters.

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Text plus target labels | Text without task-specific target labels |
| Objective | Learn to predict a specified answer | Discover structure or patterns |
| Example output | `fees` for an incoming email | Cluster 2 for suggestions with related content |
| Main challenge | Obtaining accurate, representative labels | Interpreting whether discovered patterns are useful |

The main difference is **whether a target answer is supplied during training**, not whether the data contains text or numbers.

## Question 5. Why Text Needs Numerical Representation

Most standard ML algorithms perform mathematical operations on vectors, such as calculating distances, weighted sums, or probabilities. Raw words cannot directly be used in these operations as consistent numerical features. A **text representation** converts documents into numbers while keeping feature positions consistent across examples (Manning et al., 2008).

A simple method is **Bag of Words (BoW)** with word-count features: choose a vocabulary, then count each vocabulary word in every document. For this original example, consider two short event descriptions:

- Document A: “garden garden opens”
- Document B: “museum opens”

After lowercasing and separating words on spaces, use the ordered vocabulary **[`garden`, `museum`, `opens`]**.

| Document | `garden` | `museum` | `opens` | Numerical vector |
|---|---:|---:|---:|---|
| A | 2 | 0 | 1 | `[2, 0, 1]` |
| B | 0 | 1 | 1 | `[0, 1, 1]` |

For A, `garden` appears twice, `museum` zero times, and `opens` once. Both documents therefore have comparable three-dimensional vectors. These can be used as inputs to a classifier or to a document-comparison algorithm. A new document must use the **same ordered vocabulary**, rather than creating new feature positions at prediction time.

**Main limitations:** BoW disregards word order, so different sentences containing the same words can produce the same vector despite conveying different meanings. It also treats related words as separate features and may create large, sparse vectors. More advanced representations can address some, but not all, of these weaknesses (Jurafsky & Martin, 2026; Manning et al., 2008).

## Question 6. Challenges for Low-Resource Languages Such as Somali

A language is **low-resource in NLP** when suitable digital corpora, annotated training examples, evaluation sets, or language-specific tools are scarce relative to the needs of a particular task. This designation describes the availability of computational resources—not the expressive richness of the language or the number of people who speak it. Research on Somali documents and morphological analysis identifies several practical obstacles (Badel et al., 2023; Gedi et al., 2026).

### Challenge 1: Limited task-specific datasets and annotations

High-quality NLP systems need representative examples. Collecting and legally reusing digital Somali text can be difficult, and tasks such as named-entity recognition or search evaluation also require labor-intensive human annotation. Badel et al. (2023) specifically developed a Somali information-retrieval test corpus because suitable resources were limited; their collection required document cleaning and human relevance judgments. Gedi et al. (2026) likewise identify a shortage of human-annotated Somali lemmatization resources.

**Effect on performance:** With few or narrow examples, models may learn patterns specific to the available news articles or websites rather than performing reliably on other kinds of writing. Small evaluation sets can also make it harder to judge how well a system generalizes. Collecting texts from diverse sources and carefully checking annotations are important responses.

### Challenge 2: Morphologically complex word forms

**Morphology** concerns how words change form to express grammatical information. Somali uses productive inflectional and derivational processes, so multiple surface forms may be related to a common base. Gedi et al. (2026) built a Somali resource containing **5,584 root words and 78,629 derived forms**, illustrating how many related forms must be handled in computational analysis.

**Effect on performance:** A simple word-level model can treat related forms as unrelated vocabulary entries. This increases sparsity, makes unseen forms harder to process, and can reduce matches in search systems. Lemmatization, morphology-aware processing, and subword methods may help, but require careful language-specific evaluation.

### Challenge 3: Variation in spelling and limited preprocessing tools

The same Somali expression may appear with different spellings in digital text. Badel et al. (2023) explicitly report inconsistent written forms and the lack of a fully functional Somali stemmer as limitations of their information-retrieval study. A **stemmer** attempts to reduce related words to a shared form; without dependable normalization tools, variants can remain separate in a search index.

**Effect on performance:** A search engine may miss a relevant document if the query and document use different spellings. Similar mismatches can reduce the effectiveness of classifiers built on exact word matches. A possible response is to test spelling normalization and language-specific tokenization against real Somali data rather than assume that tools built for English will work unchanged.

Together, these issues reinforce one another: sparse data makes it harder to build language-aware tools, while inconsistent word forms make small datasets less effective.

## Question 7. Research Case Study: Somali Information Retrieval

### Paper and research objective

**Paper:** *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*  
**Authors:** Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou  
**Publication:** *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462

Badel et al. (2023) aimed to create a dedicated evaluation corpus for **Somali-language information retrieval** and test a method for improving how documents are found in response to Somali queries. This addresses the practical problem of searching Somali documents without depending entirely on translation into other languages.

### Data and sources

The authors compiled **2,335 Somali documents**, amounting to **578,216 words**, from sources including Hiiraan Online, Dhacdo Net, and publicly available cultural and historical books. They prepared **16 search queries** and document-relevance judgments with input from native speakers, linguists, and NLP researchers. Their cleaning removed items such as HTML and non-Somali content (Badel et al., 2023).

### Method

The researchers built a Python-based inverted index after tokenization and the removal of Somali stop words. They used the **BM25** ranking model and tested **pseudo-relevance feedback (PRF)**: retrieve initial high-ranking documents, select **five expansion terms**, add them to the original query, and retrieve again. They compared the PRF-based approach with a TF-IDF baseline, reporting precision, recall, and F-measure at the top 10 and top 20 retrieved results (Badel et al., 2023).

### Findings

The paper's **Table 3** reports the following F-measures:

| Method | F@10 | F@20 |
|---|---:|---:|
| TF-IDF baseline | 0.537 | 0.436 |
| PRF approach | **0.721** | **0.572** |

PRF performed better than the baseline on both reported measures in this evaluation. Its reported **recall at 20** was **0.917**, versus **0.689** for the baseline. These are findings for the study's own corpus and test setup, **not** a general accuracy rate for every Somali search application (Badel et al., 2023, Table 3).

### Limitations and task category

The authors identified **inconsistent spelling** of Somali terms and the **absence of a fully functional Somali stemmer** as limitations. They also proposed expanding the corpus in future work. Since the experiments evaluated only 16 queries on a specifically collected dataset, further evaluation on different text sources and queries would be valuable; this last point is an inference from the study's scope rather than a separately reported experimental result (Badel et al., 2023).

This case study belongs to **search and similarity** (Question 3). The **input** is a Somali search query plus a document collection; the **output** is a ranked list of documents judged likely to answer the query. Its significance is both practical and methodological: it offers an evaluation resource for Somali search and demonstrates a measurable retrieval improvement in the reported setting.

---

## References

Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information retrieval corpus: Bridging the gap between query translation and dedicated language resources. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 7463–7469). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.462

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. In *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)* (pp. 179–189). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.africanlp-main.17

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models* (3rd ed., online manuscript, August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071
