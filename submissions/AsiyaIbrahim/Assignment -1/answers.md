# Assignment 1 – NLP Foundations

## 1. Natural Language Processing, AI and Machine Learning

**Definition.** Natural Language Processing (NLP) is the field concerned with building computational systems that can analyse, understand and generate human language, in written or spoken form (Jurafsky & Martin, 2024; Hirschberg & Manning, 2015). It combines linguistics, computer science and statistics.

**Relationship.** The three fields are closely related, but they are not strictly nested:

| Field | Scope | Role relative to NLP |
|---|---|---|
| Artificial Intelligence (AI) | Building systems that perform tasks normally requiring human intelligence (Russell & Norvig, 2021) | The broadest field. NLP is one of its major subfields, alongside computer vision and robotics. |
| Machine Learning (ML) | Algorithms that improve at a task by learning patterns from data instead of following hand-written rules (Mitchell, 1997) | A major method used in modern NLP. |
| NLP | Language-specific AI | Uses ML (especially deep learning) to handle text and speech. |

NLP also draws on non-ML techniques, such as rule-based grammars, so it is not entirely dependent on ML.

**Real-life example: spam filtering in an email service.**

1. *NLP* reads the email text and turns it into a form a computer can work with (cleaning, tokenizing, extracting features).
2. *ML* learns from thousands of past emails marked "spam" or "not spam" and fits a classifier, such as logistic regression or a neural network.
3. *AI* is the overall system: it processes the incoming message, makes a decision, and moves the email to the junk folder without human intervention.

## 2. Core terms, illustrated with one dataset

**Example dataset:** a small Somali text-topic dataset. The sample below is invented for illustration.

| ID | Text (document) | Label |
|---|---|---|
| 1 | Ardayda  ayaa imtixaanka ku paasay | Education |
| 2 | Dowladda ayaa soo saartay miisaaniyad cusub | Politics |
| 3 | Dhakhaatiirta ayaa tallaal bixiyay | Health |

| Term | Definition | Application to the dataset |
|---|---|---|
| **Corpus** | A large, structured collection of texts used for linguistic or computational analysis (Jurafsky & Martin, 2024) | The entire collection of texts, i.e. all three rows here and thousands in a real dataset. |
| **Document** | One unit of text within a corpus; its size is defined by the task | One text, such as row 1. In other tasks a document could be a tweet, article or paragraph. |
| **Token** | The smallest unit a system processes, such as a word, subword or character (Manning et al., 2008) | Row 1 splits into the tokens *ardayda, jaamacadda, ayaa, imtixaanka, galay*. |
| **Vocabulary** | The set of unique tokens in the corpus (the "types") | All distinct tokens across the three texts. *Ayaa* appears in all three but counts once in the vocabulary. |
| **Feature** | A measurable property of the input that a model uses to make predictions | Token counts, such as "*ardayda* appears 1 time", or word-frequency weights such as TF-IDF. |
| **Label** | The target output the model should predict (in supervised learning) | The topic: Education, Politics or Health. |

**Key distinction:** features are inputs derived from the text, while the label is the answer the model is trained to predict. The model learns that features such as *ardayda* correlate with the label "Education".

## 3. Four common NLP tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A document (or sentence) | One label from a fixed set | A text → "Education" (topic); a tweet → "toxic"/"not toxic" |
| **Sequence labeling** | A sequence of tokens | One label per token | "Muhidin lives in Hargeisa" → PERSON, –, –, LOCATION (named entity recognition); part-of-speech tagging |
| **Search and similarity** | A query plus a collection of documents (or two texts) | A ranked list of documents, or a similarity score | A search engine ranks pages for a query (Manning et al., 2008); duplicate-question detection |
| **Text generation** | A prompt or source text | New text | Machine translation, summarization, chatbots (Jurafsky & Martin, 2024) |

The main difference is the granularity of the output. Classification gives one label per text, sequence labeling gives one label per token, search gives a ranking, and generation gives a whole new sequence.

## 4. Supervised vs. unsupervised learning in NLP

| Aspect | Supervised | Unsupervised |
|---|---|---|
| Training data | Inputs paired with human-provided labels | Inputs only, with no labels |
| Goal | Learn a mapping from input to a known target | Discover hidden structure in the data |
| NLP example | Spam or fake-news detection with a classifier | Topic modelling with Latent Dirichlet Allocation (Blei et al., 2003); clustering similar documents |
| Evaluation | Compare predictions with gold labels (accuracy, F1) | Indirect: coherence, human inspection |

**Why the difference?**

- **Supervised tasks need target labels** because the algorithm has to be told what the correct answer is. Its error is measured by comparing predictions with those labels. To teach a model to detect fake news, humans must first mark which posts are fake.
- **Unsupervised tasks do not** because the objective is defined by the data itself. LDA, for instance, groups words that co-occur frequently into "topics" without anyone saying what the topics should be. The categories emerge from patterns, not from a prior answer key.

Modern language models blur this line: they are pretrained with *self-supervised* learning, where the label (the next or masked word) comes from the text itself, and are then fine-tuned with labelled data (Devlin et al., 2019).

## 5. Why text must become numbers

**Reason.** ML models are mathematical functions: they multiply, add and optimize numeric values (matrices and vectors). Raw strings have no inherent arithmetic meaning, so words must first be mapped to numerical vectors (Goldberg, 2017; Jurafsky & Martin, 2024). Representation quality largely determines model quality.

**Simple method: Bag-of-Words (BoW).** Each document becomes a vector with one position per vocabulary word. The value is how many times that word appears (Manning et al., 2008).

Using the three texts above, and a vocabulary of just three words *[ardayda, dowladda, ayaa]*:

- Text 1 → [1, 0, 1]
- Text 2 → [0, 1, 1]
- Text 3 → [0, 0, 1]

**Main limitation.** BoW discards word order and meaning. "Dog bites man" and "man bites dog" get identical vectors, and the representation cannot tell that two different words are synonyms. Vectors are also very long and sparse because they have one dimension per vocabulary word (Jurafsky & Martin, 2024). Dense embeddings such as word2vec (Mikolov et al., 2013) and contextual models such as BERT (Devlin et al., 2019) were developed to address this.

## 6. Challenges of NLP for low-resource languages (Somali)

Somali has many speakers but few digital resources, so it counts as low-resource. Joshi et al. (2020) show that the world's languages are extremely unevenly served by NLP, with most African languages in the lowest tiers.

**Challenge 1: Limited datasets and annotated corpora.**

- **Description:** there is little digitized text, and very few labelled datasets for tasks like sentiment analysis or fake-news detection.
- **Effect:** data-hungry models cannot learn reliably. Multilingual models trained mostly on other languages often under-capture Somali syntax, semantics and morphology because Somali gets little training data and vocabulary in them (Mohamed et al., 2025). Evaluation is also hard without standard benchmarks.

**Challenge 2: Rich morphology and word-form variation.**

- **Description:** Somali is a Cushitic language with inflection and affixation. A single root can appear in many surface forms, and it also has articles and particles that attach to words.
- **Effect:** a word-based vocabulary balloons and becomes sparse, so many forms are rare or unseen in training. Without good lemmatizers or stemmers, search and classification quality drops. Recent work builds Somali lemmatization datasets and hybrid rule-based methods for exactly this reason (Gedi et al., 2026).

**Challenge 3: Spelling variation and informal, mixed-language text.**

- **Description:** social-media Somali often has inconsistent spelling, dialect differences, and code-switching with English, Arabic or Italian.
- **Effect:** identical words are treated as different tokens, which hurts tokenization and classification, and language-identification tools may mislabel the text. Models trained on formal news text (the most available source) perform worse on this noisy data.

## 7. Case study

**Reference:** Mohamed, M. A., et al. (2025). *Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models.* arXiv:2503.18117. https://arxiv.org/abs/2503.18117

**Task.** The authors built two human-annotated, social-media-sourced Somali datasets for fake news and toxicity classification, and developed a transformer-based monolingual Somali language model named SomBERTa. The motivation is that low-resource languages like Somali struggle with misinformation and toxic-content detection because of scarce annotated data and the lack of language-specific pretrained models.

**Data.** The paper reports three data contributions: a large, diverse Somali text corpus for pre-training, human-annotated classification datasets for fine-tuning, and the SomBERTa model itself, evaluated on fake news, toxicity and news-topic classification.

**Method.** SomBERTa uses self-supervised pretraining followed by multi-task supervised fine-tuning. This follows the unsupervised-then-supervised pattern discussed in Q4: unlabelled Somali text teaches the language, and labelled data teaches the specific tasks. The model was compared with multilingual models such as AfriBERTa and AfroXLMR.

**Findings.** Compared with the multilingual models, SomBERTa consistently outperformed them in both fake news and toxic content classification. The results support the paper's argument that a language-specific model can do better than broad multilingual ones for Somali. Exact scores are not reported here, so consult the paper's results tables if needed. Note also that the paper is an arXiv preprint, so check whether a peer-reviewed version exists.

**Which task from Q3?** **Text classification.** The input is a Somali social-media post, and the output is a label (fake or real, toxic or not toxic, or a news topic).

## References

- Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. *Proceedings of IJCNLP-AACL 2023*.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*.
- Goldberg, Y. (2017). *Neural Network Methods for Natural Language Processing*. Morgan & Claypool.
- Gedi, A., et al. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of AfricaNLP 2026* (ACL Anthology, 2026.africanlp-main.17).
- Hirschberg, J., & Manning, C. D. (2015). Advances in natural language processing. *Science, 349*(6245), 261–266.
- Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of ACL 2020*.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
- Mohamed, M. A., et al. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. *arXiv:2503.18117*.
- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
