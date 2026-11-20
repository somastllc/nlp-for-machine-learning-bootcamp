# Assignment 1 – NLP Foundations

**Student:** Liibaan1212 (GitHub username)
**Course:** NLP for Machine Learning Bootcamp

---

## 1. What is Natural Language Processing (NLP)?

**Definition.** Natural Language Processing is the field of computer science that builds systems able to read, interpret, and produce human language, in written or spoken form. It sits at the meeting point of computer science, linguistics, and artificial intelligence (Jurafsky & Martin, 2024). The central difficulty is that human language is ambiguous and depends on context, while computers only work with precise, numeric instructions.

**Relationship to AI and ML.**

| Term | Meaning | Where it fits |
|---|---|---|
| Artificial Intelligence (AI) | The broad goal of building machines that perform tasks that normally need human intelligence, such as reasoning, perception, and language (Russell & Norvig, 2021). | The umbrella field. |
| Machine Learning (ML) | A set of methods where a program improves at a task by learning patterns from data instead of following hand-written rules (Mitchell, 1997). | A sub-area of AI. |
| NLP | AI applied specifically to human language. | An application area of AI that today relies heavily on ML. |

NLP is not *only* machine learning. Early NLP systems were rule-based (hand-written grammars and dictionaries). Modern systems mostly use ML because learning from large amounts of text copes with the messiness of real language better than fixed rules do.

**Real-life example: an email spam filter.**

1. *NLP* handles the language side: it splits the email into words, cleans the text, and turns it into numbers the computer can use.
2. *ML* does the learning: a classifier is trained on thousands of emails already marked "spam" or "not spam" and learns which word patterns (for example "free prize", "click now") are linked to spam.
3. *AI* is the overall system: an intelligent tool that makes a decision on a new email without a human reading it.

So NLP prepares and interprets the text, ML learns the decision rule, and together they form an AI application.

---

## 2. Core terms: corpus, document, token, vocabulary, feature, label

**Example dataset: MasakhaNEWS, Somali subset.** MasakhaNEWS is a news-topic classification dataset covering 16 African languages (Adelani et al., 2023). Its Somali portion contains 1,463 news articles split into 1,021 for training, 148 for development, and 294 for testing, labelled with 7 topics: business, entertainment, health, politics, religion, sports, and technology (Adelani et al., 2023).

To show tokens and vocabulary at small scale, I use one short **illustrative** Somali sentence (written by me, not taken from the dataset):

> *Kooxdu waxay ku guulaysatay ciyaarta.* ("The team won the game.")

| Term | Definition (in my words) | Applied to this example |
|---|---|---|
| **Corpus** | A large, organised collection of text used for NLP work. | The full set of 1,463 Somali news articles in MasakhaNEWS. |
| **Document** | One unit of text inside the corpus; what counts as a "document" depends on the task. | One news article (headline plus body). |
| **Token** | A small unit that a text is split into: usually a word, sometimes a sub-word or punctuation mark. | The sentence above has 5 word tokens: `Kooxdu`, `waxay`, `ku`, `guulaysatay`, `ciyaarta`. |
| **Vocabulary** | The set of *unique* tokens found in the corpus (or allowed by the tokenizer). | The unique words across all 1,463 articles. Subword models fix this size in advance; for example SomBERTa uses 70,000 sub-word tokens (Mohamed et al., 2025). |
| **Feature** | A measurable input property the model uses to make its prediction. | For a classical model: how many times each vocabulary word appears in an article (word counts or TF-IDF scores). For a neural model: the numeric embeddings of the tokens. |
| **Label** | The correct answer (target) attached to a document, which the model must learn to predict. | The topic of the article, e.g. `sports` or `politics`. |

**Note:** the features are *derived from* the text, while the label is *given with* the text (assigned by human annotators). The model learns the link between features and labels.

---

## 3. Four common NLP tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A piece of text (sentence, post, article). | One label (or several) for the whole text. | Classifying a Somali news article as `sports` or `politics`; flagging a comment as toxic or not. |
| **Sequence labeling** | A sequence of tokens (a sentence). | One label *per token*. | Named Entity Recognition: tagging each word as person, place, organisation, or other (e.g. "Muqdisho" → location). Also part-of-speech tagging. |
| **Search and similarity** | A query (or a document) plus a collection of documents. | A ranked list of the most relevant documents, or a similarity score between two texts. | A web search engine; finding duplicate or related questions. |
| **Text generation** | A prompt or source text. | New text. | Machine translation (English → Somali), summarisation, chatbots. |

The key difference is the *shape of the output*: one label per text, one label per token, a ranked list or score, or a new sequence of text.

---

## 4. Supervised vs. unsupervised learning in NLP

| | Supervised learning | Unsupervised learning |
|---|---|---|
| **Data** | Text **with** target labels. | Text **without** target labels. |
| **Goal** | Learn to predict the label for new text. | Discover hidden structure, such as groups or topics. |
| **How it learns** | Compares its prediction to the known correct label and adjusts to reduce the error. | Looks for patterns (similarity, co-occurrence) in the data itself. |
| **NLP example** | Spam detection or news-topic classification (e.g. the MasakhaNEWS labels). | Topic modelling with Latent Dirichlet Allocation (Blei et al., 2003), or clustering similar news articles. |

**Why one needs labels and the other does not.** A supervised model needs a target because its training works by measuring the gap between what it predicted and the correct answer; with no correct answer there is nothing to measure that gap against. An unsupervised model has no "right answer" to reach. Its job is to organise the data, so it only needs the text itself. For instance, a topic model finds that words like "goal", "match", and "team" often occur together and groups them, without anyone telling it that the topic is "sports".

**A useful middle case.** Modern language models such as BERT are pre-trained with *self-supervised* learning: the "label" is created automatically from the text itself by hiding a word and asking the model to predict it (Devlin et al., 2019). This is how huge unlabelled corpora can be used.

---

## 5. Why text must be converted to numbers

Machine learning models are mathematical functions. They multiply matrices, compute distances, and adjust numeric weights through calculus. A raw string such as "guul" has no meaning in arithmetic, so a model cannot add, compare, or differentiate it. Text must therefore be turned into numerical vectors that still carry useful information about the text (Jurafsky & Martin, 2024).

**Simple method: Bag of Words (BoW).**

1. Build the vocabulary of all unique words in the corpus.
2. Represent each document as a vector with one position per vocabulary word.
3. Fill each position with the number of times that word appears in the document.

Example with vocabulary `[bites, dog, man]`:

| Document | bites | dog | man |
|---|---|---|---|
| "dog bites man" | 1 | 1 | 1 |
| "man bites dog" | 1 | 1 | 1 |

**Main limitation: it ignores word order and context.** The two sentences above mean very different things, yet get identical vectors. BoW also produces very large, mostly-zero (sparse) vectors for big vocabularies, and treats every word form as unrelated. This is why TF-IDF weighting (Manning et al., 2008) and, later, dense word embeddings and contextual models like BERT were developed.

---

## 6. Challenges of NLP for low-resource languages such as Somali

A language is called *low-resource* when it has little digital text, few labelled datasets, and few ready-made tools. Joshi et al. (2020) showed that language technology is very unevenly distributed, with most of the world's languages receiving little attention. Somali is spoken by an estimated 22 million or more people (Mohamed et al., 2025), yet it still faces this problem. Below are four challenges.

### 6.1 Limited datasets

- **What it is.** Modern NLP models need large amounts of text, and supervised tasks need *labelled* text. For Somali both are scarce. As examples, the Somali subset of MasakhaNEWS has only 1,463 articles (Adelani et al., 2023), and the multilingual model XLM-R included only about 62 million Somali tokens in its training data (as reported by Mohamed et al., 2025).
- **Effect on an NLP system.** Models trained on little data perform worse, are less reliable on new topics or writing styles, and are hard to evaluate fairly because the test sets are small. Multilingual models also tend to under-represent Somali grammar and meaning because Somali is only a small share of their training data (Mohamed et al., 2025).

### 6.2 Rich word forms (morphology)

- **What it is.** Somali words change form by adding endings. For example, the noun *buug* ("book") can appear as *buugga* ("the book") or *buugag* ("books"). (Illustrative examples; please check with a Somali speaker.) Handling this requires specialised tools such as lemmatizers, which reduce a word to its base form; Somali lemmatizers have only recently been built (Mohamed & Mohamed, 2023).
- **Effect on an NLP system.** A word-based system like Bag of Words treats each form as a separate, unrelated word. This enlarges the vocabulary, spreads the counts thinly across forms, and makes the model fail on forms it did not see in training (the "out-of-vocabulary" problem). Sub-word tokenizers (such as WordPiece) reduce this problem but do not remove it.

### 6.3 Spelling variation and informal writing

- **What it is.** Somali has a standard Latin-script spelling (Saeed, 1999), but writing on social media is often informal: letters or doubled vowels may be dropped, abbreviations are common, and people spell the same word in different ways.
- **Effect on an NLP system.** Every spelling variant becomes a different token, so the system sees many "new" words. This adds noise, lowers accuracy on social-media tasks such as toxicity or fake-news detection, and makes data cleaning harder.

### 6.4 Mixed-language text (code-switching)

- **What it is.** Somali speakers often mix Somali with English, Arabic, or other languages in the same sentence (e.g. "Waan busy ahay": illustrative).
- **Effect on an NLP system.** A model trained on a single language sees unknown words and may misread or ignore the foreign parts. Language-identification and tokenization both become harder when languages switch mid-sentence.

---

## 7. Case study: Somali fake-news and toxic-message detection

**Source:** Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models.* arXiv:2503.18117. (Authors from Aston University, Jamhuriya University, Somali National University, and the University of Djibouti.)

**Problem.** Social media is a major news source for Somali speakers, and it spreads misinformation and harmful content. Human moderation does not scale, but Somali lacks labelled datasets and language models built for it (Mohamed et al., 2025).

**Data.**

- *Pre-training corpus:* about 160 million tokens of Somali text merged from sources including books, BBC Somali, MasakhaNEWS, CC100-Somali, and the CIRAL collection.
- *Fake-news dataset:* collected from Facebook news channels; 3,710 comments were first gathered, and after fact-checking only 1,897 instances were kept as clearly real or fake.
- *Toxicity dataset:* comments from three heavily debated topics (constitutional amendments, a celebrity court case, a maritime dispute), roughly 3,000 comments in total, labelled toxic or non-toxic and then sub-categorised (abuse, insult, identity hate, threat, and others).
- Two human annotators labelled each item, and an item was kept only when both agreed.

**Method.** The authors built **SomBERTa**, described as the first monolingual Somali BERT-based model. They (1) cleaned and merged the corpora, (2) trained a WordPiece tokenizer with a 70,000-token vocabulary and pre-trained the model with masked-word prediction, and (3) fine-tuned it for fake-news detection, toxicity classification, and news-topic classification. They compared it against BERT, mBERT, AfriBERTa, AfroXLMR, RoBERTa, DistilBERT, and ALBERT.

**Findings (accuracy, %).**

| Model | News topics (MasakhaNEWS) | Toxicity (binary) | Toxicity (multi-label) | Fake news | Average |
|---|---|---|---|---|---|
| SomBERTa (large) | 94.17 | 84.0 | 78.80 | 95.0 | **87.99** |
| AfriBERTa (large) | 95.24 | 83.33 | 77.05 | 94.47 | 87.52 |
| AfroXLMR (base) | 94.77 | 81.67 | 75.63 | 92.63 | 86.18 |
| BERT (base) | 92.57 | 79.0 | 75.19 | 90.53 | 84.32 |

SomBERTa achieved the best results on fake-news and toxicity detection and the best average, but ranked third on news-topic classification behind AfriBERTa and AfroXLMR (Mohamed et al., 2025). The authors conclude that a language-specific model with targeted training data can beat general multilingual models on Somali tasks.

**Which task from Question 3?** **Text classification**: each whole post or comment gets one label (fake/real, toxic/non-toxic, or a topic category).

**My critical notes.** (1) It is an arXiv preprint, so I treat it with some caution. (2) Only two annotators labelled the data, and the datasets are fairly small. (3) The margins over AfriBERTa are small (for example 95.0 vs 94.47 on fake news), so I would not claim a big advantage without more testing. Even so, the work is valuable because it creates new Somali resources, which directly addresses the data-scarcity challenge discussed in Question 6.

---

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). *MasakhaNEWS: News topic classification for African languages.* arXiv:2304.09972. https://arxiv.org/abs/2304.09972

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019.* https://arxiv.org/abs/1810.04805

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of ACL 2020.*

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill.

Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models.* arXiv:2503.18117. https://arxiv.org/abs/2503.18117

Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and rule-based word lemmatization approach for the Somali language.* arXiv:2308.01785. https://arxiv.org/abs/2308.01785

Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

Saeed, J. I. (1999). *Somali.* John Benjamins.
