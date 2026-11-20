# Assignment 1 – NLP Foundations

**Student:** Sagal Mohamed Yusuf
**Course:** Natural Language Processing

---

## 1. Natural Language Processing, Artificial Intelligence and Machine Learning

**Natural Language Processing (NLP)** is the field concerned with building computer systems that can analyse, interpret and produce human language, in written or spoken form (Jurafsky & Martin, 2024). It combines ideas from linguistics, computer science and statistics. Typical products of NLP are search engines, machine translation, spam filters and chatbots.

**Artificial Intelligence (AI)** is the broader goal of building systems that act intelligently, such as reasoning, planning, perceiving and communicating (Russell & Norvig, 2021). **Machine Learning (ML)** is one way of reaching that goal: instead of programming every rule by hand, we let a program improve its performance on a task by learning from data (Mitchell, 1997).

The three are nested rather than separate:

| Concept | Role | Relationship |
|---|---|---|
| AI | The overall goal: intelligent behaviour in machines | The widest field |
| NLP | The AI area that deals specifically with language | An application area of AI |
| ML | A learning method that finds patterns in data | The main toolset used by modern NLP |

NLP does not depend only on ML. Early systems were built from hand-written grammar rules and dictionaries. Today, however, most NLP systems are trained with ML, and the most successful ones use deep learning (Devlin et al., 2019).

**Real-life example: detecting fake news in Somali on Facebook.** A research team built a system that reads Somali Facebook posts and decides whether each one is real or fake news (Mohamed et al., 2025).
- *AI* is the goal: automate a judgement that normally needs a human fact-checker.
- *NLP* is the language handling: the system must read Somali text, split it into units and represent its meaning.
- *ML* is the learning: a model is trained on posts that humans labelled "real" or "fake", and then applies the patterns it learned to new posts.

Remove any one of the three and the system does not work: without NLP the computer cannot handle the text, and without ML it cannot learn the decision from examples.

---

## 2. Core terms: corpus, document, token, vocabulary, feature, label

**Example dataset.** I use a small toy news-classification dataset, modelled on real ones such as MasakhaNEWS (Adelani et al., 2023). Real datasets work the same way, only at larger scale.

| ID | Text | Label |
|---|---|---|
| 1 | "Parliament approves the new budget" | politics |
| 2 | "Local team wins the cup final" | sports |
| 3 | "Parliament debates the cup security law" | politics |

**Corpus.** A corpus is the full collection of text used in a study or to train a model (Jurafsky & Martin, 2024). Here the corpus is all three headlines together. A real corpus can hold millions of words; for instance, the SomBERTa pre-training corpus contains roughly 160 million tokens (Mohamed et al., 2025).

**Document.** A document is one unit of text inside the corpus, and its size depends on the task. Here each headline is one document (3 documents). In another project a document could be a full article, a tweet or a single comment.

**Token.** A token is a basic unit produced by splitting a document, usually a word, a punctuation mark or a sub-word piece. The process is called tokenisation. Document 2 gives the tokens `Local`, `team`, `wins`, `the`, `cup`, `final` (6 tokens). Modern models often use sub-word tokenisers such as WordPiece, which can split a long Somali word into smaller meaningful pieces (Mohamed et al., 2025).

**Vocabulary.** The vocabulary is the set of *distinct* tokens in the corpus. After lower-casing, the vocabulary here is {parliament, approves, the, new, budget, local, team, wins, cup, final, debates, security, law}, which is 13 unique words from 16 running tokens in total. The difference between the two counts is worth remembering: tokens count every occurrence, the vocabulary counts each word type once.

**Feature.** A feature is a measurable property of the input that the model uses to make its decision (Manning et al., 2008). In a simple model, each vocabulary word can be a feature, with its count in the document as the value. Document 1 would have `parliament = 1`, `budget = 1`, `cup = 0`, and so on. Features can also be richer, such as sentence length or embedding values.

**Label.** A label (or target) is the correct answer attached to a document, supplied by humans, which the model must learn to predict. Here the labels are `politics` and `sports`. In the Somali study above, the labels are `fake`/`real` for news and `toxic`/`non-toxic` for comments.

**How they connect:** the corpus contains documents → documents are split into tokens → the distinct tokens form the vocabulary → tokens or vocabulary entries become features → the label is what the model learns to predict from those features.

---

## 3. Four common NLP tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A whole text (document, post, headline) | One label (or several labels) from a fixed set | Deciding whether a Somali Facebook comment is toxic or non-toxic |
| **Sequence labelling** | A sequence of tokens (a sentence) | One label for *each* token | Named entity recognition: tagging "Mogadishu" as `LOCATION` and "Hassan" as `PERSON`; part-of-speech tagging |
| **Search and similarity** | A query (or a text) plus a collection of documents | A ranked list of documents, or a similarity score | A search engine returning the most relevant articles for a query; finding duplicate questions |
| **Text generation** | A prompt or source text | A new sequence of text | Machine translation, summarisation, chatbot answers |

The main difference is the *shape of the output*. Classification gives one decision for the whole text. Sequence labelling gives one decision per token. Search gives an ordering. Generation gives new text whose length is not fixed in advance.

---

## 4. Supervised versus unsupervised learning in NLP

**Supervised learning** trains a model on examples where the correct answer (the label) is already provided. The model learns the mapping from input to label, and is then tested on unseen examples (Jurafsky & Martin, 2024).
- *NLP example:* classifying news articles into topics such as politics, sport and health, trained on articles that humans have already categorised.

**Unsupervised learning** works with data that has *no labels*. The goal is to discover structure that is already inside the data (Manning et al., 2008).
- *NLP example:* topic modelling with Latent Dirichlet Allocation (LDA), which groups words that tend to appear together into "topics" and describes each document as a mixture of them (Blei et al., 2003). Clustering similar documents is another example.

**Why one needs labels and the other does not.** In classification the model is asked to reproduce a *specific, human-defined answer*, and it can only improve if it is told, example by example, whether it was right. The label is the signal that drives training, so it is essential. In topic modelling there is no "right topic" to match. The algorithm only looks for statistical regularities (which words co-occur), so the data itself is enough and no target is needed.

**A related third category.** Large pre-trained models such as BERT use *self-supervised* learning: the training text creates its own labels by hiding some words and asking the model to predict them (Devlin et al., 2019). This needs no human annotation, which matters a great deal for low-resource languages. SomBERTa was pre-trained this way on unlabelled Somali text and then fine-tuned with labelled data (Mohamed et al., 2025).

---

## 5. Why text must become numbers

Most ML algorithms are mathematical procedures: they multiply, add and compare numbers, and compute gradients on them. They cannot operate directly on letters or words. Text must therefore be converted into a numeric form, usually vectors (lists of numbers), before a model can use it (Jurafsky & Martin, 2024).

**One simple method: Bag of Words (BoW).**
1. Build the vocabulary from the corpus.
2. Represent each document as a vector with one position per vocabulary word.
3. Fill each position with the number of times that word occurs in the document.

Using the toy vocabulary in Question 2, Document 2 ("Local team wins the cup final") becomes a vector with `1` at *local, team, wins, the, cup, final* and `0` everywhere else. Weighting variants such as TF-IDF reduce the influence of very common words (Salton & Buckley, 1988).

**Main limitation: it ignores word order and meaning.** "Dog bites man" and "Man bites dog" produce exactly the same vector. BoW also treats every word as unrelated to every other, so "car" and "automobile" are as different as "car" and "banana". In addition, the vectors are very long and mostly zeros, because each vector has one entry per vocabulary word. Dense *embeddings* such as word2vec (Mikolov et al., 2013) and contextual models such as BERT were developed to solve these weaknesses.

---

## 6. Challenges of NLP for low-resource languages such as Somali

A language is called low-resource when it has little digital text, few annotated datasets and few ready-made tools. Somali is spoken by an estimated 22 million or more people, yet it is still classed as low-resource (Mohamed et al., 2025). Four challenges follow.

### 6.1 Limited datasets and annotated data
Modern NLP depends on large text collections and on human-labelled examples. For Somali both are scarce. The authors of SomBERTa note that multilingual models such as XLM-R included only a small amount of Somali in training (about 62 million tokens), and that multilingual models such as AfriBERTa and AfroLM often under-capture Somali syntax, meaning and morphology because the training data are limited (Mohamed et al., 2025).
**Effect on a system:** with little training data, models make more errors on Somali than on English, and supervised tasks are held back because someone must first create the labels. In the SomBERTa study, collecting 3,710 social-media comments produced only 1,897 usable real/fake examples after fact-checking and filtering, which shows how costly labelled data are to produce.

### 6.2 Complex word forms (morphology)
Somali words can carry many affixes and are formed through derivation, so one root can appear in many surface forms. A recent Somali lexicon records over 5,584 root words and 78,629 derived forms (Gedi et al., 2026).
**Effect on a system:** a simple word-level vocabulary treats each form as a separate token, which inflates the vocabulary and splits the statistics for a single concept across many entries. Normalisation steps such as lemmatisation help, but they are hard to build. The first Somali lemmatiser reached 57% accuracy on full news articles (Mohamed & Mohamed, 2023), and the larger 2026 system still reported only about 51% on full articles and about 60% on short texts (Gedi et al., 2026). This shows that even a basic preprocessing step is not solved.

### 6.3 Few ready-made tools and pre-trained models
For English, tokenisers, taggers and large pre-trained models are freely available. For Somali, the first monolingual pre-trained language model (SomBERTa) only appeared in 2025 (Mohamed et al., 2025), and lemmatisation resources are still being created (Gedi et al., 2026).
**Effect on a system:** developers must build the pipeline pieces themselves, or depend on multilingual models that were not designed around Somali.

### 6.4 Informal spelling and mixed-language text (to be checked against a source)
Social-media text is often written informally, with inconsistent spelling, abbreviations and words borrowed from English or Arabic. This is a general issue for user-generated text in many languages (Hedderich et al., 2020).
**Effect on a system:** the same word may appear under several spellings, each treated as a different token, and foreign words may be unknown to a Somali-only vocabulary. *I did not find a Somali-specific study measuring this effect, so I am presenting it as a general challenge rather than a measured Somali result.*

---

## 7. Case study

**Reference:** Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models.* arXiv:2503.18117.

**Problem.** Social media spreads misinformation and abusive content, and human moderators alone cannot keep up. Somali lacks the datasets and language-specific models needed to automate detection.

**Task.** Text classification. The paper covers (a) fake news detection (binary: fake/real), (b) toxicity detection (binary toxic/non-toxic, plus multi-label toxicity categories) and (c) news topic classification (multi-class).

**Data.**
- *Pre-training corpus:* about 160 million tokens of Somali text, merged from news, books, web-crawled text and existing datasets (including MasakhaNEWS, BBC Somali, CC100 and CIRAL).
- *Fake news dataset:* Facebook comments from news channels, of which 1,897 real/fake examples remained after fact-checking and filtering.
- *Toxicity dataset:* about 3,014 comments (1,785 + 915 + 314) on three debated topics: constitutional amendments, a celebrity court case and a maritime dispute. Comments were labelled toxic or non-toxic, and toxic ones were further placed in categories such as abuse, insult and identity-hate.
- *Labelling:* two annotators labelled each item, and an item was kept only if both agreed.

**Method.**
1. Clean and merge the corpora.
2. Train a WordPiece tokeniser (70,000-token vocabulary).
3. Pre-train a BERT-style model, **SomBERTa**, with masked-word prediction (self-supervised).
4. Fine-tune it on each labelled classification dataset (supervised).
5. Compare it with BERT, mBERT, AfriBERTa, AfroXLMR, RoBERTa, DistilBERT and ALBERT.

**Findings (accuracy, %).**

| Model | MasakhaNEWS (topics) | Toxicity (binary) | Toxicity (multi) | Fake news (binary) | Average |
|---|---|---|---|---|---|
| SomBERTa-large | 94.17 | 84.0 | 78.80 | 95.0 | **87.99** |
| AfriBERTa-large | 95.24 | 83.33 | 77.05 | 94.47 | 87.52 |
| AfroXLMR-base | 94.77 | 81.67 | 75.63 | 92.63 | 86.18 |
| BERT-base | 92.57 | 79.0 | 75.19 | 90.53 | 84.32 |

SomBERTa achieved the best average accuracy (87.99%) and the best results on fake news and toxicity, but it ranked third on news topic classification, behind AfriBERTa and AfroXLMR.

**My assessment.** The result supports the idea that a language-specific model can help for Somali. However, the margin over AfriBERTa is small (about 0.5 percentage points on average), and the labelled datasets are small and come from a single platform (Facebook) with only two annotators. The conclusion should therefore be read as promising rather than final.

**Link to Question 3:** this is **text classification**, because each whole post or comment is assigned one label (or a set of labels).

---

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). *MasakhaNEWS: News topic classification for African languages.* arXiv:2304.09972.

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*.

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)*, pp. 179–189. Association for Computational Linguistics. https://aclanthology.org/2026.africanlp-main.17/

Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., & Klakow, D. (2020). *A survey on recent approaches for natural language processing in low-resource scenarios.* arXiv:2010.12309.

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space.* arXiv:1301.3781.

Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill.

Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models.* arXiv:2503.18117.

Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and rule-based word lemmatization approach for the Somali language.* arXiv:2308.01785.

Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.
Add assignment 1 answers
