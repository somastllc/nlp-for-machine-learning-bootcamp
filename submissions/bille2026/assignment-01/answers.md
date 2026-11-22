# Assignment 1 – NLP Foundations

**Name:** bille abdul ali  
**GitHub username:** bille2026  
**Sources:** listed in the References section at the end (author–year citations in the text).

---

## 1. What is NLP, and how does it relate to AI and Machine Learning?

**Definition.** Natural Language Processing (NLP) is the field concerned with building computer systems that can read, interpret, and produce human language, in written or spoken form (Jurafsky & Martin, 2025). Human language is ambiguous, context-dependent, and constantly changing, so NLP combines ideas from linguistics, computer science, and statistics to turn raw language into something a program can work with.

**Relationship to AI and ML.** The three fields are nested rather than separate:

- **Artificial Intelligence (AI)** is the broad goal of building systems that act intelligently, including perceiving, reasoning, planning, and communicating (Russell & Norvig, 2021).
- **Machine Learning (ML)** is the subfield of AI in which programs improve at a task by learning from data instead of following hand-written rules (Mitchell, 1997).
- **NLP** is the area of AI devoted to language. Early NLP relied heavily on hand-written grammars and rules. Today most NLP systems are built with ML, so NLP is both an application area of AI and one of the main users of ML.

```
AI  ⊃  ML  ⊃  (ML-based NLP)
AI  ⊃  NLP  (NLP also includes rule-based and linguistic methods)
```

**Real-life example: email spam filtering.**

| Layer | Role in a spam filter |
|---|---|
| **AI** | The overall goal: a system that automatically decides which messages the user should see. |
| **NLP** | Handles the language: splits the email into words, normalises them, and turns the text into features such as word counts. |
| **ML** | Learns from thousands of emails already marked "spam" or "not spam" which word patterns signal spam, then applies that to new emails. |

NLP prepares the text, ML learns the decision rule from examples, and the combined system is an AI application that acts on the user's behalf. Removing any one part breaks the pipeline: without NLP the model cannot read the text, and without ML someone would have to write every spam rule by hand.

---

## 2. Core terms for text data

I use the **MasakhaNEWS** dataset as the running example. It is a news topic classification dataset covering 16 African languages, including Somali, and for Somali it draws on BBC Somali articles (Adelani et al., 2023; Mohamed et al., 2025). The sentence below is a short illustrative headline I wrote to show the terms, not a quotation from the dataset.

> *"Dowladdu waxay shaacisay qorshe cusub."* ("The government announced a new plan.")

| Term | General meaning | Applied to MasakhaNEWS (Somali) |
|---|---|---|
| **Corpus** | A large, organised collection of text used for study or training. | The whole collection of Somali news articles in the dataset. |
| **Document** | One unit of text in the corpus; its size depends on the task. | One news article (or headline plus body). The example headline is one document. |
| **Token** | A basic unit produced by splitting text: a word, a subword, or punctuation. | The headline splits into tokens such as `Dowladdu`, `waxay`, `shaacisay`, `qorshe`, `cusub`, `.` (word-level). A subword tokenizer may split rarer words further. |
| **Vocabulary** | The set of distinct tokens the system knows. | All unique tokens across the training articles, e.g. a list of a few thousand to tens of thousands of forms. Words not in it are "out-of-vocabulary". |
| **Feature** | A measurable property of a document that the model uses as input. | For example, how many times each vocabulary word appears in an article (word counts), or a TF-IDF weight for each word. |
| **Label** | The correct answer the model should predict (target). | The topic assigned to the article, e.g. politics, health, or sports. |

**How they connect:** the corpus is made of documents; documents are split into tokens; the distinct tokens form the vocabulary; features are computed from tokens for each document; and in supervised learning each document's features are paired with a label.

---

## 3. Four common NLP tasks

| Task | What it does | Input | Output | Example |
|---|---|---|---|---|
| **Text classification** | Assigns a category to a whole text. | A text (post, article, review). | One label (or several labels). | Deciding whether a Somali Facebook comment is *toxic* or *non-toxic*. |
| **Sequence labeling** | Assigns a label to each token in a sequence. | A sequence of tokens (a sentence). | A sequence of labels, one per token. | Named-entity recognition: tagging each word as PERSON, LOCATION, ORGANISATION or O (Tjong Kim Sang & De Meulder, 2003). Part-of-speech tagging works the same way. |
| **Search and similarity** | Finds or ranks texts by how relevant or similar they are. | A query and a collection of documents (or two texts). | A ranked list of documents, or a similarity score. | A search engine returning the most relevant Somali news articles for a query (Manning et al., 2008). |
| **Text generation** | Produces new text. | A prompt, source text, or context. | A newly generated sequence of text. | Machine translation from English to Somali, summarisation, or a chatbot reply. |

The key difference is the *shape of the output*: one label per text (classification), one label per token (sequence labeling), a ranking or score (search and similarity), or a whole new text (generation).

---

## 4. Supervised vs. unsupervised learning in NLP

**Supervised learning** learns from examples where the correct answer (label) is already known. The model is shown input–label pairs, learns the mapping, and is then tested on unseen inputs. **Unsupervised learning** receives only the inputs, with no labels, and must find structure in the data on its own (Mitchell, 1997; Jurafsky & Martin, 2025).

| | Supervised | Unsupervised |
|---|---|---|
| **Data needed** | Text + human-provided labels | Text only |
| **Goal** | Predict a known target | Discover hidden structure |
| **NLP example** | News topic classification: predict "sports", "politics", etc. | Topic modelling with Latent Dirichlet Allocation (Blei et al., 2003), or clustering similar articles |
| **How it is evaluated** | Compare predictions with true labels (accuracy, F1) | No single correct answer; judged by coherence or usefulness |

**Why one needs labels and the other does not.** In news classification, the categories "politics" or "sports" are human-defined concepts. Nothing in the raw words tells a model that a given group of articles should be called "sports" or what categories we want, so we must supply labelled examples to define the task and to measure success. In topic modelling the goal is different: the model only finds groups of words that tend to appear together (for example, *match, goal, team* together), and a human then reads the groups and decides what to call them. Because no predefined "right answer" exists, no label is needed.

*Related note:* modern language models such as BERT are pre-trained with **self-supervised** learning, where the training signal (a masked word) comes from the text itself, so no human labelling is required. They are then fine-tuned with labelled data for a specific supervised task (Devlin et al., 2019). The case study in Question 7 uses exactly this two-step approach.

---

## 5. Why text must become numbers

**Why.** Machine Learning models are mathematical functions. They compute with numbers (multiplying weights, summing, comparing distances) and cannot operate directly on characters or words. Text therefore has to be converted into numeric vectors that keep enough information about the language for the task (Jurafsky & Martin, 2025; Manning et al., 2008).

**Simple method: Bag-of-Words (BoW).** Build a vocabulary from the training corpus, then represent each document as a vector with one position per vocabulary word, whose value is how many times that word occurs in the document. A common refinement, **TF-IDF**, down-weights words that appear in many documents, since they carry little information (Salton et al., 1975; Manning et al., 2008).

Toy example with vocabulary [government, plan, football, match]:

| Document | government | plan | football | match |
|---|---|---|---|---|
| "government plan" | 1 | 1 | 0 | 0 |
| "football match match" | 0 | 0 | 1 | 2 |

**Main limitation.** BoW ignores word order and meaning. "Dog bites man" and "man bites dog" get identical vectors, and two different words with the same meaning (synonyms) are treated as completely unrelated. The vectors are also very long and mostly zeros (sparse) because the vocabulary can be large. For a morphologically rich language such as Somali, this is worsened because many inflected or derived forms of one word are counted as separate features (see Question 6). Dense embeddings and Transformer models were developed partly to address these weaknesses.

---

## 6. Challenges of NLP for low-resource languages such as Somali

A language is called *low-resource* when it has little digital text, few annotated datasets, and few ready-made tools, even if many people speak it. Somali is spoken by more than 22 million people, mainly in Somalia, Djibouti, Kenya, Ethiopia and the diaspora, yet it remains under-served by NLP resources (Mohamed et al., 2025). Joshi et al. (2020) and Hedderich et al. (2021) describe how unevenly the world's languages are covered in NLP.

### Challenge 1: Limited datasets and models

*What it is.* Supervised NLP needs labelled data and pre-training needs large text corpora, and for Somali both are scarce. Mohamed et al. (2025) note that multilingual models such as XLM-R included Somali with only about 0.4 GB (around 62 million tokens) of training text, and that the available Somali news classification data came from a few thousand BBC Somali articles (about 2,915 in MasakhaNEWS). These models also tend to capture Somali syntax and morphology poorly because of the small amount of Somali they were trained on.

*Effect on a system.* Fewer examples mean the model sees fewer word forms, topics and writing styles, so it generalises poorly to new domains such as social media. Researchers often have to build and annotate datasets themselves before they can even start, which is slow and expensive. Benchmarks are also smaller, making it harder to measure real progress.

### Challenge 2: Rich morphology and word forms

*What it is.* Somali has an agglutinative and derivational morphology, where words are built from roots plus affixes. The Somali lemmatization lexicon of Gedi et al. (2026) contains 5,584 root words with 78,629 derivative forms. The same root therefore shows up in many surface forms.

*Effect on a system.* Word-level methods such as Bag-of-Words treat every surface form as a separate feature, which inflates the vocabulary, makes the data sparser, and means rare forms are seen too few times to learn from. Lemmatization (reducing words to their roots) can help, but it is itself hard: the rule- and lexicon-based Somali lemmatizers reported by Mohamed and Mohamed (2023) and Gedi et al. (2026) reached roughly 44–60% accuracy on news articles and excerpts and higher accuracy on short social-media messages, depending on the test set. (The two studies use different test collections, so these numbers are not directly comparable.) Subword tokenizers such as WordPiece or Byte-Pair Encoding (Sennrich et al., 2016) are another way to cope, and SomBERTa uses WordPiece for this reason (Mohamed et al., 2025).

### Challenge 3: Informal, variable and mixed-language text

*What it is.* Much of the Somali text available online comes from social media, where people write informally, with inconsistent spelling, abbreviations, and sometimes words from other languages (for example English or Arabic) mixed in. Formal sources such as news articles look very different from this.

*Effect on a system.* A model trained on clean news text may fail on social-media comments, because spelling variants of the same word appear as different tokens and unfamiliar mixed-language words are out-of-vocabulary. This makes tasks like toxicity detection harder, and it is one reason Mohamed et al. (2025) built their datasets directly from social-media posts rather than reusing news data.

> **Note for myself:** the general observation about Somali morphology and data scarcity is supported by the cited papers. The specific points on spelling variation and code-switching in Somali social media are a reasoned discussion; I should look for a Somali-specific paper that measures them before the final submission.

---

## 7. Case study

**Reference.** Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models.* arXiv:2503.18117. https://arxiv.org/abs/2503.18117

**Problem.** Social media is a major news source for Somali speakers, but it also spreads misinformation and toxic content. Human moderation does not scale, and automatic tools are hard to build for Somali because of scarce annotated data and the lack of models tailored to the language.

**Task.** Text classification: fake vs. real news (binary), toxic vs. non-toxic (binary), toxicity type (multi-label), and news topic (multi-class). In terms of Question 3, the input is a Somali text (a post or comment) and the output is one or more labels.

**Data.**
- *Pre-training corpus* of about 160 million tokens, merged from MasakhaNEWS (Somali), BBC Somali, Somali books, CC100-Somali, the Somali part of the AfriBERTa corpus, and CIRAL.
- *Fake news dataset:* collected from public Facebook news channels. Of 3,710 comments initially gathered, 1,897 remained after fact-checking and annotation as clearly real or fake.
- *Toxicity dataset:* comments on three politically and socially debated topics (constitutional amendments, a celebrity court case, a maritime dispute), roughly 3,000 comments in total, labelled toxic/non-toxic and then by category (abuse, obscene, insult, identity hate, severe toxic, threat).
- *Annotation:* two annotators labelled each item independently, and an item was kept only if both agreed.

**Method.** The authors trained **SomBERTa**, a BERT-based Somali language model, in three stages: (1) clean and merge the corpora; (2) pre-train a WordPiece tokenizer (70,000-token vocabulary) and the model with masked language modelling, a self-supervised objective; (3) fine-tune the pre-trained model with the labelled datasets for each classification task. They compared it with BERT, mBERT, DistilBERT, RoBERTa, ALBERT, AfriBERTa and AfroXLMR using accuracy, precision, recall and F1.

**Findings.**

| Model | News topic | Toxicity (binary) | Toxicity (multi) | Fake news | Average |
|---|---|---|---|---|---|
| SomBERTa-large | 94.17 | **84.0** | **78.80** | **95.0** | **87.99** |
| AfriBERTa-large | **95.24** | 83.33 | 77.05 | 94.47 | 87.52 |
| AfroXLMR-base | 94.77 | 81.67 | 75.63 | 92.63 | 86.18 |
| mBERT | 93.01 | 79.5 | 76.67 | 91.84 | 85.26 |
| BERT-base | 92.57 | 79.0 | 75.19 | 90.53 | 84.32 |

*(Accuracy %, selected rows from the paper's Table 6.)* SomBERTa achieved the best average accuracy (87.99%) and was the best on fake news and both toxicity tasks. On news topic classification it ranked third, behind AfriBERTa and AfroXLMR. The authors conclude that a language-specific model trained on a diverse Somali corpus can outperform multilingual models, and that the pipeline can be reused for other low-resource languages.

**My assessment.** The study is valuable because it builds the missing resources (corpus, labelled datasets, model) rather than only applying existing tools. Its limits are that the labelled datasets are small (a few thousand examples), only two volunteers annotated them, the margins over AfriBERTa are small (about 0.5 points on average), and the paper is an arXiv preprint, so I treat its results as promising rather than final.

---

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). *MasakhaNEWS: News topic classification for African languages.* arXiv:2304.09972.

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)*, 179–189. https://aclanthology.org/2026.africanlp-main.17/

Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., & Klakow, D. (2021). A survey on recent approaches for natural language processing in low-resource scenarios. *Proceedings of NAACL-HLT 2021*. arXiv:2010.12309.

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of ACL 2020*, 6282–6293.

Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval.* Cambridge University Press. https://nlp.stanford.edu/IR-book/

Mitchell, T. M. (1997). *Machine learning.* McGraw-Hill.

Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). *Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models.* arXiv:2503.18117. https://arxiv.org/abs/2503.18117

Mohamed, S. A., & Mohamed, M. A. (2023). *Lexicon and rule-based word lemmatization approach for the Somali language.* arXiv:2308.01785. https://arxiv.org/abs/2308.01785

Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.

Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*, 1715–1725.

Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. *Proceedings of CoNLL-2003*, 142–147.
