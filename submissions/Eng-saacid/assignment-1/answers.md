# Assignment 1 – NLP Foundations

## Question 1. Natural Language Processing and Its Relationship to AI and ML

Natural Language Processing (NLP) enables computers to understand, interpret, and process human language (Hirschberg & Manning, 2015). It is a branch of Artificial Intelligence (AI), which focuses on creating systems that perform intelligent tasks (Russell & Norvig, 2020). Machine Learning (ML) allows computers to learn patterns from data instead of relying only on manually written rules (Mitchell, 1997).

**Relationship Between NLP, AI, and ML:**

AI is the broader field, NLP is a specialized area of AI, and ML is a technique used by NLP systems to learn from language data and make predictions. Modern NLP commonly uses ML, although rule-based methods are also possible.

**Real-life Example:**

A Somali news website uses NLP to process news articles and ML to classify them into categories such as politics, sports, and business. AI represents the overall goal of automating this task. This type of news classification is studied in the MasakhaNEWS dataset, which includes Somali news articles (Adelani et al., 2023).

## Question 2. Corpus, Document, Token, Vocabulary, Feature, and Label

The example dataset is the Somali portion of MasakhaNEWS, a news topic classification dataset. It contains 1,463 articles (1,021 train, 148 development, 294 test) in 7 topics (Adelani et al., 2023).

| Term | Definition | Application in MasakhaNEWS (Somali) |
|---|---|---|
| Corpus | A collection of texts assembled for analysis or model training. | All 1,463 Somali articles together. |
| Document | One text item treated as a single example. | One article (headline_text). |
| Token | A unit produced by splitting text; it may be a word, subword, or symbol. | In "Kooxda Soomaaliya ayaa guuleysatay", the word-level tokens are Kooxda, Soomaaliya, ayaa, guuleysatay. |
| Vocabulary | The set of distinct token types used by the system. | All different tokens found in the training articles. |
| Feature | A measurable, usually numerical, property given to a model. | How many times a word such as "kooxda" appears in an article. |
| Label | The target answer attached to an example in supervised learning. | The topic of the article, for example politics or sports. |

The seven labels are business (114 articles), entertainment (139), health (354), politics (500), religion (73), sports (148), and technology (135). The classes are unbalanced, because politics has far more articles than religion. This matters when evaluating a model, since accuracy alone can hide poor results on small classes. The sentence used for the token row is my own illustration.

## Question 3. Four Common NLP Tasks

| Task | What it does | Input | Output | Example |
|---|---|---|---|---|
| Text classification | Assigns a category to a whole text. | An article or message. | One label, such as "sports". | Somali news topic classification (Adelani et al., 2023). |
| Sequence labeling | Assigns a tag to each token. | A sentence split into tokens. | One tag per token, such as PERSON, LOCATION, or O. | Named entity recognition (Tjong Kim Sang & De Meulder, 2003). |
| Search and similarity | Finds and ranks documents related to a query. | A query and a document collection. | A ranked list of documents or similarity scores. | Searching Somali web pages for a query (Manning et al., 2008). |
| Text generation | Produces new text from a prompt or source text. | A prompt, or a sentence in another language. | A newly written sequence of words. | Machine translation or summarization. |

The main difference between the tasks is the form of the output: one label per text, one label per token, a ranking, or a new piece of text.

## Question 4. Supervised and Unsupervised Learning in NLP

Supervised learning trains a model on texts that come with target answers, so the model can compare its prediction to the known answer and correct itself. Unsupervised learning uses texts without target answers and looks for structure in them, such as groups of similar documents or recurring themes (Manning et al., 2008).

Supervised example: Somali news topic classification. The articles in MasakhaNEWS are labelled by human annotators, and a classifier learns to predict the topic of a new article. Labels are needed during training because the model must be shown what the correct category is.

Unsupervised example: Topic modeling with Latent Dirichlet Allocation (Blei et al., 2003), which discovers groups of words that tend to appear together without being told any topic names. No labels are needed because the method is optimized using the text alone, and people interpret and name the discovered topics afterwards.

| Aspect | Supervised learning | Unsupervised learning |
|---|---|---|
| Training data | Text plus target labels | Text only |
| Objective | Predict a specified answer | Discover structure |
| Main challenge | Getting accurate labels | Judging whether the patterns are useful |

Modern language models also use self-supervised learning, where words are hidden and the model predicts them from context (Devlin et al., 2019). The target is a word or token from the original text, so human annotation is not required.

## Question 5. Why Text Needs Numerical Representation

Most ML algorithms work by calculating weighted sums, distances, and probabilities on vectors of numbers. Raw words cannot be used in these calculations directly. A text representation converts each document into a list of numbers with the same positions for every example, so that the model can compare documents (Jurafsky & Martin, 2025).

A simple method is Bag of Words (Salton et al., 1975). A vocabulary is chosen and each document is described by how many times each vocabulary word appears in it.

Example with two short documents and the ordered vocabulary [lost, match, team, the, won]:

| Document | lost | match | team | the | won | Vector |
|---|---|---|---|---|---|---|
| "the team won the match" | 0 | 1 | 1 | 2 | 1 | [0, 1, 1, 2, 1] |
| "the team lost" | 1 | 0 | 1 | 1 | 0 | [1, 0, 1, 1, 0] |

Main limitation: Bag of Words ignores word order, so "dog bites man" and "man bites dog" receive the same vector. It also treats related words as completely separate features, and the vectors become long and mostly zeros as the vocabulary grows. For Somali this is especially visible, because "koox" and "kooxda" would be counted as two unrelated words.

## Question 6. Challenges for Low-Resource Languages Such as Somali

A language is considered low-resource in NLP when digital text, annotated datasets, and language tools are scarce for the task at hand. The term describes the availability of resources, not the number of speakers. Somali is spoken by over 22 million people (Mohamed et al., 2025; Joshi et al., 2020).

**Challenge 1: Limited Datasets and Annotations**

The Somali portion of MasakhaNEWS contains 1,463 news articles, including 1,021 training articles, 148 development articles, and 294 testing articles (Adelani et al., 2023). The Somali portion of the CC100 web corpus used for the multilingual XLM-R model contains approximately 62 million tokens (Mohamed et al., 2025).

**Effect on performance:** Models trained on limited or narrow datasets may not perform well on different types of text. Small test sets can also make evaluation results less reliable.

**Possible solutions:** Self-supervised pretraining on raw text, transfer learning, and community-built datasets can help improve NLP performance (Hedderich et al., 2020; Nekoto et al., 2020).

**Challenge 2: Word Forms and Morphology**

Somali words can contain attached grammatical information. For example, "koox" and "kooxda" mean "team" and "the team."

**Effect on performance:** A word-level model may treat these forms as separate vocabulary items, increasing data sparsity and creating difficulties when encountering unseen words.

**Possible solutions:** Lemmatization and subword tokenization methods such as WordPiece and BPE can help address this problem (Sennrich et al., 2016). Mohamed and Mohamed (2023) developed a rule-based Somali lemmatizer. The SomBERTa model also uses WordPiece tokenization with a vocabulary of 70,000 tokens (Mohamed et al., 2025).

**Challenge 3: Annotation Quality and Informal Text**

Agreement between annotators for Somali in MasakhaNEWS was a Fleiss’ kappa of 0.55, the lowest score reported among the languages with multiple annotators (Adelani et al., 2023).

**Effect on performance:** When human annotators disagree, the dataset may contain inconsistent labels, making it harder for a model to learn accurate patterns. Informal Somali social media text also contains spelling variations and non-standard expressions. Therefore, models trained mainly on formal news articles may perform less effectively on informal text.

**Example and possible solutions:** In the SomBERTa study, 3,710 social media comments were initially collected for the fake-news dataset. After fact-checking and verification, the dataset was reduced to 1,897 instances classified as real or fake (Mohamed et al., 2025). Improving annotation guidelines and collecting diverse Somali text can help address these challenges.

## Question 7. Research Case Study: Somali Fake News and Toxicity Detection

**Paper:** Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models

**Authors:** Muhidin A. Mohamed, Shuab D. Ahmed, Yahye A. Isse, Hanad M. Mohamed, Fuad M. Hassan, and Houssein A. Assowe

**Publication:** arXiv preprint 2503.18117 (2025).  
https://arxiv.org/abs/2503.18117

**Task:**

The authors aimed to automatically detect fake news and toxic comments in Somali-language Facebook content and classify news topics because manual content moderation is difficult to scale (Mohamed et al., 2025).

**Data:**

The pretraining corpus contains approximately 160 million tokens collected from six sources, including books, BBC Somali, CC100, and CIRAL.

The fake news dataset contains 1,897 labelled instances selected from 3,710 collected comments. The toxicity dataset includes comments about three debated topics: constitutional amendments (1,785), a celebrity court case (915), and a maritime dispute (314).

Each item was labelled independently by two annotators. Items were retained only when both annotators agreed; disagreements were discarded (Mohamed et al., 2025).

**Method:**

The authors cleaned and combined the text corpora, trained a WordPiece tokenizer, and developed a BERT-based language model called SomBERTa, with approximately 126 million parameters. They pretrained the model using Masked Language Modeling (MLM) and then fine-tuned it for different classification tasks.

They compared SomBERTa with several existing models, including BERT, mBERT, DistilBERT, RoBERTa, ALBERT, AfriBERTa, and AfroXLMR.

**Findings (Accuracy, %):**

| Model | News Topic | Toxicity (Binary) | Toxicity (Multi-label) | Fake News | Average |
|---|---:|---:|---:|---:|---:|
| SomBERTa-large | 94.17 | 84.00 | 78.80 | 95.00 | 87.99 |
| AfriBERTa-large | 95.24 | 83.33 | 77.05 | 94.47 | 87.52 |
| AfroXLMR-base | 94.77 | 81.67 | 75.63 | 92.63 | 86.18 |

SomBERTa achieved the best results in fake news detection and both toxicity classification tasks. It also achieved the highest average accuracy of 87.99%. However, it ranked third in news topic classification.

The performance differences between SomBERTa and AfriBERTa are relatively small. The paper reports accuracy scores but does not present statistical significance tests. Therefore, the results should be interpreted with caution, especially because the datasets are relatively small.

**Task Category:**

This research is an example of **Text Classification (Question 3)**. The input is a news article, social media post, or comment, and the output is a category or label, such as fake or real, toxic or non-toxic, or a news topic.

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). MasakhaNEWS: News topic classification for African languages. arXiv:2304.09972. https://arxiv.org/abs/2304.09972

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, 993–1022.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019. https://arxiv.org/abs/1810.04805

Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., & Klakow, D. (2020). A survey on recent approaches for natural language processing in low-resource scenarios. arXiv:2010.12309. https://arxiv.org/abs/2010.12309

Hirschberg, J., & Manning, C. D. (2015). Advances in natural language processing. Science, 349(6245), 261–266.

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. Proceedings of ACL 2020.

Jurafsky, D., & Martin, J. H. (2025). Speech and language processing (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to information retrieval. Cambridge University Press.

Mitchell, T. M. (1997). Machine learning. McGraw-Hill.

Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. arXiv:2503.18117. https://arxiv.org/abs/2503.18117

Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word lemmatization approach for the Somali language. arXiv:2308.01785. https://arxiv.org/abs/2308.01785

Nekoto, W., Marivate, V., Matsila, T., Fasubaa, T., et al. (2020). Participatory research for low-resourced machine translation: A case study in African languages. arXiv:2010.02353. https://arxiv.org/abs/2010.02353

Russell, S., & Norvig, P. (2020). Artificial intelligence: A modern approach (4th ed.). Pearson.

Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. Communications of the ACM, 18(11), 613–620.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. Proceedings of ACL 2016. https://arxiv.org/abs/1508.07909

Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. Proceedings of CoNLL-2003.