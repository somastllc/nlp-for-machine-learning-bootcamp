# Assignment 1 – NLP Foundations

> **Course:** Natural Language Processing
>
> **Topic:** Foundational NLP Concepts, Text Data Terminology, Learning Paradigms, Vector Representations, and Low-Resource Language Challenges (Somali)

## 1. Natural Language Processing (NLP) and Its Relationship to AI and Machine Learning

**Natural Language Processing (NLP)** is an interdisciplinary subfield of computer science, artificial intelligence, and computational linguistics concerned with designing algorithms and computational models that enable computers to parse, interpret, generate, and reason over human (natural) languages in written or spoken form (Jurafsky & Martin, 2024; Eisenstein, 2019). Unlike formal programming languages with rigid, unambiguous syntax, natural languages are inherently ambiguous, context-dependent, and constantly evolving—requiring NLP systems to model morphology, syntax, semantics, and pragmatics simultaneously.

### Relationship Between AI, Machine Learning, and NLP

These three fields operate as nested, intersecting disciplines:

* **Artificial Intelligence (AI)** is the broadest umbrella discipline focused on creating computational systems capable of performing tasks that typically require human intelligence, such as reasoning, perception, planning, and communication (Russell & Norvig, 2020).

* **Machine Learning (ML)** is a core methodology within AI that enables systems to infer statistical patterns and decision boundaries directly from empirical data rather than relying solely on hand-crafted rules (Goodfellow et al., 2016).

* **Natural Language Processing (NLP)** sits at the intersection of AI, ML, and linguistics. Modern NLP relies heavily on ML (and specifically deep learning architectures such as Transformers) to learn linguistic representations from large text corpora, while also incorporating linguistic structures and domain knowledge (Jurafsky & Martin, 2024).

### Real-Life Example: Intelligent Customer Support Routing and Response

Consider a mobile banking support portal handling thousands of incoming customer inquiries daily:

1. **AI (System-Level Goal):** The overarching system acts as an intelligent virtual agent capable of understanding customer intent, triaging urgent fraud reports, and drafting helpful responses automatically.

2. **NLP (Linguistic Processing):** When a customer types *"Lacagtaydii waa la xaday, fadlan kaadhka ii xidh"* ("My money was stolen, please block my card"), NLP techniques segment the sentence into tokens, normalize spelling variations, detect the language (Somali), and extract key entities (the action *"block"* and object *"card"*).

3. **Machine Learning (Statistical Decision-Making):** A supervised classification model trained on thousands of historical support tickets maps the numerical representation of that message to a high-priority `Fraud_Alert` category with 98% confidence, triggering an automated account freeze and routing the ticket to a security specialist.

## 2. Core Text Data Terminology

To ground these foundational terms concretely, consider an example dataset of **1,500 customer product reviews** collected from an e-commerce platform, where each review is annotated as either `Positive` or `Negative` for sentiment analysis.

| Term | Formal Definition | Application to the E-Commerce Review Dataset | 
| ----- | ----- | ----- | 
| **Corpus** | A large, structured, and machine-readable collection of authentic texts assembled for linguistic analysis or model training (Jurafsky & Martin, 2024). | The entire dataset of **1,500 customer product reviews** stored together for training and evaluating the sentiment model. | 
| **Document** | A single, self-contained unit of text within a corpus—such as a sentence, paragraph, article, tweet, or review (Manning et al., 2008). | One individual customer review in the dataset, e.g., *"The battery life is amazing, though delivery was slow."* | 
| **Token** | An atomic, discrete unit produced by segmenting raw text (typically words, subwords, punctuation marks, or numbers) (Jurafsky & Martin, 2024). | Breaking the review above into word-level units yields 11 tokens: `["The", "battery", "life", "is", "amazing", ",", "though", "delivery", "was", "slow", "."]`. | 
| **Vocabulary** | The finite, deduplicated set of all unique tokens (types) observed across the entire corpus (or retained after frequency filtering) (Eisenstein, 2019). | If the 1,500 reviews contain 45,000 total running tokens but only **4,200 distinct words**, those 4,200 unique tokens form the model's vocabulary ($\vert{}V\vert{} = 4,200$). | 
| **Feature** | A measurable, numerical attribute extracted from a document that represents its linguistic properties to a machine learning algorithm (Bird et al., 2009). | The TF-IDF weight of the word `"amazing"`, the count of exclamation marks, or a 768-dimensional dense embedding vector representing a review. | 
| **Label** | The ground-truth target output or category assigned to a document that a supervised model is trained to predict (Goodfellow et al., 2016). | The human-annotated sentiment tag attached to each review—specifically `Positive` or `Negative`. | 

## 3. Common NLP Tasks

NLP encompasses a wide range of computational problems that differ in the granularity of their inputs and outputs (Jurafsky & Martin, 2024). Four foundational tasks are detailed below:

### 1. Text Classification

* **Description:** Assigning one or more predefined categorical labels to an entire span of text (such as a sentence, email, or article). Common applications include spam detection, topic categorization, dialect identification, and sentiment analysis (Manning et al., 2008).

* **Input:** A variable-length sequence of text (a single document, e.g., an incoming email message).

* **Output:** A discrete class label (e.g., `Spam` or `Not Spam`) or a probability distribution across all candidate classes.

### 2. Sequence Labeling

* **Description:** Assigning a categorical tag to *every individual token* in an input sequence, capturing local context and sequential dependencies. Standard examples include Part-of-Speech (POS) tagging and Named Entity Recognition (NER) (Jurafsky & Martin, 2024).

* **Input:** An ordered sequence of tokens $X = (x_1, x_2, \dots, x_n)$, such as `["Dr.", "Amina", "works", "in", "Hargeisa"]`.

* **Output:** A parallel sequence of tags $Y = (y_1, y_2, \dots, y_n)$ of identical length, such as `["O", "B-PER", "O", "O", "B-LOC"]` in NER (identifying *Amina* as a Person and *Hargeisa* as a Location).

### 3. Search and Semantic Similarity

* **Description:** Measuring how closely two pieces of text align in meaning or relevance, enabling information retrieval systems to rank documents against a user query even when exact keywords do not overlap (Manning et al., 2008; Reimers & Gurevych, 2019).

* **Input:** A pair of text sequences—typically a search query $q$ and a candidate document $d$ (or a collection of candidate documents $\{d_1, d_2, \dots, d_k\}$).

* **Output:** A continuous similarity or relevance score (such as a cosine similarity value between $-1$ and $1$) or a ranked list of documents ordered from most to least relevant.

### 4. Text Generation

* **Description:** Producing coherent, contextually appropriate natural language sequences conditioned on an input prompt, structured data, or source text. Applications include machine translation, abstractive summarization, and conversational assistants (Eisenstein, 2019).

* **Input:** A conditioning prompt, context window, or source sequence (e.g., a news article in English for translation or summarization).

* **Output:** A newly generated sequence of tokens $Y = (y_1, y_2, \dots, y_m)$ predicted autoregressively one token at a time (e.g., the translated Somali article or a concise three-sentence summary).

### Summary Comparison Table

| NLP Task | Primary Unit of Prediction | Input | Output | 
| ----- | ----- | ----- | ----- | 
| **Text Classification** | Whole document / span | A text document (e.g., review, email) | A single category label or class probabilities | 
| **Sequence Labeling** | Individual token | A sequence of $n$ tokens | A sequence of $n$ token-level tags | 
| **Search & Similarity** | Document pair / query-corpus | A query text and candidate document(s) | A continuous similarity score or ranked list | 
| **Text Generation** | Token sequence (autoregressive) | A prompt, source text, or conditioning context | A newly generated sequence of $m$ tokens | 

## 4. Supervised vs. Unsupervised Learning in NLP

Machine learning paradigms in NLP are primarily distinguished by whether the training data contains explicit ground-truth annotations (Goodfellow et al., 2016; Jurafsky & Martin, 2024).

### Supervised Learning

In **supervised learning**, an algorithm learns a mapping function $f: X \rightarrow Y$ from a dataset of paired examples $\{(x_i, y_i)\}_{i=1}^N$, where each input text $x_i$ is explicitly paired with a human-annotated target label $y_i$. During training, the model compares its predictions $\hat{y}_i$ against the true labels $y_i$ using a loss function and updates its parameters to minimize prediction error.

* **NLP Example:** *Spam Email Detection.* A classifier is trained on 50,000 emails manually labeled as `Spam` or `Ham` (legitimate). By comparing predicted probabilities against these ground-truth labels, the model learns which lexical patterns (e.g., *"urgent wire transfer"*) strongly indicate spam.

### Unsupervised Learning

In **unsupervised learning**, the algorithm is given only unlabeled raw text $\{x_i\}_{i=1}^N$ without predefined target outputs. Instead of predicting external categories, the model discovers latent statistical structure, recurring co-occurrence patterns, or natural groupings inherent within the text itself (Eisenstein, 2019).

* **NLP Example:** *Topic Modeling (e.g., Latent Dirichlet Allocation) or Document Clustering.* Given 100,000 unlabeled news articles, an unsupervised topic model groups articles based on word co-occurrence distributions—automatically separating clusters corresponding to sports, politics, economics, and health without any human-assigned topic tags.

### Why One Needs Target Labels and the Other Does Not

The requirement for target labels depends on whether the task defines an **externally imposed decision boundary** or seeks **intrinsic structural discovery**:

* **Supervised tasks require labels** because categories like *"offensive speech"*, *"positive sentiment"*, or *"Named Entity: Organization"* are human-defined semantic constructs. Without explicit ground-truth labels ($y_i$) acting as a supervisory signal, the optimization algorithm has no objective error metric to tell whether a predicted boundary matches human intent.

* **Unsupervised tasks do not require labels** because their objective is defined purely over the internal geometry or probability distribution of the input data $P(X)$—such as maximizing cluster cohesion or predicting masked/adjacent words from surrounding context (self-supervision).

## 5. Numerical Representations of Text

### Why Text Must Be Converted into Numbers

Machine learning algorithms—ranging from logistic regression and Support Vector Machines to deep neural networks—are fundamentally mathematical optimization engines built on linear algebra and calculus. Operations such as dot products, matrix multiplications, distance metrics (e.g., Euclidean distance or cosine similarity), and gradient descent require continuous or discrete numerical vectors in a vector space $\mathbb{R}^d$ (Goodfellow et al., 2016; Jurafsky & Martin, 2024). Raw character strings cannot be multiplied by weight matrices or differentiated; therefore, text must be mapped into structured numerical representations before any statistical learning can occur.

### Simple Representation Method: Bag-of-Words (BoW)

The **Bag-of-Words (BoW)** model represents each document as a fixed-length vector of size $\vert{}V\vert{}$ (where $\vert{}V\vert{}$ is the corpus vocabulary size). Each dimension $j$ corresponds to a specific word in the vocabulary, and the value at index $j$ records the frequency (count) of that word in the document (Manning et al., 2008).

For example, given a vocabulary `["dog", "bites", "man"]`:

* *"dog bites man"* $\rightarrow [1, 1, 1]$

* *"man bites dog"* $\rightarrow [1, 1, 1]$

### Main Limitation of Bag-of-Words

The primary limitation of Bag-of-Words is that it **discards word order, syntax, and semantic relationships** (treating a document as an unordered "bag" of independent tokens):

1. **Loss of Word Order and Context:** As shown above, *"dog bites man"* and *"man bites dog"* produce identical vectors $[1, 1, 1]$ despite having opposite meanings. Negations (e.g., *"not good"* vs. *"good"*) are similarly lost.

2. **High Dimensionality and Sparsity:** Because vocabularies often contain tens of thousands of words while a single sentence uses only a few, BoW vectors are extremely sparse (mostly zeros) and fail to capture semantic similarity between synonyms (e.g., `"car"` and `"automobile"` occupy completely orthogonal dimensions with zero similarity).

## 6. Challenges of NLP for Low-Resource Languages (Somali)

Languages like **Somali (*Af-Soomaali*)**, spoken by over 22 million people across the Horn of Africa and the global diaspora, are classified as **low-resource languages** in NLP because they lack the massive annotated corpora, standardized lexical tools, and pretrained benchmarks available for English or Mandarin (Gedi et al., 2026). Four major linguistic and technical challenges directly impact Somali NLP systems:

### 1. Scarcity of Digitized and Annotated Datasets

* **Nature of the Challenge:** While general web text in Somali is growing, high-quality, human-annotated datasets for downstream tasks (such as NER, sentiment analysis, or machine translation) remain scarce. Furthermore, available corpora heavily skew toward the standardized **Maxaa Tiri** written variety, leaving major spoken dialects such as **Af-Maay** severely underrepresented in digital text (Gedi et al., 2026).

* **Impact on NLP Systems:** Deep learning models and Large Language Models (LLMs) trained on imbalanced multilingual corpora suffer from severe **overfitting** and poor generalization. A model trained only on standard Maxaa Tiri news articles often fails abruptly when deployed on social media posts or Maay dialect text.

### 2. Rich Inflectional and Derivational Morphology

* **Nature of the Challenge:** Somali is a morphologically rich Cushitic language that uses extensive prefixation, suffixation, internal vowel changes, and reduplication to mark tense, gender, number, focus, and definiteness (e.g., a single root like *qor* \["write"\] inflects into *qoray*, *qortay*, *qorayaan*, *qoraal*, *qoraallo*) (Gedi et al., 2026).

* **Impact on NLP Systems:** Morphological richness causes severe **vocabulary explosion and data sparsity**. In word-level models, dozens of inflected forms of the same root are treated as completely unrelated tokens, fragmenting statistical counts. In modern Transformer models using English-centric subword tokenizers (like BPE), Somali words are over-segmented into tiny character fragments, inflating sequence lengths by 2–3x, exhausting context windows, and degrading semantic accuracy.

### 3. Orthographic (Spelling) Variation and Gemination

* **Nature of the Challenge:** Although Somali adopted an official Latin orthography in 1972, informal digital writing (especially on social media and messaging platforms) exhibits widespread spelling inconsistency—most notably around **geminate (doubled) consonants** (`b, d, dh, g, l, m, n, r`) and long vs. short vowels (`a/aa`, `e/ee`, `i/ii`, `o/oo`, `u/uu`), which distinguish word meanings (e.g., *berri* \["tomorrow"\] vs. *beri* \["once/time"\], or *dhacay* vs. *dacay*) (Gedi et al., 2026).

* **Impact on NLP Systems:** Without robust text normalization or character-aware modeling, minor spelling variations map identical words to distinct vocabulary entries or out-of-vocabulary (`<UNK>`) tokens, causing sharp drops in information retrieval recall and classification accuracy.

### 4. Code-Switching and Mixed-Language Text

* **Nature of the Challenge:** In everyday digital communication, Somali speakers frequently mix Somali with **English, Arabic, or Italian** within the same sentence (e.g., *"Bro, assignment-ka maanta aan submit-gareynayaa"* — *"Bro, I am submitting the assignment today"*), often attaching Somali inflectional suffixes (`-ka`, `-gareynayaa`) directly onto English loanwords.

* **Impact on NLP Systems:** Standard monolingual tokenizers, POS taggers, and stop-word lists break down on hybrid words (such as `submit-gareynayaa`), misclassifying borrowed stems or failing to match lexicon entries.

## 7. Case Study: Automatic Somali Dialect Identification in Low-Resource Settings

### Citation

> Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., & Ebdiyow, A. U. (2026). *Somali dialect identification in low-resource settings using machine learning and deep learning*. Discover Data, 4(1), 10. https://doi.org/10.1007/s44248-026-00109-y

### Summary of the Study

* **Problem and Task:**
  The Somali language encompasses two primary dialectal varieties—**Maxaa Tiri** (the standardized variety used in official media and education) and **Af-Maay** (widely spoken in southern Somalia). Because existing Somali NLP tools and LLMs are trained almost exclusively on Maxaa Tiri, they struggle to process Maay text accurately. The authors formulate **Automatic Dialect Identification (ADI)** as an upstream preprocessing task to classify incoming Somali text into either `MAXAA TIRI` or `MAAY` before routing it to downstream applications like machine translation or sentiment analysis.

  * *Connection to Question 3:* This problem directly exemplifies **Text Classification** (specifically binary document classification), where the **input** is a raw Somali text snippet/post and the **output** is a discrete dialect label (`MAXAA TIRI` or `MAAY`).

* **Data:**
  To overcome the lack of existing Maay datasets, the researchers constructed and manually annotated a balanced corpus of **8,947 Somali text samples** gathered from diverse real-world domains—including social media posts, user comments, news outlets, blogs, and formal documents—supplemented by expert-guided translations to ensure adequate coverage of written Maay (Gedi et al., 2026).

* **Methodology:**
  The authors systematically benchmarked both classical machine learning models and deep neural architectures using character- and word-level feature representations:

  1. **Traditional ML Classifiers:** Multinomial Naive Bayes and Support Vector Machines (SVM) trained on extracted lexical and character $n$-gram features.

  2. **Deep Learning Architecture:** A Bidirectional Long Short-Term Memory (**BiLSTM**) network designed to capture sequential dependencies in both forward and backward directions across text sequences.
     To verify that models did not simply memorize topic-specific artifacts, the researchers conducted **cross-source evaluation** (testing models on domains unseen during training).

* **Key Findings:**
  Both traditional ML and deep learning approaches achieved strong dialect discrimination on the benchmark dataset, with character-level and sequence features effectively capturing phonological and morphological differences between Maxaa Tiri and Maay. However, the study demonstrated that model accuracy drops during cross-source testing when transitioning from formal news text to noisy, informal social media comments—highlighting the necessity of diverse, multi-domain training corpora and dialect-aware preprocessing pipelines for low-resource languages (Gedi et al., 2026).

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media.

2. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.

3. Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., & Ebdiyow, A. U. (2026). Somali dialect identification in low-resource settings using machine learning and deep learning. *Discover Data*, 4(1), 10. https://doi.org/10.1007/s44248-026-00109-y

4. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. In *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)* (pp. 179–189). Association for Computational Linguistics.

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

6. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Stanford University.

7. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
ABDIRAHIM HASSAN
8. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 3982–3992). Association for Computational Linguistics.

9. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
