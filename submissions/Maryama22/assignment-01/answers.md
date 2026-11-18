# Assignment 1 – NLP Foundations

---

## 1. Natural Language Processing, AI, and Machine Learning

**Natural Language Processing (NLP)** is a multidisciplinary subfield of computer science, artificial intelligence, and computational linguistics dedicated to enabling computers to read, interpret, analyze, and generate human language in both written and spoken forms (Jurafsky & Martin, 2024). Unlike programming languages, human languages are inherently ambiguous, context-dependent, and continuously evolving. NLP bridges this gap by translating unstructured human communication into structured representations that machines can systematically process.

### Relationship to Artificial Intelligence and Machine Learning
* **Artificial Intelligence (AI)** is the broadest overarching scientific discipline focused on creating computational systems capable of performing tasks that typically require human cognitive intelligence—such as logical reasoning, planning, perception, and decision-making (Russell & Norvig, 2021).
* **Machine Learning (ML)** is a foundational subset of AI in which algorithms automatically learn statistical patterns, relationships, and decision boundaries directly from historical data rather than relying exclusively on rigid, hand-crafted rules written by a human programmer.
* **Natural Language Processing (NLP)** sits at the intersection of AI, ML, and formal linguistics. While early rule-based NLP systems relied heavily on manually written grammatical rules and static dictionaries, modern NLP depends predominantly on statistical Machine Learning and deep neural networks to generalize across vast volumes of real-world text and speech.

| Concept | Core Definition | Role in Language Technology |
| :--- | :--- | :--- |
| **Artificial Intelligence (AI)** | The broad science of building systems that simulate human cognitive abilities | Defines the overarching intelligent goal (e.g., automated customer service triage) |
| **Machine Learning (ML)** | Algorithms that learn statistical patterns and generalize from training examples | Trains predictive models on historical language data without explicit hard-coded rules |
| **Natural Language Processing (NLP)** | Computational techniques for analyzing, understanding, and generating human language | Handles tokenization, syntactic parsing, semantic representation, and text generation |

### Real-Life Illustration: E-Commerce Customer Review Filter
Consider an online retail platform operating in Mogadishu that receives hundreds of Somali customer reviews every day. 
* **AI** sets the overarching business and technical goal: creating an intelligent triage system that automatically detects urgent customer complaints so support staff can resolve them before customers churn.
* **NLP** acts as the linguistic interface: it ingests each raw Somali review (for example, *"Alaabtu waa tayo xun oo waqtigii loogu talagay ma soo gaarin"*, meaning *"The product is poor quality and did not arrive on time"*), normalizes the text, splits it into meaningful tokens, and maps those tokens into numerical features.
* **Machine Learning** powers the classification decision: having been trained on thousands of historical reviews previously labeled as `Positive` or `Negative`, the ML classifier recognizes statistical patterns associated with dissatisfaction (such as the co-occurrence of `"tayo xun"` and `"ma soo gaarin"`) and automatically flags the incoming review as `Negative`.

---

## 2. Core Terms Used with Text Data

To illustrate foundational text data concepts clearly, consider a structured example dataset consisting of **500 Somali product reviews** collected from an online marketplace, annotated for binary sentiment analysis (`Positive` or `Negative`).

| Term | Formal Definition | Application to the E-Commerce Dataset |
| :--- | :--- | :--- |
| **Corpus** | A large, structured collection of authentic text documents gathered for linguistic analysis, benchmarking, or model training. | All 500 Somali customer reviews combined together into a single dataset. |
| **Document** | A single, standalone unit of text within a corpus (such as a sentence, product review, social media post, or news article). | One individual review entry, such as: *"Adeegga keenista waa mid aad u degdeg badan"* (*"The delivery service is very fast"*). |
| **Token** | A discrete, meaningful segment produced when a document is divided by a tokenizer (commonly whole words, subwords, or punctuation marks). | Splitting the document above with a word-level tokenizer produces 8 tokens: `["Adeegga", "keenista", "waa", "mid", "aad", "u", "degdeg", "badan"]`. |
| **Vocabulary** | The complete set of unique token types recognized across the entire corpus after text normalization and preprocessing. | All distinct words appearing across the 500 reviews without duplicates. For instance, if the 500 reviews contain 6,500 total words (token occurrences) that reduce to 1,850 distinct words, the **vocabulary size** is 1,850. |
| **Feature** | A measurable numerical property or attribute extracted from text that a Machine Learning model uses as input to make predictions. | The frequency count, binary presence, or TF-IDF weight of a specific term such as `"tayo"` (quality) or `"degdeg"` (fast) within a given review. |
| **Label** | The ground-truth target category or output value assigned to a document that a supervised model is trained to predict. | The sentiment tag attached to each review by human annotators—either **`Positive`** or **`Negative`**. |

In a supervised NLP pipeline, the raw customer review text serves as the input ($X$), while the sentiment label serves as the target output ($Y$). Once the model learns the statistical mapping between text features and labels during training, it can accurately predict labels for newly submitted, unseen reviews during inference.

---

## 3. Four Common NLP Tasks

NLP encompasses diverse computational tasks depending on the structure of the input data and the granularity of the required output:

| NLP Task | Input | Output | Concrete Example |
| :--- | :--- | :--- | :--- |
| **Text classification** | A complete document, paragraph, sentence, or message | A single discrete category (or multiple tags) assigned to the entire text | Classifying a Somali product review as `Positive` or `Negative` |
| **Sequence labeling** | A sequential stream of tokens (typically a sentence) | A categorical label assigned to *each individual token* in the sequence | Identifying `Khaalid` as a Person (`B-PER`) and `Hargeysa` as a Location (`B-LOC`) in *"Khaalid ayaa ku nool Hargeysa"* |
| **Search and similarity** | A user search query alongside a repository of candidate documents | A ranked list of the most relevant documents or continuous similarity scores | Retrieving customer reviews most semantically related to the query *"delivery delay"* |
| **Text generation** | A prompt, instruction, conversation history, or source document | Newly synthesized, contextually coherent human-like text | Generating a concise three-sentence summary of a lengthy company return policy |

### Detailed Breakdown of Each Task
1. **Text Classification:** Assigns one or more predefined labels to an entire piece of text based on its overall semantic content. When there are two possible categories (e.g., `Positive` vs. `Negative`), it is known as *binary classification*; when there are three or more categories (e.g., `Sports`, `Politics`, `Economy`), it is *multiclass classification*.
2. **Sequence Labeling:** Operates at the fine-grained token level rather than the document level, predicting a distinct tag for every token in a sentence while accounting for surrounding context. Prominent applications include **Named Entity Recognition (NER)**—which locates people, organizations, and geographic places—and **Part-of-Speech (POS) tagging**, which identifies nouns, verbs, and adjectives.
3. **Search and Similarity:** Evaluates lexical or semantic closeness between a search query and a collection of candidate documents to rank and retrieve the most relevant matches (Manning et al., 2008). Modern similarity systems go beyond exact keyword matching by comparing dense vector representations to capture conceptual meaning.
4. **Text Generation:** Produces brand-new sequences of natural language conditioned on an input prompt or source text. Common real-world applications include neural machine translation (translating between Somali and English), abstractive text summarization, and interactive conversational assistants.

---

## 4. Supervised vs. Unsupervised Learning in NLP

Machine Learning methodologies in NLP are broadly categorized into supervised and unsupervised learning paradigms depending on whether human-annotated target labels are available during the training phase:

| Dimension | Supervised Learning | Unsupervised Learning |
| :--- | :--- | :--- |
| **Training Data** | Text inputs paired with ground-truth target labels $(X, Y)$ | Raw, unlabeled text documents without target answers $(X)$ |
| **Primary Objective** | Learn a predictive mapping function to classify or score unseen text accurately | Discover hidden structures, thematic clusters, or latent semantic patterns |
| **Requirement for Target Labels** | **Strictly Required** | **Not Required** |
| **Evaluation Method** | Comparing model predictions against held-out ground-truth labels (using Accuracy, Precision, Recall, F1-score) | Intrinsic metrics (e.g., cluster coherence, perplexity) or qualitative human inspection |
| **NLP Example** | Training a sentiment classifier on 5,000 product reviews labeled `Positive` or `Negative` | Grouping 50,000 unlabeled customer reviews into thematic topics using Latent Dirichlet Allocation (LDA) |

### Why One Needs Target Labels and the Other Does Not
* **Supervised Learning:** In a supervised setup, the model learns through explicit error correction. For each training document, the algorithm predicts an output, compares its prediction against the ground-truth label using a mathematical loss function, and updates its internal parameters (weights) to minimize future errors. Without target labels, the model would have no objective standard to measure prediction error against.
* **Unsupervised Learning:** In an unsupervised setup, there is no predefined "correct answer" to predict. Instead, algorithms such as **Latent Dirichlet Allocation (LDA)** or **K-Means clustering** analyze intrinsic statistical properties of the text itself—such as how frequently certain words co-occur across documents—to automatically organize reviews into coherent thematic groups (e.g., shipping delays, product defects, or pricing praise). A human analyst subsequently inspects the top keywords in each cluster to interpret what topic the group represents.

---

## 5. Converting Text into Numerical Representations

### Why Numerical Conversion is Necessary
Machine Learning models—ranging from logistic regression to deep neural networks—are fundamentally mathematical optimization engines. They operate by performing linear algebra computations such as matrix multiplications, dot products, cosine similarity calculations, and gradient descent updates. Because computers cannot directly execute arithmetic operations on raw character strings like `"tayo"` or `"biyo"`, unstructured text must first be converted into structured numerical vectors—a process known as **text vectorization**.

### Simple Representation Method: Bag-of-Words (BoW)
One of the simplest and most foundational vectorization techniques is the **Bag-of-Words (BoW)** model (Bird et al., 2009). First, the system builds a fixed vocabulary of all unique words occurring in the training corpus and assigns each word a fixed column index. Second, every document is converted into a numerical vector of length $|V|$ (equal to the vocabulary size), where each entry records the frequency count (or binary presence) of that specific word in the document.

For example, suppose our fixed vocabulary is `["shirkad", "biyo", "fiican", "xun"]`:
* The review *"biyo fiican"* (*"good water"*) is represented as the vector `[0, 1, 1, 0]`.
* The review *"shirkad biyo xun"* (*"bad water company"*) is represented as `[1, 1, 0, 1]`.

### Main Limitations of Bag-of-Words
1. **Complete Loss of Word Order and Syntactic Context:** BoW treats a document as an unordered collection ("bag") of words, discarding grammar and word sequence. Consequently, two sentences with opposite meanings—such as *"Maanta ma fiicna"* (*"Today is not good"*) and *"Maanta waa fiican"* (*"Today is good"*)—share overlapping vocabulary words and produce very similar numerical vectors despite expressing contradictory sentiments.
2. **High Dimensionality, Sparsity, and Lack of Semantic Similarity:** In real-world corpora containing 50,000 or more unique words, every short sentence is represented as a 50,000-dimensional vector filled almost entirely with zeros (**sparse vectors**), which wastes memory and computational resources. Furthermore, BoW treats every word as completely independent; it cannot recognize that `"degdeg"` (*fast*) and `"dhaqso"` (*quickly*) have closely related meanings. Modern dense word embeddings such as **Word2Vec** (Mikolov et al., 2013) overcome this limitation by learning compact, continuous vectors from surrounding word contexts.

---

## 6. Challenges of NLP for Low-Resource Languages such as Somali

**Low-resource languages** are languages that lack large, publicly available, machine-readable corpora, annotated benchmarks, and pre-trained computational tools despite being spoken by tens of millions of people (Joshi et al., 2020). Building reliable NLP systems for Somali presents four major linguistic and data-related challenges:

1. **Scarcity of Annotated Datasets**
   * **Challenge:** Supervised Machine Learning and deep learning models require large, carefully curated, human-labeled datasets to generalize effectively. Compared to high-resource languages like English, Somali has far fewer publicly available annotated datasets across core tasks such as sentiment analysis, named entity recognition, toxicity detection, and machine translation. As Mohamed et al. (2025) highlight, Somali NLP faces significant limitations in annotated datasets, requiring researchers to invest substantial manual effort to construct new benchmarks from scratch.
   * **Impact on NLP Systems:** When trained on small or narrow datasets, models suffer from **overfitting**—memorizing specific phrases from the training data rather than learning generalizable linguistic patterns—leading to poor accuracy when deployed on real-world text.

2. **Rich Morphological Complexity**
   * **Challenge:** Somali is a morphologically rich language that uses extensive inflectional and derivational suffixes, clitics, and internal vowel changes to mark definiteness, gender, number, tense, and focus. For example, a single noun root such as *buug* ("book") can appear as *buugga* ("the book"), *buugag* ("books"), *buugaagta* ("the books"), or *buuggayga* ("my book"). Gedi et al. (2026) directly address this linguistic challenge in their development of a morphologically informed Somali lemmatization corpus that integrates rule-based morphological analysis, root derivation, and part-of-speech tagging.
   * **Impact on NLP Systems:** Without accurate Somali stemmers or lemmatizers, standard word-level tokenizers treat every inflected variant as an entirely unrelated vocabulary entry. This drastically inflates the vocabulary size, causes high **Out-of-Vocabulary (OOV)** rates during testing, and prevents models from recognizing that inflected variants share a core semantic root.

3. **Spelling Variation and Informal Orthography**
   * **Challenge:** Although Somali adopted an official, standardized Latin orthography in 1972 (which uses doubled vowels such as *aa*, *ee*, *ii*, *oo*, *uu* to represent long vowel sounds), informal digital communication on social media and messaging platforms exhibits widespread spelling inconsistency. Users frequently drop doubled vowels, use non-standard abbreviations, or apply phonetic spellings—for example, writing *manta* instead of *maanta* ("today"), or *somalia* instead of *Soomaaliya*.
   * **Impact on NLP Systems:** Standard tokenizers treat each spelling variant as a separate token, splitting statistical frequency across multiple sparse features. As a result, search engines fail to retrieve relevant documents and classifiers lose predictive signal.

4. **Code-Switching (Mixed-Language Text)**
   * **Challenge:** In everyday digital discourse, Somali speakers frequently alternate between Somali, English, and Arabic within the same sentence—a linguistic phenomenon known as **code-switching**. Moreover, speakers routinely attach Somali grammatical suffixes directly onto foreign root words, such as writing *"System-ka ayaa cillad ka jirtaa"* (*"The system has a fault"*) or *"File-ka ii soo dir"* (*"Send me the file"*).
   * **Impact on NLP Systems:** Monolingual tokenizers and language models trained exclusively on formal Somali or English struggle to segment hybrid morphological constructions (`System-ka`) and mixed syntactic structures, resulting in broken tokenization and degraded downstream performance.

| Somali NLP Challenge | Underlying Linguistic / Data Cause | Impact on NLP System Performance |
| :--- | :--- | :--- |
| **Scarce annotated datasets** | Limited public labeled benchmarks and corpora | Overfitting and weak generalization across unseen domains |
| **Rich morphology** | Extensive suffixation, inflections, and clitics | Bloated vocabulary size and high Out-of-Vocabulary (OOV) rates |
| **Spelling variation** | Informal online writing and dropped long vowels | Splits identical words into separate, low-frequency features |
| **Code-switching** | Mixing Somali with English/Arabic and hybrid affixes | Disrupts monolingual tokenizers, parsers, and language models |

---

## 7. Case Study: Somali Fake News and Toxicity Detection

* **Selected Research Paper:** Mohamed, et al. (2025). *Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models.* arXiv:2503.18117.
* **NLP Task Covered:** **Text Classification** (corresponding to Task #1 in Question 3), specifically encompassing binary fake news detection, binary toxicity classification, and multiclass Somali news topic categorization.
* **Problem Addressed:** With the rapid expansion of social media usage among Somali speakers, the spread of misinformation (fake news) and online toxic behavior has become a growing societal concern. However, automated detection systems were severely constrained by the lack of annotated Somali social media datasets and the absence of monolingual pretrained language models tailored to Somali's morphological and syntactic characteristics.
* **Data:** To overcome data scarcity, the researchers compiled a large-scale unlabeled Somali text corpus from diverse web sources for self-supervised pretraining. In addition, they constructed and manually annotated two new benchmark datasets collected from social media platforms—one dedicated to Somali fake news detection and the other to toxic message classification.
* **Method:** The authors developed **SomBERTa**, a Transformer-based monolingual language model pretrained specifically on the Somali corpus using masked language modeling. They subsequently fine-tuned SomBERTa on the downstream classification datasets and conducted comparative evaluations against existing multilingual Transformer architectures, including **AfriBERTa**.
* **Findings:** Experimental results demonstrated that **SomBERTa** consistently outperformed the competing multilingual baseline models—including AfriBERTa—across the Somali classification benchmarks. These findings provide strong empirical evidence that pretraining Transformer models directly on dedicated, native-language corpora yields substantially stronger representation quality and classification accuracy for low-resource languages than relying solely on cross-lingual multilingual models.

---

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media. https://www.nltk.org/book/
2. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assoweh, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)*, 179–189. https://aclanthology.org/2026.africanlp-main.17.pdf
3. Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 6282–6293. https://doi.org/10.18653/v1/2020.acl-main.560
4. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
6. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*. https://arxiv.org/abs/1301.3781
7. Mohamed, et al. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. *arXiv:2503.18117*. https://arxiv.org/abs/2503.18117
8. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
