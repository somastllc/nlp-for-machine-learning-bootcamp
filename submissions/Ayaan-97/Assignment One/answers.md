# **ASSIGNMENT 1 – NLP FOUNDATIONS**
# Assignment 1 – NLP Foundations
---
## 1. Definition of NLP and its relationship with AI and ML
---
**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that enables computers to understand, analyze, interpret, and generate human language.

NLP can be applied to Somali to help computers understand sentences, identify people's opinions, translate Somali into English, and answer questions written in Somali.

*Example: Somali AI chatbot*

Imagine a Somali student sends this message to an educational chatbot:

*"Sideen ku baran karaa Python programming?"*

Meaning: "How can I learn Python programming?"

The system works as follows:

- **AI:** Provides the intelligent chatbot that interacts with the student.
- **NLP:** Processes the Somali sentence and identifies that the student is asking for learning guidance.
- **ML:** Uses patterns learned from previous questions and answers to recognize the student's intention and select an appropriate response.

Possible chatbot response:

*"Waxaad Python ku baran kartaa adigoo ka bilaabaya variables, data types, conditions, loops iyo functions."*

Meaning: "You can learn Python by starting with variables, data types, conditions, loops, and functions."

This demonstrates how AI, NLP, and ML can work together to create a Somali-language educational assistant.

## 2. Important NLP terminology using Somali examples
---

Consider a dataset containing Somali customer reviews about a delivery company.

### **Example dataset**

| ID | Somali customer review                           | Label    |
| -- | ------------------------------------------------ | -------- |
| D1 | Adeegga shirkaddu aad buu u wanaagsan yahay.     | Positive |
| D2 | Alaabtayda si degdeg ah ayaa la ii keenay.       | Positive |
| D3 | Adeeggu aad buu u xun yahay.                     | Negative |
| D4 | Alaabtayda waqtigii loogu talagalay lama keenin. | Negative |

### **Explanation of the six terms**

**1. Corpus**

The corpus is the complete collection of Somali customer reviews.

In our example, all four reviews together form one corpus.

**2. Document**

A document is one individual text within the corpus.

Example:

*"Adeegga shirkaddu aad buu u wanaagsan yahay."*

This single review is one document.

**3. Token**

A token is a unit of text produced when a sentence is divided into smaller parts.

Example:

Original Somali sentence:

`Adeegga shirkaddu aad buu u wanaagsan yahay.`

Possible word-level tokens:

`["Adeegga", "shirkaddu", "aad", "buu", "u", "wanaagsan", "yahay"]`

**4. Vocabulary**

Vocabulary means all the unique tokens collected from the dataset.

For example, words such as:

- adeegga
- shirkaddu
- wanaagsan
- alaabtayda
- degdeg
- xun
- waqtigii

These are examples of vocabulary items found in the reviews.

**5. Feature**

Features are numerical representations of information extracted from the Somali reviews.

For example, the model can use the presence or frequency of words such as `wanaagsan`, `degdeg`, and `xun` to recognize customer opinions.

**6. Label**

A label is the expected category assigned to a document.

For this dataset:

- Positive: Good customer experience.
- Negative: Bad customer experience.

The labels help train a model to classify new Somali customer reviews.

## 3. Four common NLP tasks 
---

### **3.1 Text Classification**

Text classification is the process of assigning a category to a complete sentence or document.

**Somali application: Customer sentiment analysis**

Consider the following Somali sentences:

| Input sentence                              | Expected output |
| ------------------------------------------- | --------------- |
| Adeegga shirkaddu waa wanaagsan yahay.      | Positive        |
| Aad ayaan ugu faraxsanahay adeegga.         | Positive        |
| Adeeggu waa mid aad u xun.                  | Negative        |
| Shaqaalaha shirkaddu si fiican iima caawin. | Negative        |

**Input:** A Somali sentence or customer review.

**Output:** Positive or Negative.

A company can use this system to understand whether Somali-speaking customers are satisfied with its services.

### **3.2 Sequence Labeling**

Sequence labeling assigns a label to each individual token in a sentence.

**Somali application: Named Entity Recognition (NER)**

Consider this sentence:

*"Ayaan waxay dhigataa Jaamacadda Ummadda Soomaaliyeed ee Muqdisho."*

Meaning: "Ayaan studies at Somali National University in Mogadishu."

An NLP model can identify the following entities:

| Somali token | Label        |
| ------------ | ------------ |
| Ayaan        | PERSON       |
| waxay        | OTHER        |
| dhigataa     | VERB         |
| Jaamacadda   | ORGANIZATION |
| Ummadda      | ORGANIZATION |
| Soomaaliyeed | ORGANIZATION |
| Muqdisho     | LOCATION     |

**Input:** A Somali sentence.

**Output:** Tokens with their corresponding labels.

This can be useful for extracting people's names, university names, cities, and organizations from Somali documents.

### **3.3 Search and Similarity**

Search and similarity allow computers to identify documents that are relevant to a user's query.

**Somali application: University document search**

Imagine a student searches:

*"Jaamacadaha bixiya waxbarashada Computer Science ee Muqdisho."*

Meaning: "Universities offering Computer Science education in Mogadishu."

The search system may return documents containing:

- Macluumaadka Jaamacadda Ummadda Soomaaliyeed.
- Barnaamijyada waxbarashada Computer Science.
- Jaamacadaha ku yaalla magaalada Muqdisho.

The system can identify relevant documents even if they do not contain exactly the same words as the search query.

**Input:** A Somali search query and a collection of documents.

**Output:** A ranked list of relevant documents.

### **3.4 Text Generation**

Text generation allows computers to produce new sentences and paragraphs based on instructions.

**Somali application: Educational chatbot**

Input:

*"Sharax muhiimadda ay leedahay barashada Artificial Intelligence."*

Meaning: "Explain the importance of studying Artificial Intelligence."

Possible output:

*"Artificial Intelligence waa tiknoolajiyad muhiim ah oo ka caawisa dadka inay si fudud u xalliyaan dhibaatooyinka, u falanqeeyaan xogta, una horumariyaan adeegyada bulshada."*

Meaning: "Artificial Intelligence is an important technology that helps people solve problems easily, analyze data, and improve public services."

**Input:** A Somali question or instruction.

**Output:** A newly generated Somali response.

## 4. Supervised and Unsupervised Learning in NLP

### **4.1 Supervised Learning**

Supervised learning is a machine learning method that uses labeled data to train a model.

**Somali example: Detecting positive and negative comments**

Consider this dataset:

| Somali comment                          | Label    |
| --------------------------------------- | -------- |
| Waxbarashadu aad bay u wanaagsan tahay. | Positive |
| Macallinku si fiican ayuu wax u dhigaa. | Positive |
| Waxbarashadu ma fiicna.                 | Negative |
| Adeegga dugsigu waa mid liita.          | Negative |

The model learns the relationship between Somali words and their corresponding sentiment labels.

When it receives a new sentence:

*"Macallinku waa qof wanaagsan."*

It may predict:

`Positive`

**Why are labels necessary?**

The model needs correct answers during training so that it can learn how to distinguish positive and negative Somali sentences.

### **4.2 Unsupervised Learning**

Unsupervised learning uses data without predefined labels. The model discovers patterns and similarities independently.

**Somali example: Grouping Somali news articles**

Imagine collecting 1,000 Somali news articles without assigning categories.

An unsupervised learning algorithm such as K-Means can group similar articles.

| Cluster   | Possible topic               |
| --------- | ---------------------------- |
| Cluster 1 | Siyaasadda iyo dowladda      |
| Cluster 2 | Waxbarashada iyo jaamacadaha |
| Cluster 3 | Dhaqaalaha iyo ganacsiga     |
| Cluster 4 | Ciyaaraha iyo kubadda cagta  |

The model discovers these groups by examining similarities between the articles.

**Why are labels not necessary?**

The purpose is to discover hidden patterns rather than predict an answer that has already been provided.

## **5. Numerical representation of the text
---

### **5.1 Why must Somali text be converted into numbers?**

Machine learning models generally require numerical inputs to perform mathematical calculations and identify patterns.

For example, a computer can display this Somali sentence:

*"Waxaan jeclahay barashada Computer Science."*

However, most traditional machine learning algorithms cannot directly use the ordinary words as numerical features.

Therefore, the sentence must be converted into numbers through a process called text vectorization.

### **5.2 Bag of Words (BoW) using Somali**

Bag of Words is a simple method that represents text by counting how many times each vocabulary word appears in a document.

Consider these three Somali sentences:

- D1: Waxaan jeclahay waxbarashada.
- D2: Waxaan jeclahay tiknoolajiyada.
- D3: Waxbarashadu waa muhiim.

**Create the vocabulary**

After basic text normalization, we obtain:

`["waxaan", "jeclahay", "waxbarashada", "tiknoolajiyada", "waa", "waxbarashadu", "muhiim"]`

**Assign numerical positions**

| Word           | Position |
| -------------- | -------- |
| waxaan         | 1        |
| jeclahay       | 2        |
| waxbarashada   | 3        |
| tiknoolajiyada | 4        |
| waa            | 5        |
| waxbarashadu   | 6        |
| muhiim         | 7        |

**Convert the sentences into numerical vectors**

| Document | waxaan | jeclahay | waxbarashada | tiknoolajiyada | waa | waxbarashadu | muhiim |
| -------- | ------ | -------- | ------------ | -------------- | --- | ------------ | ------ |
| D1       | 1      | 1        | 1            | 0              | 0   | 0            | 0      |
| D2       | 1      | 1        | 0            | 1              | 0   | 0            | 0      |
| D3       | 0      | 0        | 0            | 0              | 1   | 1            | 1      |

The resulting vectors are:

- D1 = `[1, 1, 1, 0, 0, 0, 0]`
- D2 = `[1, 1, 0, 1, 0, 0, 0]`
- D3 = `[0, 0, 0, 0, 1, 1, 1]`

The numerical vectors can now be used by machine learning algorithms.

### **5.3 Limitations of Bag of Words for Somali**

Bag of Words has several limitations:

**1. It ignores word order.**

For example:

- Ardaygu macallinka ayuu caawiyaa.
- Macallinku ardayga ayuu caawiyaa.

The two sentences have different meanings, but Bag of Words may produce the same representation because they contain the same words.

**2. It does not understand context.**

A word can have different meanings depending on how it is used in a sentence.

**3. It has difficulty with different word forms.**

For example:

- Macallin
- Macallimiin

These are related words, but a basic Bag of Words model treats them as separate vocabulary items.

**Conclusion:** Numerical representation allows machine learning algorithms to process Somali text, but simple methods such as Bag of Words cannot fully understand Somali grammar, word order, and contextual meaning.

## 6. Challenges of Natural Language Processing for Somali
---
Somali is often described as a **low-resource language in NLP** because, compared with high-resource languages such as English, it has historically had fewer publicly available annotated datasets, language-processing tools, and widely used evaluation benchmarks.

Somali is an important example of a language that requires more specialized NLP research and resources.

Although millions of people communicate in Somali, developing accurate NLP systems remains challenging because of limitations in digital datasets, linguistic tools, and language-processing resources.

### 6.1: Limited Somali Datasets

One major challenge is the limited availability of high-quality Somali datasets, particularly datasets that have been manually labeled for machine learning.

For example, developing a Somali sentiment analysis system requires many labeled sentences such as:

| Somali sentence              | Label    |
| ---------------------------- | -------- |
| Adeegga waa wanaagsan yahay. | Positive |
| Aad ayaan ugu faraxsanahay.  | Positive |
| Adeegga waa mid liita.       | Negative |
| Ma jecli adeeggan.           | Negative |

If a researcher only collects 100 examples, the model may not learn enough different Somali expressions.

**Effect on NLP:**

- Low classification accuracy.
- Poor understanding of unfamiliar sentences.
- Difficulty identifying different expressions.
- Increased risk of overfitting.

**Possible solution:** Researchers should collect more Somali text from educational materials, news articles, public discussions, and other appropriate sources, and have native speakers annotate the data.

### 6.2: Spelling and Writing Variations

Somali speakers sometimes use different writing styles, especially in informal online communication.

For example, consider these possible forms:

- Aad baad u mahadsan tahay.
- Aad baad u mahadsantahay.

The writing differs, although the intended meaning is similar.

A model that has learned only one form may have difficulty recognizing the other.

**Effect on NLP:**

- The same expression may be treated as different tokens.
- Search systems may miss relevant results.
- Classification models may produce incorrect predictions.

**Possible solution:** Use text normalization to standardize common spelling variations while preserving the original meaning.

### 6.3: Complex Word Forms and Morphology

Somali words can have different forms depending on grammatical structure and number.

For example:

| Word        | Meaning      |
| ----------- | ------------ |
| Macallin    | Teacher      |
| Macallimiin | Teachers     |
| Arday       | Student      |
| Ardayda     | The students |

An NLP system must learn relationships between related forms instead of treating every form as completely unrelated.

**Effect on NLP:**

- Increased vocabulary size.
- Difficulty recognizing related words.
- Reduced search accuracy.
- More complicated model training.

**Possible solution:** Use stemming and lemmatization.

Lemmatization attempts to identify the appropriate base form of a word, while stemming reduces words using simpler morphological rules.

### 6.4: Mixed-Language Text (Code-Switching)

Many Somali speakers combine Somali and English in everyday communication.

For example:

*"Maanta waxaan leeyahay assignment ku saabsan Machine Learning."*

Meaning: "Today I have an assignment about Machine Learning."

This sentence contains Somali and English vocabulary.

**Effect on NLP:**

- The system may fail to recognize some words.
- It may apply incorrect grammatical rules.
- Translation may become inaccurate.
- Sentiment analysis may misunderstand the sentence.

**Possible solution:** Develop multilingual NLP models that recognize Somali and English within the same sentence.

### Summary of Somali NLP Challenges

| Challenge          | Somali example or issue      | Effect                                    |
| ------------------ | ---------------------------- | ----------------------------------------- |
| Limited datasets   | Few annotated Somali reviews | Poor model performance                    |
| Spelling variation | Different writing forms      | Incorrect token recognition               |
| Morphology         | Macallin / Macallimiin       | Difficulty identifying word relationships |
| Code-switching     | Somali mixed with English    | Incorrect language processing             |

Research has started addressing these problems. For example, the Somali lemmatization research by Gedi et al. (2026) created a lexical resource containing more than 5,584 root words and 78,629 derivative forms, demonstrating the importance of building Somali-specific linguistic resources. 

## 7. Case Study: A Real NLP Research Project for the Somali Language
---

## **Research paper: Morphologically-Informed Somali Lemmatization Corpus Built with a Web-Based Crowdsourcing Platform**

Published research – AfricaNLP 2026

**Authors:** Abdifatah Ahmed Gedi, Shafie Abdi Mohamed, Yusuf A. Yusuf, Muhidin A. Mohamed, Fuad Mire Hassan, and Houssein A. Assowe.

**Publication:** Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026).

**Research link:** [Read the original research paper](https://aclanthology.org/2026.africanlp-main.17/).

### 7.1 Background of the Research

Somali has many related word forms, making it difficult for computers to identify the base form of a word.

For example:

- Macallin – Teacher.
- Macallimiin – Teachers.

A computer may treat these as two completely different words unless it has learned their relationship.

The researchers focused on developing a Somali lemmatization resource to help computers identify the root forms of Somali words.

### 7.2 Research Task

The main task is **Somali Lemmatization**.

Lemmatization is the process of identifying the base or dictionary form of a word.

For example, a Somali NLP system may be designed to associate:

| Input word  | Expected base form |
| ----------- | ------------------ |
| Macallimiin | Macallin           |
| Ardayda     | Arday              |
| Buugaagta   | Buug               |

These are illustrative examples of the lemmatization task, not individual reported evaluation outputs from the paper.

### 7.3 Data Used

The researchers developed a Somali lemmatization lexicon containing:

# **5,584+**

Root words

# **78,629+**

Derivative forms

The lexical resource also includes part-of-speech annotations for the words.

The researchers evaluated their approach using a 294-document corpus containing news articles, social media posts, and short messages. 

### 7.4 Method Used

The researchers used a combination of linguistic resources and computational techniques.

The process included:

1. **Lexicon development:** Creating a dictionary containing Somali root words and their related forms.
2. **Crowdsourcing:** Developing a web-based platform to support the expansion of the lexical resource.
3. **Part-of-speech annotation:** Associating words with their grammatical categories.
4. **Rule-based processing:** Applying linguistic rules to identify base forms.
5. **Evaluation:** Testing the lemmatization approach using different types of Somali documents.

### 7.5 Research Findings

The study reported the following lemmatization accuracy:

| Document type              | Accuracy |
| -------------------------- | -------- |
| Full articles              | 51.27%   |
| Article excerpts           | 44.14%   |
| Short texts such as tweets | 59.51%   |

These results show that Somali lemmatization remains challenging, particularly when dealing with longer documents and complex word forms.

The study also demonstrates the value of combining lexical resources, linguistic rules, and human contributions.

### 7.6 Which NLP Task Does This Research Cover?

This research mainly covers **Sequence Labeling and word-level text normalization**.

The system processes individual Somali words and associates their different forms with their appropriate base forms.

It is related to sequence labeling because a sentence can be processed token by token, with each token receiving a corresponding lemma.

### 7.7 Importance of the Research

This research is important because:

- It contributes to Somali language technology.
- It provides linguistic resources for future NLP researchers.
- It helps improve Somali information retrieval.
- It supports the development of better Somali machine learning models.
- It demonstrates how human knowledge can help solve NLP problems in low-resource languages.

**My understanding:** This research shows that developing Somali NLP systems requires more than simply collecting sentences. Researchers must also understand how Somali words are formed and how different forms are related.

## Conclusion
---
Natural Language Processing has significant potential to improve technology for Somali-speaking communities.

Through the examples in this assignment, I have learned how computers can process Somali sentences, identify people's opinions, recognize names and locations, search Somali documents, and generate Somali responses.

I have also understood that developing Somali NLP systems requires special attention to language structure, vocabulary, spelling variations, and the availability of training datasets.

The Somali lemmatization case study demonstrates that researchers are already developing language-specific resources to address these challenges.
---

## References (APA 7th Edition)
---
1. Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of the 7th Workshop on African Natural Language Processing*, 179–189. [https://doi.org/10.18653/v1/2026.africanlp-main.17](https://doi.org/10.18653/v1/2026.africanlp-main.17)
2. Bahar, K., & Ramaha, N. T. (2023). Exploring Somali sentiment analysis: A resource-light approach for small-scale text classification. *International Conference on Applied Engineering and Natural Sciences*. [https://doi.org/10.59287/icaens.1069](https://doi.org/10.59287/icaens.1069)
3. Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing* (3rd ed. draft). Stanford University. [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
4. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word lemmatization approach for Somali language. *AfricaNLP Workshop at ICLR 2023*.
5. Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.
6. Nakatumba-Nabende, J., et al. (2024). Building text and speech benchmark datasets and models for low-resourced East African languages: Experiences and lessons. *Applied AI Letters, 5*(2), e92. [https://doi.org/10.1002/ail2.92](https://doi.org/10.1002/ail2.92)
