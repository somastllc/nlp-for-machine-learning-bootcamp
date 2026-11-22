# Assignment 1 – NLP Foundations

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a field of computing that helps computers understand, analyze, process, and generate human language such as Somali, English, and Arabic.

NLP is related to Artificial Intelligence (AI) and Machine Learning (ML). Artificial Intelligence is the broad field of creating computer systems that perform tasks that normally require human intelligence. Machine Learning is a part of AI that allows computers to learn patterns from data instead of depending only on manually written rules. NLP focuses specifically on human language.

For example, an email spam detection system uses NLP to analyze email messages. Machine Learning can train the system using examples of spam and normal emails. After training, the system can predict whether a new email is spam or not.

Therefore, AI is the broader field, Machine Learning is one approach within AI, and NLP focuses on processing human language.

---

## 2. Explain Corpus, Document, Token, Vocabulary, Feature, and Label

These terms are important when working with text data in NLP.

| Term       | Definition                                                                   | Example                                       |
| ---------- | ---------------------------------------------------------------------------- | --------------------------------------------- |
| Corpus     | A collection of text documents used for analysis or training.                | A collection of 10,000 SMS messages.          |
| Document   | A single piece of text in a corpus.                                          | "Are you coming today?"                       |
| Token      | A unit of text produced by a tokenizer.                                      | "Python", "is", or "."                        |
| Vocabulary | The collection of unique tokens found in the text.                           | `['Python', 'is', 'useful']`                  |
| Feature    | A numerical value representing information used by a Machine Learning model. | The number of times the word "prize" appears. |
| Label      | The correct category assigned to a particular document.                      | `spam` or `ham`.                              |

### Example Dataset

Consider the following SMS dataset:

| Document | Text                     | Label |
| -------- | ------------------------ | ----- |
| D1       | Are you coming today?    | ham   |
| D2       | Claim your cash prize!   | spam  |
| D3       | Call me when you arrive. | ham   |
| D4       | Win money now!           | spam  |

The entire collection of these messages is called a corpus. Each SMS is a document. The words and punctuation produced after tokenization are tokens. The unique tokens form the vocabulary. Numerical values extracted from these messages are features, while spam and ham are labels.

---

## 3. Describe Four Common NLP Tasks

### A. Text Classification

Text classification assigns a category to an entire text.

**Input:** A text message or document.

**Output:** A category or class.

Example:

Input: "Congratulations! You won a prize."

Output: `Spam`

Other examples include sentiment analysis, topic classification, and email classification.

### B. Sequence Labeling

Sequence labeling assigns labels to individual tokens or parts of a sentence.

**Input:** A sentence containing several words.

**Output:** Labels assigned to words or tokens.

Example:

Sentence: "Amina lives in Muqdisho."

| Word     | Label    |
| -------- | -------- |
| Amina    | Person   |
| lives    | Other    |
| in       | Other    |
| Muqdisho | Location |

This task can be used for Named Entity Recognition (NER), which identifies people, places, and organizations.

### C. Search and Similarity

Search and similarity systems find relevant or related documents based on a query or another document.

**Input:** A search query such as "battery problems."

**Output:** A ranked list of relevant documents or reviews.

For example, a system may return customer reviews discussing batteries that fail quickly.

### D. Text Generation

Text generation is the process of producing new human-readable text using a computer system.

**Input:** A prompt, question, or existing text.

**Output:** Newly generated text.

Examples include summarizing an article, answering questions, translating sentences, or generating a response in a chatbot.

---

## 4. Compare Supervised and Unsupervised Learning in NLP

### Supervised Learning

Supervised learning uses training data that contains input examples and their correct target labels. The model learns patterns between the text and its labels to make predictions on new data.

Example:

| Message                | Label |
| ---------------------- | ----- |
| Win money now!         | Spam  |
| Call me tomorrow.      | Ham   |
| Claim your free prize! | Spam  |

A model learns from these labeled examples and predicts whether new messages are spam or ham.

**Why are labels needed?**

Labels provide the correct answers during training. The model uses them to learn which patterns are associated with each class.

### Unsupervised Learning

Unsupervised learning works with data that does not have target labels for the task. The system attempts to find patterns, groups, or similarities within the data.

Example:

A dataset contains thousands of customer reviews without positive or negative labels. A system can represent the reviews as numerical vectors and find similar reviews or group related documents.

**Why are labels not needed?**

The system is not learning to predict predefined answers. Instead, it discovers patterns or relationships within the data.

### Comparison Table

| Supervised Learning                | Unsupervised Learning              |
| ---------------------------------- | ---------------------------------- |
| Uses labeled data.                 | Uses data without target labels.   |
| Learns to predict target answers.  | Finds patterns or similarities.    |
| Example: Spam classification.      | Example: Grouping similar reviews. |
| Labels are needed during training. | Target labels are not required.    |

---

## 5. Why Must Text Be Converted into Numerical Representations?

Most Machine Learning models work with numerical values rather than raw human language. Therefore, text must be converted into numbers before many models can process it.

A vectorizer converts text into numerical features using a consistent vocabulary.

### Bag-of-Words (BoW)

Bag-of-Words is a simple text representation method that counts or records the occurrence of words in a document.

For example, suppose the vocabulary is:

`['free', 'meeting', 'prize']`

We can represent two messages using binary values:

| Message      | free | meeting | prize | Vector    |
| ------------ | ---: | ------: | ----: | --------- |
| free prize   |    1 |       0 |     1 | [1, 0, 1] |
| free meeting |    1 |       1 |     0 | [1, 1, 0] |

Each position represents one word in the vocabulary. A value of 1 means the word appears, while 0 means it does not appear.

### Main Limitation

Bag-of-Words does not understand the meaning or context of words. It mainly records whether words appear or how often they appear. It also does not naturally preserve word order.

For example:

- "The dog bites the man."
- "The man bites the dog."

These sentences contain similar words but have different meanings. A basic Bag-of-Words representation may not distinguish their word order.

---

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

Somali is considered a low-resource language for some NLP applications because there may be fewer available datasets, tools, and labeled examples compared with languages such as English.

### A. Limited Labeled Data

Many NLP models require large datasets for training. Somali may have fewer ready-made labeled datasets for tasks such as spam classification, sentiment analysis, and Named Entity Recognition.

This makes it difficult for models to learn enough language patterns. Researchers may need to collect text and manually label examples, which requires time and resources.

### B. Spelling Variation

Somali text can contain spelling variations.

For example:

- Soomaaliya
- Somalia

A basic word-based vectorizer may treat these as two different vocabulary entries.

This can reduce model performance because the model may not recognize that both forms refer to the same country.

### C. Word Forms

Somali words can have different grammatical forms.

For example:

- `nin` – man
- `ninka` – the man

A simple word-based model may treat them as different tokens, even though they are related in meaning.

This increases vocabulary size and can make learning more difficult.

### D. Mixed-Language Text

Somali speakers sometimes combine Somali and English in the same message.

Example:

"Meeting-ka maanta ayaan leenahay."

This combines the English word "Meeting" with a Somali ending.

A tokenizer or model designed mainly for English may not process this type of text correctly, which can reduce its accuracy.

---

## 7. Case Study: Natural Language Processing for African Languages

### Case Study: Masakhane – Machine Translation for African Languages

Masakhane is a research community that works to improve Natural Language Processing for African languages, including languages with limited training data.

One of its important research areas is machine translation, which aims to translate text from one language into another.

### Task

The task is Machine Translation. The system receives text in one language and produces translated text in another language.

This is a form of Text Generation because the system generates a new sentence in the target language.

### Data

The research uses parallel text datasets. Parallel datasets contain sentences in one language and their corresponding translations in another language.

For example:

| English       | Target Language        |
| ------------- | ---------------------- |
| Good morning. | A translated sentence. |
| How are you?  | A translated sentence. |

Such datasets help the model learn relationships between languages.

### Method

Machine translation systems commonly use Machine Learning and neural network approaches. Models learn language patterns and relationships from translated sentence pairs.

The model processes the source-language text and generates an appropriate translation in the target language.

### Findings and Importance

Research efforts such as Masakhane demonstrate the importance of developing NLP technologies for African languages. They encourage researchers to create datasets, build language tools, and improve translation systems for languages that have limited digital resources.

This work helps make language technology more accessible to African communities.

### NLP Task Covered

This case study covers **Text Generation**, specifically Machine Translation, because the system produces new text in another language.

---

# Conclusion

Natural Language Processing helps computers work with human language. It is connected to Artificial Intelligence and Machine Learning and can be used for tasks such as text classification, sequence labeling, search, similarity, and text generation.

Text must usually be converted into numerical features before Machine Learning models can process it. Methods such as Bag-of-Words provide simple representations, although they have limitations in understanding meaning and context.

NLP for Somali and other low-resource languages faces challenges such as limited datasets, spelling variations, different word forms, and mixed-language text. Improving datasets and language tools can help create better NLP systems for these languages.

---

# References

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing (3rd Edition Draft).* [https://web.stanford.edu/\~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly Media. [https://www.nltk.org/book/](https://www.nltk.org/book/)

3. Masakhane. *Masakhane: Machine Translation for African Languages.* [https://www.masakhane.io/](https://www.masakhane.io/)

4. Scikit-learn Developers. *Working with Text Data.&#x20;*[https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

5. Hugging Face. *NLP Course.* [https://huggingface.co/learn/nlp-course/](https://huggingface.co/learn/nlp-course/)