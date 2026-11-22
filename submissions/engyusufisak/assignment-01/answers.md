# Assignment 1 – NLP Foundations

**Name:** Yusuf Isak
**GitHub:** engyusufisak  
**Course:** NLP for Machine Learning Bootcamp

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is NLP and how it relates to AI and ML](#1-what-is-nlp-and-how-it-relates-to-ai-and-ml)
3. [Core terms with one dataset](#2-core-terms-with-one-dataset)
4. [Four common NLP tasks](#3-four-common-nlp-tasks)
5. [Supervised and unsupervised learning](#4-supervised-and-unsupervised-learning)
6. [Why text must become numbers](#5-why-text-must-become-numbers)
7. [Challenges of NLP for Somali](#6-challenges-of-nlp-for-somali)
8. [Case study: MasakhaNEWS](#7-case-study-masakhanews)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Introduction

Today people write a lot of text every day: news, Facebook posts, WhatsApp messages, comments and reviews. A big part of this text is in Somali. But a computer does not understand text the way we do. It needs special methods to read the text, cut it into small pieces, change it into numbers and learn from it. This area is called **Natural Language Processing (NLP)**.

In this assignment I answer seven questions about the basics of NLP. In questions 1 to 5 I explain what NLP is, the main words we use for text data, the common NLP tasks, supervised and unsupervised learning, and how text is changed into numbers. In questions 6 and 7 I focus on the Somali language: why it is hard for NLP, and a real research paper that made a Somali dataset.

To connect my answers together, I use one dataset as an example in many places: the **Somali part of MasakhaNEWS**, which is a collection of BBC Somali news articles where each article has a topic (Adelani et al., 2023).

---

## 1. What is NLP and how it relates to AI and ML

### 1.1 Definition

In my own words, **NLP is the part of computer science that teaches computers to work with human language**, like Somali, English or Arabic. "Natural" language means the languages people speak, not programming languages like Python (Jurafsky & Martin, 2025). An NLP system gets text (or speech) as input and gives something useful as output, for example a category, a list of search results, or new text.

### 1.2 AI, ML and NLP

- **Artificial Intelligence (AI)** is the big field. Its goal is to make computers do things that need intelligence, like thinking, learning and making decisions (Russell & Norvig, 2021).
- **Machine Learning (ML)** is one way to do AI. Instead of a programmer writing all the rules, the computer learns the rules from examples (data).
- **NLP** is the part of AI that works with language.

When I first studied this, I thought NLP is completely inside ML, but that is not correct. **NLP and ML overlap.** The first NLP programs did not learn anything. For example, ELIZA (1966) was a chatbot that only used rules written by people. Later, NLP started using statistics, and now most NLP uses machine learning and deep learning (Jurafsky & Martin, 2025). So today most NLP is ML-based, but rule-based NLP is still NLP.

```
┌──────────────── Artificial Intelligence ────────────────┐
│    ┌──── Machine Learning ─────┐                        │
│    │  (images, numbers   ┌─────┼──── NLP ─────┐         │
│    │   prediction…)      │ML-based│ rule-based │         │
│    │                     │  NLP   │    NLP     │         │
│    └─────────────────────┼────────┘            │         │
│                          └─────────────────────┘         │
└─────────────────────────────────────────────────────────┘
Figure 1. NLP and ML overlap. Rule-based NLP is outside ML.
```

| Time | How NLP worked | Example |
| --- | --- | --- |
| 1950s–1980s | Rules written by people | Georgetown–IBM translation test (1954), ELIZA (1966) |
| 1990s–2000s | Statistics and machine learning | Spam filters, statistical translation |
| 2010s–now | Deep learning | Word2Vec (2013), transformers, ChatGPT |

### 1.3 Real-life example: the keyboard on my phone

When I type "Waan ku" on my phone, the keyboard shows words like "jeclahay" or "arkay" to help me finish the sentence. In this one small feature we can see all three:

- **AI:** the goal is to help me write faster, like a smart helper.
- **ML:** nobody wrote a rule that says "after *ku* show *jeclahay*". The keyboard learned it from a lot of text. It also learns from the words I type often.
- **NLP:** the keyboard must cut my sentence into words, change them into numbers, look at the words before, and then suggest the next word.

---

## 2. Core terms with one dataset

**My dataset:** the Somali part of **MasakhaNEWS** (Adelani et al., 2023). It has **1,463 BBC Somali news articles**. Each article has one topic from 7 topics: business, entertainment, health, politics, religion, sports and technology. The data is divided into 1,021 articles for training, 148 for checking (dev) and 294 for testing.

| Term | What it means (my words) | In my dataset |
| --- | --- | --- |
| **Corpus** | All the texts together | All 1,463 Somali news articles |
| **Document** | One text, one example | One news article |
| **Token** | A small piece of text after we cut it, usually a word or a punctuation mark | `Kooxda`, `Barcelona`, `.` |
| **Vocabulary** | The list of all *different* tokens | Every different word in the 1,021 training articles |
| **Feature** | A number the model uses as input | How many times the word `ciyaarta` ("the game") is in an article |
| **Label** | The correct answer for one example | `sports` for an article about football |

**Example.** I wrote this sentence like a sports news article:

> *Kooxda Barcelona ayaa ciyaarta ku guuleysatay.* ("The Barcelona team won the game.")

If we cut it into words, we get these tokens:
`['Kooxda', 'Barcelona', 'ayaa', 'ciyaarta', 'ku', 'guuleysatay', '.']`

That is **7 tokens**. No word is repeated, so if this was our only text, the vocabulary would also have 7 entries. When we count the words, the sentence becomes numbers (the **features**), and its **label** is `sports`.

(I learned these terms from Bird, Klein & Loper, 2009, and Jurafsky & Martin, 2025.)

---

## 3. Four common NLP tasks

| Task | Input | Output | Example |
| --- | --- | --- | --- |
| **Text classification** | A whole text | One label for the text | A news article → `politics`; a product review → `negative` |
| **Sequence labeling** | A sentence (list of tokens) | One label for **each** token | `Xasan` → PERSON, `Hargeysa` → PLACE, other words → nothing |
| **Search and similarity** | A question + many documents | A list of the most similar documents | Searching "cudurka duumada" (malaria) → health articles about malaria |
| **Text generation** | A text or a question | New text | Writing a short summary of a long news article |

- **Text classification:** the model reads the whole text and gives it one category. If there are two categories it is called binary, and if there are more it is called multiclass (Jurafsky & Martin, 2025).
- **Sequence labeling:** the model gives a label to every word, not to the whole text. Finding names of people and places is an example (this is called NER).
- **Search and similarity:** the system changes the question and the documents into numbers, checks which documents are closest to the question, and shows them in order (Manning, Raghavan & Schütze, 2008).
- **Text generation:** the model writes new text, one word after another.

---

## 4. Supervised and unsupervised learning

| | **Supervised** | **Unsupervised** |
| --- | --- | --- |
| Data | Text **and** the correct label | Text only |
| Goal | Learn to predict the label | Find groups or patterns |
| My example | Classifying Somali news by topic | Finding topics in news with LDA |
| How to check it | Compare with the correct labels | Harder, a person looks at the results |

**Supervised example.** In MasakhaNEWS, people gave every article a topic. The model sees many articles with their topics and learns, for example, that words like `ciyaarta` and `kooxda` usually mean `sports`. Then we test it on new articles it has never seen.

**Unsupervised example.** LDA is a method that finds topics by itself (Blei, Ng & Jordan, 2003). If I give it 5,000 Somali news articles **without** topics and ask it for 5 topics, it finds groups of words that often come together. For example, one group could be `doorashada`, `baarlamaanka`, `madaxweyne` (election, parliament, president) and another group could be `cudurka`, `isbitaalka`, `dhakhtar` (disease, hospital, doctor). The model doesn't know these groups are "politics" and "health". A person looks at the groups and gives them names after.

**Why one needs labels and the other doesn't.** In supervised learning, **people decide the categories**. The model can't know that a category like "religion" exists, or which articles belong to it, if we don't show it examples with labels. In unsupervised learning we only ask the model to find patterns that are already in the text, like which words come together. So it doesn't need labels. But the groups it finds may be different from the categories we wanted.

---

## 5. Why text must become numbers

### 5.1 Why

Machine learning models work with math: they add, multiply and compare numbers. They cannot multiply a word like "guuleysatay". Also, texts have different lengths. In MasakhaNEWS, a headline can be 4 or 5 words, but the article under it can be hundreds of words. Most models need every example to have the **same number of features**, like every row in an Excel table has the same columns. So we use a method that changes every text into a list of numbers with the same length.

### 5.2 One method: Bag of Words

Bag of Words counts the words. It works like this:

1. Make a vocabulary from the training texts.
2. Give every word a fixed position.
3. For every text, count how many times each word appears.

My example with three short texts:

- T1: "kooxda ciyaarta guuleysatay"
- T2: "dowladda doorashada"
- T3: "ciyaarta kooxda kooxda"

Vocabulary: `[ciyaarta, doorashada, dowladda, guuleysatay, kooxda]`

| Text | ciyaarta | doorashada | dowladda | guuleysatay | kooxda |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 1 | 0 | 0 | 1 | 1 |
| T2 | 0 | 1 | 1 | 0 | 0 |
| T3 | 1 | 0 | 0 | 0 | 2 |

Now every text is a list of 5 numbers, and a model can use it. In Python, scikit-learn's `CountVectorizer` does this (scikit-learn developers, n.d.). The MasakhaNEWS paper also used it.

### 5.3 Main limitation

Bag of Words **forgets the order of the words and their meaning**. For example, "dog bites man" and "man bites dog" get the same numbers, but the meaning is very different. Also, words with the same meaning get different positions. If a new text has a word that was not in the training texts, the model just ignores it. Later methods like word embeddings (Mikolov et al., 2013) were made partly to fix the meaning problem.

---

## 6. Challenges of NLP for Somali

Somali is called a **low-resource language** in NLP. Joshi et al. (2020) showed that a few languages like English and Chinese have a lot of data and tools, but most languages in the world, including many African languages, have very little. These are the challenges I found:

### 6.1 Not enough data

There is much less labelled Somali data than English. Even the MasakhaNEWS team could only collect about 2,900 Somali BBC articles, and in the end only 1,463 were labelled (Adelani et al., 2023). English datasets often have hundreds of thousands of examples.
**Effect:** the model sees fewer examples, so it learns less and makes more mistakes. Also, with a small test set, we can't be very sure about the results.

### 6.2 Different spellings

Somali got its official Latin writing in 1972, but on Facebook and WhatsApp many people don't follow the official spelling. Long vowels are sometimes written with two letters and sometimes with one: `caafimaad` / `cafimaad` (health), `dhaqaale` / `dhaqale` (economy). Some people write `waxaan` and others write `wax aan`.
**Effect:** the computer thinks every spelling is a different word. One word becomes many rare words, and if a new spelling comes later, the model doesn't recognise it.

### 6.3 Many forms of one word

In Somali we add endings to words. For example `buug` (book), `buugga` (the book), `buuggii` (the book we talked about), and `buugaag` (books).
**Effect:** the computer sees these as four different words, even though they are related. English tools (like stopword lists) don't know these Somali endings.

### 6.4 Mixed languages and dialects

Many Somali people mix Somali with Arabic and English in one sentence, for example: *"InshaAllah berri ayaan kuu soo diri doonaa file-ka"* ("God willing, I will send you the file tomorrow"). This sentence has Arabic, Somali, and an English word with a Somali ending. There are also different kinds of Somali, like Maxaa tiri, Benaadir and Maay.
**Effect:** a model that learned from formal news Somali can make mistakes on social media Somali or on Maay, because the text looks different from what it learned.

### 6.5 People don't always agree on labels

In MasakhaNEWS, the three Somali annotators often disagreed about topics. Their agreement score was 0.55 out of 1, which was one of the lowest in the dataset (Adelani et al., 2023).
**Effect:** if the labels are not consistent, the model learns from confusing examples.

---

## 7. Case study: MasakhaNEWS

### 7.1 The paper

Adelani, D. I., Masiak, M., et al. (2023). *MasakhaNEWS: News Topic Classification for African Languages*. It was made by **Masakhane**, a group of African NLP researchers. About 50 people worked on it.

### 7.2 The problem

For English there are many datasets for classifying news by topic, but for African languages there were almost none. So nobody could really measure how well NLP models work for these languages.

### 7.3 The task

**Classifying news by topic:** the model reads a news article and chooses one topic from 7 (business, entertainment, health, politics, religion, sports, technology).

### 7.4 The data

- News in **16 languages** spoken in Africa, for example Somali, Hausa, Swahili, Amharic and Yorùbá, plus English and French.
- The articles came from **BBC** and **Voice of America** websites.
- Volunteers labelled the articles. First they labelled 200 articles by hand. After that, a model suggested topics and the people corrected them. At least two people had to agree on the topic.
- For each language, 70% of the data was for training, 10% for checking and 20% for testing.

| Somali part | |
| --- | --- |
| Source | BBC Somali |
| Articles | 1,463 (1,021 train / 148 dev / 294 test) |
| Topics | 7 |
| Annotators | 3 |

### 7.5 The method

They tested three types of models. The results are F1 scores (out of 100, higher is better).

1. **Simple ML models** like Naive Bayes, using Bag of Words (the same method I explained in question 5).
2. **Big pretrained language models** like AfroXLMR, which were trained more on the news data.
3. **Models with very few or zero examples:** for example, training with only 5 to 50 articles per topic, training on Hausa or Swahili and testing on Somali, or asking ChatGPT and GPT-4 directly without any training.

### 7.6 The results

| | All languages (average) | Somali |
| --- | ---: | ---: |
| Best simple ML model | 85.7 | 74.8 |
| Best pretrained model (AfroXLMR-large) | **92.6** | **86.9** |
| GPT-4 with no training | 81.7 | 77.3 |

- The big pretrained models were much better than the simple Bag of Words models.
- With only **10 examples per topic**, one of the few-example methods reached 86.0 on average. That is more than 90% of the result with full training. I think this is very good news for languages like Somali that don't have a lot of data.
- GPT-3.5 did badly on languages that don't use Latin letters, like Amharic and Tigrinya.
- Somali scores were **lower than the average**. I think this matches the challenges in question 6: less data, and annotators who didn't agree much.

### 7.7 Which NLP task is it?

It is **text classification** (question 3). The input is a whole news article and the output is one topic. There are 7 topics, so it is **multiclass** classification.

### 7.8 Why it is important

- Now Somali has a public dataset with labels checked by people, so new Somali models can be tested and compared. (There was an older dataset called ANTC with Somali, but its labels came from the websites' own categories, not from people.)
- It shows we don't always need a huge dataset. Pretrained models and few-example methods can still work well with small Somali data.
- **A limitation I noticed:** all the Somali articles are from BBC, so the Somali is formal and correct. I think a model trained on it will do worse on Facebook or WhatsApp Somali, because of the spelling and mixed-language problems from question 6.

---

## Conclusion

In this assignment I learned the basics of NLP. NLP is the part of AI that works with human language. Today it mostly uses machine learning, but rule-based programs are also NLP (Q1). We describe text data with words like corpus, document, token, vocabulary, feature and label (Q2). The four common NLP tasks are different mainly in their output: one label, a label for every word, a list of results, or new text (Q3). Supervised learning needs labels because people decide the categories, but unsupervised learning finds patterns in the text by itself (Q4). Models only understand numbers, so we change text into numbers, for example with Bag of Words, which is simple but forgets word order and meaning (Q5).

For Somali, all of this is harder because there is little data, spelling is different from person to person, words have many forms, and people mix languages (Q6). The MasakhaNEWS paper shows this: Somali results were lower than average, but modern models still worked well with small data (Q7). I think Somali NLP needs more good Somali data and tools made for our language, and Somali speakers like us can help build them.

---

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J. O., Tonja, A. L., Mwase, C., Ogundepo, O., Dossou, B. F. P., et al. (2023). *MasakhaNEWS: News topic classification for African languages*. arXiv:2304.09972. <https://arxiv.org/abs/2304.09972>

Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly. <https://www.nltk.org/book/>

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 6282–6293). <https://doi.org/10.18653/v1/2020.acl-main.560>

Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft). <https://web.stanford.edu/~jurafsky/slp3/>

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. <https://nlp.stanford.edu/IR-book/>

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space*. arXiv:1301.3781. <https://arxiv.org/abs/1301.3781>

Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.

scikit-learn developers. (n.d.). *Feature extraction: Text feature extraction*. <https://scikit-learn.org/stable/modules/feature_extraction.html>
