# Assignment 1: NLP Foundations

## 1. What is NLP, and how does it relate to AI and Machine Learning?

Natural Language Processing (NLP) is the field that teaches computers to work with human language. The work includes reading text, finding meaning in it, and writing text back. Jurafsky and Martin describe it as a mix of computer science and linguistics, and they treat speech and text as two sides of the same field [1].

Artificial Intelligence (AI) is the wider goal. It covers any system that does tasks we normally link to human thinking, such as planning, seeing, and speaking [2]. Machine Learning (ML) is one way to reach that goal. In ML, a program improves at a task by learning from examples, not by following rules that a person wrote by hand [3].

The three sit inside each other. AI is the largest circle. ML is a part of AI. NLP is the part of AI that deals with language, and today most NLP systems use ML to do their work. Older NLP systems used hand written rules, so NLP does not always need ML, but the modern field leans on it heavily [1].

**A real life example: a spam filter.** Think of an email service that moves junk mail out of your inbox.

* The NLP part is reading the email text and turning the words into something the system can study.
* The ML part is learning from thousands of emails that people already marked as spam or not spam. The model finds word patterns that often show up in spam, such as "free prize" or "send your bank details".
* The AI part is the full system acting on its own. It decides, without a human, where each new email should go.

So the three work as one chain. NLP prepares the language, ML learns from it, and the whole system behaves in a smart way that we call AI.

## 2. Key terms, shown with one dataset

I will use the Somali news dataset from MasakhaNEWS. It is a collection of news articles in 16 African languages, and each article has a topic label [5]. The Somali part has 2,915 articles taken from BBC Somali [6]. The sentence I use below is a short headline that I made up in the style of this data, so that the terms are easy to see.

Example headline: *Dowladda ayaa soo saartay sharci cusub.* (The government has issued a new law.)

| Term | Meaning | In this dataset |
|:---|:---|:---|
| Corpus | The full collection of text we study | All 2,915 Somali articles together |
| Document | One unit of text inside the corpus | One news article, with its headline and body |
| Token | A small piece of text that the system treats as one unit, usually a word or part of a word | The headline above has six word tokens: Dowladda, ayaa, soo, saartay, sharci, cusub |
| Vocabulary | The list of all different tokens found in the corpus | Every unique token across all articles. The word "sharci" counts once, even if it appears in 500 articles |
| Feature | A measurable input that the model uses to make its choice | How many times each vocabulary word appears in an article. The count of "sharci" is one feature |
| Label | The correct answer we want the model to predict | The topic of the article, such as politics, health, or sports |

Two points matter here. First, a token is a choice made by the person building the system. You can split on spaces, or you can split words into smaller pieces, and Jurafsky and Martin give a full chapter to this choice [1]. Second, features are made from the text, but labels are added by people. In this dataset the labels come from the news categories, so they cost effort to collect, and that is why labelled data is scarce for many languages.

## 3. Four common NLP tasks

| Task | Input | Output | Small example |
|:---|:---|:---|:---|
| Text classification | A piece of text, such as an article, a comment, or an email | One label (or a few labels) for the whole text | A Somali article goes in, and "politics" comes out |
| Sequence labeling | A sentence, seen as a list of tokens | One label for every token | A sentence goes in, and each word is tagged as a person, a place, an organisation, or other. This is named entity recognition |
| Search and similarity | A query, plus a set of documents to look through | A ranked list of the best matching documents, or a score that says how alike two texts are | You type a question, and the system returns the most useful pages first |
| Text generation | A prompt or a source text | New text, written one token at a time | A question goes in, and an answer or a summary comes out |

Some notes on each. Text classification is the simplest and the most common task, and it covers spam filters, sentiment checks, and news topic sorting [1]. Sequence labeling is different because the output is as long as the input. The MasakhaNER project built such data for ten African languages, and it shows how much work it takes to label every word by hand [10]. Search systems often turn both the query and the documents into numbers and then compare them [4]. Text generation is what large language models do, and the other three tasks can also be written as generation tasks, which is one reason the field has moved toward it [1].

## 4. Supervised and unsupervised learning in NLP

In supervised learning the model learns from examples that already have the right answer attached. Each training item is a pair: the text, and its label. The model guesses, compares the guess with the label, measures the error, and changes itself to do better next time.

In unsupervised learning the data has no labels. The model looks for structure by itself, such as groups of similar texts or words that tend to appear together.

| | Supervised | Unsupervised |
|:---|:---|:---|
| Data | Text with labels | Text only |
| Goal | Predict the label for new text | Find hidden patterns or groups |
| NLP example | Sorting Somali news into topics, like the MasakhaNEWS task | Clustering thousands of unlabelled comments into themes, or topic modelling |

**Why one needs labels and the other does not.** The reason is the way each one measures success. A supervised model needs a target, because the learning step works by checking its guess against that target. Without the right answer there is no error to measure, so there is nothing to correct. An unsupervised method never asks "is this the correct answer?" It asks a different question, such as "which of these texts are close to each other?" That question can be answered from the text alone, so no labels are needed.

There is a middle case worth knowing. Language models are trained on plain text, but the training still has a target. The target is the next word in the text, and that word is already there in the data, so no person has to label it. This is often called self supervised learning [1].

## 5. Why text must become numbers

Machine Learning models are built from arithmetic. They multiply inputs by weights, add the results, and compare numbers to find errors. A word such as "sharci" cannot be multiplied by anything. So before a model can use text, we must change the text into numbers, and we usually need a vector of fixed length so that every document fits the same shape [1].

**One simple method: bag of words.** We list the vocabulary, then for each document we count how many times each word appears. Take two short documents.

Document A: "dog bites man"
Document B: "man bites dog"

The vocabulary is: bites, dog, man.

| | bites | dog | man |
|:---|:---:|:---:|:---:|
| Document A | 1 | 1 | 1 |
| Document B | 1 | 1 | 1 |

The two rows are the same, even though the two sentences mean very different things.

**Main limitation.** Bag of words throws away word order, so it loses much of the meaning [1][4]. It has a second weakness too. It treats every word as unrelated to every other word, so "car" and "automobile" look as different as "car" and "banana". The vectors are also very long and mostly zeros when the vocabulary is large. Later methods such as word embeddings and transformer models like BERT were made to fix these problems by giving similar words similar numbers and by looking at the words around each token [1][13].

## 6. Challenges of NLP for low resource languages such as Somali

Joshi and colleagues showed that language technology is very uneven. Only a small number of the world's more than 7,000 languages get most of the data and attention, and many systems that are called language neutral are really built for a few languages [8]. Somali is spoken by an estimated 22 million people, yet it still counts as low resource [6]. Here are three challenges that I found in the research.

**Challenge 1: Limited datasets.** Modern models need a lot of text, and labelled text is even harder to find. Some numbers show the gap. In the SomBERTa study, the authors gathered about 160 million Somali tokens from six sources to train their language model [6]. The same paper says the Somali share of the large multilingual model XLM R is only about 62 million tokens [6]. The labelled sets are smaller still. The fake news set has 1,897 items after filtering [6]. Also, much of the available text comes from news sites. The dataset card for the AfriBERTa corpus says most of its text comes from the BBC, so models trained on it are likely to lean toward the news style [11]. The effect on a system is clear. A model trained on small or one sided data can fail on text from social media or daily speech, and a small test set makes it hard to know how good the model really is.

**Challenge 2: Rich word forms (morphology).** Somali builds many word forms from one root by adding parts to it. A lab at Jamhuriya University reports a lemmatization corpus with over 5,584 root words and 78,629 derived forms [7]. I only read the project summary of that work, so I treat the numbers as reported by the lab. The effect on an NLP system is that one idea shows up in many surface forms. A model that counts whole words sees each form as a new word, so the counts for each one become tiny, and the vocabulary grows large. The SomBERTa team dealt with this by training a subword tokenizer with a vocabulary of 70,000 pieces, because splitting words into smaller parts helps with words that have many prefixes and endings [6].

**Challenge 3: Dialects and mixed language in online text.** The SomaliNLP project, which works on a dialect identification tool, lists at least three groups to separate: Benaadir, Northern, and Maay [14]. This is a project website and not a peer reviewed paper, so I use it only as a pointer. Online writing adds another layer. Code switching, which means using more than one language in the same conversation, is common in text from social media [12]. For Somali I did not find a study that measures this, so I will not give a number. The likely effect is easy to explain though. A system trained on formal news Somali sees words and spellings in social posts that it never met in training, and its accuracy drops. The SomBERTa paper makes the same point from the other side. It built its test data from Facebook comments because news data alone did not cover how people write online [6].

Several methods exist to help with all three problems, such as transfer learning from other languages and creating extra training data. Hedderich and colleagues survey them [9]. They help, but they do not replace real Somali data collected with the community.

## 7. Case study

**Paper:** Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., and Assowe, H. A. (2025). *Detection of Somali Written Fake News and Toxic Messages on the Social Media Using Transformer Based Language Models.* arXiv preprint 2503.18117 [6]. Some of the authors work at Jamhuriya University in Mogadishu and at Somali National University.

**Problem.** Anyone can post on social media, so false news and harmful comments spread fast. People cannot check every post by hand. The authors want an automatic way to catch fake news and toxic comments written in Somali, a language with few tools of this kind.

**Task.** The paper covers three jobs: fake news detection (fake or real), toxic comment detection (toxic or nontoxic, and also which kind of toxic), and news topic classification. All three are **text classification**, the first task in question 3. A text goes in and a label comes out.

**Data.**

* For training the language model, the authors joined six existing Somali text sources into one corpus of about 160 million tokens. The sources include BBC Somali, Somali books, CC100 Somali, the Somali part of MasakhaNEWS, the AfriBERTa corpus, and CIRAL.
* For fake news, they collected 3,710 Facebook comments from news pages, including BBC News Somali and Jowhar news site, plus five pages known for false posts. After fact checking, 1,897 items were clear enough to label as fake or real.
* For toxicity, they collected public comments on three debates: changes to the Somali constitution (1,785 comments), a celebrity court case (915), and the maritime dispute (314). Toxic comments were put into categories such as abuse, insult, and identity hate.
* Two volunteers labelled each item on their own, and an item was kept only if both agreed.

**Method.** The authors built SomBERTa, a BERT style model made for Somali. They trained a subword tokenizer with 70,000 pieces. Then they pretrained the model by hiding words in the text and asking the model to guess them, which uses plain text and needs no labels. After that they fine tuned the model on each labelled task. They compared it with BERT, mBERT, RoBERTa, DistilBERT, ALBERT, AfriBERTa, and AfroXLMR.

**Findings.** All numbers below are accuracy in percent, taken from the paper's Table 6.

| Model | News topics | Toxicity (two classes) | Toxicity (categories) | Fake news | Average |
|:---|:---:|:---:|:---:|:---:|:---:|
| SomBERTa large | 94.17 | 84.0 | 78.80 | 95.0 | 87.99 |
| AfriBERTa large | 95.24 | 83.33 | 77.05 | 94.47 | 87.52 |
| AfroXLMR base | 94.77 | 81.67 | 75.63 | 92.63 | 86.18 |
| BERT base | 92.57 | 79.0 | 75.19 | 90.53 | 84.32 |

SomBERTa had the best average, 87.99, and the best scores on both toxicity tasks and on fake news. On news topics it came third, behind AfriBERTa and AfroXLMR. The authors conclude that a model trained only on Somali can beat larger multilingual models on some Somali tasks.

**My own reading of the limits.** The gaps between the top models are small, in some cases under one point, and the paper text I read does not state the size of the test sets, so I would not claim that SomBERTa is clearly better everywhere. Only two annotators labelled the data, and I did not find an agreement score in the version I read. All the social media data comes from Facebook, so it may not match other platforms. Even so, the paper is useful because it links back to the challenges in question 6. It answers the data problem by building new labelled data, and it answers the word form problem with a subword vocabulary made for Somali.

## References

[1] Jurafsky, D. and Martin, J. H. (2026). *Speech and Language Processing*, 3rd edition draft, released August 19, 2026. https://web.stanford.edu/~jurafsky/slp3/

[2] Russell, S. and Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*, 4th edition. Pearson.

[3] Mitchell, T. M. (1997). *Machine Learning*. McGraw Hill.

[4] Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

[5] Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). MasakhaNEWS: News Topic Classification for African Languages. arXiv preprint 2304.09972. https://arxiv.org/abs/2304.09972

[6] Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., and Assowe, H. A. (2025). Detection of Somali Written Fake News and Toxic Messages on the Social Media Using Transformer Based Language Models. arXiv preprint 2503.18117. https://arxiv.org/abs/2503.18117

[7] Somali language AI and Innovation Lab, Jamhuriya University of Science and Technology (2026). Project page for "Morphologically informed Somali Lemmatization Corpus built with a Web based Crowdsourcing Platform". Figures are as reported on the project page.

[8] Joshi, P., Santy, S., Budhiraja, A., Bali, K., and Choudhury, M. (2020). The State and Fate of Linguistic Diversity and Inclusion in the NLP World. Proceedings of ACL 2020. https://arxiv.org/abs/2004.09095

[9] Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., and Klakow, D. (2020). A Survey on Recent Approaches for Natural Language Processing in Low Resource Scenarios. arXiv preprint 2010.12309. https://arxiv.org/abs/2010.12309

[10] Adelani, D. I., Abbott, J., Neubig, G., D'souza, D., Kreutzer, J., Lignos, C., et al. (2021). MasakhaNER: Named Entity Recognition for African Languages. Transactions of the Association for Computational Linguistics, 9. https://doi.org/10.1162/tacl_a_00416

[11] Dataset card for the AfriBERTa corpus (Castorini group), Hugging Face. States that most of the text comes from the BBC news website.

[12] Winata, G. I., Aji, A. F., Yong, Z. X., and Solorio, T. (2022). The Decades Progress on Code Switching Research in NLP: A Systematic Survey on Trends and Challenges. arXiv preprint 2212.09660. https://arxiv.org/abs/2212.09660

[13] Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL. https://arxiv.org/abs/1810.04805

[14] SomaliNLP project website. Description of the Somali dialect identification project (Benaadir, Northern, and Maay dialects). Project website, not peer reviewed.
