# Assignment 1 – NLP Foundations

## Introduction
In this assignment I look at the basics of Natural Language Processing (NLP). I explain what NLP is, the main words we use when we talk about text data, common NLP tasks, and how models learn from text. I also explain why computers need text as numbers, what makes NLP hard for Somali, and I summarize one Somali research paper. I tried to explain everything in my own words and I list my sources at the end.

## Question 1. Natural Language Processing and Its Relationship to AI and ML

### What is NLP?
Natural Language Processing (NLP) is a part of computer science that helps computers read, understand, and write human language. This can be text or speech. For example, NLP is what lets a computer translate Somali into English, or understand what you typed into a search box (Jurafsky & Martin, 2025).

### How NLP, AI, and ML Are Connected

NLP, AI, and ML are connected like circles inside each other:

Artificial Intelligence (AI) is the biggest circle. It means making computers do things that normally need human intelligence, like thinking, seeing, or using language (Russell & Norvig, 2021).
Machine Learning (ML) is one way to build AI. Instead of writing all the rules by hand, we give the computer many examples and it learns the patterns by itself (Russell & Norvig, 2021).
NLP is the part of AI that works with language. Old NLP systems used hand-written rules, but most NLP today uses ML to learn from large amounts of text (Jurafsky & Martin, 2025).
Field	Main question	Example
Artificial Intelligence	Can a computer act smart?	A chess program, a self-driving car, a chatbot
Machine Learning	Can a computer learn from data instead of rules?	Predicting house prices from past sales
NLP	Can a computer understand and use human language?	Translating a Somali news article into English.

### Real-life Example: A Spam SMS Filter
Imagine a phone company in Mogadishu that wants to stop scam text messages.

AI is the goal: the system should decide by itself if a message is safe or a scam, like a human would.
NLP prepares the text: it cleans the message, splits it into words, and turns the words into numbers.
ML does the learning: the model is trained on thousands of old messages that staff already marked as "spam" or "not spam". It learns that words like lacag (money) and abaalmarin (prize) often show up in scams.
Result: when a new message comes in, the model checks it in less than a second and blocks it if it looks like spam.

So NLP handles the language, ML does the learning, and together they give us an AI system that filters messages without a person reading each one. (This is my own example to explain the idea, not a real study.)

## Question 2. Corpus, Document, Token, Vocabulary, Feature, and Label

### Example Dataset
To explain these words, I made a small example dataset of Somali news headlines. It is based on MasakhaNEWS, which is a real dataset of news articles in 16 African languages, including Somali. Each article in it has a topic like politics, sports, or health (Adelani et al., 2023).

ID	Text (document)	Label
1	Kooxda Barcelona ayaa ku guuleysatay ciyaarta	sports
2	Dowladda ayaa ansixisay miisaaniyadda cusub	politics
3	Kooxda waxay ciyaarta ku guuleysatay gool	sports

### The Terms Explained
Term	What it means	Example from my dataset
Corpus	The full collection of text we use.	All three headlines together are the corpus.
Document	One single piece of text in the corpus. It can be a tweet, a sentence, an email, or a full article.	Each headline is one document.
Token	One small piece of a document after we split it, usually a word. Splitting text like this is called tokenization.	Headline 1 has 6 tokens: Kooxda, Barcelona, ayaa, ku, guuleysatay, ciyaarta.

Vocabulary	The list of all the different words in the corpus, with no repeats.	kooxda, barcelona, ayaa, ku, guuleysatay, ciyaarta, dowladda, ansixisay, miisaaniyadda, cusub, waxay, gool = 12 words.
Feature	Something we can measure in a document and give to the model as a number.	How many times each word appears. In headline 1, ciyaarta = 1 and dowladda = 0.
Label	The correct answer we want the model to predict.	sports or politics. The model learns that kooxda (team) and ciyaarta (match) usually mean sports.

To sum up: a corpus is made of documents, documents are split into tokens, the different tokens make the vocabulary, we use the vocabulary to make number features, and each document gets a label (Jurafsky & Martin, 2025).

## Question 3. Four Common NLP Tasks

### NLP tasks are different mainly in what goes in and what comes out (Jurafsky & Martin, 2025).

Task	What it does	Input	Output	Example
Text classification	Puts a whole text into a category.	A text (sentence, review, article)	One label	A Facebook comment → toxic or not toxic
Sequence labeling	Gives a label to every word in the text.	A list of words	A list of labels, one for each word	"Xasan wuxuu tagay Hargeysa" → [PERSON, O, O, LOCATION]
Search and similarity	Finds texts that are close in meaning and ranks them.	A search query and many documents	A ranked list of results, or a similarity score	Searching "qiimaha bariiska" (price of rice) shows the most useful reports first
Text generation	Writes new text.	A question, prompt, or a text to translate or summarize	New text	Translating Somali to English, or a chatbot answering a question

The main difference between the first two is size: text classification gives one label for the whole text, but sequence labeling gives one label for each word. Search compares texts with each other (Manning, Raghavan & Schütze, 2008). Text generation is the only one where the output is new sentences instead of a label or a score.

## Question 4. Supervised vs Unsupervised Learning in NLP

### Supervised Learning
Supervised learning means the model learns from examples that already have the right answer (a label). It learns to connect the text to the label, and we test it by checking how many new examples it gets right (Jurafsky & Martin, 2025).

Example: sentiment analysis of Somali product reviews. A person marks each review as positive or negative. The model learns that phrases like aad u fiican (very good) usually appear in positive reviews.

### Unsupervised Learning
Unsupervised learning means the model only gets raw text, with no answers. It has to find patterns by itself, for example groups of documents that are similar (Jurafsky & Martin, 2025).

Example: topic discovery (clustering or topic modelling). We give the model 10,000 Somali news articles with no labels. It puts articles with words like doorashada (the election) and baarlamaanka (parliament) in one group, and articles with roob (rain) and abaar (drought) in another group. After that, a person looks at each group and gives it a name.

Comparison
	Supervised	Unsupervised
Training data	Text and labels	Text only
Goal	Predict a known answer	Find hidden groups or patterns
NLP examples	Classification, sentiment, translation	Clustering, topic modelling
How we check it	Compare with the right answers (accuracy)	Harder, often a person has to judge
Cost	Labels take time and money	Cheaper, raw text is enough
Why Does One Need Labels and the Other Doesn't?

In sentiment analysis, we already know the answers we want: "positive" or "negative". The model can't guess what "positive" means unless we show it examples, and without the right answers it can't tell when it makes a mistake. In topic discovery, there is no fixed answer. We want the model to show us what groups already exist in the data, so the similarity between the texts is enough. Simply put: we need labels when we already know the categories, and we don't need them when we want the data to show us the categories.

Many modern models use both. They first learn from lots of unlabeled text, and then they get trained a bit more on a small labeled dataset. The paper in Question 7 does exactly this.

## Question 5. Why Text Must Be Converted into Numbers
Why Numbers?

Machine learning models work with math. They add, multiply, and compare numbers. A word like ciyaarta is just letters, and the model can't do math with letters. So we have to turn each text into a list of numbers (called a vector). Then texts that are similar will have similar numbers, and the model can find patterns (Jurafsky & Martin, 2025).

### A Simple Method: Bag-of-Words (BoW)

First we make the vocabulary. Then for each document, we count how many times each vocabulary word appears (Manning, Raghavan & Schütze, 2008). Here is BoW for my three headlines from Question 2:

Document	kooxda	barcelona	ayaa	ku	guuleysatay	ciyaarta	dowladda	ansixisay	miisaaniyadda	cusub	waxay	gool
Headline 1 (sports)	1	1	1	1	1	1	0	0	0	0	0	0
Headline 2 (politics)	0	0	1	0	0	0	1	1	1	1	0	0
Headline 3 (sports)	1	0	0	1	1	1	0	0	0	0	1	1

You can see that headlines 1 and 3 (both sports) share four words, so their numbers look alike. Headline 2 (politics) only shares ayaa. Now a model can learn that kooxda and ciyaarta point to sports.

### Main Limitation of Bag-of-Words

BoW ignores word order and meaning. It just counts words, so "the dog bit the man" and "the man bit the dog" look exactly the same, even though they mean very different things. It also doesn't know that some words are related. For example, kooxda and kooxdii (both mean "the team"), or guuleysatay and guulaysatay (two spellings of "won"), are treated as completely different words. Also, when the vocabulary gets very big, the vectors become very long and mostly zeros. Because of these problems, people later made better methods like word2vec (Mikolov et al., 2013) and BERT (Devlin et al., 2019), which understand that similar words have similar meanings.

## Question 6. Challenges of NLP for Low-Resource Languages such as Somali

Somali is spoken by about 25 million people in the Horn of Africa and is written in the Latin alphabet (SomaliWeb, 2026). But for NLP it is a low-resource language. This means there is not much labeled data, there are few tools, and only a small amount of Somali text is used to train big AI models (Joshi et al., 2020). Below are the main problems and how they affect NLP systems.

### 6.1 Limited Datasets and Tools

NLP models learn from data, so the more data, the better they usually work. Somali has very few labeled datasets and very few models made for it. Mohamed et al. (2025) say this makes things like translation, sentiment analysis, and text classification hard for Somali. Even basic tools, like part-of-speech taggers, are hard to find.

How it affects NLP: if a model only sees a few thousand examples, it sees each word too few times to learn it well. It may memorize the training data and then do badly on new text. Big multilingual models like mBERT saw very little Somali, so they break Somali words into many small pieces. This makes them slower and less accurate for Somali than for English (SomaliWeb, 2026).

### 6.2 Spelling Variation

Somali only got an official way of writing with Latin letters in 1972 (Saeed, 1999). Today, especially on social media, people spell the same word in different ways. Long vowels are written with double letters (aa, ee, oo), but people often skip or add letters. For example: wanaagsan / wanagsan (good), guuleysatay / guulaysatay (won), and Muqdisho / Mogadishu / Xamar for the capital city.

How it affects NLP: the computer sees each spelling as a different word. So the vocabulary gets bigger, and each spelling appears fewer times. If the model learned a word with one spelling, it may not recognize it with another spelling. Search also suffers: if you search for wanaagsan, you may not find texts that wrote wanagsan. One fix is normalization, which means changing all the spellings of a word into one standard form before training.

### 6.3 Many Word Forms

In Somali, we add endings to words to show things like "the", "my", or past and present (Saeed, 1999). For example, the word buug (book) can become buugga (the book), buuggii (the book we talked about), buugaag (books), and buuggayga (my book).

How it affects NLP: every form counts as a separate word, so one idea is spread over many words that each appear only a few times. A simple model like Bag-of-Words can't see that buugga and buuggii are about the same thing. Possible fixes are lemmatization (turning each word back into its base form) and sub-word tokenization (splitting words into smaller parts), which the model in Question 7 uses.

### 6.4 Mixed-Language Text

Many Somali speakers, especially online and outside Somalia, mix Somali with English or Arabic in the same sentence. For example: "Meeting-ka waa la cancel gareeyay" (the meeting was cancelled). Here, English words get Somali endings.

How it affects NLP: a model trained only on clean Somali text will see English words with Somali endings and get confused. It may give the wrong label, or not even know which language the sentence is in. There is also very little mixed-language data available for training.

Problem	What happens	Effect on the NLP system	Possible fix
Limited data	Few labeled examples	Low accuracy, model memorizes	Build and share datasets, use pre-trained models
Spelling variation	One word, many spellings	Too many words, unknown words, bad search	Normalization, sub-word models
Many word forms	One root, many endings	Related words look unrelated	Lemmatization, sub-word tokenization
Mixed language	Somali + English/Arabic together	Wrong language, wrong labels	Multilingual models, mixed-language data

## Question 7. Case Study: Detecting Somali Fake News and Toxic Messages

### The Paper
Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models. The authors are from Aston University (UK), Jamhuriya University, Somali National University, and the University of Djibouti.

### The Problem (Task)
Many Somalis get their news from social media, but social media also spreads fake news and abusive comments. There are too many posts for people to check by hand, and most AI tools don't work well for Somali. So the researchers built a system to check:

Is a post fake or real news?
Is a comment toxic or not toxic? If it is toxic, what type is it (insult, abuse, threat, hate, and so on)?

### The Data
Dataset	Where it came from	Size	Labels
Fake news	Facebook pages like BBC Somali and Jowhar	3,710 collected, 1,897 kept after checking	fake / real
Toxic comments	Facebook comments on three topics (constitution changes, a celebrity court case, the sea border dispute)	3,014 comments	toxic / not toxic, plus toxic type
Training text (no labels)	BBC Somali, MasakhaNEWS, books, and other Somali web text	about 160 million tokens	none

Two people labeled each post or comment. They only kept the ones where both people agreed, so the labels are more reliable.

### The Method
The researchers built their own Somali language model called SomBERTa. It is based on BERT. First, they trained it on the 160 million tokens of plain Somali text. In this step, the model learns Somali by guessing hidden words in sentences. Then they trained it again on the labeled datasets for each task. SomBERTa splits words into smaller pieces, which helps with Somali word endings (see Question 6.3). They compared it with other models that know many languages, like mBERT, AfriBERTa, and AfroXLMR.

### The Results (Accuracy %)
Model	News topic	Toxic or not	Toxic type	Fake news	Average
SomBERTa	94.17	84.00	78.80	95.00	87.99
AfriBERTa-large	95.24	83.33	77.05	94.47	87.52
AfroXLMR-base	94.77	81.67	75.63	92.63	86.18
mBERT-base	93.01	79.50	76.67	91.84	85.26

SomBERTa got the best average score. It was the best at fake news and both toxic tasks. AfriBERTa was a little better at guessing news topics. This shows that a model made only for Somali can do as well as, or better than, big models made for many languages. Guessing the type of toxic comment was the hardest task (78.8%). This is probably because some types had very few examples. For example, there were only 74 obscene comments. The paper also gives new labeled Somali datasets that other researchers can use.

### Which NLP Task Is This?

This is text classification. The input is a whole post or comment, and the output is one label (fake or real, toxic or not toxic, or the toxic type). It also connects to Question 4, because the model first learns from unlabeled text and then from labeled text. And it connects to Question 6, because the whole reason for the project is that Somali has so little data.

## Conclusion

NLP is the part of AI that deals with language, and today it mostly works using machine learning. Text is collected into a corpus, split into tokens, and turned into numbers, because models can only work with numbers. NLP tasks are different in their output: one label for a text, one label per word, a ranked list, or new text. Supervised learning needs labels because it has to learn the right answers, while unsupervised learning finds patterns by itself. For Somali, there are big challenges like little data, different spellings, many word forms, and mixed languages. But the SomBERTa paper shows that building Somali datasets and Somali models can really improve results.

## References
Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. Proceedings of IJCNLP-AACL 2023. https://arxiv.org/abs/2304.09972
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019. https://arxiv.org/abs/1810.04805
Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. Proceedings of ACL 2020. https://arxiv.org/abs/2004.09095
Jurafsky, D., & Martin, J. H. (2025). Speech and Language Processing (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. https://nlp.stanford.edu/IR-book/
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. https://arxiv.org/abs/1301.3781
Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. https://arxiv.org/abs/2503.18117
Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.
Saeed, J. (1999). Somali. London Oriental and African Language Library. John Benjamins.
SomaliWeb (2026). SomaliWeb v1: A quality-filtered Somali web corpus with a matched tokenizer and a public language-identification benchmark. https://arxiv.org/pdf/2605.18232