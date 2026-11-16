1: Question 1 Answer
NLP: a branch of AI that lets computers comprehend human language

AI is the broad discipline that makes machines intelligent
ML is a branch of AI that lets machines learn from data
NLP: a field that lets computers comprehend human language

Real-world example: ChatGPT
- AI is what makes ChatGPT intelligent.
- ML trains it with massive amounts of text data.
- NLP lets it grasp users' questions and produce human-like responses.

2: Question 2 Answer

A corpus is a large, structured collection of written or spoken language data stored digitally, used for linguistic research, dictionary-making, and natural language processing

Example: 5,000 student feedback comments.

A document is a body of text handled as a single unit of information.

- Example: "The teacher explains lessons clearly.

A token in Natural Language Processing (NLP) is a small, basic unit of text, such as a word, subword, or character.
- Example: "Students enjoy learning" = Students, enjoy, learning.

Vocabulary: The set of unique words used in the text corpus is known as the vocabulary

A feature is the input data pulled from text that a model uses to make decisions.

- Example: The number of positive words in each comment.

A label is the output, or correct answer, that the model tries to predict.

Example: Positive, Negative, or Neutral.


3: Question 3 Answer

Text classification is the process of assigning predefined categories to textual data.

input: a sentence, email, or document
output: a category or label
example: classifying an email as spam or not

Sequence Labeling assigns a label to every word or token within a sentence.
input: a sentence that contains multiple words
output: a label for each word
example: identifying names, locations, and dates within a sentence

Search and Similarity: identifies documents or sentences that are relevant to or resemble a query.

Input: a search query together with a collection of documents.
Output: a ranked list of documents that are relevant or similar.
Example: searching for articles on artificial intelligence.

Text Generation: produces new human-readable text from an input or instruction.

Input: a prompt, a question, or some starting text.
Output: newly generated text.
Example: ChatGPT producing an answer to a user's question


4 Question 4 Answer

Supervised learning in Natural Language Processing (NLP) trains models on labeled text data so they can predict known outcomes.
Example: classifying emails as spam or not spam.
This requires target labels, since the model learns from examples that have known correct answers.
Unsupervised learning uncovers concealed patterns within raw, unlabeled text.
Example: Sorting news articles into groups based on similar topics.
Target Labels: No target labels are needed, since the model identifies patterns and relationships by itself.

5 Question  5 Answer

text needs to be converted into numbers because most ML models work with numerical data

Example: Bag of Words turns text into numbers by tallying how often each word appears.

Limitation: It disregards word order and context.


6: Question 6 Answer

Low recourse languages such as Somali face several challenges in nlp

Limited Dataset: there are no enough text data set available for Somali 


spelling variation: Somali speakers sometimes write the same word differently that makes nlp systems hard to recognize words and understand their meaning.




7: Question 7 Answer

Case Study: Sentiment Analysis in Hausa and Igbo

Article: Seals_Lab at SemEval-2023 Task 12 (2023).

Task: The researchers developed an NLP system to identify positive, negative, and neutral opinions in Hausa and Igbo social media posts.

Data: They used human-labeled Twitter posts from the AfriSenti dataset.

Method: They used AfriBERTa, a language model designed for African languages, to classify sentiments.

Findings: The model achieved an F1-score of 80.85% for Hausa and 80.82% for Igbo.

sourse: https://aclanthology.org/2023.semeval-1.208
