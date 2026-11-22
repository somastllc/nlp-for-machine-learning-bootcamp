# Assignment 1 – NLP Foundations 
## 1. What is NLP and how does it connect to AI and ML?
Natural Language Processing (NLP) is the part of computer science that helps computers read, understand, and write human languages like English or Somali.

**Artificial Intelligence** (AI) is the big umbrella for any smart computer system.

**Machine Learning (ML)** is a smaller piece inside AI where computers learn from past data instead of us writing strict rules for them.

How they work together (Example - Mobile Money Chatbot):

NLP reads the customer’s text message and figures out what words mean.

ML uses past chat data to guess what the customer wants (like checking a delayed transaction).

AI runs the whole system to reply back or send the issue to a human agent.

## 2. Six Key NLP Terms (Explained with a Somali News Example)
Imagine a dataset of 5,000 Somali news headlines from websites:

**Corpus: The whole collection of all 5,000 news headlines grouped together.

Document: Just one single headline from that collection (e.g., "Xulka qaranka Soomaaliya ayaa guul gaaray").

Token: Breaking that single sentence down into individual words or pieces: ["Xulka", "qaranka", "Soomaaliya", "ayaa", "guul", "gaaray"].

Vocabulary: The list of every unique word used across all 5,000 headlines (no repeated words).

Feature: A number given to a word (like its importance or score) so the computer can do math with it.

Label: The correct category tag given to the headline, like Sports or Politics.

3. Four Common NLP Tasks
Text Classification:

Input: A customer review or message.

Output: A category tag (e.g., Positive or Negative).

Sequence Labeling:

Input: A sentence of words.

Output: A matching tag for every single word (e.g., telling which word is a Person's name and which is a City).

Search and Similarity:

Input: A search question and a list of articles.

Output: A score showing how closely the article matches the meaning of your question.

Text Generation:

Input: A prompt or text in another language.

Output: Newly written, smooth sentences.

4. Supervised vs. Unsupervised Learning
Supervised Learning: The computer is trained using text plus correct answer labels provided by humans (Example: Email spam filter, where emails are pre-labeled as Spam or Not Spam). It needs labels so it knows if its guess is right or wrong.

Unsupervised Learning: The computer looks at unlabeled raw text and groups similar things together by finding patterns on its own (Example: Grouping thousands of news articles into topics automatically without human labels). It doesn't need labels because it's just exploring data patterns.

5. Why Convert Text to Numbers? (And Bag-of-Words)
Why numbers? Computers and machine learning models only understand math (numbers, matrices, and equations). They cannot multiply or calculate raw alphabet letters like "buug".

Simple Method (Bag-of-Words): It counts how many times each word from our vocabulary appears in a sentence.

Main Limitation: It ignores word order and sentence meaning. For example, "The dog bit the man" and "The man bit the dog" would look like the exact same numerical data to a basic Bag-of-Words model, even though their meanings are completely opposite!

6. Challenges of NLP for Low-Resource Languages (Like Somali)
Challenge 1: Lack of Data and Datasets: Somali doesn't have huge digitized training texts or pre-made labeled datasets compared to English. This causes models to memorize training data instead of learning general rules, leading to mistakes on new words.

Challenge 2: Complex Word Forms and Spelling Variations: Somali words change a lot by adding prefixes and suffixes (e.g., buug, buugga, buugag), which makes the vocabulary huge. Also, people spell informal words differently on social media (like writing "maxaa" as "mxaa"), which confuses computer models.

7. Case Study (Somali Fake News & Toxicity Detection)
Paper/Article: Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models (Mohamed et al., 2025).

Task: Text Classification (detecting fake news and toxic hate speech on social media).

Data: A collection of 160 million Somali tokens for pre-training, plus human-annotated social media datasets for fake news and toxicity.

Method: Built and trained SomBERTa, a custom monolingual RoBERTa AI model specifically designed for the Somali language.

Findings: SomBERTa successfully outperformed other multilingual models, achieving about 87.99% accuracy in detecting fake news and toxic messages in Somali.

References
Eisenstein, J. (2019). Introduction to Natural Language Processing. MIT Press.

Gedi, A. A., et al. (2026). Morphologically-informed Somali Lemmatization Corpus. AfricaNLP 2026.

Goldberg, Y. (2017). Neural Network Methods for NLP.

Hedderich, M. A., et al. (2021). A Survey on Recent Approaches for NLP in Low-Resource Scenarios. NAACL-HLT.

Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing (3rd ed.).

Manning, C. D., et al. (2008). Introduction to Information Retrieval. Cambridge.

Mohamed, M. A., et al. (2025). Detection of Somali-written Fake News and Toxic Messages Using Transformer-based Models. AfricaNLP 2025 / arXiv:2503.18117.

Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.