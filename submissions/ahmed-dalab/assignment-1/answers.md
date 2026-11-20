# Assignment 1 for NLP bootcamp

# Question 1:
- NLP is a field that allows computers to     understand, analyze, generate or work with human language.
- The relationship of the artifical intelligence and natural language processing is that artificial intelligence is the broader goal of making machines to perform intelligent tasks, while natural language processing is one of the area of artifical intelligence focused on language.
- It is also related to machine learning because machine learning provides many of the algorithms and techniques that NLP uses to process and understand language.
- Real-World example: **Gmail detecting spam emails**
- imagine a person sends you this: 
   "Congratulations! You have won iPhone 17 pro max, click this link to get your prize."
   - The artifical intelligence is the broader system designed to make intelligent decision, i.e, determine whether the email is spam or legitimate.
   - The natural language processing examines the email's language and identifies suspicious expressions.
   - The machine learning uses patterns learned from thousands or millions of previously labeled emails to estimate whether this email is spam.
  
  # Question 2:
  - **Corpus**: is a collection of text and written materials gathered for language analysis or train and evaluate machine learning models.
  - **Document**: is one individual portion of text inside the corpus.
  - **Token**: is a small unit of text produced when text is broken into smaller pieces.
  - **Vocabulary**: is a set of all unique tokens found inside the corpus.
  - **Feature**: is a measurable part of information that helps machine learning model to identify patterns and make predictions.
  - **Label**: is the correct category or answer assigned to a piece of data.
  

 # Questions 3:

 - **Text classification**: This is the process of teaching computer to read a piece of text and determine which category it belongs to.  
  *input*: A somali news article
  *output*: A category such Sports, Politics or Health
 - **Sequence labeling**: This means assigning a label to each word or token in a sentence.
   *input*: A sentence or paragraph
   *output*: A label for each word or token
 - **Search and similarity**: this defines finding text that is related to what someone is looking for.
  *input*: A search query and a collection of documents
  *output*: Relevent or similar documents
 - **Text generation**: allowing computer to produce new sentences, paragraphs or complete documents.
  *input*: A prompt, question or existing text
  *output*: New text, an answer or a summary


 # Question 4: 
  - 1. Supervised Learning
  - Is a machine learning approach where a computer learns from data that is already contains the correct answers known as labels.
  - Example: Spam detection
  - email                             Lable
  - "You won $10,000"                   spam
  - "Claim you free prize now"          spam
  - "University results are ready"      not spam
- When we use this the computer learns from the label in million times, and later when asked to tell if email is spam or not it will use the trained data to get the result.
 
 - 2. Unsupervised Learning
 - Is a machine learning approach where the computer receives data without predefined labels and tries to discover hidden patterns, relationships or structures within it.
 - Example: Grouping somali news articles
 - Discovered Group       Common content
 - Group 1                 football, players, tournaments
 - Group 2                 Government, elections, parliament
 - Group 3                 Business, trade, economy
- the model discovers these groups based on the similarities in the text.


  # Question 5
  - Machine Learning models generally work with numbers rather than raw text because computers perform mathematical operations to identify patterns, calculate relationships, and make predictions. Therefore, text must first be converted into numerical values that a machine learning algorithm can understand.
  
  - One simple representation method is Bag-of-Words (BoW).
  - **How it works**
     Bag-of-Words converts text into numbers by counting how many times each word appears in a document. It creates a vocabulary of unique words and represents each document using the frequency of those words.
     Example: 
    Suppose we have two sentences:
    - Sentence 1: "I like Somali news"
    - Sentence 2: "I like Somali football"
  - Step 1: Create a vocabulary
    Collect all unique words:

       ["I", "like", "Somali", "news", "football"]
  - Step 2: Convert sentences into word-count vectors
   Word	    I	  like	Somali	news	football
Sentence1	1	  1	     1	     1	          0
Sentence2	1	  1	     1 	     0	          1

  The conversion: "I like Somali news"  -> [1, 1, 1, 1, 0]

  **Main limitation**
    Bag-of-Words does not understand the meaning or order of words; it only counts their frequency. For example, "Ahmed likes Mohamed" and "Mohamed likes Ahmed" produce the same vector even though their meanings are different.

    It can also create very large vectors when the vocabulary contains thousands or millions of unique words. This is called the high-dimensionality problem.

   # Question 6
**Challenges for Somali NLP**: Somali is considered a low-resource language because there is a limited amount of digitally available text, annotated datasets, language-processing tools, and research resources compared with languages such as English, French, and Arabic. This makes it more difficult to develop accurate NLP applications for Somali.

**Challenge 1: Limited datasets**
 Somali has relatively few large, high-quality text datasets that have been collected, cleaned, and labeled for machine learning.

For example, imagine you want to train a model to classify Somali news into Sports, Politics, Business, and Health. You might find thousands of English articles already categorized, but finding an equally large and reliable collection of labeled Somali articles is much harder.
*How this affects NLP systems*:
- Models have fewer examples from which to learn.
- Models may struggle to recognize unfamiliar words and expressions.- Models may perform poorly on new data that differs from their training data.
- Training modern language models from scratch becomes more difficult and expensive.
**Challenge 2: Spelling and dialect variation**
Somali text can have differences in spelling, word formation, dialects, and formatting. The same word or meaning may appear in different forms depending on the writer, dialect, or online platform.For example, Somali has different varieties, including MAXAA TIRI and MAAY, which can differ in vocabulary, pronunciation, and grammatical structures.

*How this affects NLP systems*:
- The same meaning may be represented by different words or spellings.
- A model trained on one dialect may struggle to understand another.
- Different spellings increase the number of unique tokens in the vocabulary.
- Models may incorrectly treat similar words as completely unrelated.

**Challenge 3: Mixed-language text (Code-switching)**
Code-switching happens when people combine two or more languages within the same conversation, sentence, or piece of writing.

For example, Somali speakers may combine Somali and English:
- "Maanta meeting ayaan leenahay." (We have a meeting today.)
- "Assignment-ka waan dhammeeyay." (I finished the assignment.)
- "Internet-ka maanta wuu gaabis yahay." (The internet is slow today.)
  
*How this affects NLP systems*:
- A model trained only on standard Somali may not recognize English words mixed into Somali sentences.
- The model may struggle to identify the correct language of individual words.
- Borrowed words and informal expressions can make tokenization and language identification more difficult.
- The model may misunderstand the meaning of a sentence when languages are mixed.

 # Question 7

 ## Case Study: Somali Dialect Identification

 **Authors**: Abdifatah Ahmed Gedi, Yusuf Mohamed Ahmed, Shafie Abdi Mohamed, Yusuf Ahmed Yusuf, and Abdénuur Umur Ebdiyow.
 **Year**: 2026.
 **Source**: Discover Data, Volume 4, Article 10.

 **Task**
 The study focuses on automatic Somali dialect identification. Its objective is to develop a system that can examine a Somali text and determine whether it belongs to the MAXAA TIRI or MAAY dialect.

This is important because the two dialects have differences in vocabulary, grammar, and spelling. Identifying the dialect can help other NLP applications process Somali text more accurately.

**Data**
The researchers created a manually annotated dataset containing 8,947 Somali text samples collected from different sources.

The dataset contained:

***Dialect**	    **Number of samples**	**Percentage**
MAXAA TIRI	          5,505	                   61.5%
MAAY	              3,442	                   38.5%
Total	              8,947	                   100%

The researchers collected text from:
- Social media posts and comments, particularly Facebook.
- Somali news websites, including BBC and VOA.
- Blogs and online publications.
- Books and formal documents, including the Southwest State Constitution.
  

Three Somali linguists participated in manually labeling the text according to its dialect. They achieved a Cohen's Kappa agreement score of 0.92, indicating strong agreement between annotators.

**Method**
The researchers used both traditional Machine Learning and Deep Learning approaches to compare their performance.

1. Traditional Machine Learning
   
They tested four algorithms:
   - Naive Bayes
   - Support Vector Machine (SVM)
   - Random Forest
   - Logistic Regression
They used TF-IDF to convert Somali text into numerical features before training the models.

2. Deep Learning
   
They also tested six neural network architectures:
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Multilayer Perceptron (MLP)
- CNN-LSTM
- 
The researchers used Python, Scikit-learn, and TensorFlow/Keras. They divided the dataset into 80% training data and 20% testing data.

**Findings**

The study demonstrated that both traditional Machine Learning and Deep Learning can identify Somali dialects with strong performance under controlled experimental conditions.

Some reported results include:

Model	                        Reported accuracy
Naive Bayes	                        99.05%
SVM	                                98.55% in one results section
Random Forest	                    94.75% in one results section
BiLSTM	                            98%
CNN	                                98%
RNN	                                90.84%

The researchers found that Naive Bayes performed particularly well, suggesting that word patterns and vocabulary differences are useful for distinguishing Somali dialects.

However, there are some limitations:
- Some MAAY samples were translated rather than collected naturally.
- Short sentences and shared vocabulary between dialects caused classification errors.
- Mixed-language text and inconsistent spelling made identification more difficult.
- The study did not evaluate transformer models such as BERT or SomBERTa.
- Its high reported accuracy should not be treated as proof of equivalent real-world performance.

**Research note**: The paper contains inconsistent SVM accuracy figures between its results sections (98.55% and 99.57%). It is better to acknowledge this rather than present one figure as definitive

**NLP task covered**

This case study mainly covers Text Classification because the system receives a piece of Somali text as input and predicts which dialect category it belongs to.


**References**

1. Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., & Ebdiyow, A. U. (2026). Somali dialect identification in low-resource settings using machine learning and deep learning. Discover Data, 4, Article 10. https://doi.org/10.1007/s44248-026-00109-y

2. Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word lemmatization approach for the Somali language. arXiv. https://arxiv.org/abs/2308.01785

3. Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). Detection of Somali-written fake news and toxic messages on social media using transformer-based language models. arXiv. https://arxiv.org/abs/2503.18117