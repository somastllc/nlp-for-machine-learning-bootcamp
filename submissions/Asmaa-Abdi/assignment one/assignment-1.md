1. What is Natural Language Processing (NLP)?
Natural language processing (NLP) is a part of artificial intelligence (AI) that helps
computers understand human language. It turns unstructured language into formats that
computers can classify, search, compare, extract information from, or generate. This can
involve several processing layers, including tokenization, parsing, named entity recognition,
embeddings, semantic analysis, and model inference.
Artificial Intelligence is the broader field of computer science focused on creating systems
that can perform tasks that typically require human intelligence.
Machine Learning is a subset of Artificial Intelligence. It enables systems to learn patterns
from data and improve performance without being explicitly programmed for every scenario.
Natural Language Processing is a branch of Artificial Intelligence focused on human
communication and computer systems by allowing machines to process text and speech in
meaningful ways. NLP focuses specifically on language-related tasks.
Real-Life Example:
A Smartphone Voice Assistant
1. NLP processes the raw audio data(converted from speech to text)to understand
human language.
2. ML models analyse historical user data such us location. ml predicts the correct
location based on past pattern.
3. AI executes the speech recognition, location services and reminders app.

   
2..Corpus, Document, Token, Vocabulary, Feature, and Label
ID	            Raw Text Document 	                                  Label
Report 1	“Revenue increased by 10% in January”.	                Positive
Report 2	“February saw a slight decline of 5% in expenditure.”	Negative
Report 3	“march revenue increased steadily due to new sales.”	Positive

corpus is a collection of text that we use for analysis. Think of it as the entire body of text you’re working with.

Example: Imagine you’re analyzing monthly financial reports for your company. If you have fifty reports, all these reports combined make up your corpus.

A document represents a self-contained unit of information within the corpus.

Example:

Report 1: “Revenue increased by 10% in January compared to December. The marketing campaign contributed significantly to this growth.”

Report 2: “February saw a slight decline of 5% in expenditure. The main reason was reduced spending on operational costs.”

Vocabulary refers to the set of unique words present in your corpus. Think of it as a list of all the distinct words in your text without duplicates.

Example: the financial report was published. the annual revenue increased

vocabulary: [the , financial,  report, was, published,  annual, revenue,  increased ]

Tokenization is a fundamental process in NLP where a large piece of text is divided into smaller units, called tokens. 

Example: Revenue increased by 10% in January, breaking it down into indivitual tokens yields: [“revenue” , “increased”, “by”, “10%”, “in” , “january”, “.”]



Feature is a measurable, numerical representation of text that a machine learning model uses to make predictions.

Example: the frequancy count of word “increased” (which is 1) or the percentage figure “10%” acts as a numerical feature indicating positive performance.

Label is the target category or output tag assigned to predict for each document to train the model what the text represents.

Example: if we are classifying financial reports by company performance, (“Revenue increased by 10%....”) would be assigned the label “positive Growth”, while  (“February saw a slight decline of 5% ....”) would be assigned the label “Decline”.



3.Four Common NLP Tasks
Text classification is like sorting different pieces of text into specific groups or categories. Imagine you have a bunch of emails and you want to separate them into "important" and "not important" piles. Text classification helps computers do this automatically by categorizing text into predefined groups.
Input: A text sequence or document (e.g., "This movie was an absolute masterpiece!").

Output: A single categorical label or a probability distribution across available classes (e.g., Positive, with a 98% confidence score).

Text generation involves creating coherent and contextually relevant text based on input prompts. It spans a wide range of applications, from creative writing to automated reporting.

Input: A prompt, a set of instructions, conversation history, or source data (e.g., "Summarize the following article in three bullet points...").

Output: A newly generated sequence of natural language text responding to or fulfilling the prompt.

Sequence Labeling unlike text classification, which assigns one label to an entire document, sequence labeling assigns tag or label to every indivitual token in a sequential order.

Input: A sequence of tokens or words (e.g., "Apple is expanding its headquarters in London.").

Output: A sequence of labels aligned token-by-token with the input (e.g., [Organization, O, O, O, O, O, Location, O]).

Search and Semantic Similarity is The task of measuring how semantically close or relevant two or more pieces of text are, or retrieving the most relevant documents from a database given a user query. Common applications include semantic search engines, duplicate detection, and recommendation systems.

Input: A query text paired with a target text, or a query paired with a large corpus of documents.

Output: A numerical similarity score (typically between 0 and 1, like a cosine similarity metric) or a ranked list of relevant documents matching the query.

NLP Tasks 
Task	             Core Focus	                   Input	                   Output
Text Classification	Categorizing whole texts	A document or sentence	    A category label / class
Sequence Labeling	Tagging individual tokens	A sequence of words	        Token-level label mapping
Search & Similarity	Matching meaning / ranking	A query and text corpus	    Similarity scores or ranked documents
Text Generation	   Creating new language	  A prompt or instructions  	A generated text response



4. Supervised vs. Unsupervised Learning in NLP
Supervised learning is the model is trained with labeled data where each input has a corresponding output. e.g., text paired with a target category or sentiment.
Why it needs target labels: Supervised learning requires labels because it relies on explicit supervision to guide optimization. 
Example: sentiment analysis
Unsupervised learning involves training the model with unlabeled data which helps to uncover patterns, structures or relationships within the data without predefined outputs. e.g., millions of unannotated web pages or articles.
Why it does not need labels: Unsupervised learning focuses on data exploration and representation rather than prediction.
Example: Topic modeling

5.Why Does Text Need to Become Numerical?
In Natural Language Processing (NLP), the ability to transform textual data into numerical representations is a fundamental concept. This process, known as text vectorization, is integral to NLP tasks, enabling machines to comprehend and process human language effectively. Text vectorization converts raw text data into structured numerical formats that machine learning algorithms can understand and process.
Example: Binary Term Frequency (BTF)
Binary Term Frequency, often referred to as binary encoding, is a fundamental text vectorization method. 
 let's consider these 2 sentences 
Sentence 1: “I really enjoy reading. I enjoy reading books.”
Sentence 2: “I prefer watching movies, not reading.”

Below is the Binary Term Frequency vector representation
	         books	enjoy	prefer	reading	movies	really	not	watching
sentence 1	  1	      1	    0	     1	      0	      1	     0	   0
sentence 2	  0	      0	    1	     1	      1       0	     1	   1
the text has now been connverted in numerical vector 

example: sentence 1 = [1,1,0,1,0,1,0,0]

limitation

Information Loss: It completely disregards term frequency, which means we lose valuable information about how often a term appears in a document.

No Semantics Understanding: Binary Term Frequency doesn’t capture the semantic meaning of terms or their relationships.

High Dimensionality: For large vocabularies, vectors can become high-dimensional and sparse, which can be challenging to work with.

Handling OOV Terms: Binary Term Frequency relies on a predefined vocabulary, which means that terms not present in the vocabulary are effectively ignored. This limitation can lead to information loss, especially when dealing with OOV terms.

6.Challenges of NLP for Low-Resource Languages Such as Somali
low-resource languages—languages that lack large digitized datasets, pre-trained models, and standardized linguistic tools—presents major technical hurdles compared to high-resource languages like English.

Spelling Variations

somali text can contain different spelling in the same word .Writers frequently spell words phonetically based on regional pronunciation or inconsistently split and merge compound words(e.g: “jabuuti” or “jibuuti” , “gabar” or “gabadh” and “wexey” or “waxay”)

limitation 

Low-resource languages have far smaller collections of raw text, speech recordings, and labeled datasets than high-resource languages like English. In addition, Somali has a strong oral tradition, meaning much everyday communication is spoken rather than captured in large digital archives.

morphology

Somali is a morphologically rich language. A single root word can take on dozens of different forms through prefixes, suffixes, and internal vowel changes to express tense, gender, number, definiteness, and focus markers. 

Mixed-Language Text 

In online spaces, business communications, and diaspora communities, Somali speakers frequently engage in mixing Somali seamlessly with English, Arabic, or Italian within the same sentence or even within the same word (e.g., attaching Somali grammatical suffixes onto English verbs or nouns).

7.case study:Nlp in somali language

Study Focus: Research on Somali Text Classification and Sentiment Analysis using Machine Learning algorithms.

​Task: Text Classification (categorizing unstructured Somali text documents into predefined classes).

​Data: A curated corpus of Somali text reviews and news articles collected from digital media and web sources.

​Method: Applying text preprocessing (tokenization and stop word removal), extracting numerical features using TF-IDF, and training classifiers such as Support Vector Machines (SVM) and Naive Bayes to categorize the text.

​Findings: The study demonstrated that despite the low-resource nature and complex morphology of Somali, traditional machine learning models combined with TF-IDF can effectively classify text when proper cleaning and tokenization are applied.

​Matched NLP Task from Question 3: This case study directly covers Text Classification.

Conclusion
Natural Language Processing (NLP) bridges the gap between human communication and computational understanding by combining artificial intelligence, machine learning, and linguistics. Throughout this assignment, we explored how unstructured text is structured into tokens, features, and numerical representations so that both supervised and unsupervised models can extract meaningful patterns. Furthermore, examining the challenges and real-world applications of NLP in low-resource languages like Somali highlights the ongoing need for richer linguistic datasets and tailored preprocessing techniques to build inclusive language technologies.



References
Muhidin A.Mohamed(Aston University)- Research and Publications on Somali language precessing, lemmatization and African language technologies(available via google scholar / AlphaXiv

Shafie Abdi Mohamed and Muhidin Abdullahi Mohamed. 2023. Lexicon and rule-based word lemmatization approach for the somali language. arXiv preprint arXiv:2308.01785

Gedi, A. A., Mohamed, Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assoweh, H. A. (2026). Morphologically-informed Somali Lemmatization Corpus built with a Web-based Crowdsourcing Platform. Jamhuriya University of Science & Technology, Mogadishu, Somalia & Aston University, UK.



