A1.1: NLP -Natural Language Processing, a field focused on enabling computers to process and work with human language

A1.2: NLP, AI, and ML are closely related fields, but they have different purposes
  Ai : a broad field focused on developing computer systems capable of performing tasks that normally require human intelligence.
  ML : a branch of AI that enables computers to learn patterns from data and improve their performance without requiring every rule to be explicitly programmed.
  NLP : a field focused on enabling computers to process and work with human language. It uses techniques from AI, ML,          linguistics, and other areas.

  Machine Learning is one of the main approaches used in modern NLP systems. Instead of manually programming every language rule, developers can train models using large collections of text data. However, NLP does not always require Machine Learning. Some NLP applications use predefined linguistic rules, while others use statistical methods or machine learning models

A1.3: Scenario: AI-powered Customer Support Chatbot for a Real Estate Company
      A customer sends a message: "I'm looking for a two-bedroom apartment in Mogadishu with a monthly rent of $500. Do you have any available?."
      NLP: nderstands the customer's message and extracts the requirements: two bedrooms, Mogadishu, and a $500 budget.
      ML: Analyzes available property data and recommends apartments that best match the customer's requirements.
      AI: Coordinates the process, retrieves matching properties, and generates a personalized response.
      Output: The customer receives a list of available apartments matching their requirements, including prices, locations, and property details.
    
A2: Scenario: Real Estate Customer Reviews Dataset
     Imagine a real estate company collects customer reviews about its property management services.
     A2.1.1 : Corpus 
        Defination : A collection of text documents used for NLP analysis.
        Example : All 1,000 customer reviews collected by a real estate company.
     A2.1.2 : Document 
        Defination : A single piece of text within a corpus.
        Example : "The apartment is spacious and well-maintained".
     A2.1.3 : Token 
        Defination : A small unit of text, usually a word or punctuation mark.
        Example : "The apartment is good" → The, apartment, is, good.
     A2.1.4 : Vocabulary 
        Defination :All unique tokens found in a corpus.
        Example : Apartment, rent, property, good, location, management.
     A2.1.5 : Label 
        Defination: The target category assigned to a document that a model learns to predict.
        Example : Positive, Negative, or Neutral review.
    Action :
       Customer review:
        "The apartment is beautiful, spacious, and well-maintained."
         Corpus: All customer reviews.
         Document: This particular review.
         Tokens: The, apartment, is, beautiful, spacious, and, well-maintained.
         Vocabulary: All unique words across the reviews.
         Features: Word frequency, such as beautiful = 1 and spacious = 1.
         Label: Positive
    
A3: Using a Real Estate Company as an example:
     A3.1: Text Classification
              Description : Assigns text to predefined categories
              Input : Customer review: "The apartment is excellent.
              Output : Positive sentiment.
     A3.2: Sequence Labeling
              Description : Assigns a label to each word or token in a sentence.
              Input : "I need an apartment in Mogadishu."
              Output : Apartment = Property type; Mogadishu = Location.
     A3.3: Search and Similarity
              Description : Finds relevant documents or compares how similar two texts are.
              Input : Search query: "Two bedroom apartment in Mogadishu."
              Output : A list of similar or matching property listings.
     A3.4: Sequence Labeling
              Description : Generates new human-like text based on an input or instruction.
              Input :Property details: 2 bedrooms, Mogadishu, $500/month.
              Output : Discover this beautiful two bedroom apartment in Mogadishu for just $500 per month!
    Summary :
       Text Classification: What category does this text belong to?
       Sequence Labeling: What does each word represent?
       Search and Similarity: Which text or document matches my query?
       Text Generation: What relevant text can I create?

A4: Supervised : Learns from labeled text data.
        Example 
             Input: "The apartment is beautiful."
             Target label: Positive.
    Unsupervised : Learns patterns from unlabeled text data.
        Example : a real estate company provides 1,000 customer reviews. The model groups similar reviews into topics such as rent, location, and property maintenance without predefined categories.

A5: Most Machine Learning models cannot directly understand human language because they work with 
     mathematical operations and numerical data. Therefore, text must be converted into numbers so that ML models can process it, identify patterns, and make predictions.

    Simple Representation Method: Bag of Words (BoW)
    Bag of Words is a simple NLP technique that converts text into numerical vectors by counting how many times each word appears in a document.

    Real Estate Example:
    Suppose we have two customer reviews:
      Review 1: "The apartment is beautiful."
      Review 2: "The apartment is expensive."

    Our vocabulary is:
      apartment, beautiful, expensive

    The numerical representation becomes:
         Document       Apartment    Beautiful    Expensive
         Review 1           1              1              0
         Reviwe 2           1              0              1
    Now, the ML model can use these numbers as input for training and prediction.
    Main Limitation of Bag of Words
    It ignores word order and grammar.
    It does not understand the meaning or context of words.
    It can produce large, sparse vectors when the vocabulary is large.
    Example: "The apartment is good" and "The apartment is not good" may have very similar numerical representations, even though their meanings are different.

A6: Challenges of NLP for Low-Resource Languages (Somali)

    1. Limited Datasets
    Somali has fewer publicly available digital text datasets and labeled resources compared with widely studied languages like English.

    How it affects NLP: ML models have less training data to learn from, which can reduce accuracy and make it difficult to understand Somali language patterns.

    Example: A Somali sentiment analysis model may struggle to correctly classify customer reviews because it has limited examples of positive and negative Somali expressions.

    2. Spelling Variation
    Somali users may write the same word differently, especially in informal online communication.

    How it affects NLP: The system may treat different spellings of the same word as separate words, making text classification and search less accurate.

    Example: A customer may write a word using different spellings or informal abbreviations, causing the NLP system to misunderstand the intended meaning.

    3. Complex Word Forms (Morphology)
    Somali words can have different forms depending on grammatical structure, tense, gender, and other linguistic features.

    How it affects NLP: The system may fail to recognize that different forms of a word share the same root meaning. This affects search, translation, and text classification. Somali NLP research has identified morphological analysis and lemmatization as important challenges. 

    Example: A property search system may fail to connect different grammatical forms of a Somali word when searching for related property information.

    4. Mixed-Language Text (Code-Switching)
    Somali speakers sometimes mix Somali and English in the same sentence, especially on social media.

    How it affects NLP: The system may struggle to identify the correct language, understand the sentence structure, and interpret the meaning accurately.

    Example: A customer writes: "Gurigan location-kiisa waa good, laakiin rent-kiisu waa expensive."

    An NLP system may struggle to correctly interpret the mixture of Somali and English.

    Conclusion
    Somali NLP faces challenges due to limited datasets, spelling variations, complex word forms, and mixed-language communication. These challenges can reduce the accuracy of NLP systems. More Somali datasets, language-specific tools, and better preprocessing techniques can help improve performance.

    References
    Somali Information Retrieval Corpus – EMNLP 2023. https://aclanthology.org/2023.emnlp-main.462/?utm_source
    Morphologically-informed Somali Lemmatization Corpus – AfricaNLP 2026.https://aclanthology.org/2026.africanlp-main.17/?utm_source
    Somali Dialect Identification in Low-Resource Settings https://link.springer.com/article/10.1007/s44248-026-00109-y?utm_source

A6 : Case Study: Offensive Language Detection in Somali Using NLP
    Research Paper: Benchmarking Explainable Offensive Language Detection in Somali with Human-Annotated Rationales (2026).

    Authors: Abdisalam Mahamed Badel and colleagues.

    Research Problem: The researchers wanted to develop an NLP system that could identify offensive language in Somali social media comments and explain why a comment was classified as offensive.

    1. Task
       Detect whether a Somali social media comment contains offensive language or not.
    2. Data
       The researchers collected 10,175 Somali text samples from TikTok, YouTube, and X. Human annotators labeled the samples as:
       Offensive and Not offensive
       They also identified the words or phrases that justified each classification.
    3. Method
       The researchers:
        Developed a dataset with human annotations.
        Used transformer-based language models.
        Developed an explainable framework called SomOffXplain.
        Compared its performance with other models and Large Language Models (LLMs).
    4. Findings
        The proposed system achieved 87% accuracy.
        It performed better than the compared approaches in classification and explainability.
        Several tested LLMs struggled to identify some implicit offensive expressions in Somali. 
      
    5. Which NLP Task Does It Cover?
        Text Classification
        The study covers Text Classification because the model receives a text comment and assigns it to one of two predefined categories: Offensive or Not Offensive.

    Conclusion
    This research demonstrates how NLP can help improve online safety by automatically identifying offensive Somali content. It also contributes to Somali NLP research by providing a new dataset and an explainable detection system.

    Research source: https://www.nature.com/articles/s41598-026-53781-0?utm_source