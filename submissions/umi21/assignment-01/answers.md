## 1. Define **Natural Language Processing (NLP)**. What is its relationship to **Artificial Intelligence** and **Machine Learning**? Use a real-life example to illustrate how they work together.

**Natural Language Processing (NLP)** is a subfield of Computer Science and Artificial Intelligence that uses Machine Learning to enable computers to understand and communicate with human language[^1]. **Artificial Intelligence** is the overarching field of automating intelligence, what humans would generally and easly but is hard for computers. Whereas **NLP** uses **Machine Learning** as a tool, along with computational linguistics methods, to achieve its human language processing goal.

### A real-world example: Language Translation
    1. **NLP** is the goal, the application of using **machine learning**, which is a subfield of **AI**, to translate from one human language to another.

## 2. Explain the terms **corpus**, **document**, **token**, **vocabulary**, **feature** and **label**. Use one example dataset of your choice to show how each term applies.

- **Corpus** : is an entire collection of text "dataset" prepared for a specific project or task.
- **Document** : is a single individual example from the corpus.
- **Token** : is the basic unit of text that a model processes.
- **Vocabulary** : a set of **unique** tokens mapped to an ID for the model to process.
- **Feature** : it is a numerical representation extracted from a text so an ML model uses it to make predictions and decisions. 
- **Label** : is a tag or a target value for an example (Document) that the machine learning model tries to predict.

## 3. Describe four common **NLP tasks**, such as text classification, sequence labeling, search and similarity, and text generation. For each, state what the input is and what the output is.

- **Machine Translation**: 
    - Input: A sentence, paragraph, or document written in a source language.
    - Ouput: The equivalent text written in a target language.
- **Text Summarization** :
    - Input: A lengthy document, news article, or meeting transcript.
    - Ouput: A brief paragraph or a short list of bullet points highlighting the key takeaways.
- **Question Answering** : 
    - Input: A question, often paired with a reference passage or a knowledge database.
    - Ouput: A direct answer, which is either a short text span extracted from the passage or a newly written response.
- **Autoformalization** : 
    - Input: Informal mathematical statements, theorems, or step-by-step proofs written the way mathematicians write in textbooks or research papers
    - Ouput: Strict, logically precise code that a computer program can automatically check for errors without human help

## 4. Compare **supervised** and **unsupervised** learning in NLP, giving an example of each. Explain why one task needs target labels and the other does not.

- **Supervised learning** trains a model on examples that already carry the correct answer (a label). The model learns a mapping from text to label and is then used on new text.

    - Example: Sentiment classification. Each review or tweet is labelled positive, negative or neutral, and the model learns which wording signals each class.

- **Unsupervised learning** works on raw text with no labels. The model looks for structure in the data itself, such as groups or recurring patterns.

    - Example: Topic modelling (e.g. LDA) or clustering news articles. The algorithm groups articles that use similar vocabulary without being told the topics in advance.

## 5. Explain why text must be converted into **numerical representations** before most Machine Learning models can use it. Describe one simple representation method and its main limitation.

Machine learning models are mathematical functions. They multiply, add and compare numbers, and they cannot operate on raw characters or words. To feed text into a model, we need a numerical representation that preserves something of its meaning, so that a computer can measure similarity, find patterns and compute errors.

- **Bag of Words (BoW).**: Build a vocabulary of all words in the corpus. Each document becomes a vector whose length equals the vocabulary size, and each position holds the count of that word in the document. For example, with vocabulary [good, bad, movie], the text “good good movie” becomes [2, 0, 1].

- Main limitation: BoW ignores word order and meaning. “The dog bit the man” and “The man bit the dog” get identical vectors, and “good” and “great” are treated as completely unrelated. The vectors are also very large and mostly zeros (sparse) for big vocabularies. Word embeddings and transformers were developed to address these problems.

## 6. Research the challenges of NLP for **low-resource languages** such as Somali. Discuss at least two challenges (for example, limited datasets, spelling variation, word forms or mixed-language text) and explain how each affects an NLP system.

1. Limited datasets and annotated data. Somali has far fewer digital texts, and fewer labelled corpora, than English. Research on Somali notes that scarce annotated training data and the lack of language models tailored to Somali limit AI automation. Effect: Supervised models cannot be trained well because there are too few labelled examples, so accuracy is lower and models overfit. Researchers often have to build datasets from scratch or rely on multilingual models that were trained on little Somali.

2. Spelling variation and non-standard writing. Social media Somali is written informally, with inconsistent spellings, abbreviations and no fixed convention for some words. Effect: The same word appears as several different tokens, which inflates the vocabulary and splits the statistics for that word. A model that has never seen a variant treats it as unknown (the out-of-vocabulary problem), so tokenisation and classification are less reliable.

3. Rich word forms (morphology). Somali words change through affixes and other grammatical markers, so one root can appear in many surface forms. Effect: Word-level models face data sparsity because each form is rare. Stemming, lemmatisation or subword tokenisation tools are needed, and these are less developed for Somali than for English.

4. Mixed-language text (code-switching). Somali speakers often mix Somali with English or Arabic in online writing. Effect: Models trained on one language misread the other words, and language identification becomes harder.

## 7. Find one **case study** (research paper or article) that applies NLP to a real problem, preferably in Somali or another African language. Summarize its task, data, method and findings, and identify which NLP task from question 3 it covers.

**This Paper**: Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models, Mohamed, Ahmed, Isse, Mohamed, Hassan and Assowe, arXiv, March 2025 (arxiv.org/abs/2503.18117).

Mohamed and colleagues (2025) wrote a paper on detecting fake news and toxic messages in Somali social media posts. The problem is that anyone can post online, so false and harmful content spreads easily, while Somali has very little labelled data and no language model built for it. The authors created two human-labelled datasets from social media, one for fake news and one for toxic messages, and gathered a large collection of Somali text to train a model. They built SomBERTa, which they describe as the first Somali-only transformer model, and tested it on fake news, toxicity and news topic classification. They compared it with multilingual models such as AfriBERTa and AfroXLMR. Their results show that SomBERTa did better than these models on both fake news and toxic content detection. This task is text classification, and it is supervised learning because the model was trained on labelled examples.


## References

[^1] [What is NLP?](https://www.ibm.com/think/topics/natural-language-processing)