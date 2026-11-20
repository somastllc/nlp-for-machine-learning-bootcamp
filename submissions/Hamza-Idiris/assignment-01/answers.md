# Assignment 1 -- NLP Foundations

**Student:** Hamza Mohamed Idiris\
**Course:** Natural Language Processing\
**Assignment:** 1 -- NLP Foundations\

------------------------------------------------------------------------

## 1. What Is Natural Language Processing, and How Is It Related to AI and Machine Learning?

Natural Language Processing (NLP) is a field of computer science and
artificial intelligence that develops methods for enabling computers to
work with human language. It includes tasks such as understanding
written text, identifying information, translating languages, analysing
opinions, answering questions, and generating responses. Unlike
structured data, human language is flexible and often ambiguous. The
same word can have different meanings depending on its context, and
people may express the same idea in many different ways. NLP combines
ideas from linguistics, computer science, statistics, and machine
learning to address these problems (Stanford HAI, n.d.; Harrison &
Sidey-Gibbons, 2021).

NLP, Artificial Intelligence (AI), and Machine Learning (ML) are closely
connected, but they do not mean the same thing:

-   **Artificial Intelligence (AI)** is the broad area concerned with
    building computer systems that perform tasks associated with
    intelligent behaviour, such as reasoning, understanding, planning,
    or decision-making.
-   **Machine Learning (ML)** is one approach within AI. Instead of
    writing every rule manually, developers train a model using data so
    that it can learn patterns and make predictions on new examples.
-   **Natural Language Processing (NLP)** is a specialised area of AI
    concerned with human language. Many modern NLP systems use machine
    learning, especially deep learning, to learn patterns from large
    collections of text or speech.

In simple terms, AI is the wider goal, ML is one way to build
intelligent systems, and NLP applies computational methods to language.
Not every NLP system must use machine learning: some older systems rely
mainly on dictionaries and manually written linguistic rules. However,
machine learning is central to many current NLP applications.

### Real-life example: analysing university student feedback

Imagine a university that receives thousands of written comments from students about its courses, library, registration process, and campus services. The university could use an AI-based feedback analysis system to help staff understand the main concerns.

NLP is used to process the written comments, identify important words and phrases, and recognise what each comment is discussing. A machine-learning model can be trained on examples that have already been reviewed and categorised by people. It can then classify new comments into topics such as course quality, library services, or registration problems, and may also estimate whether the feedback is positive or negative. The wider AI application presents these results in a dashboard so university staff can identify common issues and decide what needs attention.

This example shows how the three areas work together: NLP processes the students' language, ML learns patterns from previously reviewed comments, and AI describes the broader system that supports staff decision-making. The system should assist staff rather than replace human judgement, because comments can contain sarcasm, unclear wording, or context that an automated model may misunderstand.

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

These terms describe different levels of a text dataset. To make the
distinctions clear, imagine a small dataset of customer reviews for a
fictional online shop.

  ------------------------------------------------------------------------------------------------------------
  Term                    Meaning                 Example in the review dataset
  ----------------------- ----------------------- ------------------------------------------------------------
  **Corpus**              A collection of text    All 1,000 customer reviews collected by the shop.
                          documents gathered for  
                          a particular purpose or 
                          study.                  

  **Document**            One individual unit of  "The delivery was fast and the product was excellent."
                          text inside a corpus.   
                          It may be a review,     
                          email, article, or      
                          message.                

  **Token**               A basic unit produced   `The`, `delivery`, `was`, `fast`, `and`, `the`, `product`,
                          when text is divided    `was`, `excellent`, `.`
                          during tokenization.    
                          Tokens are often words  
                          or punctuation marks,   
                          depending on the        
                          tokenizer.              

  **Vocabulary**          The set of distinct     `{delivery, fast, product, excellent, slow, damaged, ...}`
                          tokens recognised in a  
                          corpus after the chosen 
                          preprocessing steps.    

  **Feature**             A measurable            The number of times `excellent` appears in a review, or its
                          representation or       TF-IDF value.
                          property of a document  
                          that a model can use.   

  **Label**               The target category or  `Positive`, `Negative`, or `Neutral` review sentiment.
                          answer associated with  
                          an example in a         
                          labelled dataset.       
  ------------------------------------------------------------------------------------------------------------

The terms are related, but they should not be treated as
interchangeable. A corpus contains documents; documents are divided into
tokens; the vocabulary collects the distinct tokens found across the
corpus. Features represent information from each document in a form that
a model can process. Labels provide the expected answer when the dataset
is being used for a supervised learning task.

For example, the review "The product is excellent" could be represented
by features indicating whether the words `product` and `excellent`
occur. If a human reviewer has marked it as positive, its label is
`Positive`. The words are part of the vocabulary, their occurrences
contribute to the features, and the sentiment category is the label.

## 3. Four Common NLP Tasks

NLP covers several types of tasks. The input and output depend on what
the system is expected to do.

  --------------------------------------------------------------------------------
  NLP task           What it does   Input          Output         Example
  ------------------ -------------- -------------- -------------- ----------------
  **Text             Assigns one or A sentence,    A class or set Classifying a
  classification**   more           review, email, of classes.    customer review
                     categories to  or full                       as positive,
                     a text.        document.                     negative, or
                                                                  neutral.

  **Sequence         Assigns a      A sentence     A label for    Named Entity
  labeling**         label to each  divided into   each token,    Recognition
                     token in a     tokens.        usually in the (NER), which
                     sequence.                     same order.    identifies names
                                                                  of people,
                                                                  places, and
                                                                  organisations.

  **Search and       Finds          A search query Ranked         Searching a
  similarity**       documents that or reference   documents or   university
                     are relevant   document,      similarity     website for
                     or similar to  together with  scores.        pages about
                     a query or     a collection                  admission
                     another        of documents.                 requirements.
                     document.                                    

  **Text             Produces new   A prompt,      A generated    A chatbot
  generation**       language based question, or   sentence,      generating an
                     on an          partial text.  paragraph,     answer to a
                     instruction,                  summary, or    user's question.
                     prompt, or                    response.      
                     previous text.                               
  --------------------------------------------------------------------------------

### Further explanation

**Text classification:** The system considers the whole text and chooses
a category. For example, a bank could classify incoming messages as
questions about account access, card services, or transactions. The
output is a category rather than a label for every individual word.

**Sequence labeling:** The system examines the words in their order and
predicts a label for each one. In the sentence "Amina lives in
Mogadishu," a named-entity recognition system might label `Amina` as a
person and `Mogadishu` as a location. The output is therefore a sequence
of token-level labels.

**Search and similarity:** A search system compares a user's query with
stored documents and returns the most relevant results. Similarity may
be calculated using word overlap, TF-IDF vectors, or semantic
embeddings. A useful system should retrieve documents that match the
meaning of the query, not only documents containing exactly the same
words.

**Text generation:** A generative system predicts and produces a
sequence of language in response to an input. Examples include drafting
an email, answering a question, or summarising an article. Generated
text should still be checked because fluent language is not a guarantee
that every statement is correct.

## 4. Supervised and Unsupervised Learning in NLP

Supervised and unsupervised learning differ mainly in the type of
training information they receive.

### Supervised learning

Supervised learning uses examples that contain both input data and a
known target label. During training, the model compares its predictions
with the correct labels and adjusts its parameters to reduce errors.
After learning from the labelled examples, it can predict labels for new
data.

**NLP example:** A sentiment classifier is trained on 5,000 customer
reviews. Each review has been labelled by a person as `Positive` or
`Negative`. The text is the input, and the sentiment label is the
target. The model learns which language patterns are associated with
each category and then predicts the sentiment of an unseen review.

### Unsupervised learning

Unsupervised learning uses data without target labels. The algorithm
attempts to discover structure or patterns in the data by itself.
Depending on the method, it may group similar documents, identify
topics, or find unusual patterns.

**NLP example:** A university collects thousands of student feedback
comments but has not assigned categories to them. A clustering algorithm
can group comments that use similar language or discuss similar
subjects. The groups might reveal themes such as teaching quality,
registration, library services, or internet access. These themes are
discovered from the text rather than supplied as correct labels
beforehand.

  -----------------------------------------------------------------------
  Comparison              Supervised learning     Unsupervised learning
  ----------------------- ----------------------- -----------------------
  Training data           Labelled examples       Unlabelled examples

  Target labels           Required for the        Not required
                          training objective      

  Main purpose            Learn to predict a      Discover hidden
                          known outcome           structure or patterns

  NLP example             Spam detection using    Grouping documents by
                          spam/not-spam labels    similar topics

  Main limitation         Preparing reliable      Discovered groups may
                          labels can be expensive be difficult to
                          and time-consuming      interpret or may not
                                                  match useful human
                                                  categories
  -----------------------------------------------------------------------

A supervised task needs target labels because the model must learn the
relationship between the input and a known answer. Without examples of
the expected output, it cannot calculate the usual supervised prediction
error against those target answers. An unsupervised task has a different
objective: it looks for patterns in the input data, so predefined target
labels are not necessary. This does not mean that unsupervised results
never need human evaluation; people may still need to interpret and
assess the patterns found.

## 5. Why Text Must Be Converted into Numerical Representations

Most machine-learning algorithms operate on numbers rather than raw
sentences. A computer can store text as characters, but a typical ML
model needs a consistent mathematical representation to calculate
distances, weights, probabilities, or other operations. Text documents
also have different lengths, so they must be transformed into a
representation that the selected model can process.

This transformation is often called **text vectorization** or **feature
extraction**. It converts documents into numerical vectors. For example,
a model cannot directly calculate the importance of a word from a
sentence in the same way it calculates with numerical values. A vector
representation makes those calculations possible.

### One simple method: Bag of Words

The Bag of Words (BoW) method represents a document by counting how
often each vocabulary word appears. It creates a vocabulary from the
collection of documents and assigns each vocabulary item a position in a
vector.

Suppose the dataset contains these three short documents:

1.  "NLP is useful"
2.  "Machine learning is useful"
3.  "NLP supports learning"

The vocabulary, in one possible order, is:

`[NLP, is, useful, Machine, learning, supports]`

The documents can then be represented using word counts:

  Document                       NLP   is   useful   Machine   learning   supports
  ---------------------------- ----- ---- -------- --------- ---------- ----------
  NLP is useful                    1    1        1         0          0          0
  Machine learning is useful       0    1        1         1          1          0
  NLP supports learning            1    0        0         0          1          1

Each row is a document vector, and each column represents a vocabulary
item. The numbers indicate how many times each word occurs in the
document. These vectors can be supplied to many traditional
machine-learning models.

### Main limitation of Bag of Words

The main weakness of basic Bag of Words is that it ignores word order
and much of the context in which words appear. For example, "the student
helped the teacher" and "the teacher helped the student" contain the
same words and may receive the same BoW vector, even though their
meanings are different. It can also create very large, sparse vectors
when the vocabulary contains many words. More advanced approaches, such
as n-grams, TF-IDF, and contextual word embeddings, address some of
these limitations, although each has its own trade-offs (scikit-learn
developers, n.d.; Jurafsky & Martin, 2024).

## 6. Challenges of NLP for Low-Resource Languages Such as Somali

A low-resource language is a language for which comparatively few
digital resources are available for building and evaluating language
technologies. These resources may include large text collections,
dictionaries, annotated datasets, spelling tools, speech recordings, and
language-specific software. A language can have many speakers and a rich
literary tradition while still being underrepresented in computational
research.

Somali is an important example. It is widely spoken in the Horn of
Africa, but the amount of publicly available, carefully labelled Somali
text used to train NLP systems is limited compared with major languages
such as English. This creates several practical challenges.

### 6.1 Limited datasets and annotation resources

Many machine-learning approaches require large amounts of representative
data. Supervised NLP tasks also need examples labelled by people who
understand the language and the task. For Somali, researchers may have
difficulty finding large, balanced, publicly available datasets for
sentiment analysis, named-entity recognition, question answering, and
other tasks.

Creating a reliable dataset requires more than collecting text. The data
may need cleaning, consent and privacy checks, language identification,
and annotation by people who understand Somali. Annotation can be
expensive and time-consuming. If the training dataset is small or does
not represent different topics and speakers, a model may memorise
patterns in the training data and perform poorly on new text. It may
also make more errors on dialects or topics that are missing from the
dataset.

Research on Somali information retrieval illustrates this resource gap.
Badel et al. (2023) describe the lack of a dedicated Somali retrieval
corpus and introduce a collection of 2,335 documents from sources
including Somali news websites and poetry books. Building such a corpus
is an important step because search systems need relevant
language-specific documents for development and evaluation.

### 6.2 Spelling variation and informal writing

People do not always write Somali in exactly the same way, especially in
informal digital communication. A word may be shortened, misspelled,
written with different spacing, or mixed with English expressions.
Social media messages may also contain emojis, repeated letters,
abbreviations, and non-standard punctuation.

This variation creates a problem for tokenization and vocabulary
matching. A model may treat two differently written forms of the same
word as separate tokens. As a result, the system may fail to connect
them, even when a human reader understands that they express the same
idea. This can reduce the quality of text classification, search, and
similarity calculations.

A practical response is to develop Somali-aware text normalization tools
and datasets that include real examples of informal writing. However,
normalization must be designed carefully: changing a word incorrectly
can remove meaning or dialect information.

### 6.3 Morphological complexity and word forms

Somali has morphological patterns in which words can change form through
grammatical and derivational processes. A single root may appear in
several related forms. If an NLP system treats every form as completely
unrelated, its vocabulary becomes larger and the available examples for
each form become fewer.

This can make it harder for a model to recognise related words,
particularly when some forms are rare or absent from its training data.
It can affect search, text classification, and language modelling.

Lemmatization is one possible solution. It attempts to map related forms
to a common dictionary form, or lemma. However, this requires accurate
knowledge of Somali morphology. Mohamed and Mohamed (2023) describe a
lexicon- and rule-based approach to Somali lemmatization, showing that
language-specific resources and rules are useful starting points. More
recent work by Gedi et al. (2026) also highlights the continuing need
for annotated Somali lemmatization resources.

### 6.4 Dialect differences and code-switching

Somali includes dialect variation, including Maxaa-tiri and Maay.
Speakers may also mix Somali with English or Arabic, particularly in
online conversations, education, business, and technology-related
discussions. This practice is often called **code-switching**.

A model trained mostly on one dialect may not understand another dialect
equally well. Shared words may have different meanings or uses, while
different words may express similar meanings. When Somali and English
appear in the same sentence, a system may also identify the wrong
language or apply unsuitable tokenization and language-specific rules.

Gedi et al. (2026) studied Somali dialect identification and reported
challenges involving shared vocabulary, code-switching, mixed dialect
use, and inconsistent social-media spelling. Their work shows why Somali
datasets should represent different dialects and real writing practices
rather than relying on one narrow source.

### Why these challenges matter

These problems are connected. Limited datasets make it difficult to
represent spelling variation, morphology, and dialects adequately. When
a model does not see enough examples of a word form or dialect, it may
produce unreliable predictions. Improving Somali NLP therefore requires
not only more data, but also better annotation, language-specific
preprocessing, dialect representation, and evaluation with native
speakers.

## 7. Case Study: Explainable Offensive Language Detection in Somali

### 7.1 Research selected

For this case study, I selected the 2026 research paper by Badel et
al. titled **"Benchmarking explainable offensive language detection in
Somali with human-annotated rationales,"** published in *Scientific
Reports*. The study addresses the challenge of identifying offensive
language in Somali online content while also explaining which parts of a
text support a model's decision.

This is a relevant real-world problem because online platforms need ways
to identify potentially harmful or abusive content. A system that only
returns a category may be difficult for moderators to trust. An
explanation that highlights the words or phrases behind a prediction can
make the result easier to inspect.

### 7.2 Task

The main task is **text classification**: the system classifies a Somali
text sample as offensive or not offensive. The research also includes a
sequence-level explanation component. It identifies words or phrases
that support the classification, which is related to span-level
rationale extraction.

Therefore, the main NLP task from Question 3 is text classification,
with an additional token- or phrase-level explanation.

### 7.3 Data

The researchers introduced a dataset called **XOLDS**, containing 10,175
Somali text samples. The samples were manually collected from TikTok and
YouTube, with additional samples from X. Three annotators labelled each
sample as offensive or not offensive. Two of the three annotators also
selected relevant text spans and provided explanations for their
decisions.

Human annotation is important in this task because offensiveness can
depend on context, meaning, and how a phrase is used. A word that
appears in an offensive message may also appear in a harmless context.
The dataset therefore provides both category labels and human-selected
rationales for evaluation.

### 7.4 Method

The researchers proposed an explainable framework called
**SomOffXplain**. They evaluated five fine-tuned pretrained models and
used Local Interpretable Model-Agnostic Explanations (LIME) as one
explainability baseline. They also adapted four large language models
using zero-shot and few-shot prompting to examine how well they could
identify and explain offensive content in Somali.

The framework focuses on two related questions:

1.  Can a model correctly identify whether a Somali text sample is
    offensive?
2.  Can it provide a useful explanation by identifying the words or
    phrases that support its prediction?

### 7.5 Findings

The study reports that SomOffXplain performed better than the compared
baselines in predictive accuracy and in the quality of its explanations,
including plausibility and faithfulness. The authors also found that
approximately half of the evaluated state-of-the-art large language
models failed to produce high-quality rationales that aligned with the
human-annotated explanations. LIME-based explanations were also found to
be weak for Somali text in this evaluation.

The findings are significant because they show that strong performance
in a widely used language does not automatically transfer to a
low-resource language. A model may produce fluent explanations while
missing the actual meaning or context of a Somali message.
Human-annotated data and careful evaluation are therefore essential.

The researchers also acknowledge limitations. Collecting and annotating
the dataset required considerable manual effort, and models can still
miss implicit offensive meaning or misclassify words whose
interpretation depends on context. These limitations suggest that more
Somali language resources and further evaluation are needed.

### 7.6 Connection to the NLP tasks in Question 3

The central task is **text classification**, because the system assigns
each text sample to one of two categories: offensive or not offensive.
The explanation component adds a sequence- or span-labelling element by
identifying the specific words and phrases that influenced the
prediction.

### 7.7 My assessment

I consider this study useful because it does not treat classification
accuracy as the only measure of success. In a real moderation system, a
wrong decision can affect users, so it is important to understand why a
model produced a particular result. The use of human-provided
explanations also makes it possible to compare model reasoning with the
judgement of Somali speakers.

However, such a system should support human moderators rather than make
every moderation decision automatically. Somali dialects, sarcasm,
cultural context, and mixed-language writing can be difficult to
interpret. The study demonstrates progress, but it also shows that
responsible Somali NLP requires broader datasets, careful testing, and
continued involvement from native speakers.

------------------------------------------------------------------------

## Conclusion

NLP provides methods for analysing and generating human language, and it
is an important area within artificial intelligence. Machine learning
supports many NLP applications by learning patterns from text data.
Understanding the basic terms---such as corpus, document, token,
vocabulary, feature, and label---is necessary before building or
evaluating NLP systems.

The four tasks discussed in this assignment demonstrate that NLP can
classify whole documents, label individual words, retrieve relevant
information, and generate new text. Supervised learning uses labelled
examples to learn a prediction task, while unsupervised learning
searches for patterns without predefined target labels. Since most
machine-learning models require numerical input, text must also be
converted into representations such as Bag of Words or TF-IDF.

The discussion of Somali shows that language technology cannot be
developed equally well for every language using the same resources and
assumptions. Limited datasets, spelling variation, morphology, dialect
differences, and code-switching can all affect system performance. The
selected Somali offensive-language detection study demonstrates both the
progress being made and the need for more high-quality resources and
human evaluation. Overall, developing useful NLP for Somali requires
technical methods as well as attention to the language and the people
who use it.

------------------------------------------------------------------------

## References

Badel, A., Zhong, T., Xu, X., Tai, W., Zhou, F., & Dahir, A. A. (2026).
Benchmarking explainable offensive language detection in Somali with
human-annotated rationales. *Scientific Reports, 16*, Article 24406.
https://doi.org/10.1038/s41598-026-53781-0

Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information
retrieval corpus: Bridging the gap between query translation and
dedicated language resources. In *Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing* (pp. 7463--7469).
Association for Computational Linguistics.
https://aclanthology.org/2023.emnlp-main.462/

Gedi, A. A., Ahmed, Y. M., Mohamed, S. A., Yusuf, Y. A., & Ebdiyow, A.
U. (2026). Somali dialect identification in low-resource settings using
machine learning and deep learning. *Discover Data, 4*, Article 10.
https://doi.org/10.1007/s44248-026-00109-y

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F.
M., & Assowe, H. A. (2026). Morphologically-informed Somali
lemmatization corpus built with a web-based crowdsourcing platform. In
*Proceedings of the 7th Workshop on African Natural Language
Processing*. Association for Computational Linguistics.
https://aclanthology.org/2026.africanlp-main.17/

Harrison, C. J., & Sidey-Gibbons, C. J. (2021). Machine learning in
medicine: A practical introduction to natural language processing. *BMC
Medical Research Methodology, 21*, Article 158.
https://doi.org/10.1186/s12874-021-01347-1

Jurafsky, D., & Martin, J. H. (2024). *Speech and language processing*
(3rd ed. draft). https://web.stanford.edu/\~jurafsky/slp3/

Mohamed, S. A., & Mohamed, M. A. (2023). Lexicon and rule-based word
lemmatization approach for the Somali language. In *Proceedings of the
6th Workshop on NLP for Internet Freedom (NLP4IF 2023)*.
https://arxiv.org/abs/2308.01785

scikit-learn developers. (n.d.). *Feature extraction: Text feature
extraction*. scikit-learn documentation.
https://scikit-learn.org/stable/modules/feature_extraction.html

Stanford Institute for Human-Centered Artificial Intelligence. (n.d.).
*What is natural language processing (NLP)?*
https://hai.stanford.edu/ai-definitions/what-is-nlp
