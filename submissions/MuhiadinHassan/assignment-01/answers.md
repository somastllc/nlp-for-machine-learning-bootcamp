# Assignment 1 – NLP Foundations

## 1. Natural Language Processing, AI, and Machine Learning

Natural Language Processing (NLP) is the area of computing concerned with analyzing, interpreting, producing, or otherwise working with human language. It combines ideas from computer science, linguistics, statistics, and artificial intelligence. Typical examples include search, translation, speech transcription, sentiment analysis, and text generation (Jurafsky & Martin, 2026; Eisenstein, 2019).

Artificial Intelligence (AI) is the broad goal and field of building systems that perform tasks associated with intelligent behavior. Machine Learning (ML) is one family of methods within AI: instead of writing every decision as a fixed rule, developers train a model to learn useful patterns from examples. NLP is a problem area where AI and ML methods are often applied. The terms are related but not interchangeable: some NLP systems use hand-written rules, and not every AI system processes language (NIST, 2023).

**Example: an email spam filter.** NLP processes the words, punctuation, and other signals in each email. For supervised ML, developers provide many example emails marked “spam” or “not spam.” The model learns patterns associated with each class. The AI-powered email service then uses the model’s prediction to place a new message in the inbox or spam folder. The language-processing step prepares the email; ML learns from labeled examples; and the larger AI application uses the prediction to help manage email.

## 2. Corpus, Document, Token, Vocabulary, Feature, and Label

Consider a small, invented dataset of three customer messages. The example is for explanation only; it is not a published dataset.

| Document ID | Message | Label |
|---|---|---|
| D1 | “Delivery was fast and helpful.” | positive |
| D2 | “The package arrived late.” | negative |
| D3 | “Helpful support, but delivery was late.” | negative |

- **Corpus:** a collection of text assembled for analysis. The three messages together form this small corpus.
- **Document:** one unit of text in a corpus. D1 is one document; a document could also be a whole article or a conversation, depending on the project.
- **Token:** an element produced by splitting text according to a tokenizer’s rules. For a simple word tokenizer, D1 might become `Delivery`, `was`, `fast`, `and`, `helpful`, `.`. Other tokenizers may split punctuation differently or break words into subword pieces.
- **Vocabulary:** the set of distinct tokens recognized in a corpus. Ignoring punctuation and treating upper- and lower-case forms as the same, this example’s vocabulary includes `delivery`, `was`, `fast`, `and`, `helpful`, `the`, `package`, `arrived`, `late`, `support`, `but`.
- **Feature:** a measurable input used by a model. A simple feature could be whether the word `late` occurs in a document; alternatively, each vocabulary term can be a feature whose value is its count in that document.
- **Label:** the answer/category attached to an example that a model is meant to predict. Here, `positive` or `negative` is the message’s sentiment label.

These definitions depend somewhat on the analysis: a document might be a sentence or an entire post, and tokens depend on the tokenizer. The corpus supplies the text examples; features represent them for a model; labels provide the desired answers in a labeled training set.

## 3. Four Common NLP Tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A text unit, such as an email or review | One or more categories | Review → positive, neutral, or negative |
| **Sequence labeling** | A sequence of tokens | A label for each token | “Amina lives in Nairobi” → person / other / location labels (named-entity recognition) |
| **Search and similarity** | A query and a collection of documents (or two texts to compare) | Ranked documents or a similarity score | Query “Somali language lessons” → relevant pages ranked first |
| **Text generation** | A prompt, conversation history, or other conditioning input | A generated sequence of words | Question → draft answer or summary |

Classification predicts a category for a whole text (or span). Sequence labeling makes local predictions while retaining the order of tokens. Search retrieves and ranks existing items, often using lexical or semantic relevance. Generation produces new text conditioned on an input; it does not necessarily guarantee that the output is factually correct.

## 4. Supervised and Unsupervised Learning

**Supervised learning** trains on examples paired with target answers. For example, a sentiment classifier might learn from reviews labeled positive or negative. During training, the algorithm compares its predictions with the supplied labels and adjusts its parameters to reduce errors. Labels are needed because the goal is to learn a mapping from text to a known target category.

**Unsupervised learning** uses examples without target labels and seeks structure in the data. For example, a clustering method could group news articles by recurring vocabulary and themes without being told in advance which topic each article belongs to. It does not need target labels because its immediate goal is to discover patterns or groupings, rather than reproduce a supplied answer. A person may interpret the resulting clusters afterward. The distinction is about the training signal, not whether the raw text exists: both approaches need input data, but only supervised training requires the target answers.

## 5. Why Text Becomes Numbers: A Simple Method

Most ML algorithms perform mathematical operations on numeric inputs. Words are symbolic strings, so a computer system typically converts text into numbers before a model can compare examples and learn patterns. A representation method determines what information is retained and what is lost.

A basic method is **bag of words (BoW)**. First choose a vocabulary; then represent each document by the count of each vocabulary term. For the vocabulary `[fast, helpful, late]`, “fast and helpful” becomes `[1, 1, 0]`, while “late but helpful” becomes `[0, 1, 1]`. These vectors can be used as model features.

The main limitation is that ordinary BoW ignores word order and context. “Dog bites person” and “person bites dog” can have the same word counts even though their meanings differ. It also treats a word as the same feature in different contexts and may not connect related words such as “help” and “helpful.” More advanced representations, such as word or contextual embeddings, aim to encode richer information, though they have their own data and modeling limitations (Eisenstein, 2019; Jurafsky & Martin, 2026).

## 6. Challenges for Low-Resource Languages such as Somali

A language is often called **low-resource** in NLP when there are comparatively few accessible, digitized, and carefully labeled datasets, dictionaries, tokenizers, or other tools for building and evaluating systems. “Low-resource” describes the technology and data situation, not the language’s value, sophistication, or number of speakers. Research on African NLP has documented a substantial imbalance in resources and research attention across languages (Nekoto et al., 2020; Adelani et al., 2025).

**Limited datasets and annotation.** Somali NLP researchers have noted that dedicated Somali datasets have been scarce. For example, Badel et al. (2023) created an information-retrieval corpus of 2,335 documents from Somali online sources and poetry books, motivated in part by the lack of a dedicated corpus. With little representative training data, a model has fewer examples from which to learn spelling, topics, and language patterns. It may perform well on familiar news text but poorly on social media, dialects, or new subjects. Small test sets also make it harder to know whether an evaluation score generalizes.

**Spelling and orthographic variation.** People may write the same word in different ways, particularly in informal digital text, and spelling conventions may vary. A word-based model can treat variants as separate vocabulary entries. That fragments evidence: the system sees fewer examples of each spelling, which can hurt search recall, classification, and token matching. Aggressive normalization might help join variants, but could also erase meaningful distinctions; it should be checked with knowledgeable Somali speakers.

**Morphology and word forms.** Somali has rich inflectional and derivational morphology, so a root may appear in multiple surface forms. A model with a fixed word vocabulary may rarely see each form and may treat related forms as unrelated features. This can lead to out-of-vocabulary words and poorer retrieval or classification. The 2026 Somali lemmatization study describes the need for morphological resources and reports that its pilot method’s accuracy varied by text type, illustrating that a single resource or rule set does not solve every case (Gedi et al., 2026).

**Mixed-language text and domain shift.** Somali users may mix Somali with English, Arabic, or other languages, especially online. A system trained only on formal Somali may tokenize or interpret borrowed words, names, abbreviations, and language switches poorly. It can misclassify sentiment or miss relevant search documents. Training and evaluation data should therefore reflect the language varieties and settings where the system will actually be used.

## 7. Case Study: Somali Information Retrieval

Badel, Zhong, Tai, and Zhou (2023) studied **information retrieval**: finding relevant documents in response to a user’s query. Their work directly covers the “search and similarity” task in Question 3.

**Problem and data.** Somali-language search research had often relied on translating queries into another language because a dedicated Somali corpus was missing. The authors worked with language experts and NLP researchers to build an annotated corpus of 2,335 Somali documents gathered from sources including Hiiraan Online, Dhacdo.net, and Somali poetry books.

**Method.** They developed a Somali information-retrieval system using **pseudo-relevance feedback (PRF)** for query expansion. In general, PRF takes the top results from an initial search as provisional evidence of relevance, extracts useful terms from them, and adds those terms to the query before searching again. This can help when the user’s query is short or does not use the same terms as relevant documents. The paper frames its corpus as a resource for future Somali NLP applications as well.

**Findings and significance.** The authors’ main contribution was a dedicated Somali retrieval corpus and an initial retrieval system, addressing the shortage of digital language resources. Their result is important as infrastructure for Somali NLP, though a corpus of this size and source mix cannot represent every Somali dialect, genre, or spelling style. The study therefore illustrates both how a concrete NLP task can be built and why wider, representative data collection and careful evaluation remain necessary.

## References

Alabi, J. O., Hedderich, M. A., Adelani, D. I., & Klakow, D. (2025). Charting the landscape of African NLP: Mapping progress and shaping the road ahead. *Proceedings of EMNLP 2025*. https://doi.org/10.18653/v1/2025.emnlp-main.1414

Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). Somali information retrieval corpus: Bridging the gap between query translation and dedicated language resources. *Proceedings of EMNLP 2023*, 7463–7469. https://doi.org/10.18653/v1/2023.emnlp-main.462

Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press. https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing/

Gedi, A. A., Mohamed, S. A., Yusuf, Y. A., Mohamed, M. A., Hassan, F. M., & Assowe, H. A. (2026). Morphologically-informed Somali lemmatization corpus built with a web-based crowdsourcing platform. *Proceedings of the 7th Workshop on African Natural Language Processing*, 179–189. https://doi.org/10.18653/v1/2026.africanlp-main.17

Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

Nekoto, W., et al. (2020). Participatory research for low-resourced machine translation: A case study in African languages. *Findings of EMNLP 2020*, 2144–2160. https://aclanthology.org/2020.findings-emnlp.195/

National Institute of Standards and Technology (NIST). (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1*. https://doi.org/10.6028/NIST.AI.100-1

