# Assignment 1 – NLP Foundations

**Student:** [cabdilaahi]  
**GitHub username:** [cabdilaahim043-rgb]

## 1. Natural Language Processing, Artificial Intelligence, and Machine Learning

Natural Language Processing (NLP) is the field of computing concerned with analyzing, interpreting, retrieving, and producing human language, including written text and speech. It combines ideas from linguistics and computer science. Artificial Intelligence (AI) is the broader effort to develop systems that perform tasks associated with intelligence; NLP focuses on the language-related part of that effort. Machine Learning (ML) is a collection of methods that learn patterns from examples rather than relying only on manually written rules. Many modern NLP systems use ML, although NLP can also use rule-based approaches (Jurafsky & Martin, 2026).

For example, a customer-support service might receive the message, “My order arrived damaged.” NLP methods split and represent the message; an ML classifier trained on earlier messages predicts the category *damaged product*; and an AI-enabled workflow routes the complaint to the appropriate support team. Language processing defines the problem, machine learning supplies a learned prediction, and the larger AI application uses that prediction to assist a person. The classifier can make mistakes, so important decisions may still require human review.

## 2. Core terminology illustrated with a dataset

Consider this **invented miniature dataset** of customer messages. It is an illustration, not a published research dataset.

| Document ID | Text | Label |
|---|---|---|
| D1 | My parcel arrived late | Delivery issue |
| D2 | The package arrived damaged | Damage issue |
| D3 | My parcel is late | Delivery issue |

| Term | Meaning | Application to this dataset |
|---|---|---|
| **Corpus** | A collection of texts used for analysis or model development. | All three messages together form the corpus. |
| **Document** | One text unit within a corpus; its size depends on the task. | D2 is one document. |
| **Token** | A unit produced by tokenization, such as a word or subword. | With simple whitespace tokenization, D1 contains `My`, `parcel`, `arrived`, `late`. |
| **Vocabulary** | The set of distinct tokens in a corpus under a chosen tokenizer and normalization rule. | Lowercasing and splitting on spaces gives `{my, parcel, arrived, late, the, package, damaged, is}`: eight types. |
| **Feature** | A measurable property used as model input. | The count of `late` in a message is one possible feature. |
| **Label** | A target answer assigned to an example for supervised learning. | D2 has the label `Damage issue`. |

The distinction between *token* and *vocabulary item* matters: `parcel` occurs twice in the corpus but contributes only one distinct vocabulary item. Tokenization rules also matter: subword tokenizers would produce a different vocabulary (Jurafsky & Martin, 2026).

## 3. Four common NLP tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Text classification** | A document or message | One or more categories | Complaint → `Delivery issue` |
| **Sequence labeling** | A sequence of tokens | A label for each token or span | “Amina visited Hargeisa” → `Amina: PERSON`, `Hargeisa: LOCATION` |
| **Search and similarity** | A query and candidate documents, or two texts | Ranked relevant documents or a similarity score | “late delivery” → the most relevant customer messages |
| **Text generation** | A prompt, context, or source text | Newly generated text | Complaint details → draft response |

Sequence labeling includes named entity recognition (NER), which identifies expressions such as people, organizations, and locations. Search may use lexical matching or learned semantic representations. Text generation can produce fluent but inaccurate statements, so its outputs should be checked (Jurafsky & Martin, 2026).

## 4. Supervised versus unsupervised learning

**Supervised learning** learns a relationship between inputs and known target outputs. For instance, a complaint classifier can learn from messages that humans have marked `Delivery issue` or `Damage issue`. Those labels are necessary because the training objective measures how closely predictions match the desired categories.

**Unsupervised learning** examines data without task-specific target labels. For example, a clustering algorithm can group thousands of unlabelled customer messages according to their text representations. The algorithm seeks patterns of similarity, not agreement with a supplied complaint category. A person may subsequently interpret a cluster as “delivery problems,” but that interpretation was not a training label.

| Aspect | Supervised | Unsupervised |
|---|---|---|
| Training information | Inputs and target labels | Inputs without target labels |
| Example | Complaint classification | Grouping similar complaints |
| Typical objective | Predict known targets | Discover structure or patterns |
| Main challenge | Obtaining reliable labels | Interpreting and evaluating discovered groups |

Unsupervised learning does not mean that no assumptions or objectives are used: clustering still requires a representation and a definition of similarity.

## 5. Why numerical text representations are needed

Most ML algorithms calculate with numerical arrays: they use operations such as addition, multiplication, distances, and gradients. Raw character strings cannot directly serve as these numerical inputs. A representation method therefore maps text to numbers that a model can process (Jurafsky & Martin, 2026).

A simple method is **bag-of-words (BoW)**. First build a vocabulary, then represent each document by the number of times each vocabulary word appears. Suppose the vocabulary is `[parcel, late, damaged]`:

| Text | parcel | late | damaged |
|---|---:|---:|---:|
| “parcel late” | 1 | 1 | 0 |
| “parcel damaged” | 1 | 0 | 1 |

The resulting vectors are `[1, 1, 0]` and `[1, 0, 1]`. A major limitation is that ordinary BoW discards word order: “dog bites person” and “person bites dog” receive identical counts even though they describe different events. BoW also does not directly capture contextual meaning. More advanced representations, such as contextual embeddings, address some of these limitations but have their own costs and constraints.

## 6. Challenges for Somali and other low-resource languages

A language is often called *low-resource* in NLP when comparatively little suitable digital text, annotated data, evaluation material, or language-specific tooling is available for a particular task. This is about technological resources, not the value or complexity of the language.

**Limited labelled datasets.** A Somali sentiment classifier needs examples of Somali text with trustworthy sentiment labels. If few such examples are available, the model may overfit to a small sample and perform poorly on unfamiliar topics or dialects. Evaluation also becomes unreliable if test sets are tiny or unrepresentative. Building carefully reviewed datasets and exploring multilingual transfer learning can help, but transfer from other languages must be tested on Somali data.

**Spelling and orthographic variation.** Informal messages can contain typos, inconsistent spacing, and different ways of writing the same expression. A word-count model may treat variants as unrelated features, fragmenting the evidence available for each word. Normalization, character-level features, and subword tokenization may reduce this problem, although aggressive normalization can erase meaningful distinctions.

**Rich word formation.** Somali uses grammatical morphology, so related forms of a word can differ in their affixes or grammatical marking. A model that memorizes complete words may fail to recognize an unfamiliar inflected form. Morphological analysis or subword representations can help share information across related forms; their effectiveness should be evaluated rather than assumed.

**Mixed-language text and domain differences.** Online conversations may combine Somali with English or Arabic expressions. Models trained only on carefully edited monolingual text can misinterpret code-switching, borrowed words, or informal writing. Training and testing on representative real-world material, with appropriate privacy protections, can improve robustness.

These are plausible technical challenges for Somali; their severity depends on the dataset, application, and users. Research on African-language NLP, including MasakhaNER, illustrates the broader need for locally developed datasets and language-aware evaluation (Adelani et al., 2021).

## 7. Research case study: MasakhaNER

**Study:** Adelani et al. (2021), *MasakhaNER: Named Entity Recognition for African Languages*, published in *Transactions of the Association for Computational Linguistics*.

**Problem and task.** The researchers addressed the shortage of high-quality resources for named entity recognition (NER) in African languages. NER identifies mentions of entities in text, such as people, places, organizations, and dates. It belongs to the **sequence labeling** task described in Question 3.

**Data.** The project assembled and released annotated NER data for ten African languages: Amharic, Hausa, Igbo, Kinyarwanda, Luganda, Luo, Nigerian Pidgin, Swahili, Wolof, and Yorùbá. The annotations include person (`PER`), organization (`ORG`), location (`LOC`), and date (`DATE`) categories. **Somali was not one of the ten languages in this original study.** Community contributors participated in dataset creation and annotation (Adelani et al., 2021; Masakhane, n.d.).

**Method.** The authors evaluated contemporary NER approaches under supervised and transfer-learning settings, using annotated text to train and assess entity recognition. Transfer learning is especially relevant where a target language has limited training data because knowledge from other data or languages may help.

**Findings and significance.** The study produced publicly available annotated datasets, code, and models and reported empirical comparisons of NER methods. Its main practical contribution was to make systematic NER research and benchmarking more feasible for the included languages, rather than assuming models developed for high-resource languages would automatically work well. For Somali NLP, the lesson is methodological: community-led annotation and language-specific evaluation can help address resource shortages, but the paper's results should not be represented as direct evidence of Somali NER performance.

## Conclusion

NLP provides techniques for working with human language, while machine learning supplies many of the predictive methods used in current NLP applications. Fundamental concepts such as tokens, corpora, labels, and numerical features determine how a language problem is formulated. Different tasks require different inputs, outputs, and evaluation methods. For low-resource languages such as Somali, high-quality representative data and careful testing are as important as model choice.

## References

Adelani, D. I., et al. (2021). *MasakhaNER: Named entity recognition for African languages*. Transactions of the Association for Computational Linguistics, 9. https://aclanthology.org/2021.tacl-1.66/

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models* (3rd ed., online draft, August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/

Masakhane. (n.d.). *MasakhaNER dataset card*. Hugging Face. https://huggingface.co/datasets/masakhane/masakhaner
