  Part A: Research Questions

Q1. How does text preprocessing influence the quality of machine learning models?

Text preprocessing influences machine learning models by changing the way information is represented before it is used for training. Poorly prepared data can introduce noise, inconsistent representations, and unnecessary features

For example a dataset containing customer reviews may include misspelled words, repeated characters, unusual symbols, or inconsistent formats. These issues may affect how a machine learning model learns patterns

However, preprocessing does not always improve performance. Some information removed during preprocessing may be useful for a specific task. Therefore, preprocessing techniques should be selected according to the dataset, the task, and the model being used

Q2. What is information loss in text preprocessing, and why can it be a problem?

Information loss occurs when preprocessing removes or changes information that may be meaningful to a machine learning task.

For example, in a medical text classification system, the expressions "symptoms present" and "no symptoms present" have different meanings. If a preprocessing operation removes a word that expresses negation, the resulting text may become misleading

Similarly removing numbers from financial documents could eliminate important information such as prices, percentages, dates, or transaction amounts.

The goal of preprocessing should therefore be to improve data representation without unnecessarily destroying useful information.

Q3. What is the relationship between tokenization design and languages with complex word structures?

Tokenization design is important because languages differ in how words are formed written and separated. Some languages use spaces to separate words, while others may combine several meaningful components into a single written word

Languages with complex morphological structures may require tokenizers that can represent prefixes, suffixes, roots, and other word components effectively. A tokenizer that works well for one language may not work equally well for another

Subword tokenization method such as Byte-Pair Encoding (BPE) WordPiece, and Unigram are widely used in modern NLP systems because they can represent words that are not present in a model's vocabulary as complete words

However  tokenization quality should be evaluated using the target language and task rather than assuming that one tokenizer is suitable for every language.

Q4. Why should preprocessing strategies be designed according to the domain of the dataset?

Preprocessing should be adapted to the domain because the same text feature may have different importance in different fields.

For example dates article numbers, and section references may be important in legal documents. In financial documents, currencies, numerical values and percentages may be essential In medical documents    abbreviations and dosage information may affect interpretation. In social media analysis, emojis, repeated characters, and informal expressions may carry emotional information

A general preprocessing pipeline may not be appropriate for every dataset. Domain-specific preprocessing helps preserve information relevant to the task and avoid unnecessary modifications.

Q5. What are the trade-offs between preprocessing techniques and machine learning performance?

Preprocessing techniques involve trade-offs between reducing unnecessary information and preserving meaningful features.

Preprocessing decision| Potential benefit| Potential drawback
Removing numerical information| Reduces irrelevant numerical features| May remove prices   dates or quantities
Normalizing unusual text formats| Creates more consistent input| May hide differences between original expressions
Removing domain-specific terms| May simplify the dataset| May remove important technical vocabulary
Combining preprocessing methods| May improve model performance| Interactions between methods may produce unexpected results
Keeping original text| Preserves original information| May retain noise or inconsistencies

Preprocessing methods should be compared experimentally. Researchers can evaluate different preprocessing pipelines using the same dataset, model architecture, training conditions, and evaluation metrics to determine which approach performs best.

Q6. How can regular expressions be used to identify specific patterns in text data?

Regular expressions (Regex) can be used to identify patterns in unstructured text, including identifiers, dates, email addresses, numbers, and other structured elements.

Pattern| Purpose| Example
"\b[A-Z]{2,5}-\d{3,6}\b"| Identifies reference IDs| AB-2024
"\b\d{4}-\d{2}-\d{2}\b"| Identifies dates in YYYY-MM-DD format| 2026-11-22
"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"| Identifies email addresses| student@example.com
"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"| Identifies numbers with commas or decimals| 12,500.75

Regex can be used to extract, identify, validate, or selectively replace text patterns. It is useful when a dataset contains predictable structures that can be described using rules.

Q7. Case Study: How preprocessing affects Transformer and traditional machine learning models

   Research paper:

Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. Information Systems, 121, 102342.

Research objective:

The researchers investigated whether preprocessing still affects classification performance when using modern Transformer models and traditional machine learning models.

Methodology

The researchers tested preprocessing techniques and combinations of these techniques across four publicly available datasets from different domains. They evaluated nine machine learning models, including Transformer-based architectures.

Findings:

Preprocessing can significantly influence the performance of modern Transformer models. Different datasets and models respond differently to preprocessing. In certain experiments, preprocessing choices produced accuracy differences of up to 25%. A traditional Naïve Bayes classifier with an appropriate preprocessing strategy outperformed a Transformer in one reported comparison. The best preprocessing strategy depends on the task, dataset, and model.

Conclusion:

The researchers concluded that preprocessing remains an important part of NLP. Modern models do not eliminate the need to carefully evaluate text preparation methods. Researchers should select preprocessing strategies according to experimental evidence instead of applying the same techniques to every dataset.

References

1. Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. Information Systems, 121, 102342. https://doi.org/10.1016/j.is.2023.102342

2. Goldman, O., et al. (2024). Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.134/

3. Schmidt, C. W., et al. (2024). Tokenization Is More Than Compression. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.40/

4. Hugging Face. Normalization and pre-tokenization. https://huggingface.co/docs/course/main/chapter6/4
