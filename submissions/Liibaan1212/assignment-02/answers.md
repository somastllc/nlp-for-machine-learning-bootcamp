
# Part B – Practical Results

## Task 1 – Lowercasing

Lowercasing converts all uppercase letters into lowercase. It makes text more consistent, but it can remove useful information about emphasis and urgency. In T1, converting "URGENT!!!" and "WON" into lowercase removes signals that may help a spam detector identify suspicious messages.

## Task 2 – Whitespace Normalization

Whitespace normalization replaces unnecessary spaces with a single space and removes spaces at the beginning and end of the text. This makes text easier to tokenize and process. The script uses `repr()` to make the original whitespace differences visible.

## Task 3 – Regex Cleaning

The pattern `[^a-z\s]` removes every character that is not a lowercase English letter or whitespace. In T1, this removes exclamation marks, the pound symbol, digits, commas, and the full phone number. This demonstrates why deleting every non-letter character can damage important information.

The custom regex approach is safer because it replaces the phone number with `PHONE_NUMBER`, the money amount with `MONEY`, and repeated exclamation marks with one exclamation mark. It preserves the information category without retaining the original sensitive-looking number.

## Task 4 – Tokenization

The `split()` method separates text using whitespace, while NLTK's `word_tokenize()` also handles punctuation and contractions. Therefore, the two methods can produce different token counts. Vocabulary size is calculated by counting unique tokens in T4 after lowercasing them.

## Task 5 – Stopwords and Negation

Default stopword removal can remove words that are important for understanding sentiment. In T2, removing "not" changes the meaning of "Not happy" and can make the customer's negative opinion less clear. The custom stopword list protects `not`, `n't`, `no`, and `never`, allowing the system to retain important negative expressions.

## Task 6 – Stemming

The three algorithms use different suffix-removal rules, so their results are not always identical.

The following table must be copied directly from the STEMMING section of my generated `output.txt`:

[INSERT THE GENERATED STEMMING TABLE HERE]

Different original words can produce the same stem. For example, connection, connected, and connecting may be reduced to a common stem. Some resulting stems, such as `univers`, are not valid English dictionary words.

## Task 7 – Lemmatization

WordNetLemmatizer uses noun as its default part of speech. Providing the correct POS hint allows it to identify the intended grammatical form and produce more meaningful base words.

The expected important changes include:

- connected → connect
- connecting → connect
- organized → organize
- better → good
- ran → run

The exact comparison is available in the generated output.

## Task 8 – Sentence Splitting

The `sent_tokenize()` function uses NLTK's sentence tokenizer to identify sentence boundaries. T3 contains two sentences. The abbreviations "Mr." and "a.m." should not cause incorrect sentence boundaries in this example. The actual result is printed by the script.

## Task 9 – POS Tagging and Named Entity Recognition

POS tagging identifies the grammatical category of each token, such as noun, verb, adjective, and proper noun.

Named Entity Recognition identifies real-world entities, such as people, locations, and organizations.

### Entity evaluation table

Complete the returned-label column using the exact entities printed by your script.

| Entity | Expected label | Returned label | Correct? |
|---|---|---|---|
| Mr. Smith / Smith | PERSON | Check output | Check |
| London | GPE | Check output | Check |
| Apple | ORGANIZATION | Check output | Check |
| Friday | DATE | Check output | Check |
| 9 a.m. | TIME | Check output | Check |

The NLTK named entity chunker does not necessarily recognize every entity. In particular, dates and times may not be classified because its traditional named entity categories are not a complete temporal information extraction system.

Apple is especially important in this example because it refers to a company rather than a fruit. A model that does not recognize the company correctly demonstrates the limitations of automatic entity recognition.

## Task 10 – Full Preprocessing Pipeline

The first pipeline uses NLTK's default English stopwords, while the second preserves negation words.

The default pipeline can remove "not", changing the meaning of the customer's complaint. The custom pipeline preserves negation and therefore provides a more faithful representation of the original negative review.

The custom pipeline is preferable for sentiment analysis because preserving negation helps the system distinguish positive and negative opinions.

---

## Spam Detector: Recommended Preprocessing Steps

For a spam detector, I would use whitespace normalization, tokenization, carefully designed regular expressions, and selected text normalization techniques. T1 demonstrates why phone numbers and monetary amounts should be replaced with meaningful placeholders rather than deleted. I would also preserve capitalization, repeated punctuation, and suspicious words as potential spam signals, or extract them as separate features before normalization. I would avoid aggressive stopword removal and the pattern `[^a-z\s]` because they can remove important evidence such as currency symbols, phone numbers, and urgency indicators. The goal should be to remove irrelevant noise while preserving information that helps distinguish spam from legitimate messages.
