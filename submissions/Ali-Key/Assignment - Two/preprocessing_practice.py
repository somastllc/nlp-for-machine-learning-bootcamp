# Assignment 2 — Preprocessing Practice
# Part B – Practical

import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import (
    LancasterStemmer,
    PorterStemmer,
    SnowballStemmer,
    RegexpStemmer,
    WordNetLemmatizer,
)
from nltk.tokenize import sent_tokenize, word_tokenize


# --------------------------------
# TEXTS
# Discussion: What are the differences between the texts?
# --------------------------------

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."


print("\nDiscussion: Differences Between the Texts")

print(
    "T1 contains uppercase words, repeated punctuation, "
    "a phone number, and a money amount. It looks like a spam message."
)

print(
    "T2 contains negative sentiment, contractions, and an emoji. "
    "It expresses dissatisfaction with a product."
)

print(
    "T3 contains named entities such as a person's name, "
    "a location, an organization, and time information."
)

print(
    "T4 contains unnecessary spaces and punctuation. "
    "It needs whitespace normalization before further processing."
)

print(
    "These examples show that different types of text may need "
    "different preprocessing methods."
)


# --------------------------------
# WORDS FOR STEMMING / LEMMATIZATION
# Discussion: How do word forms change?
# --------------------------------

words = [
    "connection",
    "connected",
    "connecting",
    "organization",
    "organized",
    "university",
    "universal",
    "flies",
    "wolves",
    "better",
    "ran",
]

print("\nDiscussion: Words for Stemming and Lemmatization")

print(
    "These words contain different forms and grammatical categories. "
    "Some have similar meanings or word roots, while others have "
    "different meanings despite having similar spellings."
)

print(
    "Stemming reduces words by removing suffixes, while lemmatization "
    "uses vocabulary and grammar rules to find dictionary base forms."
)


# --------------------------------
# 1) LOWERCASE T1

# --------------------------------

print("\n" + "=" * 70)
print("1) LOWERCASE T1")
print("=" * 70)

lower_T1 = T1.lower()

print("Before:")
print(repr(T1))

print("\nAfter:")
print(repr(lower_T1))

print("\nDiscussion:")

print(
    "Lowercasing converts uppercase letters into lowercase letters. "
    "For example, 'HELLO' becomes 'hello'."
)

print(
    "It helps an NLP system treat words with different capitalization "
    "as the same word."
)

print(
    "However, it removes capitalization information. "
    "In spam detection, uppercase words such as URGENT, WON, and NOW "
    "can be useful signals."
)

print(
    "Therefore, lowercasing is useful when capitalization is not important, "
    "but it should be used carefully depending on the task."
)


# --------------------------------
# 2) WHITESPACE NORMALIZATION T4
# --------------------------------

print("\n" + "=" * 70)
print("2) WHITESPACE NORMALIZATION T4")
print("=" * 70)

normalized_T4 = " ".join(T4.split())

print("Before:")
print(repr(T4))

print("\nAfter:")
print(repr(normalized_T4))

print("\nDiscussion:")

print(
    "Whitespace normalization removes unnecessary spaces "
    "and replaces multiple spaces with a single space."
)

print(
    "For example, 'The   delivery   was fast' becomes "
    "'The delivery was fast'."
)

print(
    "It makes text cleaner and easier to process. "
    "It does not change the original meaning of the sentence."
)


# --------------------------------
# 3) REGEX CLEANING
# --------------------------------

print("\n" + "=" * 70)
print("3) REGEX CLEANING")
print("=" * 70)

lower_T1 = T1.lower()

regex_cleaned = re.sub(r"[^a-z\s]", "", lower_T1)

print("Before:")
print(repr(lower_T1))

print("\nAfter using [^a-z\\s]:")
print(repr(regex_cleaned))


# Find the characters deleted by the regex
deleted_characters = [
    char for char in lower_T1
    if not re.match(r"[a-z\s]", char)
]

print("\nDeleted characters:")
print(deleted_characters)

print("\nImportant:")

print(
    "This regex keeps lowercase English letters and whitespace. "
    "It removes punctuation, numbers, currency symbols, emojis, "
    "and other non-letter characters."
)

print(
    "This can remove useful information from spam messages, "
    "such as phone numbers, money amounts, and repeated exclamation marks."
)


# ------------------------------------------------------------
# Own replacement rules
# phone number -> PHONE_NUMBER
# money amount -> MONEY
# !!! -> !
# ------------------------------------------------------------

replacement_T1 = lower_T1

replacement_T1 = re.sub(
    r"\b\d{10}\b",
    "PHONE_NUMBER",
    replacement_T1
)

replacement_T1 = re.sub(
    r"£[\d,]+",
    "MONEY",
    replacement_T1
)

replacement_T1 = re.sub(
    r"!{3,}",
    "!",
    replacement_T1
)

print("\nOwn replacement rules:")
print("Phone number -> PHONE_NUMBER")
print("Money amount -> MONEY")
print("!!! -> !")

print("\nAfter replacement:")
print(repr(replacement_T1))

print("\nDiscussion:")

print(
    "Regular expressions (Regex) help find, remove, or replace "
    "specific patterns in text."
)

print(
    "Instead of deleting important information, we can replace "
    "phone numbers and money amounts with labels."
)

print(
    "This keeps useful information while making the text easier "
    "for an NLP system to process."
)


# --------------------------------
# 4) TOKENIZATION
# --------------------------------

print("\n" + "=" * 70)
print("4) TOKENIZATION")
print("=" * 70)

# T2 split()
split_T2 = T2.split()

# T2 word_tokenize()
tokenized_T2 = word_tokenize(T2)

print("T2:")
print(repr(T2))

print("\nUsing split():")
print(split_T2)

print("\nUsing word_tokenize():")
print(tokenized_T2)

print("\nToken counts:")
print("split() count:", len(split_T2))
print("word_tokenize() count:", len(tokenized_T2))


# T4 token count and vocabulary size
T4_tokens = word_tokenize(normalized_T4)

T4_vocabulary = set(
    token.lower()
    for token in T4_tokens
    if re.search(r"[A-Za-z]", token)
)

print("\nT4 tokens:")
print(T4_tokens)

print("\nT4 token count:")
print(len(T4_tokens))

print("\nT4 vocabulary:")
print(sorted(T4_vocabulary))

print("\nT4 vocabulary size:")
print(len(T4_vocabulary))

print("\nDiscussion:")

print(
    "Tokenization divides text into smaller units called tokens. "
    "These units can be words or punctuation marks."
)

print(
    "The split() method separates text using whitespace. "
    "It is simple but does not handle punctuation and contractions "
    "as carefully as an NLP tokenizer."
)

print(
    "The word_tokenize() function uses language rules to separate "
    "words and punctuation into individual tokens."
)

print(
    "Tokenization helps NLP systems process and analyze text."
)


# --------------------------------
# 5) STOPWORDS AND NEGATION
# --------------------------------

print("\n" + "=" * 70)
print("5) STOPWORDS AND NEGATION")
print("=" * 70)

stop_words = set(stopwords.words("english"))

T2_tokens = word_tokenize(T2)

default_stopword_result = [
    token
    for token in T2_tokens
    if token.lower() not in stop_words
]

print("T2 tokens:")
print(T2_tokens)

print("\nDefault NLTK stopword removal:")
print(default_stopword_result)


# Custom stopword list:
# keep not, n't, no, never, don't

custom_stop_words = stop_words.copy()

for word in ["not", "n't", "no", "never", "don't"]:
    custom_stop_words.discard(word)

custom_stopword_result = [
    token
    for token in T2_tokens
    if token.lower() not in custom_stop_words
]

print("\nCustom stopword removal:")
print("Keeping: not, n't, no, never, don't")

print(custom_stopword_result)

print("\nDiscussion:")

print(
    "Stopwords are common words that are sometimes removed "
    "during text preprocessing."
)

print(
    "Examples include 'the', 'is', 'and', and 'a'. "
    "Removing them can reduce unnecessary words."
)

print(
    "However, negation words are important because they can change "
    "the meaning of a sentence."
)

print(
    "For example, 'I am happy' and 'I am not happy' "
    "have different meanings."
)

print(
    "If we remove 'not', the negative meaning may be lost."
)

print(
    "Therefore, sentiment analysis and complaint detection should "
    "usually preserve important negation words such as 'not', "
    "'no', 'never', and 'n't'."
)


# --------------------------------
# 6) STEMMING
# --------------------------------

print("\n" + "=" * 70)
print("6) STEMMING COMPARISON")
print("=" * 70)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

regexp = RegexpStemmer(
    r"ing$|ed$|s$|ion$|ation$|ly$",
    min=4
)

print(
    f"{'Word':<15}"
    f"{'Porter':<15}"
    f"{'Lancaster':<15}"
    f"{'Snowball':<15}"
    f"{'Regexp':<15}"
)

print("-" * 75)

for word in words:

    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)
    regexp_result = regexp.stem(word)

    print(
        f"{word:<15}"
        f"{porter_result:<15}"
        f"{lancaster_result:<15}"
        f"{snowball_result:<15}"
        f"{regexp_result:<15}"
    )


print("\nMerged pairs / similar stems:")

stem_groups = {}

for word in words:

    stem = porter.stem(word)

    if stem not in stem_groups:
        stem_groups[stem] = []

    stem_groups[stem].append(word)


for stem, group in stem_groups.items():

    if len(group) > 1:
        print(stem, "->", group)


print("\nNon-word stems:")

print(
    "Some stemmers produce shortened forms that are not real English words. "
    "For example, 'universal' may become 'univers'."
)

print("\nDiscussion:")

print(
    "Stemming reduces words to their root-like forms by removing "
    "common prefixes or suffixes."
)

print(
    "Porter Stemmer applies common suffix-removal rules. "
    "Snowball Stemmer uses an improved version of the stemming approach."
)

print(
    "Lancaster Stemmer is more aggressive and may shorten words more."
)

print(
    "RegexpStemmer follows only the suffix rules specified by the user."
)

print(
    "Some stemming results are not real dictionary words. "
    "Stemming focuses on reducing word forms rather than "
    "finding the correct dictionary word."
)


# --------------------------------
# 7) LEMMATIZATION
# --------------------------------

print("\n" + "=" * 70)
print("7) LEMMATIZATION")
print("=" * 70)

lemmatizer = WordNetLemmatizer()

print(
    f"{'Word':<15}"
    f"{'Default':<15}"
    f"{'Verb (v)':<15}"
    f"{'Noun (n)':<15}"
    f"{'Adjective (a)':<15}"
)

print("-" * 75)

for word in words:

    default = lemmatizer.lemmatize(word)
    verb = lemmatizer.lemmatize(word, pos="v")
    noun = lemmatizer.lemmatize(word, pos="n")
    adjective = lemmatizer.lemmatize(word, pos="a")

    print(
        f"{word:<15}"
        f"{default:<15}"
        f"{verb:<15}"
        f"{noun:<15}"
        f"{adjective:<15}"
    )


print("\nExamples of POS hints:")

print(
    "running:",
    lemmatizer.lemmatize("running"),
    "->",
    lemmatizer.lemmatize("running", pos="v")
)

print(
    "better:",
    lemmatizer.lemmatize("better"),
    "->",
    lemmatizer.lemmatize("better", pos="a")
)

print(
    "studies:",
    lemmatizer.lemmatize("studies"),
    "->",
    lemmatizer.lemmatize("studies", pos="v")
)


print("\nDiscussion:")

print(
    "Lemmatization converts words into their dictionary base forms, "
    "which are called lemmas."
)

print(
    "Unlike stemming, lemmatization uses vocabulary and language rules "
    "to produce meaningful words."
)

print(
    "For example, 'flies' can become 'fly', and 'wolves' can become 'wolf'."
)

print(
    "The result can depend on the Part of Speech (POS). "
    "For example, 'better' becomes 'good' when treated as an adjective."
)

print(
    "The default WordNet lemmatizer treats words as nouns. "
    "Providing a POS hint can produce a more suitable result."
)


# --------------------------------
# 8) SENTENCE TOKENIZATION
# --------------------------------

print("\n" + "=" * 70)
print("8) SENTENCE TOKENIZATION")
print("=" * 70)

sentences_T3 = sent_tokenize(T3)

print("T3:")
print(repr(T3))

print("\nSentences:")

for number, sentence in enumerate(sentences_T3, start=1):
    print(number, "->", sentence)

print("\nSentence count:")
print(len(sentences_T3))

print(
    "\nNote: sent_tokenize tries not to split at abbreviations "
    "such as Mr. and a.m."
)

print("\nDiscussion:")

print(
    "Sentence tokenization divides a paragraph into individual sentences."
)

print(
    "The sent_tokenize() function uses language rules to identify "
    "sentence boundaries."
)

print(
    "It tries to avoid splitting at common abbreviations "
    "such as 'Mr.' and 'a.m.'."
)

print(
    "This is useful when an NLP system needs to analyze "
    "sentences separately."
)


# --------------------------------
# 9) POS TAGGING + NAMED ENTITY RECOGNITION
# --------------------------------

print("\n" + "=" * 70)
print("9) POS TAGGING + NAMED ENTITY RECOGNITION")
print("=" * 70)

T3_tokens = word_tokenize(T3)

tagged_T3 = pos_tag(T3_tokens)

print("Tokens:")
print(T3_tokens)

print("\nPOS tags:")

for token, tag in tagged_T3:
    print(f"{token:<15} {tag}")


print("\nNamed Entity Recognition:")

ner_result = ne_chunk(tagged_T3)

print(ner_result)


print("\nExpected / important entities:")

print("PERSON: Mr. Smith")
print("GPE: London")
print("ORGANIZATION: Apple")


print(
    "\nNote: NLTK's English NER model can make mistakes. "
    "Always inspect the result instead of assuming every entity is correct."
)

print("\nDiscussion:")

print(
    "POS tagging identifies the grammatical role of each word "
    "in a sentence."
)

print(
    "Examples include nouns, verbs, adjectives, and proper nouns."
)

print(
    "Named Entity Recognition (NER) identifies important names "
    "such as people, locations, and organizations."
)

print(
    "In T3, Mr. Smith is a person, London is a location, "
    "and Apple refers to an organization in this example."
)

print(
    "These techniques help NLP systems understand sentence structure "
    "and identify important information."
)

print(
    "However, automatic tagging and entity recognition can make mistakes. "
    "Their results should always be checked."
)


# --------------------------------
# 10) FULL PREPROCESSING PIPELINE ON T2
# --------------------------------

print("\n" + "=" * 70)
print("10) FULL PIPELINE ON T2")
print("=" * 70)

print(
    "Pipeline:"
    " lowercase -> regex cleaning -> tokenize -> stopword removal "
    "-> lemmatization"
)


# ------------------------------------------------------------
# First version: default stopwords
# ------------------------------------------------------------

pipeline_lower = T2.lower()

pipeline_clean = re.sub(
    r"[^a-z\s']",
    "",
    pipeline_lower
)

pipeline_tokens = word_tokenize(pipeline_clean)

pipeline_no_stop = [
    token
    for token in pipeline_tokens
    if token.lower() not in stop_words
]

pipeline_lemmas = [
    lemmatizer.lemmatize(token)
    for token in pipeline_no_stop
]


print("\nOriginal:")
print(repr(T2))

print("\n1. Lowercase:")
print(repr(pipeline_lower))

print("\n2. Regex cleaned:")
print(repr(pipeline_clean))

print("\n3. Tokens:")
print(pipeline_tokens)

print("\n4. Stopwords removed:")
print(pipeline_no_stop)

print("\n5. Lemmatized:")
print(pipeline_lemmas)


# ------------------------------------------------------------
# Second version: preserve negation
# ------------------------------------------------------------

pipeline_keep_negation = [
    token
    for token in pipeline_tokens
    if token.lower() not in custom_stop_words
]

pipeline_keep_negation_lemmas = [
    lemmatizer.lemmatize(token)
    for token in pipeline_keep_negation
]

print("\nPipeline keeping negation words:")
print(pipeline_keep_negation_lemmas)


print("\nDiscussion:")

print(
    "A preprocessing pipeline combines several steps "
    "to prepare raw text for an NLP task."
)

print(
    "In this example, the pipeline uses lowercase conversion, "
    "regex cleaning, tokenization, stopword removal, and lemmatization."
)

print(
    "Each step has a different purpose. Lowercasing reduces "
    "capitalization differences, regex cleaning removes selected characters, "
    "tokenization separates words, stopword removal filters common words, "
    "and lemmatization finds dictionary base forms."
)

print(
    "However, removing negation words such as 'not' and 'don't' "
    "can change the meaning of a complaint."
)

print(
    "The second pipeline preserves important negation information. "
    "This can be more suitable for sentiment analysis and complaint detection."
)

print(
    "The best preprocessing pipeline depends on the NLP task "
    "and the information that needs to be preserved."
)


print("\nComplaint meaning:")

print(
    "The complaint contains negative expressions such as "
    "'don't think' and 'not happy'."
)

print(
    "Removing these expressions can make the complaint meaning less clear. "
    "Therefore, a sentiment or complaint classifier should usually preserve "
    "important negation words."
)


# --------------------------------
# FINAL SUMMARY
# --------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Preprocessing prepares raw text for NLP systems. "
    "Different techniques help clean, organize, and analyze text."
)

print(
    "Lowercasing and whitespace normalization can reduce unnecessary variation, "
    "but aggressive regex cleaning or stopword removal can delete useful information."
)

print(
    "For spam detection, preserve useful signals such as phone numbers, "
    "money amounts, capitalization, and repeated punctuation when appropriate."
)

print(
    "For sentiment or complaint detection, preserve negation words such as "
    "'not', 'no', 'never', and important forms of 'n't'."
)

print(
    "Stemming reduces words to root-like forms, while lemmatization "
    "finds meaningful dictionary base forms."
)

print(
    "The most suitable preprocessing methods depend on the purpose "
    "of the NLP application."
)
