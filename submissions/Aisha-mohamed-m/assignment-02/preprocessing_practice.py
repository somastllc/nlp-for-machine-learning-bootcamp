# Assignment 2 — Preprocessing Practice



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


# ============================================================
# TEXTS
# ============================================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."


# ============================================================
# WORDS FOR STEMMING / LEMMATIZATION
# ============================================================

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


# ============================================================
# 1) LOWERCASE T1
# ============================================================

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
    "Lowercasing removes capitalization information. "
    "For spam detection, words such as URGENT, WON, and NOW "
    "being written in all capitals can be a useful spam signal."
)


# ============================================================
# 2) WHITESPACE NORMALIZATION T4
# ============================================================

print("\n" + "=" * 70)
print("2) WHITESPACE NORMALIZATION T4")
print("=" * 70)

normalized_T4 = " ".join(T4.split())

print("Before:")
print(repr(T4))

print("\nAfter:")
print(repr(normalized_T4))


# ============================================================
# 3) REGEX CLEANING
# ============================================================

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
    "This regex deletes punctuation, numbers, currency symbols, "
    "and other non-letter characters. This can remove useful spam signals "
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


# ============================================================
# 4) TOKENIZATION
# ============================================================

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


# ============================================================
# 5) STOPWORDS AND NEGATION
# ============================================================

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
# keep not, n't, no, never

custom_stop_words = stop_words.copy()

for word in ["not", "n't", "no", "never"]:
    custom_stop_words.discard(word)

custom_stopword_result = [
    token
    for token in T2_tokens
    if token.lower() not in custom_stop_words
]

print("\nCustom stopword removal:")
print("Keeping: not, n't, no, never")

print(custom_stopword_result)

print("\nWhy?")
print(
    "Negation words can change sentiment and meaning. "
    "For example, 'happy' and 'not happy' have opposite meanings. "
    "Removing 'not' can therefore damage a sentiment or complaint classifier."
)


# ============================================================
# 6) STEMMING
# ============================================================

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
    "Some stemmers may produce forms such as 'univers', "
    "'connect', or other shortened forms that are not normal English words."
)

print("\nConclusion:")
print(
    "Porter and Snowball are generally less aggressive than Lancaster. "
    "Lancaster can shorten words more aggressively. "
    "RegexpStemmer follows only the suffix rules we specify."
)


# ============================================================
# 7) LEMMATIZATION
# ============================================================

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


# ============================================================
# 8) SENTENCE TOKENIZATION
# ============================================================

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


# ============================================================
# 9) POS TAGGING + NAMED ENTITY RECOGNITION
# ============================================================

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


# ============================================================
# 10) FULL PREPROCESSING PIPELINE ON T2
# ============================================================

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


print("\nComplaint meaning:")
print(
    "The complaint contains negation such as 'don't' and 'not'. "
    "Removing negation words can make the complaint meaning less clear. "
    "Therefore, a sentiment or complaint classifier should usually preserve "
    "important negation words."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Preprocessing is task-dependent. "
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
