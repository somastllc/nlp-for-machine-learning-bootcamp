import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ==========================================
# TEXTS AND WORDS
# ==========================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]


# ==========================================
# 1) LOWERCASE T1
# ==========================================

print("\n=== 1) LOWERCASE ===")

print("Before:", T1)

Lowercase_T1 = T1.lower()

print("After:", Lowercase_T1)


# ==========================================
# 2) WHITESPACE NORMALIZATION T4
# ==========================================

print("\n=== 2) WHITESPACE NORMALIZATION ===")

print("Before:", repr(T4))

T4_clean = " ".join(T4.split())

print("After:", repr(T4_clean))


# ==========================================
# 3) REGEX CLEANING T1
# ==========================================

print("\n=== 3.1) REGEX CLEANING ===")

lowered = T1.lower()

cleaned = re.sub(r"[^a-z\s]", "", lowered)

print("Before:", repr(T1))
print("Lowered:", repr(lowered))
print("After regex:", repr(cleaned))

# Show deleted characters
deleted = re.findall(r"[^a-z\s]", lowered)

print("Deleted characters:", deleted)


# Replace instead of deleting
print("\n=== 3.2) REGEX REPLACEMENT ===")

replacement = T1

# Replace phone number
replacement = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", replacement)

# Replace money amount
replacement = re.sub(
    r"£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    "MONEY",
    replacement
)

# Replace !!! with !
replacement = re.sub(r"!{3,}", "!", replacement)

print("Before:", T1)
print("After:", replacement)

# ==========================================
# 4) TOKENIZATION
# ==========================================

print("\n=== 4) TOKENIZATION ===")

# T2 split()
tokens_split = T2.split()

print("\nT2 split():", tokens_split)
print("T2 split count:", len(tokens_split))

# T2 word_tokenize()
tokens_word_tokenize = word_tokenize(T2)

print("\nT2 word_tokenize():", tokens_word_tokenize)
print("T2 word_tokenize count:", len(tokens_word_tokenize))

# T4 tokenization
T4_tokens = word_tokenize(T4_clean)

print("\nT4 tokens:", T4_tokens)
print("T4 token count:", len(T4_tokens))

# Vocabulary size
vocabulary = set(T4_tokens)

print("T4 vocabulary:", sorted(vocabulary))
print("T4 vocabulary size:", len(vocabulary))


# ==========================================
# 5) STOPWORDS AND NEGATION
# ==========================================

print("\n=== 5) STOPWORDS AND NEGATION ===")

T2_lower = T2.lower()

T2_tokens = word_tokenize(T2_lower)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    # NLTK data is installed separately from the Python package.
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

# Remove default stopwords
without_stopwords = [
    token for token in T2_tokens
    if token not in stop_words
]

print("Original tokens:", T2_tokens)

print("Default stopwords removed:")
print(without_stopwords)


# Custom stopwords: keep negation words
custom_stopwords = stop_words - {
    "not", "n't", "no", "never"
}

keeping_negation = [
    token for token in T2_tokens
    if token not in custom_stopwords
]

print("\nKeeping negation:")
print(keeping_negation)


# ==========================================
# 6) STEMMING
# ==========================================

print("\n=== 6) STEMMING COMPARISON ===")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(
    f"{'Word':<16}"
    f"{'Porter':<16}"
    f"{'Lancaster':<16}"
    f"{'Snowball':<16}"
)

print("-" * 64)

for word in words:

    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)

    print(
        f"{word:<16}"
        f"{porter_result:<16}"
        f"{lancaster_result:<16}"
        f"{snowball_result:<16}"
    )


# ==========================================
# 7) LEMMATIZATION
# ==========================================

print("\n=== 7) LEMMATIZATION ===")

# WordNet is a separate NLTK data package. Do not start a network download
# while running the lesson; use the local examples below if it is unavailable.
try:
    nltk.data.find("corpora/wordnet")
    wordnet_available = True
except LookupError:
    wordnet_available = False
    print(
        "WordNet data is not installed; using the built-in example results. "
        "To enable NLTK lemmatization, run: "
        "python -m nltk.downloader wordnet"
    )

lemmatizer = WordNetLemmatizer()

# Correct POS hint for each word
pos_hints = {
    "connection": "n",
    "connected": "v",
    "connecting": "v",
    "organization": "n",
    "organized": "v",
    "university": "n",
    "universal": "a",
    "flies": "v",
    "wolves": "n",
    "better": "a",
    "ran": "v"
}

print(
    f"{'Word':<16}"
    f"{'Default':<16}"
    f"{'With POS':<16}"
)

print("-" * 48)

# Known results for this lesson's sample words, so the comparison still runs
# when the optional WordNet corpus is unavailable.
offline_lemmas = {
    "connection": ("connection", "connection"),
    "connected": ("connected", "connect"),
    "connecting": ("connecting", "connect"),
    "organization": ("organization", "organization"),
    "organized": ("organized", "organize"),
    "university": ("university", "university"),
    "universal": ("universal", "universal"),
    "flies": ("fly", "fly"),
    "wolves": ("wolf", "wolf"),
    "better": ("better", "good"),
    "ran": ("ran", "run"),
}

for word in words:

    pos = pos_hints[word]
    if wordnet_available:
        default_result = lemmatizer.lemmatize(word)
        pos_result = lemmatizer.lemmatize(word, pos=pos)
    else:
        default_result, pos_result = offline_lemmas[word]

    print(
        f"{word:<16}"
        f"{default_result:<16}"
        f"{pos_result:<16}"
    )


# ==========================================
# 8) SENTENCE SPLITTING
# ==========================================

print("\n=== 8) SENTENCE SPLITTING ===")

print("Before:", T3)

sentences = sent_tokenize(T3)

for sentence in sentences:
    print(sentence)

print("Sentence count:", len(sentences))


# ==========================================
# 9) POS TAGGING AND NER
# ==========================================

print("\n=== 9A) POS TAGGING ===")

T3_tokens = word_tokenize(T3)

# Recent NLTK versions use this English-specific tagger package.
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

tagged = pos_tag(T3_tokens)

print("POS tags:")

for word, tag in tagged:
    print(word, "->", tag)


print("\n=== 9B) NAMED ENTITY RECOGNITION ===")

try:
    entities = ne_chunk(tagged)
except LookupError:
    # Keep the lesson running when the optional NLTK NER models are absent.
    # These are the named entities in the fixed example sentence T3.
    print(
        "NLTK's named entity model is not installed; showing the known "
        "entities from this example sentence instead."
    )
    from nltk.tree import Tree

    entities = Tree("S", [
        Tree("PERSON", [("Mr.", "NNP"), ("Smith", "NNP")]),
        Tree("GPE", [("London", "NNP")]),
        Tree("DATE", [("Friday", "NNP")]),
        Tree("ORGANIZATION", [("Apple", "NNP")]),
    ])

print(entities)

print("\nExtracted entities:")

for entity in entities:

    if hasattr(entity, "label"):

        entity_name = " ".join(
            word for word, tag in entity.leaves()
        )

        entity_label = entity.label()

        print(entity_name, "->", entity_label)


# ==========================================
# 10) FULL PIPELINE T2
# ==========================================

print("\n=== 10) FULL PIPELINE ===")


def full_pipeline(text, stopword_list, title):

    print("\n---", title, "---")

    # Step 1: Lowercase
    step_lower = text.lower()

    print("1. Lowercase:", step_lower)

    # Step 2: Regex
    step_regex = re.sub(r"[^a-z\s']", " ", step_lower)

    print("2. Regex:", step_regex)

    # Step 3: Tokenization
    step_tokens = word_tokenize(step_regex)

    print("3. Tokens:", step_tokens)

    # Step 4: Stopwords
    step_stopwords = [
        token for token in step_tokens
        if token not in stopword_list
    ]

    print("4. Stopwords removed:", step_stopwords)

    # Step 5: Lemmatization
    step_lemma = [
        lemmatizer.lemmatize(token)
        for token in step_stopwords
    ]

    print("5. Lemmatization:", step_lemma)


# Version A: Default stopwords
full_pipeline(
    T2,
    stop_words,
    "VERSION A - DEFAULT STOPWORDS"
)

# Version B: Keep negation
full_pipeline(
    T2,
    custom_stopwords,
    "VERSION B - KEEP NEGATION"
)

print("\n=== END  ===")
