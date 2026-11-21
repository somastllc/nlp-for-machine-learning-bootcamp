
# ============================================================
# NLP ASSIGNMENT 02 - PART B
# Text Preprocessing with NLTK
# ============================================================

import re
import nltk
import sys

sys.stdout.reconfigure(encoding="utf-8")

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ------------------------------------------------------------
# Texts and words
# ------------------------------------------------------------

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]


# ============================================================
# 1) Lowercasing
# ============================================================

print("\n=== 1) Lowercasing ===")

print("Before:", T1)

lowered = T1.lower()

print("After: ", lowered)


# ============================================================
# 2) Whitespace normalization
# ============================================================

print("\n=== 2) Whitespace normalization ===")

print("Before:", repr(T4))

normalized = " ".join(T4.split())

print("After: ", repr(normalized))


# ============================================================
# 3) Regex cleaning
# ============================================================

print("\n=== 3) Regex cleaning ===")

# Apply the lesson regex pattern to lowercase T1

print("Before:", lowered)

cleaned = re.sub(r"[^a-z\s]", " ", lowered)

print("After: ", cleaned)

# Assignment requirement:
# Find the characters deleted by the regex

deleted = re.findall(r"[^a-z\s]", lowered)

print("Deleted characters:", deleted)

# Custom regex replacement
# Replace money, phone number and !!!

replaced = lowered

replaced = re.sub(r"£[\d,]+", "MONEY", replaced)

replaced = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", replaced)

replaced = re.sub(r"!{3,}", "!", replaced)

print("\nCustom regex replacement:")

print("Before:", lowered)

print("After: ", replaced)


# ============================================================
# 4) Tokenization
# ============================================================

print("\n=== 4) Tokenization ===")

# Tokenization using split()

tokens_split = T2.split()

print("Using split():", tokens_split)
print("Token count:", len(tokens_split))

# Tokenization using word_tokenize()

tokens = word_tokenize(T2)

print("\nUsing word_tokenize():", tokens)
print("Token count:", len(tokens))

# Tokenization of T4

tokens_T4 = word_tokenize(T4)

print("\nT4 tokens:", tokens_T4)
print("T4 token count:", len(tokens_T4))

# Vocabulary size

vocabulary = set(tokens_T4)

print("T4 vocabulary:", vocabulary)
print("Vocabulary size:", len(vocabulary))


# ============================================================
# 5) Stopwords and negation
# ============================================================

print("\n=== 5) Stopwords and negation ===")

tokens = word_tokenize(T2)

print("Original tokens:", tokens)

# Remove stopwords

stop_words = set(stopwords.words("english"))

without_stopwords = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("Without stopwords:", without_stopwords)

# Keep negation word

stop_words_keep_not = set(stopwords.words("english"))

stop_words_keep_not.discard("not")
stop_words_keep_not.discard("n't")
stop_words_keep_not.discard("no")
stop_words_keep_not.discard("never")

keeping_not = [
    word for word in tokens
    if word.lower() not in stop_words_keep_not
]

print("Keeping negation:", keeping_not)


# ============================================================
# 6) Comparing stemmers
# ============================================================

print("\n=== 6) Comparing stemmers ===")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("Word | Porter | Lancaster | Snowball")

for word in words:
    print(
        word,
        "|", porter.stem(word),
        "|", lancaster.stem(word),
        "|", snowball.stem(word)
    )

# Find words that Porter Stemmer merges into the same stem

print("\nWords with the same Porter stem:")

for word1 in words:
    for word2 in words:
        if word1 != word2:
            if porter.stem(word1) == porter.stem(word2):
                print(
                    word1,
                    "and",
                    word2,
                    "->",
                    porter.stem(word1)
                )


# ============================================================
# 7) Lemmatization with a part-of-speech hint
# ============================================================

print("\n=== 7) Lemmatization ===")

lemmatizer = WordNetLemmatizer()

# POS hints required by the assignment

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

print("Word | Default | With POS hint")

for word in words:

    default_result = lemmatizer.lemmatize(word)

    pos_result = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    print(word, "|", default_result, "|", pos_result)


# ============================================================
# 8) Sentence splitting
# ============================================================

print("\n=== 8) Sentence splitting ===")

print("Before:", T3)

sentences = sent_tokenize(T3)

print("After:")

for sentence in sentences:
    print(sentence)

print("Number of sentences:", len(sentences))


# ============================================================
# 9) POS tagging and Named Entity Recognition
# ============================================================

print("\n=== 9) POS tagging ===")

tokens = word_tokenize(T3)

tagged = pos_tag(tokens)

print(tagged)


print("\n=== Named Entity Recognition ===")

entities = ne_chunk(tagged)

print(entities)

print("\nExtracted entities:")

for entity in entities:
    if hasattr(entity, "label"):
        print(
            " ".join(word for word, tag in entity.leaves()),
            ":", entity.label()
        )


# ============================================================
# 10) Full pipeline
# lowercase -> regex -> tokenize -> stopwords -> lemmatize
# ============================================================

print("\n=== 10) Full preprocessing pipeline ===")

sample = T2

print("Original:", sample)


# ------------------------------------------------------------
# Pipeline 1: Default stopwords
# ------------------------------------------------------------

print("\n--- Pipeline 1: Default stopwords ---")

# Step 1: Lowercase

step_lower = sample.lower()

print("Lowercase:", step_lower)

# Step 2: Regex cleaning

step_clean = re.sub(r"[^a-z\s]", " ", step_lower)

print("Regex cleaning:", step_clean)

# Step 3: Tokenization

step_tokens = word_tokenize(step_clean)

print("Tokenization:", step_tokens)

# Step 4: Stopword removal

step_no_stop = [
    word for word in step_tokens
    if word not in stop_words
]

print("Without stopwords:", step_no_stop)

# Step 5: Lemmatization

step_lemmas = [
    lemmatizer.lemmatize(word)
    for word in step_no_stop
]

print("Lemmatization:", step_lemmas)


# ------------------------------------------------------------
# Pipeline 2: Keeping negation
# ------------------------------------------------------------

print("\n--- Pipeline 2: Keeping negation ---")

# Step 1: Lowercase

step_lower = sample.lower()

print("Lowercase:", step_lower)

# Step 2: Regex cleaning

step_clean = re.sub(r"[^a-z\s]", " ", step_lower)

print("Regex cleaning:", step_clean)

# Step 3: Tokenization

step_tokens = word_tokenize(step_clean)

print("Tokenization:", step_tokens)

# Step 4: Stopword removal while keeping negation

step_keep_not = [
    word for word in step_tokens
    if word not in stop_words_keep_not
]

print("Keeping negation:", step_keep_not)

# Step 5: Lemmatization

step_lemmas = [
    lemmatizer.lemmatize(word)
    for word in step_keep_not
]

print("Lemmatization:", step_lemmas)



