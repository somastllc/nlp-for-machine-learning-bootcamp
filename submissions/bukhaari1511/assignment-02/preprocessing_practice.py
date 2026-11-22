
# ============================================================
# Assignment 2 - Text Preprocessing
# NLP for Machine Learning Bootcamp
# Student: bukhaari1511
# ============================================================

import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ------------------------------------------------------------
# INPUT DATA
# ------------------------------------------------------------

T1 = "URGENT!!! You have WON a \u00A31,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]


# ============================================================
# TASK 1: LOWERCASE
# ============================================================

print("\n" + "=" * 70)
print("TASK 1: LOWERCASE")
print("=" * 70)

print("Before:")
print(T1)

lowercase_T1 = T1.lower()

print("\nAfter:")
print(lowercase_T1)


# ============================================================
# TASK 2: WHITESPACE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 70)

print("Before:")
print(repr(T4))

normalized_T4 = re.sub(r"\s+", " ", T4).strip()

print("\nAfter:")
print(repr(normalized_T4))


# ============================================================
# TASK 3: REGEX CLEANING
# ============================================================

print("\n" + "=" * 70)
print("TASK 3: REGEX CLEANING")
print("=" * 70)

pattern = r"[^a-z\s]"

regex_result = re.sub(pattern, "", lowercase_T1)

deleted_characters = re.findall(pattern, lowercase_T1)

print("Original lowercase text:")
print(lowercase_T1)

print("\nPattern:", pattern)
print("After regex cleaning:")
print(regex_result)

print("\nDeleted characters:")
print(deleted_characters)

# Custom regex rules: replace instead of delete

custom_result = lowercase_T1

# Replace phone number
custom_result = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", custom_result)

# Replace money amount
custom_result = re.sub(r"£[\d,]+(?:\.\d{2})?", "MONEY", custom_result)

# Replace three or more exclamation marks with one
custom_result = re.sub(r"!{3,}", "!", custom_result)

print("\nCustom regex replacement:")
print(custom_result)


# ============================================================
# TASK 4: TOKENIZATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 4: TOKENIZATION")
print("=" * 70)

print("T2 original:")
print(T2)

split_tokens = T2.split()
word_tokens = word_tokenize(T2)

print("\nUsing split():")
print(split_tokens)
print("Token count:", len(split_tokens))

print("\nUsing word_tokenize():")
print(word_tokens)
print("Token count:", len(word_tokens))

# Tokenize T4
T4_tokens = word_tokenize(normalized_T4)

print("\nT4 tokens:")
print(T4_tokens)

print("T4 token count:", len(T4_tokens))
print("T4 vocabulary size:", len(set(T4_tokens)))


# ============================================================
# TASK 5: STOPWORDS AND NEGATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 5: STOPWORDS AND NEGATION")
print("=" * 70)

T2_tokens = word_tokenize(T2)

english_stopwords = set(stopwords.words("english"))

# Default stopword removal
default_filtered = [
    token for token in T2_tokens
    if token.lower() not in english_stopwords
]

print("Original tokens:")
print(T2_tokens)

print("\nAfter default stopword removal:")
print(default_filtered)

# Custom stopword list that keeps negation words
negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

custom_filtered = [
    token for token in T2_tokens
    if token.lower() not in custom_stopwords
]

print("\nAfter custom stopword removal:")
print(custom_filtered)


# ============================================================
# TASK 6: STEMMING
# ============================================================

print("\n" + "=" * 70)
print("TASK 6: STEMMING")
print("=" * 70)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(
    f"{'Word':<15}"
    f"{'Porter':<15}"
    f"{'Lancaster':<15}"
    f"{'Snowball':<15}"
)

print("-" * 60)

stem_results = {
    "Porter": {},
    "Lancaster": {},
    "Snowball": {}
}

for word in words:

    porter_stem = porter.stem(word)
    lancaster_stem = lancaster.stem(word)
    snowball_stem = snowball.stem(word)

    stem_results["Porter"][word] = porter_stem
    stem_results["Lancaster"][word] = lancaster_stem
    stem_results["Snowball"][word] = snowball_stem

    print(
        f"{word:<15}"
        f"{porter_stem:<15}"
        f"{lancaster_stem:<15}"
        f"{snowball_stem:<15}"
    )

# Find different words merged into the same stem
print("\nDifferent words merged into the same stem:")

for algorithm, results in stem_results.items():

    groups = {}

    for word, stem in results.items():
        groups.setdefault(stem, []).append(word)

    for stem, original_words in groups.items():

        if len(original_words) > 1:
            print(
                f"{algorithm}: {original_words} -> {stem}"
            )

print("\nExample of a stem that is not a real English word:")

print("university ->", porter.stem("university"))
print("universal  ->", porter.stem("universal"))


# ============================================================
# TASK 7: LEMMATIZATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 7: LEMMATIZATION")
print("=" * 70)

lemmatizer = WordNetLemmatizer()

# WordNet POS hints
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
    f"{'Word':<15}"
    f"{'Default':<15}"
    f"{'With POS hint':<20}"
)

print("-" * 50)

default_lemmas = {}
hinted_lemmas = {}

for word in words:

    default = lemmatizer.lemmatize(word)

    hinted = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    default_lemmas[word] = default
    hinted_lemmas[word] = hinted

    print(
        f"{word:<15}"
        f"{default:<15}"
        f"{hinted:<20}"
    )

print("\nWords changed only after giving a POS hint:")

for word in words:

    if default_lemmas[word] != hinted_lemmas[word]:
        print(
            f"{word}: {default_lemmas[word]} -> {hinted_lemmas[word]}"
        )


# ============================================================
# TASK 8: SENTENCE SPLITTING
# ============================================================

print("\n" + "=" * 70)
print("TASK 8: SENTENCE SPLITTING")
print("=" * 70)

print("Original T3:")
print(T3)

sentences = sent_tokenize(T3)

print("\nSentences:")

for index, sentence in enumerate(sentences, start=1):
    print(f"Sentence {index}: {sentence}")

print("\nTotal number of sentences:", len(sentences))


# ============================================================
# TASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION
# ============================================================

print("\n" + "=" * 70)
print("TASK 9: POS TAGGING AND NER")
print("=" * 70)

T3_tokens = word_tokenize(T3)

# POS tagging
tagged_words = pos_tag(T3_tokens)

print("POS Tags:")
print(tagged_words)

# Named Entity Recognition
named_entities = ne_chunk(tagged_words)

print("\nNamed Entity Tree:")
print(named_entities)

# Extract entities
entity_rows = []

for subtree in named_entities:

    if hasattr(subtree, "label"):

        entity_name = " ".join(
            word for word, tag in subtree.leaves()
        )

        entity_label = subtree.label()

        entity_rows.append((entity_name, entity_label))

print("\nEntity Table:")
print(f"{'Entity':<25}{'Label':<20}")
print("-" * 45)

for entity, label in entity_rows:
    print(f"{entity:<25}{label:<20}")

# Check important expected entities
expected_entities = [
    ("Mr. Smith", "PERSON", ["Smith"]),
    ("London", "GPE", ["London"]),
    ("Friday", "DATE", ["Friday"]),
    ("Apple", "ORGANIZATION", ["Apple"])
]

print("\nExpected Entity Recognition Check:")
print(f"{'Entity':<20}{'Expected Label':<20}{'Recognized?':<15}")
print("-" * 55)

for name, expected_label, aliases in expected_entities:

    matches = [
        (entity, label)
        for entity, label in entity_rows
        if any(alias.lower() in entity.lower() for alias in aliases)
    ]

    recognized = bool(matches)

    print(
        f"{name:<20}"
        f"{expected_label:<20}"
        f"{'Yes' if recognized else 'No':<15}"
    )

    if matches:
        print("  Actual:", matches)


# ============================================================
# TASK 10: FULL PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("TASK 10: FULL PREPROCESSING PIPELINE")
print("=" * 70)


def run_pipeline(text, stopword_list, pipeline_name):

    print("\n" + "-" * 60)
    print(pipeline_name)
    print("-" * 60)

    # Stage 1: Lowercase
    stage1 = text.lower()

    print("\n1. Lowercase:")
    print(stage1)

    # Stage 2: Regex cleaning
    # Preserve apostrophes so that n't can remain a token
    stage2 = re.sub(r"[^a-z\s']", " ", stage1)
    stage2 = re.sub(r"\s+", " ", stage2).strip()

    print("\n2. Regex cleaning:")
    print(stage2)

    # Stage 3: Tokenization
    stage3 = word_tokenize(stage2)

    print("\n3. Tokenization:")
    print(stage3)

    # Stage 4: Stopword removal
    stage4 = [
        token for token in stage3
        if token.lower() not in stopword_list
    ]

    print("\n4. Stopword removal:")
    print(stage4)

    # Stage 5: Lemmatization
    stage5 = [
        lemmatizer.lemmatize(token)
        for token in stage4
    ]

    print("\n5. Lemmatization:")
    print(stage5)

    return stage5


# Pipeline 1: Default NLTK stopwords
pipeline_default = run_pipeline(
    T2,
    english_stopwords,
    "PIPELINE 1: DEFAULT NLTK STOPWORDS"
)

# Pipeline 2: Negation-keeping stopwords
pipeline_custom = run_pipeline(
    T2,
    custom_stopwords,
    "PIPELINE 2: CUSTOM NEGATION-KEEPING STOPWORDS"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nDefault pipeline result:")
print(pipeline_default)

print("\nNegation-keeping pipeline result:")
print(pipeline_custom)

print("\nWhich pipeline keeps the customer's complaint?")

print("Negation-keeping pipeline preserves important negation words.")
print("This helps retain the negative meaning of the review.")

print("\nALL TASKS COMPLETED!")
print("=" * 70)
