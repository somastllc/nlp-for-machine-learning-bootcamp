
import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# =====================================================
# DATA
# =====================================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# =====================================================
# TASK 1: LOWERCASE
# =====================================================

heading("TASK 1: LOWERCASE")

print("BEFORE:", T1)

lower_T1 = T1.lower()

print("AFTER:", lower_T1)


# =====================================================
# TASK 2: WHITESPACE NORMALIZATION
# =====================================================

heading("TASK 2: WHITESPACE NORMALIZATION")

print("BEFORE:", repr(T4))

clean_T4 = re.sub(r"\s+", " ", T4).strip()

print("AFTER:", repr(clean_T4))


# =====================================================
# TASK 3: REGEX CLEANING
# =====================================================

heading("TASK 3: REGEX CLEANING")

print("Original lowercase text:")
print(lower_T1)

pattern = r"[^a-z\s]"

regex_result = re.sub(pattern, "", lower_T1)

print("\nPattern:", pattern)
print("AFTER:", regex_result)

deleted = re.findall(pattern + r"+", lower_T1)

print("\nDeleted sequences:")
print(deleted)


print("\nCUSTOM REGEX REPLACEMENT")

custom_result = lower_T1

# Replace phone number
custom_result = re.sub(
    r"\b\d{10}\b",
    "PHONE_NUMBER",
    custom_result
)

# Replace money amount
custom_result = re.sub(
    r"£[\d,]+(?:\.\d{2})?",
    "MONEY",
    custom_result
)

# Replace multiple exclamation marks
custom_result = re.sub(
    r"!{2,}",
    "!",
    custom_result
)

print("BEFORE:", lower_T1)
print("AFTER:", custom_result)


# =====================================================
# TASK 4: TOKENIZATION
# =====================================================

heading("TASK 4: TOKENIZATION")

print("T2:", T2)

split_tokens = T2.split()

print("\nUsing split():")
print(split_tokens)
print("Token count:", len(split_tokens))

nltk_tokens = word_tokenize(T2)

print("\nUsing word_tokenize():")
print(nltk_tokens)
print("Token count:", len(nltk_tokens))


print("\nT4 TOKENIZATION")

T4_tokens = word_tokenize(clean_T4)

print("Tokens:", T4_tokens)
print("Token count:", len(T4_tokens))

unique_tokens = set(token.lower() for token in T4_tokens)

print("Unique tokens:", unique_tokens)
print("Vocabulary size:", len(unique_tokens))


# =====================================================
# TASK 5: STOPWORDS AND NEGATION
# =====================================================

heading("TASK 5: STOPWORDS AND NEGATION")

english_stopwords = set(stopwords.words("english"))

print("Original tokens:")
print(nltk_tokens)

default_filtered = [
    token for token in nltk_tokens
    if token.lower() not in english_stopwords
]

print("\nAfter default stopword removal:")
print(default_filtered)


# Keep negation words
negations = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negations

custom_filtered = [
    token for token in nltk_tokens
    if token.lower() not in custom_stopwords
]

print("\nAfter custom stopword removal:")
print(custom_filtered)


# =====================================================
# TASK 6: STEMMING
# =====================================================

heading("TASK 6: STEMMING")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(
    f"{'WORD':<15}"
    f"{'PORTER':<15}"
    f"{'LANCASTER':<15}"
    f"{'SNOWBALL':<15}"
)

print("-" * 60)

stem_results = {}

for word in words:

    p = porter.stem(word)
    l = lancaster.stem(word)
    s = snowball.stem(word)

    stem_results[word] = (p, l, s)

    print(
        f"{word:<15}"
        f"{p:<15}"
        f"{l:<15}"
        f"{s:<15}"
    )


# Find pairs merged by each stemmer

heading("STEMMER COLLISIONS")

for name, index in [
    ("PORTER", 0),
    ("LANCASTER", 1),
    ("SNOWBALL", 2)
]:

    groups = {}

    for word, stems in stem_results.items():
        stem = stems[index]
        groups.setdefault(stem, []).append(word)

    print("\n", name)

    found = False

    for stem, original_words in groups.items():

        if len(original_words) > 1:
            print(stem, "<-", original_words)
            found = True

    if not found:
        print("No collisions found.")


# =====================================================
# TASK 7: LEMMATIZATION
# =====================================================

heading("TASK 7: LEMMATIZATION")

lemmatizer = WordNetLemmatizer()

# Default POS is noun.
default_lemmas = {}

# Appropriate POS hints for the supplied words.
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
    f"{'WORD':<15}"
    f"{'DEFAULT':<15}"
    f"{'WITH POS HINT':<20}"
)

print("-" * 50)

for word in words:

    default = lemmatizer.lemmatize(word)

    hinted = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    default_lemmas[word] = default

    print(
        f"{word:<15}"
        f"{default:<15}"
        f"{hinted:<20}"
    )


# =====================================================
# TASK 8: SENTENCE SPLITTING
# =====================================================

heading("TASK 8: SENTENCE SPLITTING")

sentences = sent_tokenize(T3)

print("Original:")
print(T3)

print("\nSentences:")

for index, sentence in enumerate(sentences, start=1):
    print(f"{index}. {sentence}")

print("\nSentence count:", len(sentences))


# =====================================================
# TASK 9: POS TAGGING AND NER
# =====================================================

heading("TASK 9: POS TAGGING AND NER")

tokens_T3 = word_tokenize(T3)

tagged_words = pos_tag(tokens_T3)

print("POS TAGGING:")

for word, tag in tagged_words:
    print(f"{word:<15} {tag}")


print("\nNAMED ENTITY RECOGNITION:")

named_entities = ne_chunk(tagged_words)

print(named_entities)

print("\nEXTRACTED ENTITIES:")

def extract_entities(tree):

    entities = []

    for node in tree:

        if hasattr(node, "label"):

            entity_name = " ".join(
                leaf[0] for leaf in node.leaves()
            )

            entities.append(
                (entity_name, node.label())
            )

    return entities


entities = extract_entities(named_entities)

for entity, label in entities:
    print(f"Entity: {entity:<25} Label: {label}")


# =====================================================
# TASK 10: FULL PIPELINE
# =====================================================

heading("TASK 10: FULL PREPROCESSING PIPELINE")


def full_pipeline(text, stopword_list, pipeline_name):

    print("\n" + "#" * 60)
    print(pipeline_name)
    print("#" * 60)

    # STEP 1: LOWERCASE
    stage1 = text.lower()

    print("\n1. LOWERCASE")
    print(stage1)

    # STEP 2: REGEX
    stage2 = re.sub(r"[^a-z\s']", " ", stage1)
    stage2 = re.sub(r"\s+", " ", stage2).strip()

    print("\n2. REGEX CLEANING")
    print(stage2)

    # STEP 3: TOKENIZATION
    stage3 = word_tokenize(stage2)

    print("\n3. TOKENIZATION")
    print(stage3)

    # STEP 4: STOPWORD REMOVAL
    stage4 = [
        token for token in stage3
        if token.lower() not in stopword_list
    ]

    print("\n4. STOPWORD REMOVAL")
    print(stage4)

    # STEP 5: LEMMATIZATION
    stage5 = [
        lemmatizer.lemmatize(token)
        if token.isalpha()
        else token
        for token in stage4
    ]

    print("\n5. LEMMATIZATION")
    print(stage5)

    return stage5


# Pipeline A: Default stopwords
pipeline_default = full_pipeline(
    T2,
    english_stopwords,
    "PIPELINE A: DEFAULT STOPWORDS"
)


# Pipeline B: Negation-preserving stopwords
pipeline_custom = full_pipeline(
    T2,
    custom_stopwords,
    "PIPELINE B: NEGATION-PRESERVING STOPWORDS"
)


heading("FINAL PIPELINE COMPARISON")

print("Default stopwords:")
print(pipeline_default)

print("\nCustom stopwords:")
print(pipeline_custom)

print("\nEND OF PREPROCESSING PRACTICE")
