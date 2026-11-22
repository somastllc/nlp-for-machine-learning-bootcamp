
import re
from collections import defaultdict

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


t1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
t2 = "I don't think this jacket is worth the price. Not happy at all 😞"
t3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
t4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

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
    "ran"
]


# ============================================================
# 1. Lowercase normalization
# ============================================================

print("=" * 60)
print("1. LOWERCASE NORMALIZATION")
print("=" * 60)

print("Before:")
print(t1)

t1_lower = t1.lower()

print("After:")
print(t1_lower)


# ============================================================
# 2. Whitespace normalization
# ============================================================

print()
print("=" * 60)
print("2. WHITESPACE NORMALIZATION")
print("=" * 60)

print("Before:")
print(repr(t4))

t4_normalized = " ".join(t4.split())

print("After:")
print(repr(t4_normalized))


# ============================================================
# 3. Regular expressions
# ============================================================

print()
print("=" * 60)
print("3. REGULAR EXPRESSIONS")
print("=" * 60)

print("Before:")
print(t1_lower)

regex_result = re.sub(r"[^a-z\s]", " ", t1_lower)
regex_result = " ".join(regex_result.split())

print("After:")
print(regex_result)

removed_characters = re.findall(r"[^a-z\s]", t1_lower)

print()
print("Characters removed:")
print(removed_characters)

print()
print("Custom replacements:")

replacement_result = t1_lower

replacement_result = re.sub(
    r"\b\d{10}\b",
    "PHONE_NUMBER",
    replacement_result
)

replacement_result = re.sub(
    r"£\d{1,3}(?:,\d{3})*",
    "MONEY",
    replacement_result
)

replacement_result = re.sub(r"!{2,}", "!", replacement_result)

print(replacement_result)


# ============================================================
# 4. Tokenization
# ============================================================

print()
print("=" * 60)
print("4. TOKENIZATION")
print("=" * 60)

print("Original text:")
print(t2)

print()
print("Using split():")

t2_split = t2.split()

print(t2_split)
print("Token count:", len(t2_split))

print()
print("Using word_tokenize():")

t2_tokens = word_tokenize(t2)

print(t2_tokens)
print("Token count:", len(t2_tokens))

print()
print("T4 tokens:")

t4_tokens = word_tokenize(t4_normalized)

print(t4_tokens)
print("Token count:", len(t4_tokens))

t4_vocabulary = sorted(set(t4_tokens))

print("T4 vocabulary:")
print(t4_vocabulary)
print("Vocabulary size:", len(t4_vocabulary))


# ============================================================
# 5. Stopword removal
# ============================================================

print()
print("=" * 60)
print("5. STOPWORD REMOVAL")
print("=" * 60)

english_stopwords = set(stopwords.words("english"))

t2_lower_tokens = word_tokenize(t2.lower())

print("Before:")
print(t2_lower_tokens)

t2_without_stopwords = []

for word in t2_lower_tokens:
    if word not in english_stopwords:
        t2_without_stopwords.append(word)

print("After:")
print(t2_without_stopwords)


print()
print("After preserving negation:")

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

t2_keep_negation = []

for word in t2_lower_tokens:
    if word not in custom_stopwords:
        t2_keep_negation.append(word)

print(t2_keep_negation)


# ============================================================
# 6. Stemming
# ============================================================

print()
print("=" * 60)
print("6. STEMMING")
print("=" * 60)

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

porter_groups = defaultdict(list)
lancaster_groups = defaultdict(list)
snowball_groups = defaultdict(list)

for word in words:
    porter_stem = porter.stem(word)
    lancaster_stem = lancaster.stem(word)
    snowball_stem = snowball.stem(word)

    print(
        f"{word:<15}"
        f"{porter_stem:<15}"
        f"{lancaster_stem:<15}"
        f"{snowball_stem:<15}"
    )

    porter_groups[porter_stem].append(word)
    lancaster_groups[lancaster_stem].append(word)
    snowball_groups[snowball_stem].append(word)


print()
print("Porter shared stems:")

for stem, group in porter_groups.items():
    if len(group) > 1:
        print(stem, ":", group)


print()
print("Lancaster shared stems:")

for stem, group in lancaster_groups.items():
    if len(group) > 1:
        print(stem, ":", group)


print()
print("Snowball shared stems:")

for stem, group in snowball_groups.items():
    if len(group) > 1:
        print(stem, ":", group)


print()
print("Example of a stem that is not a real word:")

print("universal ->", porter.stem("universal"))


# ============================================================
# 7. Lemmatization
# ============================================================

print()
print("=" * 60)
print("7. LEMMATIZATION")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

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
    f"{'With POS':<15}"
)

print("-" * 45)

for word in words:
    default = lemmatizer.lemmatize(word)

    pos = pos_hints[word]

    with_pos = lemmatizer.lemmatize(word, pos=pos)

    print(
        f"{word:<15}"
        f"{default:<15}"
        f"{with_pos:<15}"
    )


print()
print("Words that changed after giving POS hints:")

for word in words:
    default = lemmatizer.lemmatize(word)

    pos = pos_hints[word]

    with_pos = lemmatizer.lemmatize(word, pos=pos)

    if default != with_pos:
        print(word, "->", default, "->", with_pos)


# ============================================================
# 8. Sentence tokenization
# ============================================================

print()
print("=" * 60)
print("8. SENTENCE TOKENIZATION")
print("=" * 60)

print("Before:")
print(t3)

sentences = sent_tokenize(t3)

print()
print("After:")

for index, sentence in enumerate(sentences, start=1):
    print(f"Sentence {index}: {sentence}")

print()
print("Number of sentences:", len(sentences))

print()
print("Abbreviations:")

for sentence in sentences:
    if "Mr." in sentence:
        print("Mr. :", sentence)

    if "a.m." in sentence:
        print("a.m. :", sentence)


# ============================================================
# 9. POS tagging and named entity recognition
# ============================================================

print()
print("=" * 60)
print("9. POS TAGGING AND NAMED ENTITY RECOGNITION")
print("=" * 60)

print("Original text:")
print(t3)

t3_tokens = word_tokenize(t3)

pos_tags = pos_tag(t3_tokens)

print()
print("POS tags:")

for word, tag in pos_tags:
    print(f"{word:<15}{tag}")


print()
print("Named entities:")

named_entities = ne_chunk(pos_tags)

print(named_entities)


print()
print("Extracted entities:")

extracted_entities = {}

for item in named_entities:
    if hasattr(item, "label"):
        entity = " ".join(word for word, tag in item.leaves())
        label = item.label()

        extracted_entities[entity] = label

        print(f"{entity:<20}{label}")


print()
print("Entity comparison:")

expected_entities = {
    "Smith": "PERSON",
    "London": "GPE",
    "Friday": "DATE",
    "Apple": "ORGANIZATION"
}

print(
    f"{'Entity':<15}"
    f"{'Predicted':<20}"
    f"{'Expected':<20}"
    f"{'Correct':<10}"
)

print("-" * 65)

for entity, expected in expected_entities.items():

    predicted = extracted_entities.get(entity, "NOT RECOGNIZED")

    correct = "Yes" if predicted == expected else "No"

    print(
        f"{entity:<15}"
        f"{predicted:<20}"
        f"{expected:<20}"
        f"{correct:<10}"
    )


# ============================================================
# 10. Complete preprocessing pipeline
# ============================================================

print()
print("=" * 60)
print("10. COMPLETE PREPROCESSING PIPELINE")
print("=" * 60)


# Pipeline A: Default stopwords

print()
print("--- PIPELINE A: DEFAULT STOPWORDS ---")

pipeline_text = t2

print()
print("Original text:")
print(pipeline_text)

pipeline_lower = pipeline_text.lower()

print()
print("1. Lowercase:")
print(pipeline_lower)

# Keep apostrophes to preserve contractions such as don't
pipeline_regex = re.sub(r"[^a-z\s']", " ", pipeline_lower)
pipeline_regex = " ".join(pipeline_regex.split())

print()
print("2. Regex cleaning:")
print(pipeline_regex)

pipeline_tokens = word_tokenize(pipeline_regex)

print()
print("3. Tokenization:")
print(pipeline_tokens)

pipeline_stopwords = []

for word in pipeline_tokens:
    if word not in english_stopwords:
        pipeline_stopwords.append(word)

print()
print("4. Stopword removal:")
print(pipeline_stopwords)

pipeline_lemmas = []

for word in pipeline_stopwords:
    pipeline_lemmas.append(lemmatizer.lemmatize(word))

print()
print("5. Lemmatization:")
print(pipeline_lemmas)

print()
print("Final result:")
print(" ".join(pipeline_lemmas))


# Pipeline B: Preserve negation

print()
print("--- PIPELINE B: PRESERVE NEGATION ---")

pipeline_text = t2

print()
print("Original text:")
print(pipeline_text)

pipeline_lower = pipeline_text.lower()

print()
print("1. Lowercase:")
print(pipeline_lower)

pipeline_regex = re.sub(r"[^a-z\s']", " ", pipeline_lower)
pipeline_regex = " ".join(pipeline_regex.split())

print()
print("2. Regex cleaning:")
print(pipeline_regex)

pipeline_tokens = word_tokenize(pipeline_regex)

print()
print("3. Tokenization:")
print(pipeline_tokens)

pipeline_stopwords = []

for word in pipeline_tokens:
    if word not in custom_stopwords:
        pipeline_stopwords.append(word)

print()
print("4. Stopword removal:")
print(pipeline_stopwords)

pipeline_lemmas = []

for word in pipeline_stopwords:
    pipeline_lemmas.append(lemmatizer.lemmatize(word))

print()
print("5. Lemmatization:")
print(pipeline_lemmas)

print()
print("Final result:")
print(" ".join(pipeline_lemmas))


# Pipeline comparison

print()
print("--- PIPELINE COMPARISON ---")

print("Original:", t2)

print("Default: ", " ".join(
    word for word in pipeline_tokens
    if word not in custom_stopwords
))

print()
print("=" * 60)
print("ALL TASKS COMPLETED")
print("=" * 60)
