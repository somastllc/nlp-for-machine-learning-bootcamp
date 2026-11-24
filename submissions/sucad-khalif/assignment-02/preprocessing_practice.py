
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    WordNetLemmatizer
)
from nltk import pos_tag, ne_chunk


# DOWNLOAD REQUIRED NLTK DATA


resources = [
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words"
]

for resource in resources:
    nltk.download(resource, quiet=True)


# ASSIGNMENT TEXTS


T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]


print("=" * 65)
print("ASSIGNMENT 2 - TEXT PREPROCESSING")
print("=" * 65)

# TASK 1: LOWERCASE NORMALIZATION


print("\nTASK 1: LOWERCASE NORMALIZATION")

lowercase_T1 = T1.lower()

print("Before:", T1)
print("After: ", lowercase_T1)


# TASK 2: WHITESPACE NORMALIZATION


print("\nTASK 2: WHITESPACE NORMALIZATION")

normalized_T4 = " ".join(T4.split())

print("Before:", repr(T4))
print("After: ", repr(normalized_T4))


# TASK 3: REGEX CLEANING

print("\nTASK 3: REGEX CLEANING")

lower_T1 = T1.lower()

deleted_characters = re.findall(r"[^a-z\s]", lower_T1)

cleaned_T1 = re.sub(r"[^a-z\s]", "", lower_T1)

print("Lowercase text:", lower_T1)
print("Deleted characters:", deleted_characters)
print("Cleaned text:", cleaned_T1)

# Custom replacements
custom_T1 = T1

custom_T1 = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", custom_T1)
custom_T1 = re.sub(r"£[\d,]+", "MONEY", custom_T1)
custom_T1 = custom_T1.replace("!!!", "!")

print("\nCustom replacements:")
print(custom_T1)


# TASK 4: TOKENIZATION

print("\nTASK 4: TOKENIZATION")

split_tokens = T2.split()
word_tokens = word_tokenize(T2)

print("T2 using split():", split_tokens)
print("Split token count:", len(split_tokens))

print("\nT2 using word_tokenize():", word_tokens)
print("Word token count:", len(word_tokens))

T4_tokens = word_tokenize(normalized_T4.lower())

print("\nT4 tokens:", T4_tokens)
print("T4 token count:", len(T4_tokens))
print("T4 vocabulary size:", len(set(T4_tokens)))


# TASK 5: STOPWORD REMOVAL

print("\nTASK 5: STOPWORD REMOVAL")

tokens_T2 = word_tokenize(T2.lower())

english_stopwords = set(stopwords.words("english"))

without_stopwords = [
    word for word in tokens_T2
    if word not in english_stopwords
]

print("Original tokens:", tokens_T2)
print("Without standard stopwords:", without_stopwords)

# Custom stopword list that preserves negation
custom_stopwords = english_stopwords - {"not", "no", "never", "n't"}

preserve_negation = [
    word for word in tokens_T2
    if word not in custom_stopwords
]

print("\nWith negation preserved:", preserve_negation)


# TASK 6: STEMMING COMPARISON

print("\nTASK 6: STEMMING COMPARISON")

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

for word in words:
    print(
        f"{word:<15}"
        f"{porter.stem(word):<15}"
        f"{lancaster.stem(word):<15}"
        f"{snowball.stem(word):<15}"
    )

# TASK 7: LEMMATIZATION


print("\nTASK 7: LEMMATIZATION")

lemmatizer = WordNetLemmatizer()

print(
    f"{'Word':<15}"
    f"{'Default':<15}"
    f"{'Verb (v)':<15}"
    f"{'Noun (n)':<15}"
    f"{'Adjective (a)':<15}"
)

print("-" * 60)

for word in words:
    default_lemma = lemmatizer.lemmatize(word)
    verb_lemma = lemmatizer.lemmatize(word, pos="v")
    noun_lemma = lemmatizer.lemmatize(word, pos="n")
    adjective_lemma = lemmatizer.lemmatize(word, pos="a")

    print(
        f"{word:<15}"
        f"{default_lemma:<15}"
        f"{verb_lemma:<15}"
        f"{noun_lemma:<15}"
        f"{adjective_lemma:<15}"
    )

# TASK 8: SENTENCE TOKENIZATION


print("\nTASK 8: SENTENCE TOKENIZATION")

sentences = sent_tokenize(T3)

for number, sentence in enumerate(sentences, start=1):
    print(f"Sentence {number}: {sentence}")

print("Total sentences:", len(sentences))


# TASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION


print("\nTASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION")

tokens_T3 = word_tokenize(T3)

tagged_words = pos_tag(tokens_T3)

print("POS tagged words:")
print(tagged_words)

named_entities = ne_chunk(tagged_words)

print("\nNamed Entity Recognition:")
print(named_entities)

print("\nExtracted entities:")

for chunk in named_entities:
    if hasattr(chunk, "label"):
        entity_name = " ".join(
            token for token, tag in chunk.leaves()
        )
        print(entity_name, ":", chunk.label())

# TASK 10: COMPLETE PREPROCESSING PIPELINE


print("\nTASK 10: COMPLETE PREPROCESSING PIPELINE")

# Step 1: Lowercase
pipeline_lower = T2.lower()

# Step 2: Regex cleaning
pipeline_clean = re.sub(r"[^a-z\s']", "", pipeline_lower)

# Step 3: Tokenization
pipeline_tokens = word_tokenize(pipeline_clean)

# Step 4: Standard stopword removal
pipeline_no_stopwords = [
    word for word in pipeline_tokens
    if word not in english_stopwords
]

# Step 5: Lemmatization
pipeline_lemmas = [
    lemmatizer.lemmatize(word)
    for word in pipeline_no_stopwords
]

print("\nPipeline A: Standard stopword removal")
print("1. Lowercase:", pipeline_lower)
print("2. Regex cleaning:", pipeline_clean)
print("3. Tokens:", pipeline_tokens)
print("4. Without stopwords:", pipeline_no_stopwords)
print("5. Lemmatized:", pipeline_lemmas)


# Pipeline B: Preserve negation

pipeline_preserve = [
    word for word in pipeline_tokens
    if word not in custom_stopwords
]

pipeline_preserve_lemmas = [
    lemmatizer.lemmatize(word)
    for word in pipeline_preserve
]

print("\nPipeline B: Negation-preserving stopword removal")
print("1. Lowercase:", pipeline_lower)
print("2. Regex cleaning:", pipeline_clean)
print("3. Tokens:", pipeline_tokens)
print("4. Without selected stopwords:", pipeline_preserve)
print("5. Lemmatized:", pipeline_preserve_lemmas)



