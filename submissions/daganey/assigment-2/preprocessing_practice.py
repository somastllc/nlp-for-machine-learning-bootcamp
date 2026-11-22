
# Assignment 2 - NLP Preprocessing Practice
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

# Task 1: Lowercasing

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

print("\n========== TASK 1: LOWERCASE ==========")

print("Before:")
print(T1)

lower_T1 = T1.lower()

print("\nAfter:")
print(lower_T1)



# Task 2: Whitespace Normalization

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

print("\n========== TASK 2: WHITESPACE ==========")

print("Before:")
print(repr(T4))

clean_T4 = " ".join(T4.split())

print("\nAfter:")
print(repr(clean_T4))



# Task 3: Regex Cleaning

print("\n========== TASK 3: REGEX CLEANING ==========")

print("\nPart A: Remove special characters")

print("Before:")
print(lower_T1)

clean_regex = re.sub(r"[^a-z\s]", "", lower_T1)

print("\nAfter:")
print(clean_regex)


# Task 3: Regex Cleaning

print("\n========== TASK 3: REGEX CLEANING ==========")

print("\nPart A: Remove special characters")

print("Before:")
print(lower_T1)

clean_regex = re.sub(r"[^a-z\s]", "", lower_T1)

print("\nAfter:")
print(clean_regex)

print("\nPart B: Replace important information")

custom_clean = lower_T1

# Replace money amount
custom_clean = re.sub(r"£[\d,]+", "MONEY", custom_clean)

# Replace phone number
custom_clean = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", custom_clean)

# Replace three exclamation marks with one
custom_clean = re.sub(r"!{3,}", "!", custom_clean)

print("\nBefore:")
print(lower_T1)

print("\nAfter:")
print(custom_clean)



# Task 4: Tokenization

from nltk.tokenize import word_tokenize

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

print("\n========== TASK 4: TOKENIZATION ==========")

# Method 1: split()
split_tokens = T2.split()

print("\nUsing split():")
print(split_tokens)
print("Token count:", len(split_tokens))

# Method 2: word_tokenize()
nltk_tokens = word_tokenize(T2)

print("\nUsing word_tokenize():")
print(nltk_tokens)
print("Token count:", len(nltk_tokens))

# Tokenization of T4
print("\nT4 Tokenization:")

T4_tokens = word_tokenize(T4)

print("Tokens:")
print(T4_tokens)

print("Token count:", len(T4_tokens))

# Vocabulary size
vocabulary = set(T4_tokens)

print("Vocabulary:")
print(vocabulary)

print("Vocabulary size:", len(vocabulary))


# Task 5: Stopwords and Negation

print("\n========== TASK 5: STOPWORDS ==========")

# Get NLTK English stopwords
english_stopwords = set(stopwords.words("english"))

# Remove default stopwords
default_filtered = [
    word for word in nltk_tokens
    if word.lower() not in english_stopwords
]

print("\nOriginal tokens:")
print(nltk_tokens)

print("\nAfter removing default stopwords:")
print(default_filtered)

# Custom stopwords: keep negation words
negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

custom_filtered = [
    word for word in nltk_tokens
    if word.lower() not in custom_stopwords
]

print("\nAfter removing custom stopwords:")
print(custom_filtered)



# Task 6: Stemming

print("\n========== TASK 6: STEMMING ==========")

words = [
    "connection", "connected", "connecting",
    "organization", "organized",
    "university", "universal",
    "flies", "wolves", "better", "ran"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

for word in words:
    p = porter.stem(word)
    l = lancaster.stem(word)
    s = snowball.stem(word)

    print(f"{word:<15} {p:<15} {l:<15} {s:<15}")


# Task 7: Lemmatization

print("\n========== TASK 7: LEMMATIZATION ==========")

lemmatizer = WordNetLemmatizer()

# Default lemmatization (noun is the default)
print("\nDefault Lemmatization:")

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

# Correct POS hints
pos_mapping = {
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

print("\nLemmatization with POS hints:")

for word in words:
    pos = pos_mapping[word]
    result = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word} ({pos}) -> {result}")


# Task 8: Sentence Splitting

from nltk.tokenize import sent_tokenize

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

print("\n========== TASK 8: SENTENCE SPLITTING ==========")

sentences = sent_tokenize(T3)

print("\nOriginal text:")
print(T3)

print("\nSentences:")
for sentence in sentences:
    print(sentence)

print("\nNumber of sentences:", len(sentences))


# Task 9: POS Tagging and Named Entity Recognition

print("\n========== TASK 9: POS TAGGING AND NER ==========")

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

# Tokenization
tokens = word_tokenize(T3)

# POS Tagging
tagged_words = pos_tag(tokens)

print("\nPOS TAGGING:")
for word, tag in tagged_words:
    print(f"{word:<15} {tag}")

# Named Entity Recognition
named_entities = ne_chunk(tagged_words)

print("\nNAMED ENTITIES:")
print(named_entities)

# Extract entities
print("\nENTITY TABLE:")

for entity in named_entities:
    if hasattr(entity, "label"):
        entity_name = " ".join(
            word for word, tag in entity.leaves()
        )
        print(f"{entity_name:<20} {entity.label()}")


# Task 10: Full Preprocessing Pipeline

print("\n========== TASK 10: FULL PIPELINE ==========")

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

lemmatizer = WordNetLemmatizer()

# Function for preprocessing
def preprocessing_pipeline(text, stopword_list, title):

    print(f"\n--- {title} ---")

    # Step 1: Lowercase
    lowercase_text = text.lower()
    print("\n1. Lowercase:")
    print(lowercase_text)

    # Step 2: Regex cleaning
    regex_text = re.sub(r"[^a-z\s']", "", lowercase_text)
    print("\n2. Regex cleaning:")
    print(regex_text)

    # Step 3: Tokenization
    tokens = word_tokenize(regex_text)
    print("\n3. Tokenization:")
    print(tokens)

    # Step 4: Stopwords removal
    filtered_tokens = [
        word for word in tokens
        if word not in stopword_list
    ]
    print("\n4. Stopwords removal:")
    print(filtered_tokens)

    # Step 5: Lemmatization
    final_tokens = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]
    print("\n5. Lemmatization:")
    print(final_tokens)


# Version 1: Default stopwords
preprocessing_pipeline(
    T2,
    english_stopwords,
    "VERSION 1: DEFAULT STOPWORDS"
)

# Version 2: Custom stopwords
preprocessing_pipeline(
    T2,
    custom_stopwords,
    "VERSION 2: CUSTOM STOPWORDS"
)
