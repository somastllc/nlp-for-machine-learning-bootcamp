
import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# Task 1: Lowercasing

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"


# Print original text
print("=== Task 1: Lowercasing ===")
print("Before:", T1)

# Convert text to lowercase
lowercase_text = T1.lower()

# # Print converted text
print("After:", lowercase_text)



# Task 2: Whitespace Normalization

T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

print("\n=== Task 2: Whitespace Normalization ===")

# Print original text
print("Before:", repr(T4))

# Normalize whitespace
normalized_text = " ".join(T4.split())

# Print normalized text
print("After:", repr(normalized_text))


#Task 3A: Regex Cleaning

lowercase_T1 = T1.lower()

cleaned_text = re.sub(r"[^a-z\s]", "", lowercase_T1)

print("\n=== Task 3A: Regex Cleaning ===")
print("Before:", lowercase_T1)
print("After:", cleaned_text)


# Task 3B: Regex Replacement

replaced_text = T1

# Replace phone number
replaced_text = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", replaced_text)

# Replace money amount
replaced_text = re.sub(r"£[\d,]+", "MONEY", replaced_text)

# Replace !!! with !
replaced_text = re.sub(r"!{3}", "!", replaced_text)

print("\n=== Task 3B: Regex Replacement ===")
print("Before:", T1)
print("After:", replaced_text)



# Task 4: Tokenization

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

print("\n=== Task 4: Tokenization ===")

# Using split()
tokens_split = T2.split()
print("Using split():", tokens_split)
print("Token count:", len(tokens_split))

# Using word_tokenize()
tokens_nltk = word_tokenize(T2)
print("Using word_tokenize():", tokens_nltk)
print("Token count:", len(tokens_nltk))

#Task 5: Stopwords and Negation

print("\n=== Task 5: Stopwords and Negation ===")

# Tokenize T2
tokens = word_tokenize(T2)

# Get NLTK English stopwords
stop_words = set(stopwords.words("english"))

# Remove default stopwords
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Original tokens:", tokens)
print("After removing stopwords:", filtered_tokens)

# Custom stopwords: preserve negation words
negation_words = {"not", "n't", "no", "never"}

custom_stopwords = stop_words - negation_words

custom_filtered_tokens = [
    word for word in tokens
    if word.lower() not in custom_stopwords
]

print("After custom stopwords:", custom_filtered_tokens)

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]
# Task 6: Stemming

print("\n=== Task 6: Stemming ===")

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

# Initialize stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# # Apply stemming
print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

for word in words:
    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)

    print(f"{word:<15} {porter_result:<15} {lancaster_result:<15} {snowball_result:<15}")

# Task 7: Lemmatization

print("\n=== Task 7: Lemmatization ===")

lemmatizer = WordNetLemmatizer()

# Default lemmatization (default POS is noun)
print("Default Lemmatization:")

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

# Correct POS hints
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

print("\nLemmatization with POS hints:")

for word in words:
    pos = pos_hints[word]
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word} -> {lemma} (POS: {pos})")


# Task 8: Sentence Splitting

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

print("\n=== Task 8: Sentence Splitting ===")

sentences = sent_tokenize(T3)

print("Original text:", T3)
print("Sentences:")

for sentence in sentences:
    print(sentence)

print("Total sentences:", len(sentences))

# Task 9: POS Tagging and NER

print("\n=== Task 9: POS Tagging and NER ===")

# # Tokenize T3
tokens_T3 = word_tokenize(T3)

# # POS Tagging
pos_tags = pos_tag(tokens_T3)

print("POS Tags:")
print(pos_tags)

# # Named Entity Recognition
named_entities = ne_chunk(pos_tags)

print("\nNamed Entities:")
print(named_entities)

# # Display recognized entities
print("\nExtracted Entities:")

for entity in named_entities:
    if hasattr(entity, "label"):
        entity_name = " ".join(word for word, tag in entity.leaves())
        entity_label = entity.label()
        print(f"Entity: {entity_name} | Label: {entity_label}")




# Task 10: Full Preprocessing Pipeline

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

print("\n=== Task 10: Full Preprocessing Pipeline ===")

# Initialize tools
stop_words = set(stopwords.words("english"))
negation_words = {"not", "n't", "no", "never"}
custom_stopwords = stop_words - negation_words
lemmatizer = WordNetLemmatizer()


# Function to process T2
def preprocessing_pipeline(text, stopword_list, pipeline_name):

    print(f"\n--- {pipeline_name} ---")

    # Step 1: Lowercase
    lowercase_text = text.lower()
    print("1. Lowercase:", lowercase_text)

    # Step 2: Regex cleaning
    cleaned_text = re.sub(r"[^a-z\s']", "", lowercase_text)
    print("2. Regex:", cleaned_text)

    # Step 3: Tokenization
    tokens = word_tokenize(cleaned_text)
    print("3. Tokenization:", tokens)

    # Step 4: Stopword removal
    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stopword_list
    ]
    print("4. Stopwords:", filtered_tokens)

    # Step 5: Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]
    print("5. Lemmatization:", lemmatized_tokens)


# Original text
print("Original Text:", T2)

# Pipeline A: Default stopwords
preprocessing_pipeline(
    T2,
    stop_words,
    "Pipeline A: Default Stopwords"
)

# Pipeline B: Custom stopwords
preprocessing_pipeline(
    T2,
    custom_stopwords,
    "Pipeline B: Negation Preserved"
)