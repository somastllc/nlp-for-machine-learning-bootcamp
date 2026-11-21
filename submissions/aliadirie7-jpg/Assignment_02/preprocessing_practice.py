import re
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag, ne_chunk
# from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt_tab')

# Practical Task 1 – Lowercasing
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
print("BEFORE LOWERCASE:")
print(T1)

print("\nAFTER LOWERCASE:")
print(T1.lower())


# Task 2 – Whitespace Normalization
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

print("\n===== TASK 2: WHITESPACE NORMALIZATION =====")

print("BEFORE:")
print(repr(T4))

clean_text = re.sub(r"\s+", " ", T4).strip()

print("\nAFTER:")
print(repr(clean_text))


# Task 3 – Regex Cleaning
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
print("\n===== TASK 3: REGEX CLEANING =====")

lower_text = T1.lower()

cleaned_text = re.sub(r"[^a-z\s]", "", lower_text)

print("BEFORE:")
print(lower_text)

print("\nAFTER:")
print(cleaned_text)

print("\n===== TASK 3: CUSTOM REGEX REPLACEMENT =====")

text = T1.lower()

# Replace phone number
text = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", text)

# Replace money
text = re.sub(r"£[\d,]+", "MONEY", text)

# Replace repeated exclamation marks
text = re.sub(r"!{2,}", "!", text)

print("BEFORE:")
print(T1.lower())

print("\nAFTER:")
print(text)


T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
from nltk.tokenize import word_tokenize

print("\n===== TASK 4: TOKENIZATION =====")

print("\n--- Python split() ---")

split_tokens = T2.split()

print(split_tokens)
print("Token count:", len(split_tokens))


print("\n--- NLTK word_tokenize() ---")

nltk_tokens = word_tokenize(T2)

print(nltk_tokens)
print("Token count:", len(nltk_tokens))


T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "
print("\n--- T4 TOKENIZATION ---")

tokens_T4 = word_tokenize(clean_text)

print("Tokens:")
print(tokens_T4)

print("Token count:", len(tokens_T4))

print("Vocabulary size:", len(set(tokens_T4)))



stop_words = set(stopwords.words("english"))

tokens = word_tokenize(T2)

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("\n--- STOPWORD REMOVAL ---")
print("Original tokens:", tokens)
print("Filtered tokens:", filtered_tokens)


custom_stop_words = stop_words - {"not", "n't", "no", "never"}

custom_filtered_tokens = [
    word for word in tokens
    if word.lower() not in custom_stop_words
]

print("\n--- CUSTOM STOPWORD REMOVAL ---")
print("Custom filtered tokens:", custom_filtered_tokens)


# ===== TASK 6: STEMMING =====

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\n===== TASK 6: STEMMING COMPARISON =====")
print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")

for word in words:
    print(
        f"{word:<15} "
        f"{porter.stem(word):<15} "
        f"{lancaster.stem(word):<15} "
        f"{snowball.stem(word):<15}"
    )


# ===== TASK 7: LEMMATIZATION =====

lemmatizer = WordNetLemmatizer()

print("\n===== TASK 7: LEMMATIZATION =====")

print("\n--- Default Lemmatization ---")

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

print("\n--- Lemmatization with POS hints ---")

for word in words:
    print(word, "->", lemmatizer.lemmatize(word, pos="v"))



T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

# ===== TASK 8: SENTENCE TOKENIZATION =====

sentences = sent_tokenize(T3)

print("\n===== TASK 8: SENTENCE TOKENIZATION =====")

for sentence in sentences:
    print(sentence)

print("Sentence count:", len(sentences))

# ===== TASK 9: POS TAGGING =====

tokens_T3 = word_tokenize(T3)

pos_tags = pos_tag(tokens_T3)

print("\n===== TASK 9: POS TAGGING =====")

for word, tag in pos_tags:
    print(word, "->", tag)



# ===== NAMED ENTITY RECOGNITION =====

named_entities = ne_chunk(pos_tags)

print("\n===== TASK 9: NAMED ENTITY RECOGNITION =====")

print(named_entities)

# ===== TASK 10: FULL PREPROCESSING PIPELINE =====

print("\n===== TASK 10: FULL PREPROCESSING PIPELINE =====")

# Step 1: Lowercasing
pipeline_text = T2.lower()
print("\n1. LOWERCASE:")
print(pipeline_text)

# Step 2: Regex Cleaning
pipeline_text = re.sub(r"[^a-z\s']", " ", pipeline_text)
pipeline_text = re.sub(r"\s+", " ", pipeline_text).strip()

print("\n2. REGEX CLEANING:")
print(pipeline_text)

# Step 3: Tokenization
pipeline_tokens = word_tokenize(pipeline_text)

print("\n3. TOKENIZATION:")
print(pipeline_tokens)

# Step 4: Default Stopword Removal
default_filtered = [
    word for word in pipeline_tokens
    if word not in stop_words
]

print("\n4A. DEFAULT STOPWORD REMOVAL:")
print(default_filtered)

# Step 5: Custom Stopword Removal
custom_stop_words = stop_words - {"not", "n't", "no", "never"}

custom_filtered = [
    word for word in pipeline_tokens
    if word not in custom_stop_words
]

print("\n4B. CUSTOM STOPWORD REMOVAL:")
print(custom_filtered)

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()

default_lemmas = [
    lemmatizer.lemmatize(word)
    for word in default_filtered
]

custom_lemmas = [
    lemmatizer.lemmatize(word)
    for word in custom_filtered
]

print("\n5A. DEFAULT LEMMATIZATION:")
print(default_lemmas)

print("\n5B. CUSTOM LEMMATIZATION:")
print(custom_lemmas)