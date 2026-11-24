import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# TASK 1: LOWERCASE


t1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

lower_t1 = t1.lower()

print("\n--- Task 1: Lowercasing ---")
print("Before:", t1)
print("After:", lower_t1)



# Task 2: Whitespace Normalization

t4 = "   The   delivery   was   fast,   and the  packaging   was   great."

clean_t4 = re.sub(r"\s+", " ", t4).strip()

print("\n--- Task 2: Whitespace Normalization ---")
print("Before:", repr(t4))
print("After:", repr(clean_t4))

# Task 3: Regex Cleaning

cleaned_t1 = re.sub(r"[^a-z\s]", "", lower_t1)

print("\n--- Task 3: Regex Cleaning ---")
print("Before:", lower_t1)
print("After:", cleaned_t1)


# Task 3: Custom Regex Replacement

replaced_t1 = re.sub(r"07\d{8}", "PHONE_NUMBER", t1)
replaced_t1 = re.sub(r"£[\d,]+", "MONEY", replaced_t1)
replaced_t1 = re.sub(r"!{3,}", "!", replaced_t1)

print("\n--- Task 3: Custom Regex Replacement ---")
print("Before:", t1)
print("After:", replaced_t1)



# Task 4: Tokenization

t2 = "I don't think this jacket is worth the price. Not happy at all 😞"

tokens_split = t2.split()
tokens_nltk = word_tokenize(t2)

print("\n--- Task 4: Tokenization ---")
print("Original Text:", t2)
print("Using split():", tokens_split)
print("split() count:", len(tokens_split))

print("Using word_tokenize():", tokens_nltk)
print("word_tokenize() count:", len(tokens_nltk))

tokens_t4 = word_tokenize(clean_t4)

print("\nT4 Tokens:", tokens_t4)
print("T4 Token Count:", len(tokens_t4))
print("T4 Vocabulary Size:", len(set(tokens_t4)))



# Task 5: Stopwords and Negation

stop_words = set(stopwords.words("english"))

filtered_t2 = [word for word in tokens_nltk if word.lower() not in stop_words]

print("\n--- Task 5: Default Stopwords ---")
print("Before:", tokens_nltk)
print("After:", filtered_t2)

custom_stopwords = stop_words - {"not", "n't", "no", "never"}

filtered_t2_custom = [
    word for word in tokens_nltk
    if word.lower() not in custom_stopwords
]

print("\n--- Task 5: Custom Stopwords (Keep Negation) ---")
print("Before:", tokens_nltk)
print("After:", filtered_t2_custom)


# Task 6: Stemming

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\n--- Task 6: Stemming Comparison ---")

for word in words:
    print(
        word,
        "| Porter:", porter.stem(word),
        "| Lancaster:", lancaster.stem(word),
        "| Snowball:", snowball.stem(word)
    )


    # Task 7: Lemmatization

lemmatizer = WordNetLemmatizer()

print("\n--- Task 7: Lemmatization Default ---")

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

print("\n--- Task 7: Lemmatization with POS Hints ---")

for word in words:
    print(
        word,
        "| Verb:", lemmatizer.lemmatize(word, pos="v"),
        "| Noun:", lemmatizer.lemmatize(word, pos="n"),
        "| Adjective:", lemmatizer.lemmatize(word, pos="a")
    )

    # Task 8: Sentence Splitting

t3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

sentences = sent_tokenize(t3)

print("\n--- Task 8: Sentence Splitting ---")
print("Before:", t3)
print("Sentences:", sentences)
print("Sentence Count:", len(sentences))

# Task 9: POS Tagging and Named Entity Recognition

pos_tags = pos_tag(word_tokenize(t3))
entities = ne_chunk(pos_tags)

print("\n--- Task 9: POS Tagging ---")
print(pos_tags)

print("\n--- Task 9: Named Entity Recognition ---")
print(entities)



# Task 10: Full Preprocessing Pipeline

print("\n--- Task 10: Default Stopwords Pipeline ---")

# Step 1: Lowercase
pipeline_lower = t2.lower()
print("1. Lowercase:", pipeline_lower)

# Step 2: Regex Cleaning
pipeline_clean = re.sub(r"[^a-z\s']", "", pipeline_lower)
print("2. Regex Cleaning:", pipeline_clean)

# Step 3: Tokenization
pipeline_tokens = word_tokenize(pipeline_clean)
print("3. Tokenization:", pipeline_tokens)

# Step 4: Remove Default Stopwords
pipeline_stopwords = [
    word for word in pipeline_tokens
    if word not in stop_words
]
print("4. Stopwords Removal:", pipeline_stopwords)

# Step 5: Lemmatization
pipeline_lemma = [
    lemmatizer.lemmatize(word) for word in pipeline_stopwords
]
print("5. Lemmatization:", pipeline_lemma)


print("\n--- Task 10: Custom Stopwords Pipeline ---")

# Step 1: Lowercase
custom_lower = t2.lower()
print("1. Lowercase:", custom_lower)

# Step 2: Regex Cleaning
custom_clean = re.sub(r"[^a-z\s']", "", custom_lower)
print("2. Regex Cleaning:", custom_clean)

# Step 3: Tokenization
custom_tokens = word_tokenize(custom_clean)
print("3. Tokenization:", custom_tokens)

# Step 4: Remove Custom Stopwords
custom_filtered = [
    word for word in custom_tokens
    if word not in custom_stopwords
]
print("4. Stopwords Removal:", custom_filtered)

# Step 5: Lemmatization
custom_lemma = [
    lemmatizer.lemmatize(word) for word in custom_filtered
]
print("5. Lemmatization:", custom_lemma)