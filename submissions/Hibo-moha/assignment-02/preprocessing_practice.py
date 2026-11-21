
import re
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

# --------------------------------------------------
# 1. DOWNLOAD REQUIRED NLTK DATA
# --------------------------------------------------

resources = [
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab"
]

for resource in resources:
    nltk.download(resource, quiet=True)

print("=" * 60)
print("NLP PREPROCESSING PRACTICE")
print("=" * 60)


# --------------------------------------------------
# 2. ASSIGNMENT DATA
# --------------------------------------------------

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great. "

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]


# --------------------------------------------------
# 3. LOWERCASE NORMALIZATION
# --------------------------------------------------

print("\n1. LOWERCASE NORMALIZATION")

t1_lower = T1.lower()

print("Original:", T1)
print("Lowercase:", t1_lower)


# --------------------------------------------------
# 4. WHITESPACE NORMALIZATION
# --------------------------------------------------

print("\n2. WHITESPACE NORMALIZATION")

t4_clean = " ".join(T4.split())

print("Original:", repr(T4))
print("Cleaned:", repr(t4_clean))


# --------------------------------------------------
# 5. REGEX: REMOVE NON-ALPHABETIC CHARACTERS
# --------------------------------------------------

print("\n3. REGEX CLEANING")

regex_clean = re.sub(r"[^a-z\s]", "", t1_lower)

print("Original lowercase:", t1_lower)
print("After regex:", regex_clean)

print("Removed characters include punctuation, currency symbols and digits.")


# --------------------------------------------------
# 6. CUSTOM REGEX REPLACEMENT
# --------------------------------------------------

print("\n4. CUSTOM REGEX REPLACEMENT")

custom_clean = T1

# Replace phone number
custom_clean = re.sub(r"\b07\d{8}\b", "PHONE_NUMBER", custom_clean)

# Replace money amount
custom_clean = re.sub(r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "MONEY", custom_clean)

# Replace repeated exclamation marks
custom_clean = re.sub(r"!{2,}", "!", custom_clean)

print("Original:", T1)
print("Processed:", custom_clean)


# --------------------------------------------------
# 7. TOKENIZATION: T2
# --------------------------------------------------

print("\n5. TOKENIZATION")

split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)

print("Python split():", split_tokens)
print("NLTK word_tokenize():", nltk_tokens)


# --------------------------------------------------
# 8. TOKEN COUNT AND VOCABULARY SIZE: T4
# --------------------------------------------------

print("\n6. TOKEN COUNT AND VOCABULARY")

t4_tokens = word_tokenize(t4_clean.lower())

token_count = len(t4_tokens)
vocabulary = set(t4_tokens)
vocabulary_size = len(vocabulary)

print("Tokens:", t4_tokens)
print("Token count:", token_count)
print("Unique vocabulary:", sorted(vocabulary))
print("Vocabulary size:", vocabulary_size)


# --------------------------------------------------
# 9. STOPWORD REMOVAL
# --------------------------------------------------

print("\n7. STOPWORD REMOVAL")

english_stopwords = set(stopwords.words("english"))

t2_tokens = word_tokenize(T2.lower())

default_filtered = [
    word for word in t2_tokens
    if word not in english_stopwords
]

# Keep important negation words
negation_words = {"no", "not", "nor", "never"}
safe_stopwords = english_stopwords - negation_words

negation_safe_filtered = [
    word for word in t2_tokens
    if word not in safe_stopwords
]

print("Original tokens:", t2_tokens)
print("Default stopword removal:", default_filtered)
print("Negation-preserving removal:", negation_safe_filtered)


# --------------------------------------------------
# 10. STEMMING
# --------------------------------------------------

print("\n8. STEMMING")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

for word in words:
    print(
        f"{word:<15} "
        f"{porter.stem(word):<15} "
        f"{lancaster.stem(word):<15} "
        f"{snowball.stem(word):<15}"
    )


# --------------------------------------------------
# 11. LEMMATIZATION
# --------------------------------------------------

print("\n9. LEMMATIZATION")

lemmatizer = WordNetLemmatizer()

print("Default lemmatization:")
for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

print("\nLemmatization with POS hints:")

pos_examples = [
    ("flies", wordnet.VERB),
    ("wolves", wordnet.NOUN),
    ("better", wordnet.ADJ),
    ("ran", wordnet.VERB),
    ("organized", wordnet.VERB)
]

for word, pos in pos_examples:
    print(word, "->", lemmatizer.lemmatize(word, pos=pos))


# --------------------------------------------------
# 12. SENTENCE TOKENIZATION: T3
# --------------------------------------------------

print("\n10. SENTENCE TOKENIZATION")

sentences = sent_tokenize(T3)

for index, sentence in enumerate(sentences, start=1):
    print(f"Sentence {index}: {sentence}")


# --------------------------------------------------
# 13. POS TAGGING
# --------------------------------------------------

print("\n11. PART-OF-SPEECH TAGGING")

t3_tokens = word_tokenize(T3)
pos_results = pos_tag(t3_tokens)

for token, tag in pos_results:
    print(f"{token:<15} {tag}")


# --------------------------------------------------
# 14. NAMED ENTITY RECOGNITION
# --------------------------------------------------

print("\n12. NAMED ENTITY RECOGNITION")

named_entities = ne_chunk(pos_results)

print(named_entities)

print("\nNamed entity chunks:")

for chunk in named_entities:
    if hasattr(chunk, "label"):
        entity_name = " ".join(word for word, tag in chunk.leaves())
        print(entity_name, "->", chunk.label())


# --------------------------------------------------
# 15. COMPLETE PIPELINE: T2
# --------------------------------------------------

print("\n13. COMPLETE PREPROCESSING PIPELINE")

print("\nOriginal text:")
print(T2)

# Step 1: lowercase
pipeline_text = T2.lower()

# Step 2: tokenize
pipeline_tokens = word_tokenize(pipeline_text)

# Step 3: remove punctuation and emoji tokens
pipeline_tokens = [
    token for token in pipeline_tokens
    if re.search(r"[a-z]", token)
]

# Step 4A: default stopword removal
default_pipeline = [
    token for token in pipeline_tokens
    if token not in english_stopwords
]

# Step 4B: negation-preserving stopword removal
safe_pipeline = [
    token for token in pipeline_tokens
    if token not in safe_stopwords
]

print("\nDefault pipeline:")
print(default_pipeline)

print("\nNegation-preserving pipeline:")
print(safe_pipeline)


# --------------------------------------------------
# 16. FINAL COMPARISON
# --------------------------------------------------

print("\n14. FINAL COMPARISON")

print("Default pipeline keeps:", default_pipeline)
print("Negation-safe pipeline keeps:", safe_pipeline)

print("\nRecommendation:")
print(
    "For a sentiment analysis or spam detection system, "
    "preprocessing should preserve meaningful information. "
    "Negation words such as 'not' should not be removed "
    "because they can change the meaning of a sentence."
)

print("\nEND OF NLP PREPROCESSING PRACTICE")
