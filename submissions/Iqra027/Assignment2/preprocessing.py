# Assignment 2 — Part B: Practical Preprocessing Script
# Setup once: python code/download-nltk-data.py
# Run:        python preprocessing_practice.py > output.txt

import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Input texts
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

# Input word list
words = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]

# --------------------------------
# 1) Lowercasing
# --------------------------------
print("=== 1) LOWERCASING ===")
t1_lower = T1.lower()
print("Before:", T1)
print("After: ", t1_lower)

# --------------------------------
# 2) Whitespace normalization
# --------------------------------
print("\n=== 2) WHITESPACE NORMALIZATION ===")
t4_normalized = " ".join(T4.split())
print("Before:", repr(T4))
print("After: ", repr(t4_normalized))

# --------------------------------
# 3) Regex cleaning
# --------------------------------
print("\n=== 3) REGEX CLEANING ===")
t1_deleted = re.sub(r"[^a-z\s]", " ", t1_lower)
deleted_chars = [ch for ch in t1_lower if re.match(r"[^a-z\s]", ch)]
print("Lowercased T1: ", repr(t1_lower))
print("After [^a-z\\s]:", repr(t1_deleted))
print("Deleted chars: ", deleted_chars)

# Custom replacement rules preserving structural spam signals
t1_replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", T1)
t1_replaced = re.sub(r"£\d{1,3}(?:,\d{3})*(?:\.\d+)?", "MONEY", t1_replaced)
t1_replaced = re.sub(r"!+", "!", t1_replaced)
print("Replaced T1:   ", t1_replaced)

# --------------------------------
# 4) Tokenization
# --------------------------------
print("\n=== 4) TOKENIZATION ===")
t2_split = T2.split()
t2_word_tokens = word_tokenize(T2)
print("T2 split() tokens:        ", t2_split)
print("T2 split() count:         ", len(t2_split))
print("T2 word_tokenize() tokens:", t2_word_tokens)
print("T2 word_tokenize() count: ", len(t2_word_tokens))

t4_tokens = word_tokenize(T4)
print("T4 tokens:                ", t4_tokens)
print("T4 token count:           ", len(t4_tokens))
print("T4 vocabulary size:       ", len(set(t4_tokens)))

# --------------------------------
# 5) Stopwords and negation
# --------------------------------
print("\n=== 5) STOPWORDS AND NEGATION ===")
default_stopwords = set(stopwords.words("english"))
t2_no_stop = [tok for tok in t2_word_tokens if tok.lower() not in default_stopwords]

negation_words = {"not", "n't", "no", "never"}
custom_stopwords = default_stopwords - negation_words
t2_keep_neg = [tok for tok in t2_word_tokens if tok.lower() not in custom_stopwords]

print("Original T2 tokens:       ", t2_word_tokens)
print("Default stopwords removed:", t2_no_stop)
print("Keeping negation words:   ", t2_keep_neg)

# --------------------------------
# 6) Stemming
# --------------------------------
print("\n=== 6) STEMMING ===")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

for word in words:
    print(
        f"{word:13} -> "
        f"Porter: {porter.stem(word):10} | "
        f"Lancaster: {lancaster.stem(word):10} | "
        f"Snowball: {snowball.stem(word):10}"
    )

# --------------------------------
# 7) Lemmatization
# --------------------------------
print("\n=== 7) LEMMATIZATION ===")
lemmatizer = WordNetLemmatizer()
pos_hints = {
    "connection": "n",
    "connected": "v",
    "connecting": "v",
    "organization": "n",
    "organized": "v",
    "university": "n",
    "universal": "a",
    "flies": "n",
    "wolves": "n",
    "better": "a",
    "ran": "v",
}

for word in words:
    hint = pos_hints[word]
    default_lemma = lemmatizer.lemmatize(word)
    hinted_lemma = lemmatizer.lemmatize(word, pos=hint)
    print(f"{word:13} -> default: {default_lemma:13} | pos='{hint}': {hinted_lemma:13}")

# --------------------------------
# 8) Sentence splitting
# --------------------------------
print("\n=== 8) SENTENCE SPLITTING ===")
t3_sentences = sent_tokenize(T3)
print("Sentences:", t3_sentences)
print("Count:    ", len(t3_sentences))

# --------------------------------
# 9) POS tagging and NER
# --------------------------------
print("\n=== 9) POS TAGGING AND NER ===")
t3_tokens = word_tokenize(T3)
t3_pos_tags = pos_tag(t3_tokens)
t3_ner_tree = ne_chunk(t3_pos_tags)
print("POS tags:", t3_pos_tags)
print("NER tree:\n", t3_ner_tree)

# --------------------------------
# 10) Full pipeline on T2
# --------------------------------
print("\n=== 10) FULL PIPELINE ===")
stage_lower = T2.lower()
stage_regex = re.sub(r"[^a-z\s]", " ", stage_lower)
stage_tokens = word_tokenize(stage_regex)

# Run 1: Default NLTK stopwords
stage_stop_default = [tok for tok in stage_tokens if tok not in default_stopwords]
stage_lemma_default = [lemmatizer.lemmatize(tok) for tok in stage_stop_default]

# Run 2: Custom negation-preserving stopwords
stage_stop_custom = [tok for tok in stage_tokens if tok not in custom_stopwords]
stage_lemma_custom = [lemmatizer.lemmatize(tok) for tok in stage_stop_custom]

print("Original:                      ", T2)
print("1. Lowercase:                  ", stage_lower)
print("2. Regex [^a-z\\s]:            ", stage_regex)
print("3. Tokenize:                   ", stage_tokens)
print("4a. Stopwords (default):       ", stage_stop_default)
print("5a. Lemmatize (default):       ", stage_lemma_default)
print("4b. Stopwords (keep negation): ", stage_stop_custom)
print("5b. Lemmatize (keep negation): ", stage_lemma_custom)