# Assignment 2 — Preprocessing (practice script)
# Uses the same simple structure as the Lesson 2 teaching script.
# Setup once: python code/download-nltk-data.py
# Run:        python preprocessing_practice.py
# Save output: python preprocessing_practice.py > output.txt

import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# --------------------------------
# Texts and words given in the assignment
# --------------------------------
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]

# --------------------------------
# 1) Lowercasing
# Capital letters in URGENT, WON and NOW may help spam detection.
# --------------------------------
print("\n=== 1) LOWERCASING ===")
print("Before:", T1)
print("After: ", T1.lower())

# --------------------------------
# 2) Whitespace normalization
# split() removes extra spaces; join() adds one space between words.
# repr() shows the spaces clearly.
# --------------------------------
print("\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(T4))
print("After: ", repr(" ".join(T4.split())))

# --------------------------------
# 3) Regex cleaning
# [^a-z\s] removes everything except lowercase English letters and spaces.
# Better rules can replace useful information instead of deleting it.
# --------------------------------
lowered = T1.lower()
cleaned = re.sub(r"[^a-z\s]", " ", lowered)

replaced = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", T1)
replaced = re.sub(r"£\s?\d[\d,]*(?:\.\d+)?", "MONEY", replaced)
replaced = re.sub(r"!{2,}", "!", replaced)

print("\n=== 3) REGEX CLEANING ===")
print("Original:      ", repr(T1))
print("Lowercased:   ", repr(lowered))
print("Deleted regex:", repr(cleaned))
print("Replaced regex:", repr(replaced))
print("Deleted by [^a-z\\s]: £, 1,000, 0712345678, ! and . (symbols and digits)")

# --------------------------------
# 4) Tokenization
# split() separates spaces; word_tokenize() also handles punctuation.
# Vocabulary size means the number of unique tokens.
# --------------------------------
t2_split = T2.split()
t2_tokens = word_tokenize(T2)
t4_tokens = word_tokenize(T4)

print("\n=== 4) TOKENIZATION ===")
print("T2:              ", T2)
print("split():          ", t2_split)
print("split() count:    ", len(t2_split))
print("word_tokenize():  ", t2_tokens)
print("Word token count: ", len(t2_tokens))
print("T4 tokens:       ", t4_tokens)
print("T4 token count:  ", len(t4_tokens))
print("T4 vocabulary:   ", len(set(t4_tokens)))

# --------------------------------
# 5) Stopwords and negation
# Default stopwords may remove 'not'. Keep not, n't, no and never.
# Lowercase tokens before checking the lowercase stopword lists.
# --------------------------------
stop_words = set(stopwords.words("english"))
keep_negation = set(stop_words)
for word in ["not", "n't", "no", "never"]:
    keep_negation.discard(word)

t2_lower_tokens = [token.lower() for token in t2_tokens]
default_tokens = [token for token in t2_lower_tokens if token not in stop_words]
negation_tokens = [token for token in t2_lower_tokens if token not in keep_negation]

print("\n=== 5) STOPWORDS AND NEGATION ===")
print("Tokens:                ", t2_lower_tokens)
print("Default stopwords:     ", default_tokens)
print("Keeping negation words:", negation_tokens)

# --------------------------------
# 6) Comparing stemmers
# Porter, Lancaster and Snowball can give different stems.
# Look for merged words and stems that are not dictionary words.
# --------------------------------
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\n=== 6) COMPARING STEMMERS ===")
print(f"{'Word':<16} {'Porter':<16} {'Lancaster':<16} {'Snowball':<16}")
for word in words:
    print(f"{word:<16} {porter.stem(word):<16} {lancaster.stem(word):<16} {snowball.stem(word):<16}")

for name, stemmer in [("Porter", porter), ("Lancaster", lancaster), ("Snowball", snowball)]:
    groups = {}
    for word in words:
        stem = stemmer.stem(word)
        groups.setdefault(stem, []).append(word)
    merged = [(stem, originals) for stem, originals in groups.items() if len(originals) > 1]
    print(name, "merged word pairs/groups:", merged if merged else "None in this list")

# --------------------------------
# 7) Lemmatization with POS hints
# WordNetLemmatizer uses noun ('n') by default.
# Choose verb ('v') or adjective ('a') where suitable.
# --------------------------------
lemmatizer = WordNetLemmatizer()
pos_hints = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v", "university": "n",
    "universal": "a", "flies": "v", "wolves": "n",
    "better": "a", "ran": "v"
}

print("\n=== 7) LEMMATIZATION WITH POS HINTS ===")
print(f"{'Word':<16} {'Default':<16} {'POS':<6} {'With hint':<16}")
for word in words:
    hint = pos_hints[word]
    print(f"{word:<16} {lemmatizer.lemmatize(word):<16} {hint:<6} {lemmatizer.lemmatize(word, pos=hint):<16}")

# --------------------------------
# 8) Sentence splitting
# The abbreviations Mr. and a.m. should not cause incorrect breaks.
# --------------------------------
sentences = sent_tokenize(T3)
print("\n=== 8) SENTENCE SPLITTING ===")
print("Original:", T3)
print("Sentences:", sentences)
print("Count:    ", len(sentences))

# --------------------------------
# 9) POS tagging and named entity recognition (NER)
# NLTK may label people, places and organizations incorrectly or miss them.
# Inspect the output rather than assuming the model is always right.
# --------------------------------
tagged = pos_tag(word_tokenize(T3))
entities = ne_chunk(tagged)

print("\n=== 9) POS TAGGING AND NER ===")
print("POS tags:", tagged)
print("NER tree:", entities)
print("Entities found (text -> label):")
for item in entities:
    if hasattr(item, "label"):
        entity_text = " ".join(word for word, tag in item.leaves())
        print(entity_text, "->", item.label())
print("Check manually: Smith (person), London (place), Friday (day), Apple (company).")

# --------------------------------
# 10) Full pipeline: lowercase -> regex -> tokenize -> stopwords -> lemmatize
# Keep apostrophes in regex so "don't" becomes "do" and "n't".
# Removing all apostrophes too early can destroy the negation.
# --------------------------------
step_lower = T2.lower()
step_clean = re.sub(r"[^a-z\s']", " ", step_lower)
step_tokens = word_tokenize(step_clean)
step_default = [token for token in step_tokens if token not in stop_words]
step_default_lemmas = [lemmatizer.lemmatize(token) for token in step_default]
step_negation = [token for token in step_tokens if token not in keep_negation]
step_negation_lemmas = [lemmatizer.lemmatize(token) for token in step_negation]

print("\n=== 10) FULL PIPELINE ===")
print("Original:                    ", T2)
print("Lowercased:                  ", step_lower)
print("Regex cleaned:               ", step_clean)
print("Tokens:                      ", step_tokens)
print("Default stopwords removed:   ", step_default)
print("Default lemmatized:          ", step_default_lemmas)
print("Negation-keeping stopwords:  ", step_negation)
print("Negation-keeping lemmatized: ", step_negation_lemmas)
