#!/usr/bin/env python3
"""Assignment 2 - Preprocessing practical.

Run from the assignment folder:
    python preprocessing_practice.py > output.txt

Before running, install NLTK and the course data:
    python -m pip install nltk
    python code/download-nltk-data.py
"""
import re
import nltk
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."
WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]

def heading(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def resource_error(name, exc):
    print(f"[NLTK DATA MISSING: {name}]")
    print("Run: python code/download-nltk-data.py")
    print(f"Details: {exc}")

# 1. Lowercasing
heading("1) LOWERCASE T1")
print("Before:", T1)
print("After: ", T1.lower())
print("Possible information lost: uppercase emphasis such as URGENT and WON.")

# 2. Whitespace
heading("2) WHITESPACE NORMALIZATION - T4")
print("Before:", repr(T4))
print("After: ", repr(" ".join(T4.split())))

# 3. Regex cleaning
heading("3) REGEX CLEANING - T1")
lower_t1 = T1.lower()
deleted = re.findall(r"[^a-z\s]", lower_t1)
cleaned = re.sub(r"[^a-z\s]", " ", lower_t1)
print("Lowercase before:", repr(lower_t1))
print("Pattern [^a-z\\s] result:", repr(cleaned))
print("Characters/matches deleted:", deleted)
print("Note: the £ sign, digits, commas, periods and exclamation marks are removed.")
# Replace phone first, then currency amount, then repeated exclamation marks.
replaced = re.sub(r"(?<!\d)0712345678(?!\d)", "PHONE_NUMBER", lower_t1)
replaced = re.sub(r"£\s*[\d,]+", "MONEY", replaced)
replaced = re.sub(r"!{2,}", "!", replaced)
print("Replacement version:", replaced)

# 4. Tokenization
heading("4) TOKENIZATION")
print("T2:", T2)
print("split():", T2.split(), "| count:", len(T2.split()))
try:
    t2_word_tokens = word_tokenize(T2)
    print("word_tokenize():", t2_word_tokens, "| count:", len(t2_word_tokens))
except LookupError as exc:
    resource_error("punkt_tab / punkt", exc)
    t2_word_tokens = None
normalized_t4 = " ".join(T4.split())
t4_split_tokens = normalized_t4.split()
print("T4 normalized:", repr(normalized_t4))
print("T4 split tokens:", t4_split_tokens)
print("T4 token count:", len(t4_split_tokens))
print("T4 vocabulary size (unique tokens):", len(set(t4_split_tokens)))

# 5. Stopwords and negation
heading("5) STOPWORDS AND NEGATION - T2")
try:
    default_stop = set(stopwords.words("english"))
    if t2_word_tokens is None:
        t2_word_tokens = word_tokenize(T2)
    default_removed = [t for t in t2_word_tokens if t.lower() not in default_stop]
    keep_negation_stop = default_stop - {"not", "n't", "no", "never"}
    keep_negation = [t for t in t2_word_tokens if t.lower() not in keep_negation_stop]
    print("Tokens:", t2_word_tokens)
    print("NLTK default stopwords removed:", default_removed)
    print("Custom list keeps not / n't / no / never:", keep_negation)
except LookupError as exc:
    resource_error("stopwords", exc)

# 6. Stemming
heading("6) STEMMING COMPARISON")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
print(f"{'Word':<16} {'Porter':<16} {'Lancaster':<16} {'Snowball':<16}")
for word in WORDS:
    print(f"{word:<16} {porter.stem(word):<16} {lancaster.stem(word):<16} {snowball.stem(word):<16}")
for name, stemmer in [("Porter", porter), ("Lancaster", lancaster), ("Snowball", snowball)]:
    groups = {}
    for word in WORDS:
        groups.setdefault(stemmer.stem(word), []).append(word)
    merged = [members for members in groups.values() if len(members) > 1]
    print(f"{name} merged different words:", merged)
print("Some stems such as 'organ', 'univers', 'fli' or 'wolv' are not ordinary dictionary words.")

# 7. Lemmatization
heading("7) WORDNET LEMMATIZATION")
pos_hints = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v", "university": "n",
    "universal": "a", "flies": "v", "wolves": "n",
    "better": "a", "ran": "v"
}
lemmatizer = WordNetLemmatizer()
try:
    print(f"{'Word':<16} {'Default noun':<20} {'With POS hint':<20} {'POS':<5}")
    for word in WORDS:
        default = lemmatizer.lemmatize(word)
        hinted = lemmatizer.lemmatize(word, pos=pos_hints[word])
        print(f"{word:<16} {default:<20} {hinted:<20} {pos_hints[word]:<5}")
except LookupError as exc:
    resource_error("wordnet / omw-1.4", exc)

# 8. Sentence splitting
heading("8) SENTENCE SPLITTING - T3")
try:
    sentences = sent_tokenize(T3)
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
    print("Sentence count:", len(sentences))
except LookupError as exc:
    resource_error("punkt_tab / punkt", exc)

# 9. POS tagging and NER
heading("9) POS TAGGING AND NAMED ENTITY RECOGNITION - T3")
try:
    tagged = pos_tag(word_tokenize(T3))
    print("POS tags:", tagged)
    tree = ne_chunk(tagged)
    print("NER tree:", tree)
    print("Entities:")
    found = False
    for node in tree:
        if hasattr(node, "label"):
            found = True
            print(" -", " ".join(word for word, tag in node.leaves()), "=>", node.label())
    if not found:
        print(" - No named entities returned by the model.")
except LookupError as exc:
    resource_error("punkt_tab, averaged_perceptron_tagger_eng, maxent_ne_chunker_tab, words", exc)

# 10. Full pipeline twice
heading("10) FULL PIPELINE - T2")
def pipeline(text, stop_set, label):
    print(f"\n--- {label} ---")
    lowered = text.lower()
    print("Original:          ", text)
    print("Lowercase:         ", lowered)
    # Keep apostrophes here so n't remains available to the tokenizer.
    regex_cleaned = re.sub(r"[^a-z\s']", " ", lowered)
    print("Regex cleaned:     ", regex_cleaned)
    tokens = word_tokenize(regex_cleaned)
    print("Tokens:            ", tokens)
    filtered = [token for token in tokens if token.lower() not in stop_set]
    print("Stopwords removed: ", filtered)
    lemmas = [lemmatizer.lemmatize(token) for token in filtered]
    print("Lemmatized:        ", lemmas)

try:
    default_stop = set(stopwords.words("english"))
    custom_stop = default_stop - {"not", "n't", "no", "never"}
    pipeline(T2, default_stop, "NLTK DEFAULT STOPWORDS")
    pipeline(T2, custom_stop, "CUSTOM STOPWORDS - KEEP NEGATION")
except LookupError as exc:
    resource_error("stopwords / punkt_tab / wordnet", exc)

print("\nEND OF PRACTICAL OUTPUT")
