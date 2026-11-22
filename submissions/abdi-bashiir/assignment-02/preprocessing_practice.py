# Assignment 2 - Preprocessing (practical)
# Setup once:  python code/download-nltk-data.py
# Run:         python preprocessing_practice.py > output.txt

import re
import sys

# Print UTF-8 whatever the terminal uses (pound sign and emoji stay correct)
sys.stdout.reconfigure(encoding="utf-8")

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

T1 = "URGENT!!! You have WON a \u00a31,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all \U0001F61E"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran",
]

# --------------------------------
# 1) Lowercasing
# --------------------------------
print("\n=== 1) LOWERCASING ===")
print("Before:", T1)
print("After: ", T1.lower())

# --------------------------------
# 2) Whitespace normalization
# --------------------------------
print("\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(T4))
print("After: ", repr(" ".join(T4.split())))

# --------------------------------
# 3) Regex cleaning
# --------------------------------
lowered = T1.lower()
deleted = re.sub(r"[^a-z\s]", "", lowered)

print("\n=== 3) REGEX CLEANING ===")
print("3a) Pattern [^a-z\\s] applied to lowercased T1")
print("Before:", repr(lowered))
print("After: ", repr(deleted))
print("Deleted characters:", sorted(set(re.findall(r"[^a-z\s]", lowered))))

# 3b) Replace instead of delete (general patterns, not tied to one value)
replaced = T1
replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", replaced)       # 10 digits starting with 0
replaced = re.sub(r"\u00a3\s?\d[\d,]*(?:\.\d+)?", "MONEY", replaced)   # pound amount, e.g. 1,000 or 5.50 after the pound sign
replaced = re.sub(r"!{2,}", "!", replaced)                        # repeated ! -> one !

print("3b) Replacement rules")
print("Before:", repr(T1))
print("After: ", repr(replaced))

# --------------------------------
# 4) Tokenization
# --------------------------------
split_tokens = T2.split()
word_tokens = word_tokenize(T2)
t4_tokens = word_tokenize(T4)

print("\n=== 4) TOKENIZATION ===")
print("T2 split():", split_tokens)
print("T2 split() count:", len(split_tokens))
print("T2 word_tokenize():", word_tokens)
print("T2 word_tokenize() count:", len(word_tokens))
print("T4 word_tokenize():", t4_tokens)
print("T4 token count:", len(t4_tokens))
print("T4 vocabulary size (case-sensitive):", len(set(t4_tokens)))
print("T4 vocabulary size (after lowercasing):", len({t.lower() for t in t4_tokens}))

# --------------------------------
# 5) Stopwords and negation
# --------------------------------
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(T2)

default_result = [t for t in tokens if t.lower() not in stop_words]

negation_words = {"not", "n't", "no", "never"}
stop_words_keep_negation = stop_words - negation_words
negation_result = [t for t in tokens if t.lower() not in stop_words_keep_negation]

print("\n=== 5) STOPWORDS AND NEGATION ===")
print("Tokens:", tokens)
print("Negation words inside NLTK's default list:",
      sorted(w for w in negation_words if w in stop_words))
print("Default stopword removal:", default_result)
print("Keeping negation words:  ", negation_result)

# --------------------------------
# 6) Stemming
# --------------------------------
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\n=== 6) STEMMING COMPARISON ===")
print(f"{'Word':<14}{'Porter':<12}{'Lancaster':<12}{'Snowball':<12}")
for w in words:
    print(f"{w:<14}{porter.stem(w):<12}{lancaster.stem(w):<12}{snowball.stem(w):<12}")

for name, stemmer in [("Porter", porter), ("Lancaster", lancaster), ("Snowball", snowball)]:
    groups = {}
    for w in words:
        groups.setdefault(stemmer.stem(w), []).append(w)
    print(f"Words merged into one stem by {name}:",
          {stem: ws for stem, ws in groups.items() if len(ws) > 1})

# --------------------------------
# 7) Lemmatization
# --------------------------------
lemmatizer = WordNetLemmatizer()
pos_hints = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v", "university": "n",
    "universal": "a", "flies": "v", "wolves": "n",
    "better": "a", "ran": "v",
}

print("\n=== 7) LEMMATIZATION ===")
print(f"{'Word':<14}{'Default':<14}{'With hint':<14}{'POS'}")
changed = []
for w in words:
    default = lemmatizer.lemmatize(w)
    hinted = lemmatizer.lemmatize(w, pos=pos_hints[w])
    print(f"{w:<14}{default:<14}{hinted:<14}{pos_hints[w]}")
    if default != hinted:
        changed.append((w, default, hinted))
print("Words that changed only because of the hint:", changed)

# --------------------------------
# 8) Sentence splitting
# --------------------------------
sentences = sent_tokenize(T3)

print("\n=== 8) SENTENCE SPLITTING ===")
for i, s in enumerate(sentences, 1):
    print(i, repr(s))
print("Sentence count:", len(sentences))

# --------------------------------
# 9) POS tagging and named entity recognition
# --------------------------------
tagged = pos_tag(word_tokenize(T3))
tree = ne_chunk(tagged)

print("\n=== 9) POS TAGGING ===")
print(tagged)

print("\n=== 9) NAMED ENTITY RECOGNITION ===")
entities = []
for node in tree:
    if hasattr(node, "label"):
        text = " ".join(w for w, _ in node.leaves())
        entities.append((text, node.label()))
print("Tree:")
print(tree)
print("Entities found:")
for text, label in entities:
    print(f"  {text!r} -> {label}")

print("What was recognized? (ne_chunk has no DATE label)")
expected = [
    ("Smith", "PERSON"),
    ("London", "GPE"),
    ("Apple", "ORGANIZATION"),
    ("Friday", "a day (not available in ne_chunk)"),
]
for word, kind in expected:
    labels = [lab for text, lab in entities if word in text.split()]
    print(f"  {word:<8} expected: {kind:<35} model: {labels if labels else 'no entity'}")

# --------------------------------
# 10) Full pipeline on T2
# --------------------------------
def full_pipeline(remove_these):
    s = T2.lower()
    print("1 Lowercase :", repr(s))
    s = re.sub(r"[^a-z'\s]", " ", s)      # keep apostrophes so "n't" survives
    s = re.sub(r"\s+", " ", s).strip()
    print("2 Regex     :", repr(s))
    toks = word_tokenize(s)
    print("3 Tokenize  :", toks)
    toks = [t for t in toks if t not in remove_these]
    print("4 Stopwords :", toks)
    toks = [lemmatizer.lemmatize(t) for t in toks]
    print("5 Lemmatize :", toks)


print("\n=== 10) FULL PIPELINE ===")
print("\nA. NLTK default stopwords")
full_pipeline(stop_words)
print("\nB. Negation-keeping stopwords")
full_pipeline(stop_words_keep_negation)
