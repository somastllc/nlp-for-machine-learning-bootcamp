# Assignment 2 - Preprocessing (practical)
# Setup once:  python code/download-nltk-data.py
# Run:         python preprocessing_practice.py > output.txt

import re
from collections import Counter, defaultdict

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]


def heading(title):
    print(f"\n=== {title} ===")


# --------------------------------
# 1) Lowercasing
# --------------------------------
heading("1) LOWERCASING (T1)")
t1_lower = T1.lower()
print("Before:", T1)
print("After: ", t1_lower)
shouted = [w for w in re.findall(r"[A-Za-z]+", T1) if w.isupper() and len(w) > 1]
print("Fully-capitalised words that lose their capitals:", shouted)

# --------------------------------
# 2) Whitespace normalization
# --------------------------------
heading("2) WHITESPACE NORMALIZATION (T4)")
t4_norm = " ".join(T4.split())
print("Before:", repr(T4))
print("After: ", repr(t4_norm))

# --------------------------------
# 3) Regex cleaning
# --------------------------------
heading("3) REGEX CLEANING (T1)")
deleted_version = re.sub(r"[^a-z\s]", " ", t1_lower)
print("Lowercased:         ", repr(t1_lower))
print("After [^a-z\\s]:     ", repr(deleted_version))

removed = Counter(ch for ch in t1_lower if not re.match(r"[a-z\s]", ch))
print("Characters deleted (char: count):")
for ch, n in sorted(removed.items()):
    print(f"  {ch!r}: {n}")

replaced = T1
replaced = re.sub(r"£\d[\d,]*(?:\.\d+)?", "MONEY", replaced)   # money first
replaced = re.sub(r"\b\d{9,10}\b", "PHONE_NUMBER", replaced)    # then phone
replaced = re.sub(r"!+", "!", replaced)                         # !!! -> !
print("Replace instead of delete:")
print("Before:", T1)
print("After: ", replaced)

# --------------------------------
# 4) Tokenization
# --------------------------------
heading("4) TOKENIZATION")
t2_split = T2.split()
t2_word = word_tokenize(T2)
print("T2 split():        ", t2_split, "->", len(t2_split), "tokens")
print("T2 word_tokenize():", t2_word, "->", len(t2_word), "tokens")

t4_tokens = word_tokenize(T4)
print("T4 word_tokenize():", t4_tokens)
print("T4 token count:    ", len(t4_tokens))
print("T4 vocabulary size:", len(set(t4_tokens)), "(case-sensitive)")
print("T4 vocabulary size:", len(set(t.lower() for t in t4_tokens)), "(after lowercasing)")

# --------------------------------
# 5) Stopwords and negation
# --------------------------------
heading("5) STOPWORDS AND NEGATION (T2)")
default_stops = set(stopwords.words("english"))
NEGATIONS = {"not", "n't", "no", "never", "nor"}
my_stops = default_stops - NEGATIONS

t2_tokens = word_tokenize(T2.lower())
print("Tokens (lowercased T2):    ", t2_tokens)
print("Default NLTK stopwords:    ", [t for t in t2_tokens if t not in default_stops])
print("My list (keeps negations): ", [t for t in t2_tokens if t not in my_stops])
print("'not' in default stopword list?", "not" in default_stops)
print("\"n't\" in default stopword list?", "n't" in default_stops)

# --------------------------------
# 6) Stemming
# --------------------------------
heading("6) STEMMING")
porter, lancaster, snowball = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")
stemmers = {"Porter": porter, "Lancaster": lancaster, "Snowball": snowball}

print(f"{'word':<14}{'Porter':<14}{'Lancaster':<14}{'Snowball':<14}")
for w in WORDS:
    print(f"{w:<14}{porter.stem(w):<14}{lancaster.stem(w):<14}{snowball.stem(w):<14}")

print("\nDifferent words merged into the same stem:")
for name, st in stemmers.items():
    groups = defaultdict(list)
    for w in WORDS:
        groups[st.stem(w)].append(w)
    merged = {s: ws for s, ws in groups.items() if len(ws) > 1}
    print(f"  {name}: {merged}")

print("\nStems that are not dictionary words (no WordNet entry):")
for name, st in stemmers.items():
    fake = sorted({(w, st.stem(w)) for w in WORDS if not wordnet.synsets(st.stem(w))})
    print(f"  {name}: {fake}")

# --------------------------------
# 7) Lemmatization
# --------------------------------
heading("7) LEMMATIZATION")
lemmatizer = WordNetLemmatizer()
POS = {"connection": "n", "connected": "v", "connecting": "v",
       "organization": "n", "organized": "v", "university": "n",
       "universal": "a", "flies": "v", "wolves": "n",
       "better": "a", "ran": "v"}

print(f"{'word':<14}{'default':<14}{'pos':<5}{'with hint':<14}changed?")
changed = []
for w in WORDS:
    d = lemmatizer.lemmatize(w)
    h = lemmatizer.lemmatize(w, pos=POS[w])
    flag = "YES" if d != h else ""
    if flag:
        changed.append(w)
    print(f"{w:<14}{d:<14}{POS[w]:<5}{h:<14}{flag}")
print("Words that changed only with the hint:", changed)

# --------------------------------
# 8) Sentence splitting
# --------------------------------
heading("8) SENTENCE SPLITTING (T3)")
sents = sent_tokenize(T3)
for i, s in enumerate(sents, 1):
    print(f"Sentence {i}: {s!r}")
print("Number of sentences:", len(sents))
print("Any sentence ends with 'Mr.'?", any(s.rstrip().endswith("Mr.") for s in sents))

# --------------------------------
# 9) POS tagging and NER
# --------------------------------
heading("9) POS TAGGING AND NER (T3)")
tagged = pos_tag(word_tokenize(T3))
print("POS tags:")
for word, tag in tagged:
    print(f"  {word:<10}{tag}")

tree = ne_chunk(tagged)
entities = []
for node in tree:
    if isinstance(node, Tree):
        entities.append((" ".join(w for w, _ in node.leaves()), node.label()))

GOLD = {"Smith": "PERSON", "London": "GPE", "Apple": "ORGANIZATION"}
print("\nEntities found by ne_chunk:")
print(f"  {'text':<14}{'label':<14}correct?")
for text, label in entities:
    key = next((k for k in GOLD if k in text.split()), None)
    ok = "yes" if key and GOLD[key] == label else "NO"
    print(f"  {text:<14}{label:<14}{ok}")

found_text = " ".join(t for t, _ in entities)
print("\nMissed (expected but not found):")
for k, lab in list(GOLD.items()) + [("Friday", "DATE (day)")]:
    if k not in found_text.split():
        print(f"  {k} (should be {lab})")
print("\nFull tree:")
print(tree)

# --------------------------------
# 10) Full pipeline on T2
# --------------------------------
heading("10) FULL PIPELINE (T2)")


def pipeline(text, stops, label):
    s1 = text.lower()
    s2 = re.sub(r"[^a-z\s]", " ", s1)
    s3 = word_tokenize(s2)
    s4 = [t for t in s3 if t not in stops]
    s5 = [lemmatizer.lemmatize(t) for t in s4]
    print(f"--- {label} ---")
    print("Original:          ", text)
    print("Lowercased:        ", s1)
    print("Regex cleaned:     ", repr(s2))
    print("Tokens:            ", s3)
    print("Stopwords removed: ", s4)
    print("Lemmatized:        ", s5)
    print("'not' survived?    ", "not" in s5)
    return s5


pipeline(T2, default_stops, "A) NLTK default stopwords")
pipeline(T2, my_stops, "B) My negation-keeping list")
