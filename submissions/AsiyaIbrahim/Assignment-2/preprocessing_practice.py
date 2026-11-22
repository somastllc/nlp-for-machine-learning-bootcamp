"""
preprocessing_practice.py  --  Assignment 2 (Preprocessing), Part B

"""

import re

import nltk
from nltk import pos_tag, ne_chunk, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer

# ---------------------------------------------------------------- data
T1 = "URGENT!!! You have WON a $1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "
# (copy T4 exactly from the assignment file, including its spaces)

WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]

NEGATIONS = {"not", "n't", "no", "never"}


def heading(number, title):
    print("\n" + "=" * 70)
    print(f"TASK {number}: {title}")
    print("=" * 70)


# ------------------------------------------------------- 1. lowercasing
heading(1, "Lowercasing T1")
t1_lower = T1.lower()
print("Before:", T1)
print("After: ", t1_lower)

# ------------------------------------------------------- 2. whitespace
heading(2, "Whitespace normalization of T4")
t4_clean = re.sub(r"\s+", " ", T4).strip()
print("Before:", repr(T4))
print("After: ", repr(t4_clean))

# ------------------------------------------------------- 3. regex cleaning
heading(3, "Regex cleaning of T1")
deleted_version = re.sub(r"[^a-z\s]", " ", t1_lower)
print("3a. Pattern [^a-z\\s] on lowercased T1")
print("Before:", t1_lower)
print("After: ", repr(deleted_version))
print("Everything that was matched (deleted):", re.findall(r"[^a-z\s]+", t1_lower))

print("\n3b. Replace instead of delete (applied to the original T1)")
replaced = T1
replaced = re.sub(r"\b\d{9,10}\b", "PHONE_NUMBER", replaced)       # phone first
replaced = re.sub(r"[£$€]\s?\d[\d,]*(?:\.\d+)?", "MONEY", replaced)  # then money
replaced = re.sub(r"!{2,}", "!", replaced)                         # !!! -> !
print("Before:", T1)
print("After: ", replaced)

# ------------------------------------------------------- 4. tokenization
heading(4, "Tokenization")
split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)
print("T2 split():        ", split_tokens, "->", len(split_tokens), "tokens")
print("T2 word_tokenize():", nltk_tokens, "->", len(nltk_tokens), "tokens")

t4_tokens = word_tokenize(T4)
print("\nT4 word_tokenize():", t4_tokens)
print("T4 token count:", len(t4_tokens))
print("T4 vocabulary size (case-sensitive):", len(set(t4_tokens)))
print("T4 vocabulary size (lowercased):    ", len(set(t.lower() for t in t4_tokens)))

# ------------------------------------------------------- 5. stopwords
heading(5, "Stopwords and negation (T2)")
default_stop = set(stopwords.words("english"))
custom_stop = default_stop - NEGATIONS      # keeps not / no / never (and n't is not listed)

tokens = word_tokenize(T2.lower())
print("Tokens:", tokens)
print("Negation words in NLTK default list:", sorted(NEGATIONS & default_stop))
print("Default stopwords removed:      ", [t for t in tokens if t not in default_stop])
print("Negation-keeping list removed:  ", [t for t in tokens if t not in custom_stop])

# ------------------------------------------------------- 6. stemming
heading(6, "Stemming: Porter vs Lancaster vs Snowball")
porter, lancaster, snowball = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")
print(f"{'word':<14}{'porter':<14}{'lancaster':<14}{'snowball':<14}")
stems = {"porter": {}, "lancaster": {}, "snowball": {}}
for w in WORDS:
    p, l, s = porter.stem(w), lancaster.stem(w), snowball.stem(w)
    stems["porter"][w], stems["lancaster"][w], stems["snowball"][w] = p, l, s
    print(f"{w:<14}{p:<14}{l:<14}{s:<14}")

print("\nDifferent words merged into the same stem:")
for name, mapping in stems.items():
    groups = {}
    for w, st in mapping.items():
        groups.setdefault(st, []).append(w)
    merged = {st: ws for st, ws in groups.items() if len(ws) > 1}
    print(f"  {name}: {merged if merged else 'none'}")

print("\nStems that are not words in WordNet (rough check):")
for name, mapping in stems.items():
    try:
        fake = sorted({st for st in mapping.values() if not wordnet.synsets(st)})
    except MemoryError:
        fake = ["<WordNet unavailable: low memory>"]
    print(f"  {name}: {fake}")

# ------------------------------------------------------- 7. lemmatization
heading(7, "Lemmatization: default vs correct POS hint")
lemmatizer = WordNetLemmatizer()
POS_HINT = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v",
    "university": "n", "universal": "a",
    "flies": "v", "wolves": "n", "better": "a", "ran": "v",
}
print(f"{'word':<14}{'default(noun)':<16}{'with hint':<14}{'hint':<6}{'changed?'}")
for w in WORDS:
    try:
        d = lemmatizer.lemmatize(w)
        h = lemmatizer.lemmatize(w, pos=POS_HINT[w])
    except MemoryError:
        d, h = w, w
    print(f"{w:<14}{d:<16}{h:<14}{POS_HINT[w]:<6}{'YES' if d != h else ''}")

# ------------------------------------------------------- 8. sentence splitting
heading(8, "Sentence splitting (T3)")
sentences = sent_tokenize(T3)
print("Number of sentences:", len(sentences))
for i, s in enumerate(sentences, 1):
    print(f"  {i}. {s}")

# ------------------------------------------------------- 9. POS + NER
heading(9, "POS tagging and NER (T3)")
t3_tokens = word_tokenize(T3)
t3_tags = pos_tag(t3_tokens)
print("POS tags:")
for word, tag in t3_tags:
    print(f"  {word:<10}{tag}")

tree = ne_chunk(t3_tags)
print("\nNamed entities found by ne_chunk:")
found = []
for node in tree:
    if hasattr(node, "label"):
        text = " ".join(tok for tok, _ in node.leaves())
        found.append((text, node.label()))
        print(f"  {text:<12}{node.label()}")
if not found:
    print("  (none)")
print("\nCheck by hand: were Ali (person), Mogadisho (place), Apple (organization)")
print("and Friday (day) all recognized? Which labels are wrong?")

# ------------------------------------------------------- 10. full pipeline


def wn_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def pipeline(text, stop_set, label):
    print(f"\n--- Pipeline run: {label} ---")
    print("Original:        ", text)
    lowered = text.lower()
    print("1. Lowercase:    ", lowered)
    # keeps apostrophes so that word_tokenize can still split "don't" into do + n't
    cleaned = re.sub(r"[^a-z\s']", " ", lowered)
    print("2. Regex:        ", repr(cleaned))
    toks = word_tokenize(cleaned)
    print("3. Tokenize:     ", toks)
    no_stop = [t for t in toks if t not in stop_set]
    print("4. Stopwords:    ", no_stop)
    lemmas = []
    for w, tag in pos_tag(no_stop):
        try:
            lemma = lemmatizer.lemmatize(w, wn_pos(tag))
        except MemoryError:
            lemma = w
        lemmas.append(lemma)
    print("5. Lemmatize:    ", lemmas)


heading(10, "Full pipeline on T2")
pipeline(T2, default_stop, "NLTK default stopwords")
pipeline(T2, custom_stop, "negation-keeping stopwords")
