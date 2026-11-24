import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all "
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]


def heading(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(title)
    print(f"{'=' * 65}")


# Task 1: Lowercasing
heading("Task 1: Lowercasing (T1)")
t1_lower = T1.lower()
print("Before:", T1)
print("After :", t1_lower)


# Task 2: Whitespace normalization
heading("Task 2: Whitespace Normalization (T4)")
t4_norm = re.sub(r"\s+", " ", T4).strip()
print("Before:", repr(T4))
print("After :", repr(t4_norm))


# Task 3: Regex cleaning and custom replacements
heading("Task 3a: Regex Cleaning with [^a-z\\s] on Lowercased T1")
t1_cleaned = re.sub(r"[^a-z\s]", "", t1_lower)
deleted_chars = re.findall(r"[^a-z\s]", t1_lower)
print("Before:", t1_lower)
print("After :", t1_cleaned)
print("Deleted characters:", deleted_chars)

heading("Task 3b: Custom Regex Replacements on T1")
t1_replaced = re.sub(r"\b0\d{9}\b|\b\d{10}\b", "PHONE_NUMBER", T1)
t1_replaced = re.sub(r"£\d+(?:,\d{3})*(?:\.\d{2})?", "MONEY", t1_replaced)
t1_replaced = re.sub(r"!{3,}", "!", t1_replaced)
print("Before:", T1)
print("After :", t1_replaced)


# Task 4: Tokenization
heading("Task 4a: Tokenization of T2 (split vs word_tokenize)")
t2_split = T2.split()
t2_wt = word_tokenize(T2)
print("Before:", T2)
print("split() tokens       :", t2_split)
print("split() token count  :", len(t2_split))
print("word_tokenize tokens :", t2_wt)
print("word_tokenize count  :", len(t2_wt))

heading("Task 4b: Tokenization of T4 (Token Count & Vocabulary Size)")
t4_tokens = word_tokenize(t4_norm)
t4_vocab = set(t4_tokens)
print("Before (normalized T4):", t4_norm)
print("Tokens          :", t4_tokens)
print("Token count     :", len(t4_tokens))
print("Vocabulary size :", len(t4_vocab))
print("Unique tokens   :", sorted(t4_vocab))


# Task 5: Stopwords and negation
heading("Task 5: Stopwords and Negation on T2")
default_stops = set(stopwords.words("english"))
negations = {"not", "n't", "no", "never"}
custom_stops = default_stops - negations

t2_no_default_stops = [tok for tok in t2_wt if tok.lower() not in default_stops]
t2_no_custom_stops = [tok for tok in t2_wt if tok.lower() not in custom_stops]

print("Before:", t2_wt)
print("After (default stopwords removed):", t2_no_default_stops)
print("After (negation-keeping list)    :", t2_no_custom_stops)


# Task 6: Stemming
heading("Task 6: Stemming Comparison (Porter, Lancaster, Snowball)")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)
for w in WORDS:
    print(f"{w:<15} {porter.stem(w):<15} {lancaster.stem(w):<15} {snowball.stem(w):<15}")


# Task 7: Lemmatization
heading("Task 7: Lemmatization (Default vs POS Hint)")
lemmatizer = WordNetLemmatizer()
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
    "ran": "v",
}

print(f"{'Word':<15} {'POS':<6} {'Default':<15} {'With POS':<15} {'Changed?'}")
print("-" * 65)
for w in WORDS:
    d_lem = lemmatizer.lemmatize(w)
    p_hint = pos_hints[w]
    p_lem = lemmatizer.lemmatize(w, pos=p_hint)
    changed = "Yes" if d_lem != p_lem else "No"
    print(f"{w:<15} {p_hint:<6} {d_lem:<15} {p_lem:<15} {changed}")


# Task 8: Sentence splitting
heading("Task 8: Sentence Splitting on T3")
t3_sents = sent_tokenize(T3)
print("Before:", T3)
print("Sentence count:", len(t3_sents))
for idx, s in enumerate(t3_sents, start=1):
    print(f"  Sentence {idx}: {s}")


# Task 9: POS tagging and NER
heading("Task 9: POS Tagging and NER on T3")
t3_toks = word_tokenize(T3)
t3_pos = pos_tag(t3_toks)
t3_ner = ne_chunk(t3_pos)

print("Before:", T3)
print("POS tags:", t3_pos)
print("\nNE Chunk Tree:")
print(t3_ner)
print("\nExtracted Named Entities:")
for tree in t3_ner:
    if hasattr(tree, "label"):
        entity_text = " ".join(tok for tok, _ in tree.leaves())
        print(f"  Entity: {entity_text:<15} | Label: {tree.label()}")


# Task 10: Full pipeline on T2
heading("Task 10: Full Pipeline on T2")
def run_pipeline(text: str, stop_set: set) -> dict:
    s1_lower = text.lower()
    s2_regex = re.sub(r"\s+", " ", re.sub(r"[^a-z'\s]", " ", s1_lower)).strip()
    s3_tokens = word_tokenize(s2_regex)
    s4_stops = [t for t in s3_tokens if t not in stop_set]
    s5_lemma = [lemmatizer.lemmatize(t) for t in s4_stops]
    return {
        "1_lowercase": s1_lower,
        "2_regex": s2_regex,
        "3_tokenize": s3_tokens,
        "4_stopwords": s4_stops,
        "5_lemmatize": s5_lemma,
    }

res_default = run_pipeline(T2, default_stops)
res_custom = run_pipeline(T2, custom_stops)

print("Before:", T2)
print("\n--- Run 1: Default NLTK Stopwords ---")
for stage, val in res_default.items():
    print(f"  {stage}: {val}")

print("\n--- Run 2: Negation-Keeping Stopwords ---")
for stage, val in res_custom.items():
    print(f"  {stage}: {val}")