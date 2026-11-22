"""
preprocessing_practice.py  -  Assignment 2, Part B

Applies lowercasing, whitespace normalization, regex cleaning, tokenization,
stopword removal, stemming, lemmatization, sentence splitting, POS tagging,
NER and a full pipeline to the four texts and the word list from the
assignment. Every step prints a heading and its before/after result.

Setup once: python code/download-nltk-data.py
Run:        python preprocessing_practice.py > output.txt
"""

import re
import sys

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import (LancasterStemmer, PorterStemmer, SnowballStemmer,
                       WordNetLemmatizer)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree

# Make sure the emoji and the pound sign print correctly when redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Input data (copied from the assignment)
# ---------------------------------------------------------------------------
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]

NEGATIONS = {"not", "n't", "no", "never"}


def heading(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def show(label, value):
    print(f"{label:<16}: {value}")


# ---------------------------------------------------------------------------
# TASK 1 - Lowercasing
# ---------------------------------------------------------------------------
heading("TASK 1 - Lowercasing T1")
t1_lower = T1.lower()
show("Before", repr(T1))
show("After", repr(t1_lower))
all_caps = [w for w in T1.split() if w.isupper()]
show("ALL-CAPS words", all_caps)
show("Note", "URGENT, WON and NOW may help spam detection; lowercasing hides them")

# ---------------------------------------------------------------------------
# TASK 2 - Whitespace normalization
# ---------------------------------------------------------------------------
heading("TASK 2 - Whitespace normalization of T4")
t4_clean = " ".join(T4.split())
show("Before", repr(T4))
show("After", repr(t4_clean))
show("Length", f"{len(T4)} characters -> {len(t4_clean)} characters")

# ---------------------------------------------------------------------------
# TASK 3 - Regex cleaning
# ---------------------------------------------------------------------------
heading("TASK 3a - Deleting with the pattern [^a-z\\s] on lowercased T1")
t1_deleted = re.sub(r"[^a-z\s]", " ", t1_lower)
show("Before", repr(t1_lower))
show("After", repr(t1_deleted))
show("Deleted", "£, 1,000, 0712345678, ! and . (symbols and digits)")

heading("TASK 3b - Replacing instead of deleting on T1")
step = T1
show("Original", repr(step))
step = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", step)
show("After PHONE", repr(step))
step = re.sub(r"£\s?\d[\d,]*(?:\.\d+)?", "MONEY", step)
show("After MONEY", repr(step))
step = re.sub(r"!{2,}", "!", step)
show("After BANGS", repr(step))
show("Final", repr(step))

# ---------------------------------------------------------------------------
# TASK 4 - Tokenization
# ---------------------------------------------------------------------------
heading("TASK 4a - Tokenizing T2: split() vs word_tokenize")
t2_split = T2.split()
t2_tokens = word_tokenize(T2)
show("T2", repr(T2))
show("split()", t2_split)
show("  count", len(t2_split))
show("word_tokenize", t2_tokens)
show("  count", len(t2_tokens))
show("Only in split", [t for t in t2_split if t not in t2_tokens])
show("Only in wt", [t for t in t2_tokens if t not in t2_split])

heading("TASK 4b - Tokenizing T4: token count and vocabulary size")
t4_tokens = word_tokenize(T4)
show("T4 tokens", t4_tokens)
show("Token count", len(t4_tokens))
show("Vocab size", len(set(t4_tokens)))
show("Vocabulary", sorted(set(t4_tokens)))

# ---------------------------------------------------------------------------
# TASK 5 - Stopwords and negation
# ---------------------------------------------------------------------------
heading("TASK 5 - Stopword removal on T2 tokens")
stop_words = set(stopwords.words("english"))
keep_negation = stop_words - NEGATIONS

show("NLTK list size", len(stop_words))
show("Negations in it", sorted(w for w in NEGATIONS if w in stop_words))

t2_lower_tokens = [t.lower() for t in t2_tokens]
default_tokens = [t for t in t2_lower_tokens if t not in stop_words]
negation_tokens = [t for t in t2_lower_tokens if t not in keep_negation]

print()
show("Tokens", t2_lower_tokens)
show("Default removed", [t for t in t2_lower_tokens if t in stop_words])
show("Default result", default_tokens)
print()
show("Custom removed", [t for t in t2_lower_tokens if t in keep_negation])
show("Custom result", negation_tokens)

# ---------------------------------------------------------------------------
# TASK 6 - Stemming
# ---------------------------------------------------------------------------
heading("TASK 6 - Stemming the word list (Porter, Lancaster, Snowball)")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("| Word | Porter | Lancaster | Snowball |")
print("|---|---|---|---|")
for w in WORDS:
    print(f"| {w} | {porter.stem(w)} | {lancaster.stem(w)} | {snowball.stem(w)} |")

print()
for name, stemmer in [("Porter", porter), ("Lancaster", lancaster),
                      ("Snowball", snowball)]:
    groups = {}
    for w in WORDS:
        groups.setdefault(stemmer.stem(w), []).append(w)
    merged = {s: ws for s, ws in groups.items() if len(ws) > 1}
    show(name, merged if merged else "no merged groups")

# ---------------------------------------------------------------------------
# TASK 7 - Lemmatization
# ---------------------------------------------------------------------------
heading("TASK 7 - WordNetLemmatizer: default vs correct POS hint")
lemmatizer = WordNetLemmatizer()
POS_HINT = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v",
    "university": "n", "universal": "a",
    "flies": "v", "wolves": "n", "better": "a", "ran": "v",
}

print("| Word | Default (noun) | POS hint | With hint | Changed? |")
print("|---|---|---|---|---|")
changed_with_hint = []
for w in WORDS:
    default = lemmatizer.lemmatize(w)
    hinted = lemmatizer.lemmatize(w, pos=POS_HINT[w])
    if default != hinted:
        changed_with_hint.append(w)
    print(f"| {w} | {default} | {POS_HINT[w]} | {hinted} | "
          f"{'yes' if default != hinted else 'no'} |")
print()
show("Changed w/ hint", changed_with_hint)
show("Same either way", [w for w in WORDS if w not in changed_with_hint])

# ---------------------------------------------------------------------------
# TASK 8 - Sentence splitting
# ---------------------------------------------------------------------------
heading("TASK 8 - Sentence splitting of T3")
show("T3", repr(T3))
sentences = sent_tokenize(T3)
show("Sentence count", len(sentences))
for i, sent in enumerate(sentences, 1):
    show(f"Sentence {i}", repr(sent))
show("Split after Mr.?", any(s.endswith("Mr.") for s in sentences))
show("Split in a.m.?", any(s.startswith("m.") for s in sentences))
naive = re.split(r"(?<=\.)\s+", T3)
show("Naive re.split", f"{len(naive)} pieces -> {naive}")

# ---------------------------------------------------------------------------
# TASK 9 - POS tagging and NER
# ---------------------------------------------------------------------------
heading("TASK 9 - POS tagging and named entity recognition on T3")
t3_tokens = word_tokenize(T3)
tagged = pos_tag(t3_tokens)
show("Tokens", t3_tokens)
print("POS tags:")
for word, tag in tagged:
    print(f"  {word:<10} {tag}")

tree = ne_chunk(tagged)
print()
print("NER tree:")
print(tree)

print()
print("Entities found (text -> label):")
for item in tree:
    if isinstance(item, Tree):
        text = " ".join(word for word, _ in item.leaves())
        print(f"  {text} -> {item.label()}")
print("Check manually: Smith (person), London (place), Friday (day), "
      "Apple (company), 9 a.m. (time).")

# ---------------------------------------------------------------------------
# TASK 10 - Full pipeline on T2
# ---------------------------------------------------------------------------
heading("TASK 10 - Full pipeline on T2")
step_lower = T2.lower()
step_clean = re.sub(r"[^a-z\s']", " ", step_lower)   # keep apostrophes
step_tokens = word_tokenize(step_clean)
step_default = [t for t in step_tokens if t not in stop_words]
step_default_lemmas = [lemmatizer.lemmatize(t) for t in step_default]
step_negation = [t for t in step_tokens if t not in keep_negation]
step_negation_lemmas = [lemmatizer.lemmatize(t) for t in step_negation]

show("Original", repr(T2))
show("Lowercased", repr(step_lower))
show("Regex cleaned", repr(step_clean))
show("Tokens", step_tokens)
show("Default stop", step_default)
show("Default lemmas", step_default_lemmas)
show("Negation stop", step_negation)
show("Negation lemmas", step_negation_lemmas)

print()
print("END OF OUTPUT")