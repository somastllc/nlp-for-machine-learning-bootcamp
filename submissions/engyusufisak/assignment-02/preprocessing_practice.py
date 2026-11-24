# Assignment 2 - Part B: Preprocessing practice
# Author: engyusufisak
# Run:  python preprocessing_practice.py > output.txt

import re
import sys

import nltk
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# make sure the emoji in T2 can be printed / saved on Windows
sys.stdout.reconfigure(encoding="utf-8")

# download the NLTK data once (quiet, so it doesn't fill output.txt)
for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words"]:
    nltk.download(package, quiet=True)


def heading(number, title):
    print("\n" + "=" * 70)
    print(f"=== Task {number}: {title}")
    print("=" * 70)


# ---------------------------------------------------------------- data
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]


# ---------------------------------------------------------------- 1
heading(1, "Lowercasing (T1)")
t1_lower = T1.lower()
print("before:", T1)
print("after: ", t1_lower)
upper_words = [w for w in re.findall(r"[A-Za-z]+", T1) if w.isupper() and len(w) > 1]
print("words that were written in CAPITALS (lost after lowercasing):", upper_words)


# ---------------------------------------------------------------- 2
heading(2, "Whitespace normalization (T4)")
t4_clean = " ".join(T4.split())
print("before:", repr(T4))
print("after: ", repr(t4_clean))
print("length before:", len(T4), "| length after:", len(t4_clean))


# ---------------------------------------------------------------- 3
heading(3, "Regex cleaning (T1)")
t1_regex = re.sub(r"[^a-z\s]", " ", t1_lower)
print("pattern:      [^a-z\\s]")
print("before:      ", repr(t1_lower))
print("after:       ", repr(t1_regex))
deleted = [ch for ch in t1_lower if not re.match(r"[a-z\s]", ch)]
print("deleted characters:", deleted)
print("deleted (unique):  ", sorted(set(deleted)))

print("\n--- my own rules: replace instead of delete ---")
step = T1
print("start:         ", step)
step = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", step)        # 10-digit number starting with 0
print("phone rule:    ", step)
step = re.sub(r"[£$€]\s?\d[\d,]*(\.\d+)?", "MONEY", step)  # currency sign + amount
print("money rule:    ", step)
step = re.sub(r"!{2,}", "!", step)                        # !!! -> !
print("'!!!' rule:    ", step)
t1_replaced = step


# ---------------------------------------------------------------- 4
heading(4, "Tokenization (T2 and T4)")
t2_split = T2.split()
t2_tokens = word_tokenize(T2)
print("text:", T2)
print(f"split()        ({len(t2_split)} tokens):", t2_split)
print(f"word_tokenize  ({len(t2_tokens)} tokens):", t2_tokens)

t4_tokens = word_tokenize(T4)
print("\ntext:", repr(T4))
print("tokens:", t4_tokens)
print("token count:", len(t4_tokens))
print("vocabulary (unique tokens):", sorted(set(t4_tokens)))
print("vocabulary size:", len(set(t4_tokens)))
print("vocabulary size if lowercased first:", len(set(t.lower() for t in t4_tokens)))


# ---------------------------------------------------------------- 5
heading(5, "Stopwords and negation (T2)")
default_stops = set(stopwords.words("english"))
negations = {"not", "n't", "no", "never"}
my_stops = default_stops - negations

t2_default = [t for t in t2_tokens if t.lower() not in default_stops]
t2_mine = [t for t in t2_tokens if t.lower() not in my_stops]
print("tokens:                      ", t2_tokens)
print("NLTK default stopwords:      ", t2_default)
print("my list (keeps negations):   ", t2_mine)
print("negation words in NLTK list: ", sorted(w for w in negations if w in default_stops))
print("removed by default but kept by mine:",
      [t for t in t2_mine if t not in t2_default])


# ---------------------------------------------------------------- 6
heading(6, "Stemming (Porter, Lancaster, Snowball)")
stemmers = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english"),
}
stems = {name: {w: s.stem(w) for w in WORDS} for name, s in stemmers.items()}

print(f"{'word':<14}{'Porter':<14}{'Lancaster':<14}{'Snowball':<14}")
print("-" * 56)
for w in WORDS:
    print(f"{w:<14}{stems['Porter'][w]:<14}{stems['Lancaster'][w]:<14}{stems['Snowball'][w]:<14}")

print("\ndifferent words merged into the same stem:")
for name, table in stems.items():
    groups = {}
    for w, s in table.items():
        groups.setdefault(s, []).append(w)
    for s, ws in groups.items():
        if len(ws) > 1:
            print(f"  {name:<10} '{s}' <- {ws}")

print("\nstems that are not real English words (not found in WordNet):")
for name, table in stems.items():
    fake = sorted({s for s in table.values() if not wordnet.synsets(s)})
    print(f"  {name:<10} {fake}")


# ---------------------------------------------------------------- 7
heading(7, "Lemmatization (default vs. POS hint)")
lemmatizer = WordNetLemmatizer()
pos_hint = {
    "connection": "n", "connected": "v", "connecting": "v", "organization": "n",
    "organized": "v", "university": "n", "universal": "a", "flies": "v",
    "wolves": "n", "better": "a", "ran": "v",
}
print(f"{'word':<14}{'default':<14}{'pos':<6}{'with pos':<14}{'changed only with hint?'}")
print("-" * 70)
changed_only_with_hint = []
for w in WORDS:
    default = lemmatizer.lemmatize(w)
    hinted = lemmatizer.lemmatize(w, pos=pos_hint[w])
    flag = "YES" if default == w and hinted != w else ""
    if flag:
        changed_only_with_hint.append(w)
    print(f"{w:<14}{default:<14}{pos_hint[w]:<6}{hinted:<14}{flag}")
print("\nwords that changed only when the POS hint was given:", changed_only_with_hint)


# ---------------------------------------------------------------- 8
heading(8, "Sentence splitting (T3)")
t3_sentences = sent_tokenize(T3)
print("text:", T3)
print("number of sentences:", len(t3_sentences))
for i, s in enumerate(t3_sentences, 1):
    print(f"  sentence {i}: {s}")


# ---------------------------------------------------------------- 9
heading(9, "POS tagging and NER (T3)")
t3_tagged = pos_tag(word_tokenize(T3))
print("POS tags:", t3_tagged)

tree = ne_chunk(t3_tagged)
print("\nne_chunk tree:")
print(tree)

found = {}
for subtree in tree:
    if hasattr(subtree, "label"):
        entity = " ".join(word for word, tag in subtree.leaves())
        found[entity] = subtree.label()

# what a human would expect (my own judgement)
expected = {"Smith": "PERSON", "London": "GPE", "Friday": "DATE", "Apple": "ORGANIZATION"}

print("\nentities found by NLTK:")
print(f"{'entity':<14}{'NLTK label':<16}{'expected':<16}{'correct?'}")
print("-" * 56)
for entity, label in found.items():
    exp = next((v for k, v in expected.items() if k in entity), "(not an entity)")
    print(f"{entity:<14}{label:<16}{exp:<16}{'yes' if label == exp else 'NO'}")

print("\nexpected entities that NLTK missed:")
for entity, label in expected.items():
    if not any(entity in f for f in found):
        print(f"  {entity:<10} ({label})")


# ---------------------------------------------------------------- 10
heading(10, "Full pipeline on T2 (default vs. negation-keeping stopwords)")


def pipeline(text, stop_list, name):
    print(f"\n--- {name} ---")
    lowered = text.lower()
    print("1 lowercase: ", lowered)
    cleaned = re.sub(r"[^a-z\s']", " ", lowered)   # keep ' so "don't" survives
    print("2 regex:     ", repr(cleaned))
    tokens = word_tokenize(cleaned)
    print("3 tokenize:  ", tokens)
    kept = [t for t in tokens if t not in stop_list]
    print("4 stopwords: ", kept)
    lemmas = [lemmatizer.lemmatize(t) for t in kept]
    print("5 lemmatize: ", lemmas)
    return lemmas


result_default = pipeline(T2, default_stops, "NLTK default stopwords")
result_mine = pipeline(T2, my_stops, "my negation-keeping stopwords")
print("\nfinal (default):", result_default)
print("final (mine):   ", result_mine)
