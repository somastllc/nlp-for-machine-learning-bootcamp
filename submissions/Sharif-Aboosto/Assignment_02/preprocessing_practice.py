# Assignment 2 - Preprocessing Practice

import re
import sys
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

# NLTK RESOURCES

resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words"
]

for resource in resources:
    nltk.download(resource)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# DATA 

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]


# TASK 1: LOWERCASING

print("\n===== TASK 1: LOWERCASING (T1) =====")

t1_lower = T1.lower()

print("Before:", T1)
print("After:", t1_lower)

caps_words = re.findall(r"\b[A-Z]{2,}\b", T1)

print("All-caps words lost:", caps_words)

letters = [c for c in T1 if c.isalpha()]
upper = [c for c in letters if c.isupper()]

if letters:
    print("Uppercase ratio before:", round(len(upper) / len(letters), 2))

print("Uppercase ratio after: 0.0")


# TASK 2: WHITESPACE

print("\n===== TASK 2: WHITESPACE (T4) =====")

t4_clean = " ".join(T4.split())

print("Before:", repr(T4))
print("After:", repr(t4_clean))


# TASK 3: REGEX CLEANING

print("\n===== TASK 3: REGEX CLEANING (T1) =====")

# 3a. Delete unwanted characters

print("\n--- 3a. Delete with [^a-z\\s] ---")

t1_deleted = re.sub(r"[^a-z\s]", "", t1_lower)

deleted = re.findall(r"[^a-z\s]", t1_lower)

print("Before:", t1_lower)
print("After:", t1_deleted)
print("Deleted characters:", deleted)
print("Number deleted:", len(deleted))


# 3b. Replace instead of delete

print("\n--- 3b. Replace instead of delete ---")

t1_replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", T1)

t1_replaced = re.sub(r"£[\d,]+", "MONEY", t1_replaced)

t1_replaced = re.sub(r"!+", "!", t1_replaced)

print("Before:", T1)
print("After:", t1_replaced)


# TASK 4: TOKENIZATION

print("\n===== TASK 4: TOKENIZATION =====")

t2_split = T2.split()

print("T2 split():", t2_split)
print("Count:", len(t2_split))

t2_tokens = word_tokenize(T2)

print("T2 word_tokenize:", t2_tokens)
print("Count:", len(t2_tokens))

t4_tokens = word_tokenize(T4)

print("T4 tokens:", t4_tokens)
print("T4 token count:", len(t4_tokens))
print("T4 vocabulary size:", len(set(t4_tokens)))


# TASK 5: STOPWORDS AND NEGATION 

print("\n===== TASK 5: STOPWORDS AND NEGATION (T2) =====")

t2_lower_tokens = [t.lower() for t in t2_tokens]

print("Tokens:", t2_lower_tokens)

nltk_stops = set(stopwords.words("english"))

# NLTK stopwords

result_nltk = [
    t for t in t2_lower_tokens
    if t not in nltk_stops
]

print("With NLTK stopwords:", result_nltk)

# Custom stopwords preserving negation

negations = ["not", "n't", "no", "never", "nor"]

my_stops = set()

for w in nltk_stops:
    if w in negations or w.endswith("n't"):
        continue

    my_stops.add(w)

result_mine = [
    t for t in t2_lower_tokens
    if t not in my_stops
]

print("With my negation list:", result_mine)


# TASK 6: STEMMING

print("\n===== TASK 6: STEMMING =====")

porter = PorterStemmer()

lancaster = LancasterStemmer()

snowball = SnowballStemmer("english")

print(f"{'Word':<14}{'Porter':<12}{'Lancaster':<12}{'Snowball':<12}")

for w in words:

    print(
        f"{w:<14}"
        f"{porter.stem(w):<12}"
        f"{lancaster.stem(w):<12}"
        f"{snowball.stem(w):<12}"
    )

print(
    "\nMerged pair: university ->", porter.stem("university"),
    "| universal ->", porter.stem("universal")
)

not_real = [
    porter.stem(w)
    for w in words
    if not wordnet.synsets(porter.stem(w))
]

print("Porter stems that are not real words:", sorted(set(not_real)))


# TASK 7: LEMMATIZATION 

print("\n===== TASK 7: LEMMATIZATION =====")

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
    "ran": "v"
}

print(f"{'Word':<14}{'Default':<14}{'POS':<5}{'With POS':<14}")

changed = []

for w in words:

    default = lemmatizer.lemmatize(w)

    with_pos = lemmatizer.lemmatize(
        w,
        pos=pos_hints[w]
    )

    print(
        f"{w:<14}"
        f"{default:<14}"
        f"{pos_hints[w]:<5}"
        f"{with_pos:<14}"
    )

    if default != with_pos:
        changed.append(w)

print("Changed only with POS hint:", changed)


# TASK 8: SENTENCE SPLITTING 

print("\n===== TASK 8: SENTENCE SPLITTING (T3) =====")

sentences = sent_tokenize(T3)

for i, s in enumerate(sentences, 1):

    print(f"Sentence {i}: {s}")

print("Number of sentences:", len(sentences))


# TASK 9: POS TAGGING AND NER

print("\n===== TASK 9: POS TAGGING AND NER (T3) =====")

tags = pos_tag(word_tokenize(T3))

print("POS tags:", tags)

tree = ne_chunk(tags)

print("\nEntities:")

for chunk in tree:

    if hasattr(chunk, "label"):

        entity = " ".join(word for word, tag in chunk)

        print(f"  {entity:<10} {chunk.label()}")


# TASK 10: FULL PIPELINE

print("\n===== TASK 10: FULL PIPELINE (T2) =====")

for name, stop_list in [
    ("NLTK stopwords", nltk_stops),
    ("My negation list", my_stops)
]:

    print(f"\n--- {name} ---")

    # Step 1: Lowercase

    step1 = T2.lower()

    print("1. Lowercase:", step1)

    # Step 2: Regex cleaning

    step2 = re.sub(r"[^a-z\s']", "", step1).strip()

    print("2. Regex:", step2)

    # Step 3: Tokenization

    step3 = word_tokenize(step2)

    print("3. Tokenize:", step3)

    # Step 4: Stopwords removal

    step4 = [
        t for t in step3
        if t not in stop_list
    ]

    print("4. Stopwords:", step4)

    # Step 5: Lemmatization

    step5 = [
        lemmatizer.lemmatize(t)
        for t in step4
    ]

    print("5. Lemmatize:", step5)


print("\n===== ASSIGNMENT 2 COMPLETED =====")