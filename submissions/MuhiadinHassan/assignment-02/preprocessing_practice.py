"""Reproducible NLTK practice for Assignment 2 – Preprocessing."""

from collections import Counter
import re
import sys

import nltk
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


T1 = "URGENT!!! You have WON a \u00a31,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all \U0001f61e"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."
WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran",
]


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_pair(before, after):
    print("Before:", repr(before))
    print("After: ", repr(after))


def collect_entities(tree):
    found = []
    for node in tree:
        if isinstance(node, nltk.Tree):
            found.append((" ".join(word for word, _ in node.leaves()), node.label()))
            found.extend(collect_entities(node))
    return found


heading("1. LOWERCASE T1")
t1_lower = T1.lower()
show_pair(T1, t1_lower)

heading("2. WHITESPACE NORMALIZATION OF T4")
t4_normalized = re.sub(r"\s+", " ", T4).strip()
show_pair(T4, t4_normalized)

heading("3A. REQUIRED REGEX [^a-z\\s] ON LOWERCASED T1")
t1_regex_clean = re.sub(r"[^a-z\s]", "", t1_lower)
deleted = Counter(char for char in t1_lower if re.fullmatch(r"[^a-z\s]", char))
show_pair(t1_lower, t1_regex_clean)
print("Deleted characters with counts:", dict(sorted(deleted.items())))

heading("3B. REPLACEMENT RULES FOR PHONE, MONEY, AND EXCLAMATIONS")
t1_replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", T1)
t1_replaced = re.sub(r"\u00a3\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "MONEY", t1_replaced)
t1_replaced = re.sub(r"!{2,}", "!", t1_replaced)
show_pair(T1, t1_replaced)

heading("4. TOKENIZATION OF T2")
t2_split = T2.split()
t2_word_tokens = word_tokenize(T2)
print("split() tokens:", t2_split)
print("split() token count:", len(t2_split))
print("word_tokenize() tokens:", t2_word_tokens)
print("word_tokenize() token count:", len(t2_word_tokens))

heading("4B. TOKEN COUNT AND VOCABULARY OF T4")
t4_tokens = word_tokenize(T4)
print("word_tokenize() tokens:", t4_tokens)
print("Token count:", len(t4_tokens))
print("Vocabulary size (unique, case-sensitive tokens):", len(set(t4_tokens)))
print("Vocabulary:", sorted(set(t4_tokens)))

heading("5. STOPWORDS AND NEGATION IN T2")
english_stopwords = set(stopwords.words("english"))
t2_lower_tokens = [token.lower() for token in t2_word_tokens]
default_filtered = [token for token in t2_lower_tokens if token not in english_stopwords]
negation = {"not", "n't", "no", "never"}
custom_stopwords = english_stopwords - negation
negation_kept_filtered = [token for token in t2_lower_tokens if token not in custom_stopwords]
print("Original tokens:", t2_word_tokens)
print("NLTK stopwords removed:", [token for token in t2_lower_tokens if token in english_stopwords])
print("After NLTK default stopwords:", default_filtered)
print("After custom list keeping negation:", negation_kept_filtered)

heading("6. PORTER, LANCASTER, AND SNOWBALL STEMMING")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
print(f"{'Word':<16} {'Porter':<16} {'Lancaster':<16} {'Snowball':<16}")
print("-" * 64)
for word in WORDS:
    print(f"{word:<16} {porter.stem(word):<16} {lancaster.stem(word):<16} {snowball.stem(word):<16}")

heading("7. WORDNET LEMMATIZATION: DEFAULT NOUN VS CHOSEN POS")
lemmatizer = WordNetLemmatizer()
pos_hints = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v", "university": "n",
    "universal": "a", "flies": "n", "wolves": "n",
    "better": "a", "ran": "v",
}
print(f"{'Word':<16} {'Default (n)':<20} {'With POS hint':<20} {'POS':<5}")
print("-" * 65)
for word in WORDS:
    hint = pos_hints[word]
    print(f"{word:<16} {lemmatizer.lemmatize(word):<20} {lemmatizer.lemmatize(word, pos=hint):<20} {hint:<5}")

heading("8. SENTENCE SPLITTING OF T3")
sentences = sent_tokenize(T3)
print("Input:", T3)
print("Sentences:", sentences)
print("Sentence count:", len(sentences))

heading("9. POS TAGGING AND NAMED-ENTITY CHUNKING OF T3")
t3_tokens = word_tokenize(T3)
t3_tags = pos_tag(t3_tokens)
ner_tree = ne_chunk(t3_tags)
print("Tokens:", t3_tokens)
print("POS tags:", t3_tags)
print("NER tree:")
print(ner_tree)
entities = collect_entities(ner_tree)
print("Recognized entities (surface, label):", entities)

heading("10. FULL T2 PIPELINE: LOWERCASE -> REGEX -> TOKENIZE -> STOPWORDS -> LEMMATIZE")
pipeline_lower = T2.lower()
# Keep apostrophes so contractions such as "don't" can still expose negation.
pipeline_regex = re.sub(r"[^a-z\s']", "", pipeline_lower)
pipeline_tokens = word_tokenize(pipeline_regex)
print("Before:", repr(T2))
print("After lowercase:", repr(pipeline_lower))
print("After regex (keeps apostrophes):", repr(pipeline_regex))
print("After tokenize:", pipeline_tokens)
for name, stop_set in (("NLTK default stopwords", english_stopwords),
                       ("Custom list retaining negation", custom_stopwords)):
    filtered = [token for token in pipeline_tokens if token not in stop_set]
    lemmatized = [lemmatizer.lemmatize(token) for token in filtered]
    print(f"After stopwords ({name}):", filtered)
    print(f"After lemmatize ({name}):", lemmatized)

