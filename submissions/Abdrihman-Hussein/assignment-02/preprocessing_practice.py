# ============================================================
# Assignment 2 — Preprocessing practice
# Student: Abdrihman Hussein
# Follows the Assignment 2 tasks step by step.
#
# Setup once (only if the NLTK data is missing):
#   python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4 ^
#       averaged_perceptron_tagger averaged_perceptron_tagger_eng ^
#       maxent_ne_chunker maxent_ne_chunker_tab words
#
# Run:  python preprocessing_practice.py > output.txt
# ============================================================

import re
import sys

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords, words as nltk_words
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Windows does not use UTF-8 by default, and T2 contains an emoji.
sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------
# The texts and the word list from the assignment.
# --------------------------------
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran",
]

# --------------------------------
# 1) Lowercasing
# Lowercasing merges "WON", "Won" and "won" into one token.
# But capitals carry meaning: "NOW", "WON" and "URGENT" shout "spam".
# --------------------------------
print("\n=== 1) LOWERCASING ===")
print("Before:", T1)
print("After: ", T1.lower())

# --------------------------------
# 2) Whitespace normalization
# split() breaks on any run of spaces; " ".join() puts single spaces back.
# repr() shows the spaces, so the difference is visible.
# --------------------------------
print("\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(T4))
print("After: ", repr(" ".join(T4.split())))

# --------------------------------
# 3) Regex cleaning
# [^a-z\s] means: any character that is NOT a lowercase letter or a space.
# Here we DELETE those characters; watch the price and phone number vanish.
# --------------------------------
print("\n=== 3) REGEX CLEANING ===")
t1_lower = T1.lower()
print("Lowercased T1:      ", repr(t1_lower))
print("After [^a-z\\s]:    ", repr(re.sub(r"[^a-z\s]", "", t1_lower)))
print("Deleted characters: ", re.findall(r"[^a-z\s]", t1_lower))
print("Unique deleted:     ", sorted(set(re.findall(r"[^a-z\s]", t1_lower))))

# Now REPLACE instead of delete, so the important evidence survives.
t1_replaced = T1
t1_replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", t1_replaced)
t1_replaced = re.sub(r"£\d[\d,]*", "MONEY", t1_replaced)
t1_replaced = re.sub(r"!{2,}", "!", t1_replaced)
print("Replaced T1:        ", t1_replaced)

# --------------------------------
# 4) Tokenization
# split() keeps punctuation stuck to words ("price.").
# word_tokenize separates punctuation and splits "don't" into "do" + "n't".
# --------------------------------
print("\n=== 4) TOKENIZATION ===")
t2_split = T2.split()
t2_tokens = word_tokenize(T2)
print("T2 split():        ", t2_split)
print("T2 split() count:  ", len(t2_split))
print("T2 word_tokenize():", t2_tokens)
print("T2 token count:    ", len(t2_tokens))

t4_tokens = word_tokenize(T4)
print("T4 tokens:         ", t4_tokens)
print("T4 token count:    ", len(t4_tokens))
print("T4 vocabulary:     ", sorted(set(t4_tokens)))
print("T4 vocab size:     ", len(set(t4_tokens)))
print("T4 vocab (lower):  ", len(set(token.lower() for token in t4_tokens)))

# --------------------------------
# 5) Stopwords and negation
# NLTK's English list contains "not" and "no", but not the clitic "n't".
# Removing "not" turns "Not happy" into just "happy" — the complaint flips.
# --------------------------------
print("\n=== 5) STOPWORDS AND NEGATION ===")
stop_words = set(stopwords.words("english"))
without_stopwords = [token for token in t2_tokens if token.lower() not in stop_words]
print("Default stopwords:      ", without_stopwords)

stop_words_keep_negation = set(stopwords.words("english"))
for word in ["not", "n't", "no", "never"]:
    stop_words_keep_negation.discard(word)
with_negation = [token for token in t2_tokens if token.lower() not in stop_words_keep_negation]
print("Keeping negation words: ", with_negation)

# --------------------------------
# 6) Stemming
# Porter is the classic rule-based stemmer. Lancaster is more aggressive.
# Snowball is the improved Porter. None of them promise a real word.
# --------------------------------
print("\n=== 6) STEMMING ===")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("word          Porter      Lancaster   Snowball")
print("-" * 52)
for word in WORDS:
    print(f"{word:<14}{porter.stem(word):<12}{lancaster.stem(word):<12}{snowball.stem(word):<12}")

# Different words that Porter merges into the same stem.
porter_groups = {}
for word in WORDS:
    porter_groups.setdefault(porter.stem(word), []).append(word)
print("\nMerged by Porter:")
for stem, group in porter_groups.items():
    if len(group) > 1:
        print("  ", stem, "<-", group)

# Stems that are not real English words.
english_words = set(word.lower() for word in nltk_words.words())
print("\nPorter stems that are not real words:")
for word in WORDS:
    if porter.stem(word) not in english_words:
        print("  ", word, "->", porter.stem(word))

# --------------------------------
# 7) Lemmatization
# WordNetLemmatizer treats every word as a noun unless told otherwise.
# pos="v" = verb, "n" = noun, "a" = adjective.
# --------------------------------
print("\n=== 7) LEMMATIZATION ===")
lemmatizer = WordNetLemmatizer()
pos_hints = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v", "university": "n",
    "universal": "a", "flies": "n", "wolves": "n",
    "better": "a", "ran": "v",
}

print("word          default       with hint     hint")
print("-" * 52)
for word in WORDS:
    default_lemma = lemmatizer.lemmatize(word)
    hinted_lemma = lemmatizer.lemmatize(word, pos=pos_hints[word])
    print(f"{word:<14}{default_lemma:<14}{hinted_lemma:<14}{pos_hints[word]}")

print("\nChanged only when the POS hint was given:")
for word in WORDS:
    if lemmatizer.lemmatize(word) == word and lemmatizer.lemmatize(word, pos=pos_hints[word]) != word:
        print("  ", word, "->", lemmatizer.lemmatize(word, pos=pos_hints[word]))

# --------------------------------
# 8) Sentence splitting
# sent_tokenize does not simply cut at every period.
# Check whether "Mr." and "a.m." break the sentences in the wrong places.
# --------------------------------
print("\n=== 8) SENTENCE SPLITTING ===")
sentences = sent_tokenize(T3)
print("Sentences:", sentences)
print("Count:    ", len(sentences))

# --------------------------------
# 9) POS tagging and named entity recognition
# NNP = proper noun, VBD = past-tense verb, IN = preposition.
# ne_chunk groups tagged words into PERSON, ORGANIZATION and GPE (places).
# --------------------------------
print("\n=== 9) POS TAGGING AND NER ===")
tagged = pos_tag(word_tokenize(T3))
print("POS tags:", tagged)

tree = ne_chunk(tagged)
print("NER tree:", tree)

entities = []
for subtree in tree:
    if hasattr(subtree, "label"):
        entity = " ".join(word for word, tag in subtree.leaves())
        entities.append((entity, subtree.label()))
print("Entities:", entities)

print("\nModel check:")
print("  Mr.     -> PERSON  (part of a person name)")
print("  Smith   -> PERSON  (correct)")
print("  London  -> GPE     (correct)")
print("  Apple   -> missed; should be ORGANIZATION")
print("  Friday  -> missed; should be DATE")
print("  9 a.m.  -> missed; should be TIME")

# --------------------------------
# 10) Full pipeline on T2
# Lowercase -> regex -> tokenize -> stopwords -> lemmatize.
# Run it twice: default stopwords, then the negation-keeping list.
# --------------------------------
print("\n=== 10) FULL PIPELINE ON T2 ===")
for label, stop_list in [
    ("default stopwords", stop_words),
    ("negation-keeping stopwords", stop_words_keep_negation),
]:
    print("\n--- " + label + " ---")
    step_lower = T2.lower()
    print("Lowercased:        ", repr(step_lower))

    step_clean = re.sub(r"[^a-z\s']", " ", step_lower)
    step_clean = " ".join(step_clean.split())
    print("Regex cleaned:     ", repr(step_clean))

    step_tokens = word_tokenize(step_clean)
    print("Tokens:            ", step_tokens)

    step_no_stop = [token for token in step_tokens if token.lower() not in stop_list]
    print("Stopwords removed: ", step_no_stop)

    step_lemmas = [lemmatizer.lemmatize(token) for token in step_no_stop]
    print("Lemmatized:        ", step_lemmas)
    print("Final text:        ", " ".join(step_lemmas))
