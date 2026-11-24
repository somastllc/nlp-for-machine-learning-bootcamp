
import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "    The   delivery   was   fast,   and the  packaging   was   great. "

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

# =========================================================
# TASK 1: LOWERCASE
# =========================================================

print("\n" + "=" * 60)
print("TASK 1: LOWERCASE")
print("=" * 60)

print("BEFORE:", T1)

after_lowercase = T1.lower()

print("AFTER :", after_lowercase)


# =========================================================
# TASK 2: WHITESPACE NORMALIZATION
# =========================================================

print("\n" + "=" * 60)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 60)

print("BEFORE:", repr(T4))

after_whitespace = re.sub(r"\s+", " ", T4).strip()

print("AFTER :", repr(after_whitespace))


# =========================================================
# TASK 3: REGEX CLEANING
# =========================================================

print("\n" + "=" * 60)
print("TASK 3: REGEX CLEANING")
print("=" * 60)

lower_T1 = T1.lower()

print("BEFORE:", lower_T1)

after_regex = re.sub(r"[^a-z\s]", "", lower_T1)

print("AFTER :", after_regex)

print("\nCUSTOM REGEX REPLACEMENTS")

print("BEFORE:", T1)

custom_regex = T1
custom_regex = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", custom_regex)
custom_regex = re.sub(r"£[\d,]+", "MONEY", custom_regex)
custom_regex = re.sub(r"!{3,}", "!", custom_regex)

print("AFTER :", custom_regex)


# =========================================================
# TASK 4: TOKENIZATION
# =========================================================

print("\n" + "=" * 60)
print("TASK 4: TOKENIZATION")
print("=" * 60)

print("\nT2: split()")

print("BEFORE:", T2)

tokens_split = T2.split()

print("AFTER :", tokens_split)
print("COUNT :", len(tokens_split))

print("\nT2: word_tokenize()")

tokens_nltk = word_tokenize(T2)

print("BEFORE:", T2)
print("AFTER :", tokens_nltk)
print("COUNT :", len(tokens_nltk))

print("\nT4: TOKENIZATION AND VOCABULARY")

print("BEFORE:", normalized_text := re.sub(r"\s+", " ", T4).strip())

tokens_T4 = word_tokenize(normalized_text)

print("AFTER :", tokens_T4)
print("TOKEN COUNT:", len(tokens_T4))

vocabulary_T4 = set(tokens_T4)

print("VOCABULARY:", vocabulary_T4)
print("VOCABULARY SIZE:", len(vocabulary_T4))


# =========================================================
# TASK 5: STOPWORD REMOVAL
# =========================================================

print("\n" + "=" * 60)
print("TASK 5: STOPWORD REMOVAL")
print("=" * 60)

T2_lower = T2.lower()
tokens_T2 = word_tokenize(T2_lower)

english_stopwords = set(stopwords.words("english"))

print("\nDEFAULT STOPWORD REMOVAL")

print("BEFORE:", tokens_T2)

after_stopwords = [
    token for token in tokens_T2
    if token not in english_stopwords
]

print("AFTER :", after_stopwords)

print("\nNEGATION-PRESERVING STOPWORD REMOVAL")

custom_stopwords = english_stopwords - {"not", "no", "never"}

print("BEFORE:", tokens_T2)

after_negation = [
    token for token in tokens_T2
    if token not in custom_stopwords
]

print("AFTER :", after_negation)


# =========================================================
# TASK 6: STEMMING COMPARISON
# =========================================================

print("\n" + "=" * 60)
print("TASK 6: STEMMING")
print("=" * 60)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\nBEFORE:", words)

print("\nAFTER: STEMMER COMPARISON")

print(f"{'WORD':<15}{'PORTER':<15}{'LANCASTER':<15}{'SNOWBALL':<15}")

for word in words:
    print(
        f"{word:<15}"
        f"{porter.stem(word):<15}"
        f"{lancaster.stem(word):<15}"
        f"{snowball.stem(word):<15}"
    )


# =========================================================
# TASK 7: LEMMATIZATION
# =========================================================

print("\n" + "=" * 60)
print("TASK 7: LEMMATIZATION")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

print("BEFORE:", words)

print("\nAFTER: LEMMATIZATION COMPARISON")

print(f"{'WORD':<15}{'DEFAULT':<15}{'VERB':<15}{'NOUN':<15}{'ADJECTIVE':<15}")

for word in words:
    default_lemma = lemmatizer.lemmatize(word)
    verb_lemma = lemmatizer.lemmatize(word, pos="v")
    noun_lemma = lemmatizer.lemmatize(word, pos="n")
    adjective_lemma = lemmatizer.lemmatize(word, pos="a")

    print(
        f"{word:<15}"
        f"{default_lemma:<15}"
        f"{verb_lemma:<15}"
        f"{noun_lemma:<15}"
        f"{adjective_lemma:<15}"
    )


# =========================================================
# TASK 8: SENTENCE SPLITTING
# =========================================================

print("\n" + "=" * 60)
print("TASK 8: SENTENCE SPLITTING")
print("=" * 60)

print("BEFORE:", T3)

sentences = sent_tokenize(T3)

print("\nAFTER:")

for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

print("SENTENCE COUNT:", len(sentences))


# =========================================================
# TASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION
# =========================================================

print("\n" + "=" * 60)
print("TASK 9: POS TAGGING AND NER")
print("=" * 60)

print("BEFORE:", T3)

tokens_T3 = word_tokenize(T3)

print("\nAFTER: TOKENIZATION")
print(tokens_T3)

tags = pos_tag(tokens_T3)

print("\nAFTER: POS TAGGING")
print(tags)

named_entities = ne_chunk(tags)

print("\nAFTER: NAMED ENTITY RECOGNITION")
print(named_entities)


# =========================================================
# TASK 10: FULL PREPROCESSING PIPELINE
# =========================================================

print("\n" + "=" * 60)
print("TASK 10: FULL PREPROCESSING PIPELINE")
print("=" * 60)

def preprocess(text, preserve_negation=False):

    print("\nSTEP 1: LOWERCASE")
    print("BEFORE:", text)

    text = text.lower()

    print("AFTER :", text)

    print("\nSTEP 2: REGEX CLEANING")
    print("BEFORE:", text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    print("AFTER :", text)

    print("\nSTEP 3: TOKENIZATION")
    print("BEFORE:", text)

    tokens = word_tokenize(text)

    print("AFTER :", tokens)

    stop_words = set(stopwords.words("english"))

    if preserve_negation:
        stop_words -= {"not", "no", "never"}

    print("\nSTEP 4: STOPWORD REMOVAL")
    print("BEFORE:", tokens)

    tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    print("AFTER :", tokens)

    print("\nSTEP 5: LEMMATIZATION")
    print("BEFORE:", tokens)

    tokens = [
        lemmatizer.lemmatize(token, pos="v")
        for token in tokens
    ]

    print("AFTER :", tokens)

    return tokens


print("\nORIGINAL TEXT:")
print(T2)

print("\nPIPELINE 1: DEFAULT")

result_default = preprocess(T2, preserve_negation=False)

print("\nFINAL RESULT:", result_default)

print("\nPIPELINE 2: NEGATION PRESERVED")

result_negation = preprocess(T2, preserve_negation=True)

print("\nFINAL RESULT:", result_negation)
