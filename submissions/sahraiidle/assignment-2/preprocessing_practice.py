import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


t1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
t2 = "I don't think this jacket is worth the price. Not happy at all 😞"
t3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
t4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "


# ============================================================
# 1. Lowercasing
# ============================================================

print("=" * 60)
print("1. LOWERCASE")
print("=" * 60)

print("Before:")
print(t1)

t1_lower = t1.lower()

print("After:")
print(t1_lower)

print()
print("A spam detector could lose capitalization information such as")
print("URGENT and WON, which may be useful as a spam signal.")


# ============================================================
# 2. Whitespace normalization
# ============================================================

print()
print("=" * 60)
print("2. WHITESPACE NORMALIZATION")
print("=" * 60)

print("Before:")
print(repr(t4))

t4_normalized = " ".join(t4.split())

print("After:")
print(repr(t4_normalized))


# ============================================================
# 3. Regex cleaning
# ============================================================

print()
print("=" * 60)
print("3. REGEX CLEANING")
print("=" * 60)

print("Lowercased T1:")
print(t1_lower)

regex_result = re.sub(r"[^a-z\s]", " ", t1_lower)

print()
print("After [^a-z\\s]:")
print(regex_result)

print()
print("Characters and information deleted:")
print("£, numbers, comma, full stop, exclamation marks, and other")
print("characters that are not lowercase letters or whitespace.")


print()
print("Replacement instead of deletion:")

replacement_result = t1_lower

replacement_result = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", replacement_result)

replacement_result = re.sub(
    r"£\d{1,3}(?:,\d{3})*",
    "MONEY",
    replacement_result
)

replacement_result = re.sub(r"!{3}", "!", replacement_result)

print(replacement_result)


# ============================================================
# 4. Tokenization
# ============================================================

print()
print("=" * 60)
print("4. TOKENIZATION")
print("=" * 60)

print("T2 with split():")
t2_split = t2.split()
print(t2_split)
print("Token count:", len(t2_split))

print()
print("T2 with word_tokenize():")
t2_tokens = word_tokenize(t2)
print(t2_tokens)
print("Token count:", len(t2_tokens))

print()
print("T4 with word_tokenize():")
t4_tokens = word_tokenize(t4_normalized)
print(t4_tokens)

print("Token count:", len(t4_tokens))
print("Vocabulary size:", len(set(t4_tokens)))


# ============================================================
# 5. Stopwords and negation
# ============================================================

print()
print("=" * 60)
print("5. STOPWORDS AND NEGATION")
print("=" * 60)

english_stopwords = set(stopwords.words("english"))

t2_without_stopwords = []

for word in t2_tokens:
    if word.lower() not in english_stopwords:
        t2_without_stopwords.append(word)

print("After NLTK stopword removal:")
print(t2_without_stopwords)


negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

t2_keep_negation = []

for word in t2_tokens:
    if word.lower() not in custom_stopwords:
        t2_keep_negation.append(word)

print()
print("After stopword removal while keeping negation:")
print(t2_keep_negation)


# ============================================================
# 6. Stemming
# ============================================================

print()
print("=" * 60)
print("6. STEMMING")
print("=" * 60)

words = [
    "connection",
    "connected",
    "connecting",
    "organization",
    "organized",
    "university",
    "universal",
    "flies",
    "wolves",
    "better",
    "ran"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(
    f"{'Word':<15}"
    f"{'Porter':<15}"
    f"{'Lancaster':<15}"
    f"{'Snowball':<15}"
)

print("-" * 60)

for word in words:
    print(
        f"{word:<15}"
        f"{porter.stem(word):<15}"
        f"{lancaster.stem(word):<15}"
        f"{snowball.stem(word):<15}"
    )

print()
print("Example of different words merged into the same stem:")

print("connection ->", porter.stem("connection"))
print("connected  ->", porter.stem("connected"))
print("connecting ->", porter.stem("connecting"))

print()
print("Example of a stem that is not a real word:")

print("universal ->", porter.stem("universal"))


# ============================================================
# 7. Lemmatization
# ============================================================

print()
print("=" * 60)
print("7. LEMMATIZATION")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

print(
    f"{'Word':<15}"
    f"{'Default':<15}"
    f"{'POS':<8}"
    f"{'With POS':<15}"
)

print("-" * 53)

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

for word in words:
    default = lemmatizer.lemmatize(word)
    pos = pos_hints[word]
    with_pos = lemmatizer.lemmatize(word, pos=pos)

    print(
        f"{word:<15}"
        f"{default:<15}"
        f"{pos:<8}"
        f"{with_pos:<15}"
    )

print()
print("Words that changed after giving a POS hint:")

for word in words:
    default = lemmatizer.lemmatize(word)
    pos = pos_hints[word]
    with_pos = lemmatizer.lemmatize(word, pos=pos)

    if default != with_pos:
        print(word, "->", default, "->", with_pos)


# ============================================================
# 8. Sentence splitting
# ============================================================

print()
print("=" * 60)
print("8. SENTENCE SPLITTING")
print("=" * 60)

sentences = sent_tokenize(t3)

print("T3:")
print(t3)

print()
print("Sentences:")

for sentence in sentences:
    print(sentence)

print()
print("Number of sentences:", len(sentences))

print()
print("Mr. and a.m. did not break the sentences into wrong places.")


# ============================================================
# 9. POS tagging and NER
# ============================================================

print()
print("=" * 60)
print("9. POS TAGGING AND NER")
print("=" * 60)

t3_tokens = word_tokenize(t3)

print("POS tags:")
pos_tags = pos_tag(t3_tokens)
print(pos_tags)

print()
print("Named entities:")

named_entities = ne_chunk(pos_tags)
print(named_entities)

print()
print("Extracted entities:")

for item in named_entities:
    if hasattr(item, "label"):
        entity = " ".join(word for word, tag in item.leaves())
        label = item.label()
        print(entity, "->", label)


# ============================================================
# 10. Full pipeline with default stopwords
# ============================================================

print()
print("=" * 60)
print("10. FULL PIPELINE - DEFAULT STOPWORDS")
print("=" * 60)

pipeline_text = t2

print("Original:")
print(pipeline_text)

pipeline_lower = pipeline_text.lower()

print()
print("1. Lowercase:")
print(pipeline_lower)

pipeline_regex = re.sub(r"[^a-z\s]", " ", pipeline_lower)

print()
print("2. Regex:")
print(pipeline_regex)

pipeline_tokens = word_tokenize(pipeline_regex)

print()
print("3. Tokenize:")
print(pipeline_tokens)

pipeline_stopwords = []

for word in pipeline_tokens:
    if word not in english_stopwords:
        pipeline_stopwords.append(word)

print()
print("4. Stopwords:")
print(pipeline_stopwords)

pipeline_lemmas = []

for word in pipeline_stopwords:
    pipeline_lemmas.append(lemmatizer.lemmatize(word))

print()
print("5. Lemmatize:")
print(pipeline_lemmas)

print()
print("Final text:")
print(" ".join(pipeline_lemmas))


# ============================================================
# 10. Full pipeline with negation kept
# ============================================================

print()
print("=" * 60)
print("10. FULL PIPELINE - NEGATION KEPT")
print("=" * 60)

pipeline_text = t2

print("Original:")
print(pipeline_text)

pipeline_lower = pipeline_text.lower()

print()
print("1. Lowercase:")
print(pipeline_lower)

pipeline_regex = re.sub(r"[^a-z\s]", " ", pipeline_lower)

print()
print("2. Regex:")
print(pipeline_regex)

pipeline_tokens = word_tokenize(pipeline_regex)

print()
print("3. Tokenize:")
print(pipeline_tokens)

pipeline_stopwords = []

for word in pipeline_tokens:
    if word not in custom_stopwords:
        pipeline_stopwords.append(word)

print()
print("4. Stopwords:")
print(pipeline_stopwords)

pipeline_lemmas = []

for word in pipeline_stopwords:
    pipeline_lemmas.append(lemmatizer.lemmatize(word))

print()
print("5. Lemmatize:")
print(pipeline_lemmas)

print()
print("Final text:")
print(" ".join(pipeline_lemmas))