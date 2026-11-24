import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# #Lowercasing

text = "URGENT!!! You have WON a $1,000 gift card. Call 0712345678 NOW!"

print("=" * 50)
print("TASK 1: LOWERCASE")
print("=" * 50)

print("Before:")
print(text)

lowercase_text = text.lower()

print("\nAfter:")
print(lowercase_text)


# Whitespace Normalization

Text2 = "   The   delivery   was   fast,   and the  packaging   was   great."

print("\n" + "=" * 50)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 50)

print("Before:")
print(repr(Text2))

cleaned_text = " ".join(Text2.split())

print("\nAfter:")
print(repr(cleaned_text))



# Regex Cleaning - Delete characters


text3 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

lowercase_text3 = text3.lower()

pattern = r"[^a-z\s]"

cleaned_text3 = re.sub(pattern, "", lowercase_text3)

print("\n" + "=" * 50)
print("TASK 3A: REGEX CLEANING")
print("=" * 50)

print("Before:")
print(lowercase_text3)

print("\nAfter:")
print(cleaned_text3)



# Regex Cleaning - Replace important information

text4 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

lowercase_text4 = text4.lower()

replaced_text4 = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", lowercase_text4)

replaced_text4 = re.sub(r"£[\d,]+", "MONEY", replaced_text4)

replaced_text4 = re.sub(r"!{2,}", "!", replaced_text4)

print("\n" + "=" * 50)
print("TASK 3B: REGEX REPLACEMENT")
print("=" * 50)

print("Before:")
print(text4)

print("\nAfter:")
print(replaced_text4)



# Stopwords and Negation

text6 = "I don't think this jacket is worth the price. Not happy at all 😞"

tokens_text6 = word_tokenize(text6)

print("\n" + "=" * 50)
print("STOPWORDS AND NEGATION")
print("=" * 50)

print("\nBefore:")
print(tokens_text6)

english_stopwords = set(stopwords.words("english"))

cleaned_text6 = [
    word for word in tokens_text6
    if word.lower() not in english_stopwords
]

print("\nAfter removing NLTK stopwords:")
print(cleaned_text6)

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

cleaned_text6_custom = [
    word for word in tokens_text6
    if word.lower() not in custom_stopwords
]

print("\nAfter removing stopwords but keeping negation:")
print(cleaned_text6_custom)


# ==================================================
# 6. STEMMING
# ==================================================
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

word_list = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]


print("\n" + "=" * 50)
print("STEMMING COMPARISON")
print("=" * 50)

print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

for w in word_list:
    p_stem = porter.stem(w)
    l_stem = lancaster.stem(w)
    s_stem = snowball.stem(w)
    print(f"{w:<15} {p_stem:<15} {l_stem:<15} {s_stem:<15}")


# ==================================================
# 7. LEMMATIZATION
# ==================================================
lemmatizer = WordNetLemmatizer()

words_with_pos = [
    ("connection", "n"), ("connected", "v"), ("connecting", "v"),
    ("organization", "n"), ("organized", "v"), ("university", "n"),
    ("universal", "a"), ("flies", "n"), ("wolves", "n"),
    ("better", "a"), ("ran", "v")
]

print("\n" + "=" * 50)
print("LEMMATIZATION (DEFAULT VS WITH POS)")
print("=" * 50)

print(f"{'Word':<15} {'Default':<15} {'With POS':<15}")
print("-" * 45)

for w, pos_tag_hint in words_with_pos:
    default_lemma = lemmatizer.lemmatize(w)
    pos_lemma = lemmatizer.lemmatize(w, pos=pos_tag_hint)
    print(f"{w:<15} {default_lemma:<15} {pos_lemma:<15}")


# ==================================================
# 8. SENTENCE SPLITTING
# ==================================================
text_t7 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

print("\n" + "=" * 50)
print("SENTENCE SPLITTING")
print("=" * 50)

print("\nBefore:")
print(text_t7)

sentences_t3 = sent_tokenize(text_t7)

print("\nAfter sent_tokenize:")
for i, sent in enumerate(sentences_t3, 1):
    print(f"Sentence {i}: {sent}")
print(f"Total sentences: {len(sentences_t3)}")

# ==================================================
# 9. POS TAGGING AND NER
# ==================================================

tokens_t7 = word_tokenize(text_t7)
pos_tags_t7 = pos_tag(tokens_t7)
ner_tree_t7 = ne_chunk(pos_tags_t7)

print("\n" + "=" * 50)
print("POS TAGGING AND NER")
print("=" * 50)

print("\nPOS Tags:")
print(pos_tags_t7)

print("\nNamed Entities (NER):")
print(ner_tree_t7)


# ==================================================
# 6. STEMMING
# ==================================================
word_list = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\n" + "=" * 50)
print("STEMMING COMPARISON")
print("=" * 50)

print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

for w in word_list:
    p_stem = porter.stem(w)
    l_stem = lancaster.stem(w)
    s_stem = snowball.stem(w)
    print(f"{w:<15} {p_stem:<15} {l_stem:<15} {s_stem:<15}")


# ==================================================
# 7. LEMMATIZATION
# ==================================================
lemmatizer = WordNetLemmatizer()

# Liiska ereyada iyo nooca naxwaha (POS) ee mid walba ku habboon
words_with_pos = [
    ("connection", "n"), ("connected", "v"), ("connecting", "v"),
    ("organization", "n"), ("organized", "v"), ("university", "n"),
    ("universal", "a"), ("flies", "n"), ("wolves", "n"),
    ("better", "a"), ("ran", "v")
]

print("\n" + "=" * 50)
print("LEMMATIZATION (DEFAULT VS WITH POS)")
print("=" * 50)

print(f"{'Word':<15} {'Default':<15} {'With POS':<15}")
print("-" * 45)

for w, pos_tag_hint in words_with_pos:
    default_lemma = lemmatizer.lemmatize(w)
    pos_lemma = lemmatizer.lemmatize(w, pos=pos_tag_hint)
    print(f"{w:<15} {default_lemma:<15} {pos_lemma:<15}")


# ==================================================
# 8. SENTENCE SPLITTING
# ==================================================
text_t3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

print("\n" + "=" * 50)
print("SENTENCE SPLITTING")
print("=" * 50)

print("\nBefore:")
print(text_t3)

sentences_t3 = sent_tokenize(text_t3)

print("\nAfter sent_tokenize:")
for i, sent in enumerate(sentences_t3, 1):
    print(f"Sentence {i}: {sent}")
print(f"Total sentences: {len(sentences_t3)}")


# ==================================================
# 9. POS TAGGING AND NER
# ==================================================
tokens_t3 = word_tokenize(text_t3)
pos_tags_t3 = pos_tag(tokens_t3)
ner_tree_t3 = ne_chunk(pos_tags_t3)

print("\n" + "=" * 50)
print("POS TAGGING AND NER")
print("=" * 50)

print("\nPOS Tags:")
print(pos_tags_t3)

print("\nNamed Entities (NER):")
print(ner_tree_t3)

# ==================================================
# 10. FULL PIPELINE
# ==================================================
def run_full_pipeline(raw_text, stopword_set):
    # 1. Lowercase
    step1_lower = raw_text.lower()
    # 2. Regex (ka saar waxaan xarfo iyo bannaan ahayn)
    step2_regex = re.sub(r"[^a-z\s]", "", step1_lower)
    # 3. Tokenize
    step3_tokens = word_tokenize(step2_regex)
    # 4. Stopwords
    step4_filtered = [tok for tok in step3_tokens if tok not in stopword_set]
    # 5. Lemmatize
    step5_lemmas = [lemmatizer.lemmatize(tok) for tok in step4_filtered]
    
    return step1_lower, step2_regex, step3_tokens, step4_filtered, step5_lemmas

print("\n" + "=" * 50)
print("FULL PIPELINE (RUN 1: DEFAULT STOPWORDS)")
print("=" * 50)

s1, s2, s3, s4, s5 = run_full_pipeline(text6, english_stopwords)
print("1. Lowercase :", s1)
print("2. Regex     :", s2)
print("3. Tokenize  :", s3)
print("4. Stopwords :", s4)
print("5. Lemmatize :", s5)

print("\n" + "=" * 50)
print("FULL PIPELINE (RUN 2: KEEPING NEGATION)")
print("=" * 50)

c1, c2, c3, c4, c5 = run_full_pipeline(text6, custom_stopwords)
print("1. Lowercase :", c1)
print("2. Regex     :", c2)
print("3. Tokenize  :", c3)
print("4. Stopwords :", c4)
print("5. Lemmatize :", c5)