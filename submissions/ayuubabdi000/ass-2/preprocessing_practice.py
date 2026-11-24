import re
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

text1 = 'URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!'

print('\n===1) Lowercasing')

print('Before',text1)
print('After',text1.lower())

text4 = "    The   delivery   was   fast,   and the  packaging   was   great."

print("\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(text4))
print("After: ", repr(" ".join(text4.split())))



# --------------------------------
# 3) REGEX CLEANING
# --------------------------------

print("\n=== 3) REGEX CLEANING ===")

text1 ="URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
lowered = text1.lower()

cleaned = re.sub(r"[^a-z\s]", " ", lowered)

print("Before:", repr(lowered))
print("After: ", repr(cleaned))

# Characters deleted by the pattern

deleted = [c for c in lowered if not re.match(r"[a-z\s]", c)]

print("Deleted characters:", sorted(set(deleted)))

# Custom replacements

custom = lowered
custom = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", custom)
custom = re.sub(r"£\s?\d+(?:,\d{3})*(?:\.\d{2})?", "MONEY", custom)
custom = re.sub(r"!{2,}", "!", custom)

print("\nCustom replacement:")
print(custom)

print("\n=== 4) TOKENIZATION ===")

text2 = "I don't think this jacket is worth the price. Not happy at all 😞"
text4 = "    The   delivery   was   fast,   and the  packaging   was   great."

# Tokenize T2 using split()
tokens_split = text2.split()

print("\n--- split() ---")
print("Text:", text2)
print("Tokens:", tokens_split)
print("Token count:", len(tokens_split))


# Tokenize T2 using word_tokenize()
tokens_word = word_tokenize(text2)

print("\n--- word_tokenize() ---")
print("Text:", text2)
print("Tokens:", tokens_word)
print("Token count:", len(tokens_word))


# Tokenize T4
tokens_T4 = word_tokenize(text4)

vocabulary_T4 = set(tokens_T4)

print("\n--- T4 ---")
print("Text:", repr(text4))
print("Tokens:", tokens_T4)
print("Token count:", len(tokens_T4))
print("Vocabulary size:", len(vocabulary_T4))


print("\n=== 5) STOPWORDS AND NEGATION ===")


text2 = "I don't think this jacket is worth the price. Not happy at all 😞"
tokens = word_tokenize(text2)

stop_words = set(stopwords.words("english"))
without_stopwords = [token for token in tokens if token.lower() not in stop_words]


# Create a stopword list that keeps negation
stop_words_keep_negation = stop_words.copy()

stop_words_keep_negation.discard("not")
stop_words_keep_negation.discard("n't")
stop_words_keep_negation.discard("no")
stop_words_keep_negation.discard("never")

# Remove stopwords while keeping negation
keeping_negation = [
    token for token in tokens
    if token.lower() not in stop_words_keep_negation
]

print("Original tokens:        ", tokens)
print("Default stopwords:      ", without_stopwords)
print("Keeping negation:       ", keeping_negation)





# --------------------------------
# 6) Comparing stemmers
# --------------------------------

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

print("\n=== 6) COMPARING STEMMERS ===")

for word in words:
    print(word, "->",
          "Porter:", porter.stem(word),
          "| Lancaster:", lancaster.stem(word),
          "| Snowball:", snowball.stem(word))
    
# --------------------------------
# Finding words merged into the same stem
# --------------------------------

print("\n=== WORDS MERGED INTO THE SAME STEM ===")

for stemmer_name, stemmer in [
    ("Porter", porter),
    ("Lancaster", lancaster),
    ("Snowball", snowball)
]:

    stem_groups = {}

    for word in words:
        stem = stemmer.stem(word)

        if stem not in stem_groups:
            stem_groups[stem] = []

        stem_groups[stem].append(word)

    print("\n", stemmer_name)

    for stem, grouped_words in stem_groups.items():
        if len(grouped_words) > 1:
            print(grouped_words, "->", stem)    
            
            
            
# --------------------------------
# 7) Lemmatization with a POS hint
# --------------------------------

lemmatizer = WordNetLemmatizer()

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]

# POS hints for each word
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

print("\n=== 7) LEMMATIZATION ===")

for word in words:

    default_lemma = lemmatizer.lemmatize(word)

    pos = pos_hints[word]

    pos_lemma = lemmatizer.lemmatize(word, pos=pos)

    print(
        word,
        "-> Default:", default_lemma,
        "| POS:", pos,
        "| POS Lemma:", pos_lemma
    )            


print("\n=== WORDS CHANGED BY POS HINTS ===")

for word in words:

    default_lemma = lemmatizer.lemmatize(word)

    pos_lemma = lemmatizer.lemmatize(word, pos=pos_hints[word])

    if default_lemma != pos_lemma:

        print(word, "->", default_lemma, "->", pos_lemma)


# --------------------------------
# 8) Sentence splitting
# --------------------------------

text3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

print("\n=== 8) SENTENCE SPLITTING ===")

sentences = sent_tokenize(text3)

print("Text:", text3)
print("Sentences:", sentences)
print("Sentence count:", len(sentences))        


tagged = pos_tag(word_tokenize(text3))

print("\n=== 9) POS TAGGING ===")
print("Text:", text3)
print(tagged)

print("\n=== 9) NAMED ENTITY RECOGNITION ===")
entities = ne_chunk(tagged)
print(entities)


# --------------------------------
# 10) FULL PIPELINE
# Lowercase -> Regex -> Tokenize -> Stopwords -> Lemmatize
# --------------------------------

sample = "I don't think this jacket is worth the price. Not happy at all 😞"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# 1) Lowercasing
step_lower = sample.lower()

# 2) Regex cleaning
step_clean = re.sub(r"[^a-z\s]", " ", step_lower)

# 3) Tokenization
step_tokens = word_tokenize(step_clean)

# 4) Default stopwords
step_no_stop = [
    token for token in step_tokens
    if token not in stop_words
]

# 5) Lemmatization
step_lemmas = [
    lemmatizer.lemmatize(token) for token in step_no_stop
]


print("\n=== 10) FULL PIPELINE (DEFAULT STOPWORDS) ===")
print("Original:          ", sample)
print("Lowercased:        ", step_lower)
print("Regex cleaned:     ", step_clean)
print("Tokens:            ", step_tokens)
print("Stopwords removed: ", step_no_stop)
print("Lemmatized:        ", step_lemmas)




stop_words_keep_negation = stop_words.copy()

stop_words_keep_negation.discard("not")
stop_words_keep_negation.discard("n't")
stop_words_keep_negation.discard("no")
stop_words_keep_negation.discard("never")

step_keep_negation = [
    token for token in step_tokens
    if token not in stop_words_keep_negation
]

step_lemmas_keep_negation = [
    lemmatizer.lemmatize(token) for token in step_keep_negation
]

print("\n=== 10) FULL PIPELINE (KEEPING NEGATION) ===")
print("Original:          ", sample)
print("Lowercased:        ", step_lower)
print("Regex cleaned:     ", step_clean)
print("Tokens:            ", step_tokens)
print("Stopwords removed: ", step_keep_negation)
print("Lemmatized:        ", step_lemmas_keep_negation)