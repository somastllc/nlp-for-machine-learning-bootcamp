
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer, RegexpStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
import nltk





# INPUT DATA

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = [
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


# LOWERCASE

print("\n========== TASK 1: LOWERCASE ==========")

print("Before:", T1)

lower_T1 = T1.lower()

print("After:", lower_T1)


# WHITESPACE NORMALIZATION


print("\n========== TASK 2: WHITESPACE ==========")

print("Before:", repr(T4))

print("After:", repr(" " .join(T4.split())))


#regex cleaning
result = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", T1)

result = re.sub(r"£\s?[\d,]+(?:\.\d{2})?", "MONEY", result)

result = re.sub(r"!{3,}", "!", result)
print("\n==========  regex cleaning ==========")
print("Before:", T1)
print("After:", result)


# Tokenize T2 using split()
tokens_split = T2.split()

print("T2 using split():")
print(tokens_split)
# Tokenize T2 using word_tokenize()
tokens_word = word_tokenize(T2)

print("\nT2 using word_tokenize():")
print(tokens_word)




# Tokenize T2
tokens = word_tokenize(T2)

# Get NLTK English stopwords
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print("\n========== TASK 5: DEFAULT STOPWORDS ==========")
print("Original tokens:", tokens)
print("After removing stopwords:", filtered_words)

# Custom stopwords: keep negation words
custom_stopwords = set(stopwords.words("english"))
custom_stopwords.difference_update({"not", "n't", "no", "never"})
custom_filtered_words = [word for word in tokens if word.lower() not in custom_stopwords]
print("\n========== CUSTOM STOPWORDS ==========")
print("After keeping negation:", custom_filtered_words)




print("\n========== TASK 6: COMPARING STEMMERS ==========")


porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
regexp = RegexpStemmer("ing$|s$|e$|able$", min=4)

for word in WORDS:
    print(
        word,
        "-> Porter:", porter.stem(word),
        "| Lancaster:", lancaster.stem(word),
        "| Snowball:", snowball.stem(word),
        "| Regexp:", regexp.stem(word)
    )


# Find words merged into the same stem
print("\n========== MERGED WORDS ==========")

for name, stemmer in [
    ("Porter", porter),
    ("Lancaster", lancaster),
    ("Snowball", snowball),
    ("Regexp", regexp)
]:
    groups = {}

    for word in WORDS:
        stem = stemmer.stem(word)
        groups.setdefault(stem, []).append(word)

    for stem, group in groups.items():
        if len(group) > 1:
            print(name, ":", group, "->", stem)





lemmatizer = WordNetLemmatizer()


# Correct POS hints for each word
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

print("\n========== TASK 7: LEMMATIZATION ==========")

changed_words = []

print(f"{'Word':<15} {'Default':<15} {'With POS hint':<15}")
print("-" * 50)

for word in WORDS:
    default_result = lemmatizer.lemmatize(word)

    correct_result = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    print(
        f"{word:<15} "
        f"{default_result:<15} "
        f"{correct_result:<15}"
    )

    if default_result != correct_result:
        changed_words.append(word)

print("\n========== WORDS CHANGED WITH POS HINT ==========")
print(changed_words)



# Split T3 into sentences
sentences = sent_tokenize(T3)

print("\n========== TASK 8: SENTENCE SPLITTING ==========")

for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

print("Total sentences:", len(sentences))





# Step 1: Tokenization
tokens = word_tokenize(T3)

# Step 2: POS Tagging
pos_tags = pos_tag(tokens)

print("\n========== TASK 9: POS TAGGING ==========")
for word, tag in pos_tags:
    print(word, "->", tag)

# Step 3: Named Entity Recognition
named_entities = ne_chunk(pos_tags)

print("\n========== NAMED ENTITY RECOGNITION ==========")
print(named_entities)

# Step 4: Extract entities
def extract_entities(tree):
    entities = []

    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_label = subtree.label()

            entities.append((entity_name, entity_label))

            # Check nested entities, if any
            entities.extend(extract_entities(subtree))

    return entities


entities = extract_entities(named_entities)

print("\n========== EXTRACTED ENTITIES ==========")
for entity, label in entities:
    print(entity, "->", label)

print("\nTotal entities:", len(entities))




# Define pipeline function
def preprocessing_pipeline(text, custom=False):

    print("\nOriginal text:")
    print(text)

    # STEP 1: Lowercase
    text = text.lower()
    print("\n1. LOWERCASE:")
    print(text)

    # STEP 2: Regex cleaning
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    print("\n2. REGEX CLEANING:")
    print(text)

    # STEP 3: Tokenization
    tokens = word_tokenize(text)

    print("\n3. TOKENIZATION:")
    print(tokens)

    # STEP 4: Stopwords removal
    stop_words = set(stopwords.words("english"))

    if custom:
        negation_words = {"not", "n't", "no", "never"}
        stop_words = stop_words - negation_words

    filtered_words = [
        word for word in tokens
        if word.lower() not in stop_words
    ]

    print("\n4. STOPWORDS REMOVAL:")
    print(filtered_words)

    # STEP 5: Lemmatization
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in filtered_words
    ]

    print("\n5. LEMMATIZATION:")
    print(lemmatized_words)

    return lemmatized_words


# RUN 1: Default stopwords
print("\n======================================")
print("RUN 1: DEFAULT STOPWORDS")
print("======================================")

default_result = preprocessing_pipeline(T2, custom=False)


# RUN 2: Custom stopwords
print("\n======================================")
print("RUN 2: CUSTOM STOPWORDS")
print("======================================")

custom_result = preprocessing_pipeline(T2, custom=True)
