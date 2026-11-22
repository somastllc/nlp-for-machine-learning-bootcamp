import re
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ==========================================
# INPUT DATA
# ==========================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ==========================================
# TASK 1: LOWERCASE
# ==========================================

heading("TASK 1: LOWERCASE")

print("Before:", T1)

lower_T1 = T1.lower()

print("After: ", lower_T1)

print("Spam detector information risk:")
print("Lowercasing removes capitalization differences and may lose")
print("the emphasis shown by uppercase words such as URGENT and NOW.")


# ==========================================
# TASK 2: WHITESPACE NORMALIZATION
# ==========================================

heading("TASK 2: WHITESPACE NORMALIZATION")

print("Before:", repr(T4))

clean_T4 = re.sub(r"\s+", " ", T4).strip()

print("After: ", repr(clean_T4))


# ==========================================
# TASK 3: REGEX CLEANING
# ==========================================

heading("TASK 3: REGEX CLEANING")

print("Original lowercase T1:")
print(lower_T1)

pattern = r"[^a-z\s]"

cleaned_T1 = re.sub(pattern, "", lower_T1)

deleted = re.findall(pattern, lower_T1)

print("\nPattern:", pattern)
print("After regex:", cleaned_T1)
print("Deleted characters:", deleted)

print("\nCustom regex replacement:")

custom_T1 = lower_T1

custom_T1 = re.sub(r"0712345678", "PHONE_NUMBER", custom_T1)
custom_T1 = re.sub(r"£1,000", "MONEY", custom_T1)
custom_T1 = re.sub(r"!{3,}", "!", custom_T1)

print("Before:", lower_T1)
print("After: ", custom_T1)


# ==========================================
# TASK 4: TOKENIZATION
# ==========================================

heading("TASK 4: TOKENIZATION")

print("T2:", T2)

split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)

print("\nUsing split():")
print(split_tokens)
print("Token count:", len(split_tokens))

print("\nUsing word_tokenize():")
print(nltk_tokens)
print("Token count:", len(nltk_tokens))

print("\nT4 tokenization:")

T4_tokens = word_tokenize(T4)

print("Tokens:", T4_tokens)
print("Token count:", len(T4_tokens))
print("Vocabulary size:", len(set(T4_tokens)))


# ==========================================
# TASK 5: STOPWORDS AND NEGATION
# ==========================================

heading("TASK 5: STOPWORDS AND NEGATION")

english_stopwords = set(stopwords.words("english"))

print("Original T2 tokens:")
print(nltk_tokens)

default_filtered = [
    token for token in nltk_tokens
    if token.lower() not in english_stopwords
]

print("\nAfter default stopword removal:")
print(default_filtered)

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

custom_filtered = [
    token for token in nltk_tokens
    if token.lower() not in custom_stopwords
]

print("\nAfter keeping negation words:")
print(custom_filtered)

print("\nMeaning:")
print("Keeping negation words preserves the negative meaning")
print("of the customer's complaint.")


# ==========================================
# TASK 6: STEMMING
# ==========================================

heading("TASK 6: STEMMING")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(f"{'WORD':<15} {'PORTER':<15} {'LANCASTER':<15} {'SNOWBALL':<15}")
print("-" * 65)

for word in words:
    print(
        f"{word:<15} "
        f"{porter.stem(word):<15} "
        f"{lancaster.stem(word):<15} "
        f"{snowball.stem(word):<15}"
    )

print("\nDifferent words may be reduced to the same stem.")
print("Some stems may not be valid dictionary words.")


# ==========================================
# TASK 7: LEMMATIZATION
# ==========================================

heading("TASK 7: LEMMATIZATION")

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

print(f"{'WORD':<15} {'DEFAULT':<15} {'WITH POS HINT':<20}")
print("-" * 55)

for word in words:
    default_lemma = lemmatizer.lemmatize(word)

    hinted_lemma = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    print(
        f"{word:<15} "
        f"{default_lemma:<15} "
        f"{hinted_lemma:<20}"
    )


# ==========================================
# TASK 8: SENTENCE SPLITTING
# ==========================================

heading("TASK 8: SENTENCE SPLITTING")

print("Original T3:")
print(T3)

sentences = sent_tokenize(T3)

print("\nSentences:")

for number, sentence in enumerate(sentences, start=1):
    print(f"{number}. {sentence}")

print("\nTotal sentences:", len(sentences))


# ==========================================
# TASK 9: POS TAGGING AND NER
# ==========================================

heading("TASK 9: POS TAGGING AND NER")

tokens_T3 = word_tokenize(T3)

tags = pos_tag(tokens_T3)

print("POS Tags:")

for token, tag in tags:
    print(f"{token:<15} {tag}")

print("\nNamed Entities:")

tree = ne_chunk(tags)

for node in tree:
    if hasattr(node, "label"):
        entity_name = " ".join(
            leaf[0] for leaf in node.leaves()
        )
        print(f"{entity_name:<25} {node.label()}")


# ==========================================
# TASK 10: FULL PREPROCESSING PIPELINE
# ==========================================

heading("TASK 10: FULL PREPROCESSING PIPELINE")


def full_pipeline(text, keep_negation=False):

    print("\nOriginal text:")
    print(text)

    # Step 1: Lowercase
    text = text.lower()

    print("\n1. Lowercase:")
    print(text)

    # Step 2: Regex cleaning
    text = re.sub(r"[^a-z\s']", "", text)

    print("\n2. Regex cleaning:")
    print(text)

    # Step 3: Tokenization
    tokens = word_tokenize(text)

    print("\n3. Tokenization:")
    print(tokens)

    # Step 4: Stopword removal
    stop_words = set(stopwords.words("english"))

    if keep_negation:
        stop_words -= negation_words

    tokens = [
        token for token in tokens
        if token.lower() not in stop_words
    ]

    print("\n4. Stopword removal:")
    print(tokens)

    # Step 5: Lemmatization
    lemmatized = []

    for token, tag in pos_tag(tokens):

        if tag.startswith("V"):
            pos = "v"
        elif tag.startswith("N"):
            pos = "n"
        elif tag.startswith("J"):
            pos = "a"
        else:
            pos = "n"

        lemmatized.append(
            lemmatizer.lemmatize(token, pos=pos)
        )

    print("\n5. Lemmatization:")
    print(lemmatized)

    return lemmatized


print("\nPIPELINE A: DEFAULT NLTK STOPWORDS")

result_default = full_pipeline(T2, keep_negation=False)

print("\nPIPELINE B: KEEP NEGATION WORDS")

result_negation = full_pipeline(T2, keep_negation=True)

print("\nFINAL COMPARISON")

print("Default stopwords:", result_default)
print("Keep negation:", result_negation)

print("\nThe second pipeline preserves the customer's negative")
print("opinion by keeping negation words such as n't and not.")

print("\nALL TASKS COMPLETED SUCCESSFULLY!")