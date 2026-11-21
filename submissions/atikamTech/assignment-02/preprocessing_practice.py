import re
from collections import defaultdict

import nltk
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# ASSIGNMENT 2 - PREPROCESSING PRACTICE
# ============================================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies", "wolves",
    "better", "ran"
]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# TASK 1 - LOWERCASE
# ============================================================

print("\n" + "=" * 65)
print("TASK 1: LOWERCASE")
print("=" * 65)

print("Before:", T1)
print("After: ", T1.lower())

print("\nObservation:")
print("Lowercasing makes words consistent, but can remove capitalization")
print("signals that may help a spam detector identify urgent messages.")


# ============================================================
# TASK 2 - WHITESPACE NORMALIZATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 65)

normalized_T4 = " ".join(T4.split())

print("Before:", repr(T4))
print("After: ", repr(normalized_T4))


# ============================================================
# TASK 3 - REGULAR EXPRESSIONS
# ============================================================

print("\n" + "=" * 65)
print("TASK 3: REGEX CLEANING AND REPLACEMENT")
print("=" * 65)

lower_T1 = T1.lower()

# Find characters removed by the generic cleaning rule.
deleted_characters = re.findall(r"[^a-z\s]", lower_T1)
generic_cleaned = re.sub(r"[^a-z\s]", " ", lower_T1)

print("Original:             ", T1)
print("Lowercase:            ", lower_T1)
print("Characters removed:   ", deleted_characters)
print("Generic regex result: ", generic_cleaned)

# Replace useful information with general labels.
marked_T1 = lower_T1
marked_T1 = re.sub(r"0712345678", " PHONE_NUMBER ", marked_T1)
marked_T1 = re.sub(r"£\s?\d[\d,]*(?:\.\d{2})?", " MONEY ", marked_T1)
marked_T1 = marked_T1.replace("!!!", "!")

print("\nAfter preserving useful patterns:")
print(marked_T1)


# ============================================================
# TASK 4 - TOKENIZATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 4: TOKENIZATION")
print("=" * 65)

split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)

print("T2:", T2)
print("split() tokens:       ", split_tokens)
print("split() count:        ", len(split_tokens))
print("word_tokenize tokens: ", nltk_tokens)
print("word_tokenize count:  ", len(nltk_tokens))

T4_tokens = word_tokenize(normalized_T4)
T4_vocabulary = set(T4_tokens)

print("\nT4:", normalized_T4)
print("T4 tokens:            ", T4_tokens)
print("T4 token count:       ", len(T4_tokens))
print("T4 unique vocabulary: ", sorted(T4_vocabulary))
print("T4 vocabulary size:   ", len(T4_vocabulary))


# ============================================================
# TASK 5 - STOPWORDS AND NEGATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 5: STOPWORDS AND NEGATION")
print("=" * 65)

T2_tokens = word_tokenize(T2.lower())

default_stopword_result = [
    token for token in T2_tokens if token not in stop_words
]

# Keep negation words because they affect sentence meaning.
custom_stop_words = stop_words - {"not", "no", "never", "n't"}

negation_preserved_result = [
    token for token in T2_tokens if token not in custom_stop_words
]

print("Original tokens:             ", T2_tokens)
print("Default stopword removal:    ", default_stopword_result)
print("Keeping negation:            ", negation_preserved_result)


# ============================================================
# TASK 6 - STEMMING COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("TASK 6: COMPARING STEMMERS")
print("=" * 65)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

stemmer_results = {
    "Porter": {word: porter.stem(word) for word in words},
    "Lancaster": {word: lancaster.stem(word) for word in words},
    "Snowball": {word: snowball.stem(word) for word in words}
}

print(f"{'Word':<16}{'Porter':<16}{'Lancaster':<16}{'Snowball':<16}")

for word in words:
    print(
        f"{word:<16}"
        f"{stemmer_results['Porter'][word]:<16}"
        f"{stemmer_results['Lancaster'][word]:<16}"
        f"{stemmer_results['Snowball'][word]:<16}"
    )

print("\nWords that different stemmers combine:")
for stemmer_name, results in stemmer_results.items():
    groups = defaultdict(list)

    for original, stem in results.items():
        groups[stem].append(original)

    for stem, originals in groups.items():
        if len(originals) > 1:
            print(f"{stemmer_name}: {originals} -> {stem}")

print("\nNotice: stems are not always complete dictionary words.")


# ============================================================
# TASK 7 - LEMMATIZATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 7: LEMMATIZATION")
print("=" * 65)

print(f"{'Word':<16}{'Default':<16}{'With POS hint':<16}")

pos_examples = [
    ("running", "v"),
    ("connected", "v"),
    ("flies", "v"),
    ("wolves", "n"),
    ("better", "a"),
    ("ran", "v")
]

for word, pos in pos_examples:
    default = lemmatizer.lemmatize(word)
    hinted = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:<16}{default:<16}{hinted:<16}")


# ============================================================
# TASK 8 - SENTENCE TOKENIZATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 8: SENTENCE TOKENIZATION")
print("=" * 65)

sentences = sent_tokenize(T3)

print("Original text:", T3)
print("Sentences:", sentences)
print("Number of sentences:", len(sentences))

for abbreviation in ["Mr.", "a.m."]:
    print(f"Is {abbreviation!r} preserved in a sentence?",
          any(abbreviation in sentence for sentence in sentences))


# ============================================================
# TASK 9 - POS TAGGING AND NAMED ENTITY RECOGNITION
# ============================================================

print("\n" + "=" * 65)
print("TASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION")
print("=" * 65)

T3_tokens = word_tokenize(T3)
tagged_words = pos_tag(T3_tokens)

print("POS tagged words:")
print(tagged_words)

named_entities = ne_chunk(tagged_words)

print("\nNamed entity tree:")
print(named_entities)

print("\nExtracted entity table:")

for subtree in named_entities:
    if hasattr(subtree, "label"):
        entity_name = " ".join(word for word, tag in subtree.leaves())
        print(f"{entity_name:<25} {subtree.label()}")


# ============================================================
# TASK 10 - FULL PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 65)
print("TASK 10: FULL PREPROCESSING PIPELINE")
print("=" * 65)

def run_pipeline(text, selected_stopwords):
    step_lower = text.lower()

    # This generic rule removes punctuation, digits and emoji.
    step_regex = re.sub(r"[^a-z'\s]", " ", step_lower)

    step_tokens = word_tokenize(step_regex)

    step_stopwords = [
        token for token in step_tokens
        if token not in selected_stopwords
    ]

    step_lemmas = [
        lemmatizer.lemmatize(token)
        for token in step_stopwords
    ]

    print("\nOriginal:             ", text)
    print("Lowercase:            ", step_lower)
    print("Regex cleaned:        ", step_regex)
    print("Tokens:               ", step_tokens)
    print("After stopwords:      ", step_stopwords)
    print("After lemmatization:  ", step_lemmas)

    return step_lemmas


print("\n--- Pipeline with default stopwords ---")
default_pipeline = run_pipeline(T2, stop_words)

print("\n--- Pipeline preserving negation ---")
custom_pipeline = run_pipeline(T2, custom_stop_words)

print("\nPipeline comparison:")
print("Default result: ", default_pipeline)
print("Negation kept:  ", custom_pipeline)

print("\nEND OF ASSIGNMENT 2 PRACTICAL")