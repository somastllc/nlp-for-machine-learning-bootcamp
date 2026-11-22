import re
import sys
import nltk

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize

# Stopwords
from nltk.corpus import stopwords

# Stemming
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Lemmatization
from nltk.stem import WordNetLemmatizer

# POS tagging and Named Entity Recognition
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

nltk.data.path.insert(
    0,
    r"C:\Users\SCRPC\AppData\Roaming\nltk_data"
)

# ==========================================
# PART B: NLP PREPROCESSING PRACTICE
# STEP 1: LOWERCASE
# ==========================================

# Original text T1
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

# Display the original text
print("STEP 1: LOWERCASE")
print("Before:", T1)

# Convert all uppercase letters to lowercase
T1_lower=T1.lower()

# Display the processed text
print("After:", T1_lower)

# ==========================================
# STEP 2: WHITESPACE NORMALIZATION
# ==========================================

# Original text T4 with unnecessary spaces
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

print("\nSTEP 2: WHITESPACE NORMALIZATION")


# repr() makes spaces and other whitespace easier to see
print("Before:", repr(T4))

# split() separates words and removes extra whitespace
# join() puts the words back with one space between them
T4_clean = " ".join(T4.split())

# Display the cleaned text
print("After: ", repr(T4_clean))


# ==========================================
# STEP 3: REGEX CLEANING
# ==========================================

print("\nSTEP 3: REGEX CLEANING")

# Part A: Apply the assignment's regex pattern
print("\nA. Remove characters using [^a-z\\s]")

# Use the lowercased T1 from Step 1
print("Before:", T1_lower)

# Remove everything except lowercase letters and whitespace
T1_regex_clean = re.sub(r"[^a-z\s]", "", T1_lower)

print("After:", T1_regex_clean)

# Part B: Replace important information instead of deleting it
print("\nB. Replace phone number, money, and repeated exclamation marks")

# Start with the original lowercased message
T1_replaced = T1_lower

# Replace the phone number with PHONE_NUMBER
T1_replaced = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", T1_replaced)

# Replace the money amount with MONEY
T1_replaced = re.sub(r"£[\d,]+", "MONEY", T1_replaced)

# Replace three or more exclamation marks with one exclamation mark
T1_replaced = re.sub(r"!{3,}", "!", T1_replaced)

print("Before:", T1_lower)
print("After: ", T1_replaced)



# ==========================================
# STEP 4: TOKENIZATION
# ==========================================
print("\nSTEP 4: TOKENIZATION")

# Original customer complaint T2
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

# Part A: Tokenization using Python split()
print("\nA. Python split()")

tokens_split = T2.split()

print("Original text:", T2)
print("Tokens:", tokens_split)
print("Token count:", len(tokens_split))

# Part B: Tokenization using NLTK word_tokenize()
print("\nB. NLTK word_tokenize()")

tokens_nltk = word_tokenize(T2)

print("Original text:", T2)
print("Tokens:", tokens_nltk)
print("Token count:", len(tokens_nltk))

# Part C: Tokenization of T4
print("\nC. Tokenization of T4")

# Use the whitespace-normalized T4 from Step 2
T4_tokens = word_tokenize(T4_clean)

# Count all tokens
T4_token_count = len(T4_tokens)

# Count unique tokens
T4_vocabulary_size = len(set(T4_tokens))

print("Text:", T4_clean)
print("Tokens:", T4_tokens)
print("Token count:", T4_token_count)
print("Vocabulary size:", T4_vocabulary_size)



# ==========================================
# STEP 5: STOPWORDS AND NEGATION
# ==========================================

from nltk.corpus import stopwords

print("\nSTEP 5: STOPWORDS AND NEGATION")

# Get NLTK's built-in English stopword list
nltk_stopwords = set(stopwords.words("english"))

# Use the T2 tokens created in Step 4
print("\nOriginal tokens:")
print(tokens_nltk)

# ------------------------------------------
# A. Remove NLTK default English stopwords
# ------------------------------------------

print("\nA. NLTK DEFAULT STOPWORD REMOVAL")

tokens_without_stopwords = [
    word for word in tokens_nltk
    if word.lower() not in nltk_stopwords
]

print("Before:", tokens_nltk)
print("After: ", tokens_without_stopwords)

# ------------------------------------------
# B. Create a custom stopword list
# ------------------------------------------

print("\nB. CUSTOM STOPWORD REMOVAL")

# Negation words that must be protected
negation_words = {"not", "n't", "no", "never"}

# Remove negation words from the default stopword list
custom_stopwords = nltk_stopwords - negation_words

# Remove ordinary stopwords but keep negation
tokens_with_negation = [
    word for word in tokens_nltk
    if word.lower() not in custom_stopwords
]

print("Protected negation words:", negation_words)
print("Before:", tokens_nltk)
print("After: ", tokens_with_negation)




# ==========================================
# STEP 6: STEMMING
# ==========================================

from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

print("\nSTEP 6: STEMMING")

# Original words provided in the assignment
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

# Create the three stemming algorithms
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# Print comparison heading
print("\nWORD STEMMING COMPARISON")

print(
    f"{'Original Word':<16}"
    f"{'Porter':<16}"
    f"{'Lancaster':<16}"
    f"{'Snowball':<16}"
)

print("-" * 64)

# Apply all three stemmers to every word
for word in words:

    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)

    # Print the original word and its three stems
    print(
        f"{word:<16}"
        f"{porter_result:<16}"
        f"{lancaster_result:<16}"
        f"{snowball_result:<16}"
    )

# ------------------------------------------
# Find words that Porter merges into one stem
# ------------------------------------------

print("\nWORDS MERGED BY PORTER STEMMER")

# Store the stems and their original words
stem_groups = {}

for word in words:
    stem = porter.stem(word)

    if stem not in stem_groups:
        stem_groups[stem] = []

    stem_groups[stem].append(word)

# Display groups containing different original words
for stem, original_words in stem_groups.items():

    if len(original_words) > 1:
        print("Original words:", original_words)
        print("Common stem:", stem)

# ------------------------------------------
# Identify stems that are not dictionary words
# ------------------------------------------

print("\nEXAMPLE OF A STEM THAT IS NOT A NORMAL WORD")

for word in words:

    stem = porter.stem(word)

    if stem != word and stem not in words:
        print("Original word:", word)
        print("Stem:", stem)
        break



# ==========================================
# STEP 7: LEMMATIZATION
# ==========================================

from nltk.stem import WordNetLemmatizer

print("\nSTEP 7: LEMMATIZATION")

# Create the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Assign the correct POS hint to each word
word_pos = {
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

# ------------------------------------------
# A. Default lemmatization
# ------------------------------------------

print("\nA. DEFAULT LEMMATIZATION")
print("Default POS is noun (n)")

default_results = {}

for word in words:

    # Lemmatize without providing a POS hint
    result = lemmatizer.lemmatize(word)

    default_results[word] = result

    print(f"{word:<16} -> {result}")

# ------------------------------------------
# B. Lemmatization with correct POS hints
# ------------------------------------------

print("\nB. LEMMATIZATION WITH POS HINTS")

hinted_results = {}

for word in words:

    # Get the correct POS category for this word
    pos = word_pos[word]

    # Lemmatize using the supplied POS
    result = lemmatizer.lemmatize(word, pos=pos)

    hinted_results[word] = result

    print(f"{word:<16} -> {result:<16} POS: {pos}")

# ------------------------------------------
# C. Find words changed by POS hints
# ------------------------------------------

print("\nC. WORDS CHANGED BY POS HINTS")

for word in words:

    default = default_results[word]
    hinted = hinted_results[word]

    if default != hinted:

        print(f"Word: {word}")
        print(f"Default result: {default}")
        print(f"POS-hinted result: {hinted}")
        print("-" * 35)



# ==========================================
# STEP 8: SENTENCE SPLITTING
# ==========================================

from nltk.tokenize import sent_tokenize

print("\nSTEP 8: SENTENCE SPLITTING")

# Original text T3
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

# Split T3 into individual sentences
sentences = sent_tokenize(T3)

# Display original text
print("\nOriginal text:")
print(T3)

# Display individual sentences
print("\nDetected sentences:")

for number, sentence in enumerate(sentences, start=1):
    print(f"Sentence {number}: {sentence}")

# Count the detected sentences
print("\nTotal number of sentences:", len(sentences))

# Check whether important abbreviations remain intact
print("\nABBREVIATION CHECK")

print("Does the first sentence contain 'Mr. Smith'?",
      "Mr. Smith" in sentences[0])

print("Does the second sentence end with 'a.m.'?",
      sentences[-1].endswith("a.m."))



# ==========================================
# STEP 9: POS TAGGING AND NER
# ==========================================

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

print("\nSTEP 9: POS TAGGING AND NAMED ENTITY RECOGNITION")

# Tokenize the original T3 text
T3_tokens = word_tokenize(T3)

# ------------------------------------------
# A. PART-OF-SPEECH TAGGING
# ------------------------------------------

print("\nA. POS TAGGING")

# Identify the grammatical category of each token
T3_pos_tags = pos_tag(T3_tokens)

print("Original text:", T3)
print("\nPOS tags:")

for word, tag in T3_pos_tags:
    print(f"{word:<15} {tag}")

# ------------------------------------------
# B. NAMED ENTITY RECOGNITION
# ------------------------------------------

print("\nB. NAMED ENTITY RECOGNITION")

# Identify named entities using POS-tagged words
T3_entities = ne_chunk(T3_pos_tags)

print("\nNER Tree:")
print(T3_entities)

# Extract named entities from the NLTK tree
detected_entities = []

for subtree in T3_entities:

    # A Tree represents a named entity
    if isinstance(subtree, Tree):

        entity_name = " ".join(
            word for word, tag in subtree.leaves()
        )

        entity_label = subtree.label()

        detected_entities.append((entity_name, entity_label))

# Display detected entities
print("\nDETECTED ENTITIES")

for entity, label in detected_entities:
    print(f"Entity: {entity:<20} Label: {label}")

# ------------------------------------------
# C. CHECK ENTITY LABELS
# ------------------------------------------

print("\nC. ENTITY LABEL EVALUATION")

# Expected entities and their correct categories
expected_entities = {
    "Mr. Smith": "PERSON",
    "London": "GPE",
    "Friday": "DATE",
    "Apple": "ORGANIZATION"
}

# Compare detected entities with expected entities
print("\nExpected entity comparison:")

for expected_name, expected_label in expected_entities.items():

    matching_labels = [
        label for entity, label in detected_entities
        if entity == expected_name
    ]

    if expected_label in matching_labels:
        result = "CORRECT"
    else:
        result = "NOT RECOGNIZED CORRECTLY"

    print(
        f"{expected_name:<15} "
        f"Expected: {expected_label:<15} "
        f"Result: {result}"
    )

# ------------------------------------------
# D. CHECK WHAT THE MODEL MISSED
# ------------------------------------------

print("\nD. ENTITIES MISSED BY THE MODEL")

for expected_name, expected_label in expected_entities.items():

    matching_labels = [
        label for entity, label in detected_entities
        if entity == expected_name
    ]

    if expected_label not in matching_labels:
        print(
            f"Missing or incorrect: {expected_name} "
            f"(Expected label: {expected_label})"
        )



# ==========================================
# STEP 10: FULL NLP PREPROCESSING PIPELINE
# ==========================================

print("\nSTEP 10: FULL NLP PREPROCESSING PIPELINE")

# Original customer complaint
pipeline_text = T2

# Create the WordNet lemmatizer if not already created
lemmatizer = WordNetLemmatizer()

# ------------------------------------------
# FUNCTION: LOWERCASE AND REGEX CLEANING
# ------------------------------------------

def clean_text(text):

    # Step 1: Convert all text to lowercase
    lowercase_text = text.lower()

    # Step 2: Replace unwanted characters with spaces
    # Keep lowercase letters, whitespace, and apostrophes
    regex_text = re.sub(r"[^a-z\s']", " ", lowercase_text)

    # Remove unnecessary repeated whitespace
    regex_text = " ".join(regex_text.split())

    return lowercase_text, regex_text


# ------------------------------------------
# FUNCTION: RUN THE FULL PIPELINE
# ------------------------------------------

def run_pipeline(text, stopword_list, pipeline_name):

    print("\n" + "=" * 55)
    print(pipeline_name)
    print("=" * 55)

    # Stage 1: Lowercase
    lowercase_text = text.lower()

    print("\n1. LOWERCASE")
    print("Before:", text)
    print("After: ", lowercase_text)

    # Stage 2: Regex cleaning
    regex_text = re.sub(
        r"[^a-z\s']",
        " ",
        lowercase_text
    )

    # Normalize whitespace after regex cleaning
    regex_text = " ".join(regex_text.split())

    print("\n2. REGEX CLEANING")
    print("Before:", lowercase_text)
    print("After: ", regex_text)

    # Stage 3: Tokenization
    tokens = word_tokenize(regex_text)

    print("\n3. TOKENIZATION")
    print("Before:", regex_text)
    print("After: ", tokens)

    # Stage 4: Stopword removal
    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stopword_list
    ]

    print("\n4. STOPWORD REMOVAL")
    print("Before:", tokens)
    print("After: ", filtered_tokens)

    # Stage 5: Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    print("\n5. LEMMATIZATION")
    print("Before:", filtered_tokens)
    print("After: ", lemmatized_tokens)

    # Final processed text
    final_text = " ".join(lemmatized_tokens)

    print("\nFINAL PROCESSED TEXT:")
    print(final_text)

    return final_text


# ------------------------------------------
# PIPELINE A: DEFAULT NLTK STOPWORDS
# ------------------------------------------

default_stopwords = set(stopwords.words("english"))

default_pipeline_result = run_pipeline(
    pipeline_text,
    default_stopwords,
    "PIPELINE A: DEFAULT NLTK STOPWORDS"
)


# ------------------------------------------
# PIPELINE B: CUSTOM NEGATION-KEEPING STOPWORDS
# ------------------------------------------

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = default_stopwords - negation_words

custom_pipeline_result = run_pipeline(
    pipeline_text,
    custom_stopwords,
    "PIPELINE B: CUSTOM NEGATION-KEEPING STOPWORDS"
)


# ------------------------------------------
# FINAL COMPARISON
# ------------------------------------------

print("\n" + "=" * 55)
print("FINAL PIPELINE COMPARISON")
print("=" * 55)

print("\nDefault stopword result:")
print(default_pipeline_result)

print("\nCustom stopword result:")
print(custom_pipeline_result)

print("\nDoes the custom result preserve 'not'?")
print("not" in custom_pipeline_result.split())

print("\nDoes the default result preserve 'not'?")
print("not" in default_pipeline_result.split())
