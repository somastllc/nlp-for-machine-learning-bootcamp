
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# Download required NLTK resources
resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words"
]

for resource in resources:
    nltk.download(resource, quiet=True)


# Texts provided by the teacher
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great."


# Words provided by the teacher
words = [
    "connection", "connected", "connecting", "organization",
    "organized", "university", "universal", "flies",
    "wolves", "better", "ran"
]


print("=" * 60)
print("TASK 1: LOWERCASE")
print("=" * 60)

print("Before:", T1)
lower_T1 = T1.lower()
print("After: ", lower_T1)


print("\n" + "=" * 60)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 60)

print("Before:", repr(T4))
clean_T4 = re.sub(r"\s+", " ", T4).strip()
print("After: ", repr(clean_T4))


print("\n" + "=" * 60)
print("TASK 3: REGEX CLEANING")
print("=" * 60)

pattern = r"[^a-z\s]"

deleted_characters = re.findall(pattern, lower_T1)
regex_T1 = re.sub(pattern, "", lower_T1)

print("Before:", lower_T1)
print("After: ", regex_T1)
print("Deleted characters:", deleted_characters)
print("Unique deleted characters:", sorted(set(deleted_characters)))


print("\nCustom regex replacement:")

custom_T1 = T1
custom_T1 = re.sub(r"£[\d,]+(?:\.\d{2})?", "MONEY", custom_T1)
custom_T1 = re.sub(r"\b07\d{8}\b", "PHONE_NUMBER", custom_T1)
custom_T1 = re.sub(r"!{3,}", "!", custom_T1)

print("Before:", T1)
print("After: ", custom_T1)


print("\n" + "=" * 60)
print("TASK 4: TOKENIZATION")
print("=" * 60)

split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)

print("T2 using split():", split_tokens)
print("split() token count:", len(split_tokens))

print("\nT2 using word_tokenize():", nltk_tokens)
print("word_tokenize() token count:", len(nltk_tokens))

T4_tokens = word_tokenize(clean_T4)
T4_vocabulary = set(T4_tokens)

print("\nT4 tokens:", T4_tokens)
print("T4 token count:", len(T4_tokens))
print("T4 vocabulary size:", len(T4_vocabulary))
print("Unique T4 tokens:", sorted(T4_vocabulary))


print("\n" + "=" * 60)
print("TASK 5: STOPWORDS AND NEGATION")
print("=" * 60)

tokens_T2 = word_tokenize(T2.lower())

english_stopwords = set(stopwords.words("english"))

default_filtered = [
    word for word in tokens_T2
    if word not in english_stopwords
]

print("Original tokens:", tokens_T2)
print("Default stopword removal:", default_filtered)

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

custom_filtered = [
    word for word in tokens_T2
    if word not in custom_stopwords
]

print("Negation-preserving removal:", custom_filtered)


print("\n" + "=" * 60)
print("TASK 6: STEMMING COMPARISON")
print("=" * 60)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print(
    f"{'Original Word':<15}"
    f"{'Porter':<15}"
    f"{'Lancaster':<15}"
    f"{'Snowball':<15}"
)

for word in words:
    print(
        f"{word:<15}"
        f"{porter.stem(word):<15}"
        f"{lancaster.stem(word):<15}"
        f"{snowball.stem(word):<15}"
    )


print("\n" + "=" * 60)
print("TASK 7: LEMMATIZATION")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

pos_hints = {
    "connection": "n",
    "connected": "v",
    "connecting": "v",
    "organization": "n",
    "organized": "v",
    "university": "n",
    "universal": "a",
    "flies": "n",
    "wolves": "n",
    "better": "a",
    "ran": "v"
}

print(f"{'Original Word':<15}{'Default':<15}{'With POS hint':<15}")

for word in words:
    default_lemma = lemmatizer.lemmatize(word)
    hinted_lemma = lemmatizer.lemmatize(
        word, pos=pos_hints[word]
    )

    print(
        f"{word:<15}"
        f"{default_lemma:<15}"
        f"{hinted_lemma:<15}"
    )


print("\n" + "=" * 60)
print("TASK 8: SENTENCE SPLITTING")
print("=" * 60)

sentences = sent_tokenize(T3)

print("Original text:", T3)
print("Sentences found:", len(sentences))

for number, sentence in enumerate(sentences, start=1):
    print(f"Sentence {number}: {sentence}")


print("\n" + "=" * 60)
print("TASK 9: POS TAGGING AND NAMED ENTITY RECOGNITION")
print("=" * 60)

tokens_T3 = word_tokenize(T3)
tagged_words = pos_tag(tokens_T3)

print("POS tags:")
print(tagged_words)

named_entities = ne_chunk(tagged_words)

print("\nNamed entities:")

def print_entities(tree):
    for item in tree:
        if hasattr(item, "label"):
            entity_name = " ".join(word for word, tag in item.leaves())
            print(entity_name, ":", item.label())
            print_entities(item)
        elif isinstance(item, tuple):
            continue

print_entities(named_entities)

print("\nFull NER tree:")
print(named_entities)


print("\n" + "=" * 60)
print("TASK 10: FULL PREPROCESSING PIPELINE")
print("=" * 60)


def run_pipeline(text, stopword_list, pipeline_name):

    print("\n" + "-" * 50)
    print(pipeline_name)
    print("-" * 50)

    print("Original:", text)

    # Stage 1: Lowercase
    stage1 = text.lower()
    print("1. Lowercase:", stage1)

    # Stage 2: Regex cleaning
    stage2 = re.sub(r"[^a-z\s']", " ", stage1)
    stage2 = re.sub(r"\s+", " ", stage2).strip()
    print("2. Regex:", stage2)

    # Stage 3: Tokenization
    stage3 = word_tokenize(stage2)
    print("3. Tokenization:", stage3)

    # Stage 4: Stopword removal
    stage4 = [
        word for word in stage3
        if word not in stopword_list
    ]
    print("4. Stopword removal:", stage4)

    # Stage 5: Lemmatization
    stage5 = [
        lemmatizer.lemmatize(word)
        for word in stage4
    ]
    print("5. Lemmatization:", stage5)


run_pipeline(
    T2,
    english_stopwords,
    "PIPELINE 1: DEFAULT NLTK STOPWORDS"
)

run_pipeline(
    T2,
    custom_stopwords,
    "PIPELINE 2: NEGATION-KEEPING STOPWORDS"
)


print("\n" + "=" * 60)
print("END OF ASSIGNMENT 2 PRACTICAL")
print("=" * 60)




