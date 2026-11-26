"""
Assignment 2 – Preprocessing
Student: Idman Ahmed Ali
Course: NLP for Machine Learning Bootcamp

Run from the repository root:
python submissions/Idman4909/Assingment1/Assingment2/preprocessing_practice.py

Required NLTK data:
punkt, punkt_tab, stopwords, wordnet, omw-1.4,
averaged_perceptron_tagger_eng, maxent_ne_chunker_tab, words
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great. "


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # TASK 1
    heading("TASK 1: LOWERCASE T1")
    print("BEFORE:", T1)
    print("AFTER: ", T1.lower())

    # TASK 2
    heading("TASK 2: WHITESPACE NORMALIZATION T4")
    normalized_t4 = re.sub(r"\s+", " ", T4).strip()
    print("BEFORE:", repr(T4))
    print("AFTER: ", normalized_t4)

    # TASK 3
    heading("TASK 3: REGEX [^a-z\\s] ON LOWERCASE T1")
    lower_t1 = T1.lower()
    removed = re.findall(r"[^a-z\s]", lower_t1)
    cleaned = re.sub(r"[^a-z\s]", "", lower_t1)
    print("BEFORE:", lower_t1)
    print("REMOVED CHARACTERS:", removed)
    print("AFTER:", cleaned)

    # TASK 4
    heading("TASK 4: CUSTOM REGEX REPLACEMENTS")
    replaced = re.sub(r"\d{10}", "PHONE_NUMBER", T1)
    replaced = re.sub(r"£[\d,]+", "MONEY", replaced)
    replaced = re.sub(r"!{2,}", "!", replaced)
    print("BEFORE:", T1)
    print("AFTER: ", replaced)

    # TASK 5
    heading("TASK 5: TOKENIZATION OF T2")
    print("split():", T2.split())
    print("word_tokenize():", word_tokenize(T2))

    # TASK 5B
    heading("TASK 5B: T4 TOKEN COUNT AND VOCABULARY")
    tokens_t4 = word_tokenize(normalized_t4.lower())
    vocabulary = sorted(set(tokens_t4))
    print("TOKENS:", tokens_t4)
    print("TOKEN COUNT:", len(tokens_t4))
    print("VOCABULARY:", vocabulary)
    print("VOCABULARY SIZE:", len(vocabulary))

    # TASK 6
    heading("TASK 6: STEMMING COMPARISON")
    words = [
        "connection", "connected", "connecting", "organization", "organized",
        "university", "universal", "flies", "wolves", "better", "ran"
    ]
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")
    print(f"{'WORD':<16}{'PORTER':<16}{'LANCASTER':<16}{'SNOWBALL':<16}")
    for word in words:
        print(f"{word:<16}{porter.stem(word):<16}{lancaster.stem(word):<16}{snowball.stem(word):<16}")

    # TASK 7
    heading("TASK 7: WORDNET LEMMATIZATION")
    lemmatizer = WordNetLemmatizer()
    pos_hints = {
        "connected": "v", "connecting": "v", "organized": "v",
        "better": "a", "ran": "v"
    }
    print(f"{'WORD':<16}{'DEFAULT':<16}{'POS HINT':<16}")
    for word in words:
        default_lemma = lemmatizer.lemmatize(word)
        pos_lemma = lemmatizer.lemmatize(word, pos_hints.get(word, "n"))
        print(f"{word:<16}{default_lemma:<16}{pos_lemma:<16}")

    # TASK 8
    heading("TASK 8: SENTENCE SPLITTING AND POS TAGGING T3")
    sentences = sent_tokenize(T3)
    print("SENTENCES:")
    for sentence in sentences:
        print("-", sentence)
    tagged = pos_tag(word_tokenize(T3))
    print("POS TAGS:", tagged)

    # TASK 9
    heading("TASK 9: NAMED ENTITY RECOGNITION T3")
    tree = ne_chunk(tagged)
    print(tree)
    print("ENTITY TABLE (entity text | label):")
    for subtree in tree:
        if hasattr(subtree, "label"):
            entity_text = " ".join(token for token, tag in subtree.leaves())
            print(f"{entity_text} | {subtree.label()}")

    # TASK 10
    heading("TASK 10: FULL PIPELINE ON T2")
    lower_t2 = T2.lower()
    regex_cleaned = re.sub(r"[^a-z\s']", "", lower_t2)
    pipeline_tokens = word_tokenize(regex_cleaned)
    english_stopwords = set(stopwords.words("english"))
    default_filtered = [word for word in pipeline_tokens if word not in english_stopwords]
    negations = {"not", "no", "nor"}
    keep_negation = [
        word for word in pipeline_tokens
        if word not in english_stopwords or word in negations
    ]
    default_lemmas = [lemmatizer.lemmatize(word) for word in default_filtered]
    negation_lemmas = [lemmatizer.lemmatize(word) for word in keep_negation]

    print("ORIGINAL:", T2)
    print("LOWERCASE:", lower_t2)
    print("REGEX CLEANED:", re.sub(r"\s+", " ", regex_cleaned).strip())
    print("TOKENS:", pipeline_tokens)
    print("DEFAULT STOPWORD REMOVAL:", default_filtered)
    print("NEGATION-PRESERVING STOPWORD REMOVAL:", keep_negation)
    print("LEMMATIZED DEFAULT:", default_lemmas)
    print("LEMMATIZED KEEPING NEGATION:", negation_lemmas)


if __name__ == "__main__":
    main()
