"""Assignment 2 – NLP preprocessing practical.
Run from the repository root or this folder:
    python preprocessing_practice.py
Save output with:
    python preprocessing_practice.py > output.txt
"""
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great. "
WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def main():
    section("TASK 1: LOWERCASE T1")
    print("BEFORE:", T1)
    print("AFTER: ", T1.lower())

    section("TASK 2: WHITESPACE NORMALIZATION T4")
    print("BEFORE:", repr(T4))
    print("AFTER: ", re.sub(r"\s+", " ", T4).strip())

    section("TASK 3: REGEX [^a-z\\s] ON LOWERCASE T1")
    lowered = T1.lower()
    removed = re.findall(r"[^a-z\s]", lowered)
    cleaned = re.sub(r"[^a-z\s]", "", lowered)
    print("BEFORE:", lowered)
    print("REMOVED CHARACTERS:", removed)
    print("AFTER:", cleaned)

    section("TASK 4: CUSTOM REGEX REPLACEMENTS")
    replaced = T1
    replaced = re.sub(r"\b\d{10}\b", "PHONE_NUMBER", replaced)
    replaced = re.sub(r"£\s?[\d,]+", "MONEY", replaced)
    replaced = re.sub(r"!{2,}", "!", replaced)
    print("BEFORE:", T1)
    print("AFTER: ", replaced)

    section("TASK 5: TOKENIZATION OF T2")
    print("split():", T2.split())
    print("word_tokenize():", word_tokenize(T2))

    section("TASK 5B: T4 TOKEN COUNT AND VOCABULARY")
    tokens_t4 = word_tokenize(re.sub(r"\s+", " ", T4).strip().lower())
    print("TOKENS:", tokens_t4)
    print("TOKEN COUNT:", len(tokens_t4))
    print("VOCABULARY:", sorted(set(tokens_t4)))
    print("VOCABULARY SIZE:", len(set(tokens_t4)))

    section("TASK 6: STEMMING COMPARISON")
    porter, lancaster = PorterStemmer(), LancasterStemmer()
    snowball = SnowballStemmer("english")
    print(f"{'WORD':<16}{'PORTER':<16}{'LANCASTER':<16}{'SNOWBALL':<16}")
    for word in WORDS:
        print(f"{word:<16}{porter.stem(word):<16}{lancaster.stem(word):<16}{snowball.stem(word):<16}")

    section("TASK 7: WORDNET LEMMATIZATION")
    lemmatizer = WordNetLemmatizer()
    pos_hints = {
        "connection": wordnet.NOUN, "connected": wordnet.VERB,
        "connecting": wordnet.VERB, "organization": wordnet.NOUN,
        "organized": wordnet.VERB, "university": wordnet.NOUN,
        "universal": wordnet.ADJ, "flies": wordnet.VERB,
        "wolves": wordnet.NOUN, "better": wordnet.ADJ, "ran": wordnet.VERB
    }
    print(f"{'WORD':<16}{'DEFAULT':<16}{'POS HINT':<16}")
    for word in WORDS:
        default = lemmatizer.lemmatize(word)
        hinted = lemmatizer.lemmatize(word, pos_hints[word])
        print(f"{word:<16}{default:<16}{hinted:<16}")

    section("TASK 8: SENTENCE SPLITTING AND POS TAGGING T3")
    sentences = sent_tokenize(T3)
    print("SENTENCES:")
    for sentence in sentences:
        print("-", sentence)
    tokens_t3 = word_tokenize(T3)
    tagged = pos_tag(tokens_t3)
    print("POS TAGS:", tagged)

    section("TASK 9: NAMED ENTITY RECOGNITION T3")
    tree = ne_chunk(tagged)
    print(tree)
    print("ENTITY TABLE (entity text | label):")
    for node in tree:
        if hasattr(node, "label"):
            entity = " ".join(token for token, tag in node.leaves())
            print(entity, "|", node.label())

    section("TASK 10: FULL PIPELINE ON T2")
    print("ORIGINAL:", T2)
    lower = T2.lower()
    print("LOWERCASE:", lower)
    cleaned_t2 = re.sub(r"[^a-z\s']", " ", lower)
    cleaned_t2 = re.sub(r"\s+", " ", cleaned_t2).strip()
    print("REGEX CLEANED:", cleaned_t2)
    tokens = word_tokenize(cleaned_t2)
    print("TOKENS:", tokens)
    english_stops = set(stopwords.words("english"))
    default_filtered = [w for w in tokens if w not in english_stops]
    print("DEFAULT STOPWORD REMOVAL:", default_filtered)
    custom_stops = english_stops - {"no", "not", "nor"}
    custom_filtered = [w for w in tokens if w not in custom_stops]
    print("NEGATION-PRESERVING STOPWORD REMOVAL:", custom_filtered)
    print("LEMMATIZED DEFAULT:", [lemmatizer.lemmatize(w) for w in default_filtered])
    print("LEMMATIZED KEEPING NEGATION:", [lemmatizer.lemmatize(w) for w in custom_filtered])

if __name__ == "__main__":
    main()
