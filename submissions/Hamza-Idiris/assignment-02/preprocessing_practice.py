import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk


T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]


POS_HINTS = {
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
    "ran": "v",
}


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_before_after(label, before, after):
    print(f"Before: {before!r}")
    print(f"After : {after!r}")


def extract_entities(tree):
    """Return named-entity chunks as (text, label) pairs."""
    found = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            entity_text = " ".join(token for token, tag in subtree.leaves())
            found.append((entity_text, subtree.label()))
    return found


def main():
    # 1. Lowercasing
    heading("1. LOWERCASE T1")
    print("Before:", T1)
    t1_lower = T1.lower()
    print("After :", t1_lower)
    print("Observation: casing emphasis such as URGENT and WON is no longer visible.")

    # 2. Whitespace normalization
    heading("2. WHITESPACE NORMALIZATION FOR T4")
    t4_clean_space = re.sub(r"\s+", " ", T4).strip()
    show_before_after("T4", T4, t4_clean_space)

    # 3. Regex cleaning and replacement
    heading("3A. APPLY [^a-z\\s] TO LOWERCASED T1")
    regex_deleted = re.sub(r"[^a-z\s]", "", t1_lower)
    removed_chars = [ch for ch in t1_lower if not re.fullmatch(r"[a-z\s]", ch)]
    print("Before:", repr(t1_lower))
    print("After :", repr(regex_deleted))
    print("Deleted characters (in original order):", repr(removed_chars))
    print("Deleted character counts:", dict(Counter(removed_chars)))

    heading("3B. CUSTOM REGEX REPLACEMENTS FOR T1")
    t1_replaced = t1_lower
    t1_replaced = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", t1_replaced)
    t1_replaced = re.sub(r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "MONEY", t1_replaced)
    t1_replaced = re.sub(r"!{3,}", "!", t1_replaced)
    print("Before:", repr(t1_lower))
    print("After :", repr(t1_replaced))

    # 4. Tokenization
    heading("4A. TOKENIZE T2 WITH split() AND word_tokenize()")
    t2_split = T2.split()
    t2_word_tokens = word_tokenize(T2)
    print("split() tokens      :", t2_split)
    print("split() token count :", len(t2_split))
    print("word_tokenize tokens:", t2_word_tokens)
    print("word_tokenize count :", len(t2_word_tokens))

    heading("4B. TOKEN COUNT AND VOCABULARY SIZE FOR T4")
    t4_tokens = word_tokenize(T4)
    t4_vocab = sorted(set(token.lower() for token in t4_tokens))
    print("Tokens         :", t4_tokens)
    print("Token count    :", len(t4_tokens))
    print("Unique tokens  :", t4_vocab)
    print("Vocabulary size:", len(t4_vocab))

    # 5. Stopwords and negation
    heading("5. STOPWORDS AND NEGATION IN T2")
    english_stopwords = set(stopwords.words("english"))
    t2_lower_tokens = [token.lower() for token in t2_word_tokens]
    default_filtered = [token for token in t2_lower_tokens if token not in english_stopwords]
    negations = {"not", "n't", "no", "never"}
    custom_stopwords = english_stopwords - negations
    custom_filtered = [token for token in t2_lower_tokens if token not in custom_stopwords]
    print("Original tokens               :", t2_lower_tokens)
    print("NLTK stopwords removed        :", default_filtered)
    print("Negation-preserving stopwords :", custom_filtered)
    print("Negation words protected      :", sorted(negations))

    # 6. Stemming
    heading("6. PORTER, LANCASTER, AND SNOWBALL STEMMING")
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")
    print(f"{'WORD':<16}{'PORTER':<16}{'LANCASTER':<16}{'SNOWBALL':<16}")
    print("-" * 64)
    stem_rows = []
    for word in WORDS:
        row = (
            word,
            porter.stem(word),
            lancaster.stem(word),
            snowball.stem(word),
        )
        stem_rows.append(row)
        print(f"{row[0]:<16}{row[1]:<16}{row[2]:<16}{row[3]:<16}")

    for name, column in [("Porter", 1), ("Lancaster", 2), ("Snowball", 3)]:
        groups = {}
        for row in stem_rows:
            groups.setdefault(row[column], []).append(row[0])
        collisions = [members for members in groups.values() if len(members) > 1]
        print(f"{name} merged different words:", collisions if collisions else "No collision in this list")

    # 7. Lemmatization
    heading("7. WORDNET LEMMATIZATION: DEFAULT VS POS HINT")
    lemmatizer = WordNetLemmatizer()
    print(f"{'WORD':<16}{'DEFAULT (n)':<18}{'POS HINT':<12}{'WITH HINT':<18}")
    print("-" * 64)
    lemma_rows = []
    for word in WORDS:
        default_lemma = lemmatizer.lemmatize(word)
        pos = POS_HINTS[word]
        hinted_lemma = lemmatizer.lemmatize(word, pos=pos)
        lemma_rows.append((word, default_lemma, pos, hinted_lemma))
        print(f"{word:<16}{default_lemma:<18}{pos:<12}{hinted_lemma:<18}")
    changed_with_hint = [
        word for word, default, pos, hinted in lemma_rows if default != hinted
    ]
    print("Words changed only after POS hint:", changed_with_hint)

    # 8. Sentence splitting
    heading("8. SENTENCE SPLITTING FOR T3")
    sentences = sent_tokenize(T3)
    for index, sentence in enumerate(sentences, start=1):
        print(f"Sentence {index}: {sentence!r}")
    print("Sentence count:", len(sentences))

    # 9. POS tagging and NER
    heading("9. POS TAGGING AND NAMED ENTITY RECOGNITION FOR T3")
    t3_tokens = word_tokenize(T3)
    tags = pos_tag(t3_tokens)
    print("Tokens:", t3_tokens)
    print("POS tags:", tags)
    tree = ne_chunk(tags)
    print("NER tree:")
    print(tree)
    entities = extract_entities(tree)
    print("Extracted entities (text, NLTK label):")
    for entity_text, label in entities:
        print(f"- {entity_text!r}: {label}")
    if not entities:
        print("- No named entities returned by the NLTK chunker.")

    # 10. Full pipeline, with two stopword choices
    heading("10. FULL PIPELINE FOR T2 – TWO STOPWORD OPTIONS")
    def run_pipeline(name, stopword_set):
        print(f"\n--- {name} ---")
        stage1 = T2.lower()
        print("1. Lowercase :", repr(stage1))
        # Preserve apostrophes so contractions such as don't can be tokenized.
        stage2 = re.sub(r"[^a-z\s']", " ", stage1)
        stage2 = re.sub(r"\s+", " ", stage2).strip()
        print("2. Regex     :", repr(stage2))
        stage3 = word_tokenize(stage2)
        print("3. Tokenize  :", stage3)
        stage4 = [token for token in stage3 if token not in stopword_set]
        print("4. Stopwords :", stage4)
        # Use a simple POS-aware mapping for this sentence; WordNet defaults to nouns.
        stage5 = []
        for token in stage4:
            pos = "v" if token in {"think", "is"} else "a" if token == "happy" else "n"
            stage5.append(lemmatizer.lemmatize(token, pos=pos))
        print("5. Lemmatize :", stage5)
        return stage5

    default_pipeline = run_pipeline("NLTK DEFAULT STOPWORDS", english_stopwords)
    custom_pipeline = run_pipeline("NEGATION-PRESERVING STOPWORDS", custom_stopwords)
    print("\nPipeline comparison:")
    print("Default result:", default_pipeline)
    print("Negation-kept result:", custom_pipeline)
    print("The second pipeline should retain the negation token(s), preserving the complaint's negative meaning.")


if __name__ == "__main__":
    main()
