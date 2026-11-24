# Assignment 2 – Part B: Practical Preprocessing Script
# Setup once: python code/download-nltk-data.py
# Run:        python preprocessing_practice.py > output.txt

import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

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
    "ran",
]


def print_step(title: str, before, after) -> None:
    print(f"\n=== {title} ===")
    print("Before:", before)
    print("After: ", after)


def main() -> None:
    # 1. Lowercasing T1
    t1_lower = T1.lower()
    print_step("1) LOWERCASING T1", repr(T1), repr(t1_lower))

    # 2. Whitespace normalization on T4
    t4_space = " ".join(T4.split())
    print_step("2) WHITESPACE NORMALIZATION T4", repr(T4), repr(t4_space))

    # 3. Regex cleaning on T1
    t1_regex_delete = re.sub(r"[^a-z\s]", " ", t1_lower)
    deleted_chars = sorted(set(re.findall(r"[^a-z\s]", t1_lower)))
    print("\n=== 3a) REGEX DELETE [^a-z\\s] ON LOWERCASED T1 ===")
    print("Before:          ", repr(t1_lower))
    print("After:           ", repr(t1_regex_delete))
    print("Deleted symbols: ", deleted_chars)

    t1_replaced = re.sub(r"!+", "!", T1)
    t1_replaced = re.sub(r"£\d[\d,]*(?:\.\d+)?", "MONEY", t1_replaced)
    t1_replaced = re.sub(r"\b0\d{9,10}\b", "PHONE_NUMBER", t1_replaced)
    print_step("3b) REGEX REPLACE ON T1", repr(T1), repr(t1_replaced))

    # 4. Tokenization on T2 and T4
    t2_split = T2.split()
    t2_word_tok = word_tokenize(T2)
    print("\n=== 4a) TOKENIZATION ON T2 ===")
    print("Original:              ", repr(T2))
    print("split() tokens (count):", len(t2_split), "->", t2_split)
    print("word_tokenize (count): ", len(t2_word_tok), "->", t2_word_tok)

    t4_tok = word_tokenize(t4_space)
    t4_vocab = set(t4_tok)
    print("\n=== 4b) TOKENIZATION & VOCABULARY SIZE ON T4 ===")
    print("Normalized T4:         ", repr(t4_space))
    print("Tokens:                ", t4_tok)
    print("Token count:           ", len(t4_tok))
    print("Vocabulary size:       ", len(t4_vocab), "->", sorted(t4_vocab))

    # 5. Stopwords and negation on T2
    default_stops = set(stopwords.words("english"))
    t2_no_default_stop = [tok for tok in t2_word_tok if tok.lower() not in default_stops]

    negation_keep = {"not", "n't", "no", "never"}
    custom_stops = {w for w in default_stops if w.lower() not in negation_keep}
    t2_keep_neg = [tok for tok in t2_word_tok if tok.lower() not in custom_stops]

    print("\n=== 5) STOPWORDS AND NEGATION ON T2 ===")
    print("Before (word_tokenize):       ", t2_word_tok)
    print("After (default NLTK stopwords):", t2_no_default_stop)
    print("After (keeping negations):    ", t2_keep_neg)

    # 6. Stemming comparison
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    print("\n=== 6) STEMMING COMPARISON ===")
    for word in WORDS:
        p_stem = porter.stem(word)
        l_stem = lancaster.stem(word)
        s_stem = snowball.stem(word)
        print(
            f"{word:14s} | Porter: {p_stem:12s} | Lancaster: {l_stem:12s} | Snowball: {s_stem:12s}"
        )

    # 7. Lemmatization with default vs POS hints
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
        "ran": "v",
    }

    print("\n=== 7) LEMMATIZATION (DEFAULT VS POS HINT) ===")
    for word in WORDS:
        hint = pos_hints[word]
        default_lemma = lemmatizer.lemmatize(word)
        hint_lemma = lemmatizer.lemmatize(word, pos=hint)
        print(
            f"{word:14s} | Default (n): {default_lemma:12s} | POS={hint}: {hint_lemma:12s}"
        )

    # 8. Sentence splitting on T3
    t3_sentences = sent_tokenize(T3)
    print("\n=== 8) SENTENCE SPLITTING ON T3 ===")
    print("Before:        ", repr(T3))
    print("Sentence count:", len(t3_sentences))
    for idx, sent in enumerate(t3_sentences, start=1):
        print(f"  Sentence {idx}: {repr(sent)}")

    # 9. POS tagging and NER on T3
    t3_tokens = word_tokenize(T3)
    t3_pos = pos_tag(t3_tokens)
    t3_ner = ne_chunk(t3_pos)

    print("\n=== 9) POS TAGGING AND NER ON T3 ===")
    print("POS tags:", t3_pos)
    print("NER tree:", t3_ner)
    print("Extracted named entities:")
    for subtree in t3_ner:
        if hasattr(subtree, "label"):
            entity_text = " ".join(token for token, _ in subtree.leaves())
            print(f"  - {entity_text} ({subtree.label()})")

    # 10. Full pipeline on T2 (default stopwords vs negation-keeping stopwords)
    def run_full_pipeline(text: str, stop_set: set[str]) -> dict[str, object]:
        s1_lower = text.lower()
        s2_regex = re.sub(r"[^a-z\s]", " ", s1_lower)
        s3_tokens = word_tokenize(s2_regex)
        s4_no_stop = [tok for tok in s3_tokens if tok not in stop_set]
        s5_lemma = [lemmatizer.lemmatize(tok) for tok in s4_no_stop]
        return {
            "1_lowercase": s1_lower,
            "2_regex": s2_regex,
            "3_tokenize": s3_tokens,
            "4_stopwords": s4_no_stop,
            "5_lemmatize": s5_lemma,
        }

    pipe_default = run_full_pipeline(T2, default_stops)
    pipe_neg = run_full_pipeline(T2, custom_stops)

    print("\n=== 10a) FULL PIPELINE ON T2 (DEFAULT STOPWORDS) ===")
    print("Original:", repr(T2))
    for stage, val in pipe_default.items():
        print(f"  {stage:14s}: {val}")

    print("\n=== 10b) FULL PIPELINE ON T2 (KEEPING NEGATIONS) ===")
    print("Original:", repr(T2))
    for stage, val in pipe_neg.items():
        print(f"  {stage:14s}: {val}")


if __name__ == "__main__":
    main()

