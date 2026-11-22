"""Assignment 2 - Preprocessing practical.

First, from the bootcamp repository root, run the course setup once:
    python code/download-nltk-data.py

Then open a terminal in this assignment-02 folder and run:
    python preprocessing_practice.py > output.txt

All reported practical results should come from running this script.
"""

import re
import sys
from collections import Counter

import nltk
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import (
    LancasterStemmer,
    PorterStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree

# Keep redirected output readable on Windows terminals and in output.txt.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------
# Input texts and words supplied in Assignment 2.
# ------------------------------------------------------------
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran",
]


def heading(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_pipeline(label: str, text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> None:
    """Print every stage of the T2 preprocessing pipeline."""
    print(f"\n--- {label} ---")
    print("Original:          ", repr(text))

    lower_text = text.lower()
    print("Lowercase:         ", repr(lower_text))

    # Keep apostrophes so contractions can still be tokenized as negation.
    # This pattern removes other non-English letters, emoji, and punctuation.
    regex_text = re.sub(r"[^a-z\s']", " ", lower_text)
    regex_text = re.sub(r"\s+", " ", regex_text).strip()
    print("Regex cleaned:     ", repr(regex_text))

    tokens = word_tokenize(regex_text)
    print("Tokens:            ", tokens)

    remaining = [token for token in tokens if token.lower() not in stop_words]
    print("Stopwords removed:", remaining)

    lemmas = [lemmatizer.lemmatize(token) for token in remaining]
    print("Lemmatized:        ", lemmas)


def extract_entities(tree: Tree) -> list[tuple[str, str]]:
    """Return (entity_text, label) pairs from an NLTK chunk tree."""
    found: list[tuple[str, str]] = []
    for node in tree:
        if isinstance(node, Tree):
            entity_text = " ".join(word for word, _tag in node.leaves())
            found.append((entity_text, node.label()))
    return found


def normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def find_entity_prediction(
    target: str, predictions: list[tuple[str, str]]
) -> str | None:
    """Match an expected name to an exact or partial predicted NER span."""
    target_words = normalized_words(target)
    for predicted_text, label in predictions:
        predicted_words = normalized_words(predicted_text)
        if target_words and (target_words <= predicted_words or predicted_words <= target_words):
            return label
    return None


def main() -> None:
    # Required NLTK corpora/models are installed by the course setup script.
    try:
        english_stopwords = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
    except LookupError as error:
        print(
            "NLTK data is missing. From the bootcamp repository root, run:\n"
            "    python code/download-nltk-data.py\n"
            "Then run this script again.\n\n"
            f"NLTK detail: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # ------------------------------------------------------------
    # 1. Lowercasing: T1
    # ------------------------------------------------------------
    heading("1) LOWERCASE T1")
    print("Before:", T1)
    lower_T1 = T1.lower()
    print("After: ", lower_T1)
    print(
        "Possible information lost: all-capital emphasis in URGENT and WON, "
        "and the strong visual emphasis of NOW."
    )

    # ------------------------------------------------------------
    # 2. Whitespace normalization: T4
    # ------------------------------------------------------------
    heading("2) WHITESPACE NORMALIZATION T4")
    print("Before:", repr(T4))
    normalized_T4 = " ".join(T4.split())
    print("After: ", repr(normalized_T4))

    # ------------------------------------------------------------
    # 3. Regex cleaning: broad rule and task-specific replacement
    # ------------------------------------------------------------
    heading("3) REGEX CLEANING T1")
    broad_cleaned = re.sub(r"[^a-z\s]", " ", lower_T1)
    print("Lowercased input:     ", repr(lower_T1))
    print("Pattern [^a-z\\s]:    ", repr(broad_cleaned))

    removed_characters = list(
        dict.fromkeys(
            char for char in lower_T1
            if not ("a" <= char <= "z" or char.isspace())
        )
    )
    print("Distinct removed characters:", [repr(char) for char in removed_characters])
    print("These include the pound sign, digits, comma, full stop and exclamation marks.")

    # Replace special values with markers rather than deleting their meaning.
    marked_T1 = re.sub(r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "MONEY", T1)
    marked_T1 = re.sub(r"\b0\d{9}\b", "PHONE_NUMBER", marked_T1)
    marked_T1 = re.sub(r"!{2,}", "!", marked_T1)
    print("Task-specific replacements:", marked_T1)

    # ------------------------------------------------------------
    # 4. Tokenization: T2 and T4
    # ------------------------------------------------------------
    heading("4) TOKENIZATION")
    t2_split_tokens = T2.split()
    t2_word_tokens = word_tokenize(T2)
    print("T2 with split():       ", t2_split_tokens)
    print("T2 split() token count:", len(t2_split_tokens))
    print("T2 with word_tokenize():", t2_word_tokens)
    print("T2 word_tokenize count:", len(t2_word_tokens))

    t4_tokens = word_tokenize(T4)
    print("\nT4 word tokens:       ", t4_tokens)
    print("T4 token count:       ", len(t4_tokens))
    print("T4 vocabulary size (case-sensitive):", len(set(t4_tokens)))
    print(
        "T4 vocabulary size (after lowercasing):",
        len({token.lower() for token in t4_tokens}),
    )

    # ------------------------------------------------------------
    # 5. Stopwords and negation: T2
    # ------------------------------------------------------------
    heading("5) STOPWORDS AND NEGATION T2")
    custom_stopwords = english_stopwords - {"not", "n't", "no", "never"}

    default_remaining = [
        token for token in t2_word_tokens
        if token.lower() not in english_stopwords
    ]
    custom_remaining = [
        token for token in t2_word_tokens
        if token.lower() not in custom_stopwords
    ]
    print("Original tokens:                 ", t2_word_tokens)
    print("After default English stopwords:", default_remaining)
    print("After keeping negation words:    ", custom_remaining)
    print("Negation terms protected: not, n't, no, never")

    # ------------------------------------------------------------
    # 6. Stemming comparison
    # ------------------------------------------------------------
    heading("6) STEMMING COMPARISON")
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    print(f"{'Word':<16}{'Porter':<16}{'Lancaster':<16}{'Snowball':<16}")
    print("-" * 64)
    stem_results: dict[str, dict[str, str]] = {
        "Porter": {}, "Lancaster": {}, "Snowball": {}
    }
    for word in WORDS:
        porter_stem = porter.stem(word)
        lancaster_stem = lancaster.stem(word)
        snowball_stem = snowball.stem(word)
        stem_results["Porter"][word] = porter_stem
        stem_results["Lancaster"][word] = lancaster_stem
        stem_results["Snowball"][word] = snowball_stem
        print(f"{word:<16}{porter_stem:<16}{lancaster_stem:<16}{snowball_stem:<16}")

    print("\nExample of two different words merged by Porter:")
    print("connected ->", porter.stem("connected"), "; connecting ->", porter.stem("connecting"))
    print("Example of a non-dictionary stem: flies ->", porter.stem("flies"))

    # ------------------------------------------------------------
    # 7. Lemmatization: default noun setting and selected POS hints
    # ------------------------------------------------------------
    heading("7) LEMMATIZATION WITH POS HINTS")
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
    print(f"{'Word':<16}{'Default (noun)':<22}{'With POS hint':<22}{'POS'}")
    print("-" * 72)
    for word in WORDS:
        default_lemma = lemmatizer.lemmatize(word)
        pos = pos_hints[word]
        hinted_lemma = lemmatizer.lemmatize(word, pos=pos)
        print(f"{word:<16}{default_lemma:<22}{hinted_lemma:<22}{pos}")

    # ------------------------------------------------------------
    # 8. Sentence splitting: T3
    # ------------------------------------------------------------
    heading("8) SENTENCE SPLITTING T3")
    sentences = sent_tokenize(T3)
    for index, sentence in enumerate(sentences, start=1):
        print(f"Sentence {index}: {sentence}")
    print("Sentence count:", len(sentences))

    # ------------------------------------------------------------
    # 9. POS tagging and named entity recognition: T3
    # ------------------------------------------------------------
    heading("9) POS TAGGING AND NAMED ENTITY RECOGNITION T3")
    tagged_T3 = pos_tag(word_tokenize(T3))
    print("POS tags:")
    print(tagged_T3)

    ner_tree = ne_chunk(tagged_T3)
    print("\nNER tree:")
    print(ner_tree)

    predicted_entities = extract_entities(ner_tree)
    print("\nDetected entity spans:")
    if predicted_entities:
        for entity_text, label in predicted_entities:
            print(f"- {entity_text}: {label}")
    else:
        print("No entity spans were detected.")

    # The sentence itself provides the intended real-world context:
    # Smith is a person, London a place, Friday a day, and Apple a company.
    expected_entities = [
        ("Mr. Smith", "PERSON"),
        ("London", "GPE"),
        ("Friday", "DATE"),
        ("Apple", "ORGANIZATION"),
    ]
    print("\nEntity review table (compare the model's prediction with context):")
    print(f"{'Text':<16}{'Expected':<18}{'Predicted':<20}{'Assessment'}")
    print("-" * 72)
    for text, expected_label in expected_entities:
        predicted_label = find_entity_prediction(text, predicted_entities)
        display_prediction = predicted_label if predicted_label else "NOT RECOGNIZED"
        if predicted_label is None:
            assessment = "Missed"
        elif predicted_label == expected_label:
            assessment = "Correct"
        else:
            assessment = "Wrong label"
        print(f"{text:<16}{expected_label:<18}{display_prediction:<20}{assessment}")

    # ------------------------------------------------------------
    # 10. Full preprocessing pipeline: T2, two stopword policies
    # ------------------------------------------------------------
    heading("10) FULL PIPELINE T2")
    print_pipeline("A. Default NLTK stopwords", T2, english_stopwords, lemmatizer)
    print_pipeline("B. Stopwords that keep negation", T2, custom_stopwords, lemmatizer)

    # ------------------------------------------------------------
    # Final note for the practical report.
    # ------------------------------------------------------------
    heading("PRACTICAL SUMMARY")
    print("Review the outputs above before writing the practical report.")
    print("For the complaint in T2, compare whether 'not' and \"n't\" remain.")
    print("For T3, check the detected entity labels against the sentence context.")


if __name__ == "__main__":
    try:
        main()
    except LookupError as error:
        print(
            "A required NLTK tokenizer/model/corpus is missing. Run the course setup "
            "script (python code/download-nltk-data.py) and try again.\n"
            f"Details: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
