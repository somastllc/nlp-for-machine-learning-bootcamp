import re
import sys

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

sys.stdout.reconfigure(encoding="utf-8")

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

WORDS = ["connection", "connected", "connecting", "organization", "organized",
         "university", "universal", "flies", "wolves", "better", "ran"]


def heading(number, title):
    print("\n=== " + str(number) + ") " + title + " ===")


heading(1, "LOWERCASING")
lowered_t1 = T1.lower()
print("Before:", T1)
print("After: ", lowered_t1)
print("Words written in capitals that were lost:", re.findall(r"\b[A-Z]{2,}\b", T1))

heading(2, "WHITESPACE NORMALIZATION")
print("Before:", repr(T4))
print("After: ", repr(" ".join(T4.split())))

heading(3, "REGEX CLEANING")
cleaned_t1 = re.sub(r"[^a-z\s]", " ", lowered_t1)
deleted = re.findall(r"[^a-z\s]", lowered_t1)
print("Lowercased T1:         ", repr(lowered_t1))
print("After [^a-z\\s]:        ", repr(cleaned_t1))
print("Deleted characters:    ", deleted)
print("Different characters deleted:", sorted(set(deleted)))

money_pattern = r"[£$€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
phone_pattern = r"\b\d{10}\b"
exclaim_pattern = r"!{2,}"

replaced = re.sub(money_pattern, "MONEY", T1)
replaced = re.sub(phone_pattern, "PHONE_NUMBER", replaced)
replaced = re.sub(exclaim_pattern, "!", replaced)
print("\nMy rules, replace instead of delete")
print("Money pattern:  ", money_pattern)
print("Phone pattern:  ", phone_pattern)
print("Exclaim pattern:", exclaim_pattern)
print("Before:", T1)
print("After: ", replaced)

heading(4, "TOKENIZATION")
split_tokens = T2.split()
nltk_tokens = word_tokenize(T2)
print("T2 split():       ", split_tokens)
print("Count:            ", len(split_tokens))
print("T2 word_tokenize: ", nltk_tokens)
print("Count:            ", len(nltk_tokens))

t4_tokens = word_tokenize(T4)
t4_lower_tokens = [token.lower() for token in t4_tokens]
print("\nT4 word_tokenize: ", t4_tokens)
print("Token count:      ", len(t4_tokens))
print("Vocabulary size:  ", len(set(t4_tokens)))
print("Vocabulary:       ", sorted(set(t4_tokens)))
print("Vocabulary size after lowercasing:", len(set(t4_lower_tokens)))

heading(5, "STOPWORDS AND NEGATION")
default_stopwords = set(stopwords.words("english"))
negation_words = {"not", "n't", "no", "never"}
my_stopwords = default_stopwords - negation_words

t2_tokens = word_tokenize(T2)
with_default = [token for token in t2_tokens if token.lower() not in default_stopwords]
with_mine = [token for token in t2_tokens if token.lower() not in my_stopwords]

print("T2 tokens:                  ", t2_tokens)
print("Default NLTK stopwords:     ", with_default)
print("My list (keeps negation):   ", with_mine)
print("Negation words I kept:      ", sorted(negation_words))
print("Removed only by the default list:", [t for t in t2_tokens if t.lower() in default_stopwords and t.lower() not in my_stopwords])

heading(6, "STEMMING")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
stemmers = {"Porter": porter, "Lancaster": lancaster, "Snowball": snowball}

print("Word".ljust(14) + "Porter".ljust(14) + "Lancaster".ljust(14) + "Snowball")
for word in WORDS:
    print(word.ljust(14)
          + porter.stem(word).ljust(14)
          + lancaster.stem(word).ljust(14)
          + snowball.stem(word))

print("\nWords that share the same stem")
for name, stemmer in stemmers.items():
    groups = {}
    for word in WORDS:
        groups.setdefault(stemmer.stem(word), []).append(word)
    print(name + ":")
    for stem, members in groups.items():
        if len(members) > 1:
            print("   ", stem, "<-", members)

print("\nStems that are not found in WordNet (a quick check for 'not a real word')")
for name, stemmer in stemmers.items():
    not_real = sorted({stemmer.stem(word) for word in WORDS if not wordnet.synsets(stemmer.stem(word))})
    print(name + ":", not_real)

heading(7, "LEMMATIZATION")
lemmatizer = WordNetLemmatizer()
pos_hint = {
    "connection": "n", "connected": "v", "connecting": "v",
    "organization": "n", "organized": "v",
    "university": "n", "universal": "a",
    "flies": "v", "wolves": "n", "better": "a", "ran": "v",
}

print("Word".ljust(14) + "Default".ljust(14) + "Hint".ljust(6) + "With hint".ljust(14) + "Changed by hint")
changed_by_hint = []
for word in WORDS:
    default_lemma = lemmatizer.lemmatize(word)
    hinted_lemma = lemmatizer.lemmatize(word, pos=pos_hint[word])
    changed = default_lemma != hinted_lemma
    if changed:
        changed_by_hint.append(word)
    print(word.ljust(14) + default_lemma.ljust(14) + pos_hint[word].ljust(6)
          + hinted_lemma.ljust(14) + str(changed))
print("\nWords that changed only when the hint was given:", changed_by_hint)

heading(8, "SENTENCE SPLITTING")
sentences = sent_tokenize(T3)
print("Number of sentences:", len(sentences))
for number, sentence in enumerate(sentences, start=1):
    print(str(number) + ":", sentence)

heading(9, "POS TAGGING AND NER")
tagged = pos_tag(word_tokenize(T3))
print("POS tags:")
for word, tag in tagged:
    print("   ", word.ljust(10), tag)

tree = ne_chunk(tagged)
print("\nne_chunk tree:")
print(tree)

entities = []
not_grouped = []
for node in tree:
    if hasattr(node, "label"):
        entities.append((node.label(), " ".join(word for word, tag in node.leaves())))
    elif node[1].startswith("NNP"):
        not_grouped.append(node)

print("\nEntities found (label, text):")
for label, text in entities:
    print("   ", label.ljust(14), text)
print("Proper nouns the model did not group into an entity:", not_grouped)

heading(10, "FULL PIPELINE ON T2")


def pipeline(text, stop_set):
    step_lower = text.lower()
    step_regex = re.sub(r"[^a-z\s']", " ", step_lower)
    step_tokens = word_tokenize(step_regex)
    step_stop = [token for token in step_tokens if token not in stop_set]
    step_lemmas = [lemmatizer.lemmatize(token) for token in step_stop]
    print("Original:          ", text)
    print("Lowercased:        ", step_lower)
    print("After regex:       ", repr(step_regex))
    print("Tokens:            ", step_tokens)
    print("Stopwords removed: ", step_stop)
    print("Lemmatized:        ", step_lemmas)
    return step_lemmas


print("\nRun A: NLTK default stopwords")
result_default = pipeline(T2, default_stopwords)
print("\nRun B: my stopword list that keeps negation")
result_mine = pipeline(T2, my_stopwords)

print("\nFinal result, default list:", result_default)
print("Final result, my list:     ", result_mine)

strict = re.sub(r"[^a-z\s]", " ", T2.lower())
strict_tokens = [t for t in word_tokenize(strict) if t not in my_stopwords]
print("\nSide check, strict regex [^a-z\\s] with my list:", strict_tokens)
