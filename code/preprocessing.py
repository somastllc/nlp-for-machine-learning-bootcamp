# Lesson 2 — Preprocessing (teaching script)
# Follows lessons/02-preprocessing.md step by step.
# Setup once:  python code/download-nltk-data.py
# Run:         python code/preprocessing.py

import re

from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# --------------------------------
# 1) Lowercasing
# "Python", "PYTHON" and "python" become one token.
# But capitals can carry meaning: "US" vs "us", or "FREE PRIZE!!!" in spam.
# --------------------------------
text = "I LOVE Python"
print("\n=== 1) LOWERCASING ===")
print("Before:", text)
print("After: ", text.lower())

# --------------------------------
# 2) Whitespace normalization
# split() breaks on any run of spaces; " ".join() puts single spaces back.
# --------------------------------
text = "I   love    NLP"
print("\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(text))
print("After: ", repr(" ".join(text.split())))

# --------------------------------
# 3) Regex cleaning
# [^a-z\s] means: any character that is NOT a lowercase letter or a space.
# Every such character becomes a space. Watch the "$500" disappear,
# even though the amount could be useful evidence of spam.
# --------------------------------
text = "Win $500 NOW!!!"
lowered = text.lower()
cleaned = re.sub(r"[^a-z\s]", " ", lowered)
print("\n=== 3) REGEX CLEANING ===")
print("Before:           ", repr(text))
print("After lowercasing:", repr(lowered))
print("After regex:      ", repr(cleaned))

# --------------------------------
# 4) Tokenization
# split() keeps punctuation stuck to words ("Python!").
# word_tokenize separates it, and splits "can't" into "ca" + "n't".
# The tokens you get depend on the tokenizer's rules.
# --------------------------------
text = "I love Python!"
print("\n=== 4) TOKENIZATION ===")
print("Text:           ", text)
print("split():        ", text.split())
print("word_tokenize():", word_tokenize(text))

text = "I can't recommend this."
print("Text:           ", text)
print("word_tokenize():", word_tokenize(text))

# --------------------------------
# 5) Stopwords and negation
# NLTK's English stopword list also contains "not".
# Removing it turns "this movie is not good" into "movie good".
# --------------------------------
stop_words = set(stopwords.words("english"))
tokens = word_tokenize("this movie is not good")

without_stopwords = [token for token in tokens if token not in stop_words]

stop_words_keep_not = set(stopwords.words("english"))
stop_words_keep_not.discard("not")
keeping_not = [token for token in tokens if token not in stop_words_keep_not]

print("\n=== 5) STOPWORDS AND NEGATION ===")
print("Tokens:                ", tokens)
print("Default stopword list: ", without_stopwords)
print("Keeping 'not':         ", keeping_not)

# --------------------------------
# 6) Full pipeline: lowercase -> regex -> tokenize -> stopwords -> lemmatize
# The order matters: cleaning before tokenization decides which tokens exist.
# Check: does "not" survive? The "8" is also deleted by the regex.
# --------------------------------
sample = "Not every free offer is spam; meet me at 8 tonight!"
lemmatizer = WordNetLemmatizer()

step_lower = sample.lower()
step_clean = re.sub(r"[^a-z\s]", " ", step_lower)
step_tokens = word_tokenize(step_clean)
step_no_stop = [token for token in step_tokens if token not in stop_words]
step_lemmas = [lemmatizer.lemmatize(token) for token in step_no_stop]

print("\n=== 6) FULL PIPELINE ===")
print("Original:          ", sample)
print("Lowercased:        ", step_lower)
print("Regex cleaned:     ", step_clean)
print("Tokens:            ", step_tokens)
print("Stopwords removed: ", step_no_stop)
print("Lemmatized:        ", step_lemmas)

step_keep_not = [token for token in step_tokens if token not in stop_words_keep_not]
print("Same, keeping 'not':", [lemmatizer.lemmatize(token) for token in step_keep_not])

# --------------------------------
# 7) Comparing stemmers
# Porter: classic rules. Lancaster: more aggressive. Snowball: improved Porter.
# Regexp: removes only the endings you list. No stemmer is "correct".
# --------------------------------
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
regexp = RegexpStemmer("ing$|s$|e$|able$", min=4)

words = ["running", "studies", "fairly", "generously", "happiness", "eating"]

print("\n=== 7) COMPARING STEMMERS ===")
for word in words:
    print(word, "->",
          "Porter:", porter.stem(word),
          "| Lancaster:", lancaster.stem(word),
          "| Snowball:", snowball.stem(word),
          "| Regexp:", regexp.stem(word))

# --------------------------------
# 8) Lemmatization with a part-of-speech hint
# WordNetLemmatizer treats every word as a noun unless told otherwise.
# pos="v" means verb, pos="a" means adjective, pos="n" means noun.
# --------------------------------
print("\n=== 8) LEMMATIZATION WITH A POS HINT ===")
print("running (default):", lemmatizer.lemmatize("running"))
print("running (pos='v'):", lemmatizer.lemmatize("running", pos="v"))
print("better  (default):", lemmatizer.lemmatize("better"))
print("better  (pos='a'):", lemmatizer.lemmatize("better", pos="a"))
print("studies (default):", lemmatizer.lemmatize("studies"))
print("studies (pos='v'):", lemmatizer.lemmatize("studies", pos="v"))

# --------------------------------
# 9) Sentence splitting
# sent_tokenize does not simply cut at every period.
# --------------------------------
text = "Amina works at the United Nations in Nairobi. She sent the report on Monday."
print("\n=== 9) SENTENCE SPLITTING ===")
print(sent_tokenize(text))

# --------------------------------
# 10) POS tagging
# NNP = proper noun, VBZ = verb, NNS = plural noun, IN = preposition.
# --------------------------------
tagged = pos_tag(word_tokenize(text))
print("\n=== 10) POS TAGGING ===")
print(tagged)

# --------------------------------
# 11) Named entity recognition
# ne_chunk groups tagged words into PERSON, ORGANIZATION and GPE (places).
# The English model makes mistakes, so always read the output.
# --------------------------------
print("\n=== 11) NAMED ENTITY RECOGNITION ===")
print(ne_chunk(tagged))
