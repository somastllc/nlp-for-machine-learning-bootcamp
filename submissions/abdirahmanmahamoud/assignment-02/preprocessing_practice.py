import re
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# lowercasing
text = "I LOVE Python"
print("Before:", text)
print("After: ", text.lower())

# Whitespace normalization
text = "I   love    NLP"
print("Before:", repr(text))
print("After: ", repr(" ".join(text.split())))

# Regex cleaning
text = "Win $500 NOW!!!"
lowered = text.lower()
# cleaned [^a-z\s]
cleaned = re.sub(r"[^a-z\s]", " ", lowered)
print("Before:", repr(text))
print("After lowercasing: ", repr(lowered))
print("After cleaning: ", repr(cleaned))

# Tokenization
text = "I love Python!"
print("Text:", text)
print("split:", text.split())
print("word tokenize: ", word_tokenize(text))

text = "I can't recommend this."
print("Text:", text)
print("word tokenize:", word_tokenize(text))

# Stopwords and negation
stop_words = set(stopwords.words("english"))
tokens = word_tokenize("this movie is not good")
without_stopwords = [token for token in tokens if token not in stop_words]

stop_words_keep_not = set(stopwords.words("english"))
stop_words_keep_not.discard("not")
keeping_not = [token for token in tokens if token not in stop_words_keep_not]

print("Tokens:", tokens)
print("Without stopwords:", without_stopwords)
print("Keeping 'not':", keeping_not)

# Full pipeline: lowercase -> regex -> tokenize -> stopwords -> lemmatize
sample = "Not every free offer is spam; meet me at 8 tonight!"
lemmatizer = WordNetLemmatizer()

step_lower = sample.lower()
step_clean = re.sub(r"[^a-z\s]", " ", step_lower)
step_tokens = word_tokenize(step_clean)
step_no_stop = [token for token in step_tokens if token not in stop_words]
step_lemmas = [lemmatizer.lemmatize(token) for token in step_no_stop]

print("Original:", sample)
print("Lowercased:", step_lower)
print("Regex cleaned:", step_clean)
print("Tokens:", step_tokens)
print("Stopwords removed:", step_no_stop)
print("Lemmatized:", step_lemmas)

step_keep_not = [token for token in step_tokens if token not in stop_words_keep_not]
print("Same, keeping 'not':", [lemmatizer.lemmatize(token) for token in step_keep_not])

# Comparing stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
regexp = RegexpStemmer("ing$|s$|e$|able$", min=4)
words = ["running", "studies", "fairly", "generously", "happiness", "eating"]

for word in words:
    print(word, "->",
          "Porter:", porter.stem(word),
          "| Lancaster:", lancaster.stem(word),
          "| Snowball:", snowball.stem(word),
          "| Regexp:", regexp.stem(word))

# Lemmatization with a part-of-speech hint
print("running (default):", lemmatizer.lemmatize("running"))
print("running (pos='v'):", lemmatizer.lemmatize("running", pos="v"))
print("better  (default):", lemmatizer.lemmatize("better"))
print("better  (pos='a'):", lemmatizer.lemmatize("better", pos="a"))
print("studies (default):", lemmatizer.lemmatize("studies"))
print("studies (pos='v'):", lemmatizer.lemmatize("studies", pos="v"))

# Sentence splitting
text = "Amina works at the United Nations in Nairobi. She sent the report on Monday."
print(sent_tokenize(text))

# POS tagging
tagged = pos_tag(word_tokenize(text))
print(tagged)

# named entity recognition
print(ne_chunk(tagged))