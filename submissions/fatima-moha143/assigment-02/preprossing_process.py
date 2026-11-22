

import re
import nltk

nltk.download('stopwords')
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# Assignment data
T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great."

words = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]

# 1) Lowercasing
print("\\n=== 1) LOWERCASE ===")
print("Before:", T1)
lower_T1 = T1.lower()
print("After: ", lower_T1)
print("Spam detector warning: uppercase emphasis such as URGENT and WON may be useful evidence.")

# 2) Whitespace normalization
print("\\n=== 2) WHITESPACE NORMALIZATION ===")
print("Before:", repr(T4))
normalized_T4 = " ".join(T4.split())
print("After: ", repr(normalized_T4))

# 3) Regex cleaning
print("\\n=== 3) REGEX CLEANING ===")
regex_T1 = re.sub(r"[^a-z\\s]", "", lower_T1)
deleted_characters = re.findall(r"[^a-z\\s]", lower_T1)
print("Pattern:", r"[^a-z\\s]")
print("Before:", repr(lower_T1))
print("After: ", repr(regex_T1))
print("Deleted characters (in order):", deleted_characters)
print("Unique deleted characters:", sorted(set(deleted_characters)))

replace_T1 = T1.lower()
replace_T1 = re.sub(r"0712345678", "PHONE_NUMBER", replace_T1)
replace_T1 = re.sub(r"£\\s?1,000", "MONEY", replace_T1)
replace_T1 = re.sub(r"!{3,}", "!", replace_T1)
print("\\nReplacement rules: phone -> PHONE_NUMBER; £1,000 -> MONEY; !!! -> !")
print("Before:", T1)
print("After: ", replace_T1)

# 4) Tokenization
print("\\n=== 4) TOKENIZATION ===")
print("T2:", T2)
T2_split_tokens = T2.split()
T2_word_tokens = word_tokenize(T2)
print("T2 split():", T2_split_tokens)
print("T2 split() token count:", len(T2_split_tokens))
print("T2 word_tokenize():", T2_word_tokens)
print("T2 word_tokenize() token count:", len(T2_word_tokens))

T4_tokens = word_tokenize(T4)
print("\\nT4:", repr(T4))
print("T4 word_tokenize():", T4_tokens)
print("T4 token count:", len(T4_tokens))
print("T4 vocabulary size:", len(set(T4_tokens)))
print("T4 unique tokens:", sorted(set(T4_tokens)))

# 5) Stopwords and negation
print("\\n=== 5) STOPWORDS AND NEGATION ===")
english_stopwords = set(stopwords.words("english"))
T2_tokens = word_tokenize(T2)
T2_without_default_stopwords = [
    token for token in T2_tokens if token.lower() not in english_stopwords
]
negation_words = {"not", "n't", "no", "never"}
stopwords_keep_negation = english_stopwords - negation_words
T2_keep_negation = [
    token for token in T2_tokens if token.lower() not in stopwords_keep_negation
]
print("Original T2 tokens:             ", T2_tokens)
print("NLTK default stopwords removed: ", T2_without_default_stopwords)
print("Keeping negation words:         ", T2_keep_negation)
print("Removing 'not' can reverse the apparent sentiment; keeping negation preserves the complaint.")

# 6) Stemming comparison
print("\\n=== 6) STEMMING COMPARISON ===")
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")