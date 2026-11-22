import re
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
lemmatizer = WordNetLemmatizer()
#T1: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
#T2: I don't think this jacket is worth the price. Not happy at all 😞
#T3: Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m.
#T4:    The   delivery   was   fast,   and the  packaging   was   great.

#lower

text = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
print('before:' , text)
print('after:', text.lower())

#whitespace
text =  "The   delivery   was   fast,   and the  packaging   was   great."
print('before:', repr(text))
print('after:' , repr(" " .join(text.split())))


#regex
text = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
lowered = text.lower()
cleaner = re.sub(r"[^a-z\s]", " ", lowered)
print('before: ', repr(text))
print('after:', repr(lowered))
print('cleaning', repr(cleaner))

#tokenization
text = "I don't think this jacket is worth the price. Not happy at all 😞"
print('boefore: ', text)
print('text_split: ', text.split())
print('word_tokonize():', word_tokenize(text))

text = "The   delivery   was   fast,   and the  packaging   was   great."
print('text', text)
print('word_tokonize()', word_tokenize(text))

#stopword , nagetion
stop = set(stopwords.words('english'))
tokens= word_tokenize("I don't think this jacket is worth the price. Not happy at all 😞")

without_stop = [token for token in tokens if token.lower() not in stop]
stop_keep = set(stopwords.words("english"))
stop_keep.difference_update(["not", "n't", "no", "never"])
keep = [token for token in tokens if token.lower() not in stop_keep]
print("tokens:", tokens)
print("default:", without_stop)
print("keeping:", keep)


#connection, connected, connecting, organization, organized,
#university, universal, flies, wolves, better, ran

#Stemming
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
words = ["connection", "connected", "connecting", "organization", "organized", "university", "universal", "flies", "wolves", "better", "ran"]

print("comparing")
for word in words:
    print(word, "->",
          "porter:", porter.stem(word),
          "| snowball:", snowball.stem(word),
          "| lancater:", lancaster.stem(word))


#Lemmatization
print("connection (defualt):", lemmatizer.lemmatize("connection"))
print("connection (post hint):", lemmatizer.lemmatize("connection", pos="n"))
print("connected (defualt):", lemmatizer.lemmatize("connected"))
print("connected (post hint):", lemmatizer.lemmatize("connected", pos="v"))
print("connecting (defualt):", lemmatizer.lemmatize("connecting"))
print("connecting (post hint):", lemmatizer.lemmatize("connecting", pos="v"))
print("organization (defualt):", lemmatizer.lemmatize("organization"))
print("organization (post hint):", lemmatizer.lemmatize("organization", pos="n"))
print("organized (defualt):", lemmatizer.lemmatize("organized"))
print("organized (post hint):", lemmatizer.lemmatize("organized", pos="v"))
print("university (defualt):", lemmatizer.lemmatize("university"))
print("university (post hint):", lemmatizer.lemmatize("university", pos="n"))
print("universal (defualt):", lemmatizer.lemmatize("universal"))
print("universal (post hint):", lemmatizer.lemmatize("universal", pos="a"))
print("flies (defualt):", lemmatizer.lemmatize("flies"))
print("flies (post hint):", lemmatizer.lemmatize("flies", pos="n"))
print("wolves (defualt):", lemmatizer.lemmatize("wolves"))
print("wolves (post hint):", lemmatizer.lemmatize("wolves", pos="n"))
print("better (defualt):", lemmatizer.lemmatize("better"))
print("better (post hint):", lemmatizer.lemmatize("better", pos="a"))
print("ran (defualt):", lemmatizer.lemmatize("ran"))
print("ran (post hint):", lemmatizer.lemmatize("ran", pos="v"))

#Sentence splitting
text = 'Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m.'
print(sent_tokenize(text))

#pos tagging
taged = pos_tag(word_tokenize(text))
print(taged)

print(ne_chunk(taged))

#full pupline
full = "I don't think this jacket is worth the price. Not happy at all 😞"
step_lower = full.lower()
strp_clean = re.sub(r"[^a-z\s]", " ", step_lower)
step_tokens = word_tokenize(strp_clean)
step_no = [token for token in step_tokens if token not in stop]
step_lemma = [lemmatizer.lemmatize(token) for token in step_no]

print("original", full)
print("lower", step_lower)
print("regex", strp_clean)
print("token", step_tokens)
print("stop word remove", step_no)
print("lemma", step_lemma)

step_keep_no = [token for token in step_tokens if token not in stop_keep]
print("same", [lemmatizer.lemmatize(token) for token in step_keep_no])