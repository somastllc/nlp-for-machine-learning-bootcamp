# Assignment 2: Preprocessing

## Part A: Research

### 1. What is text preprocessing?

Text preprocessing means changing raw text with a set of rules before a model uses it. Some steps are small, such as making letters lowercase or fixing extra spaces. Others are bigger, such as cutting text into tokens, dropping some words, or reducing words to a base form [1][2].

NLP systems need it because raw text is untidy. The same word can show up as "Great", "GREAT" and "great", with odd spacing and symbols around it. A model that counts words sees each of these as a different item, so it gets fewer examples of every idea. The problem is large. Jurafsky and Martin report that Shakespeare's plays have about 884 thousand running words but 31 thousand different word types, and that a model will "constantly see unknown words" it never met before [1]. Cleaning reduces this extra variety, so the model learns from more examples of each idea.

More cleaning can make results worse because every step throws something away. If the thrown away part was a useful signal, the model never sees it. Two pieces of research show this clearly:

* Manning, Raghavan and Schütze describe how stop lists in search engines shrank over time, from about 200 to 300 words, to 7 to 12 words, and then to none. Modern web search engines mostly do not drop common words, because storing them became cheap and because dropping them breaks queries such as "President of the United States" and "flights to London" [2].
* Camacho-Collados and Pilehvar tested lowercasing, lemmatizing and multiword grouping against plain tokenization on nine datasets. They conclude that simple tokenization works equally well or better than the more complex steps, except for domain specific data [3]. I explain this study in question 7.

**A real life example.** Think of a shop that sorts customer reviews into happy and unhappy. A customer writes: "Not happy with this phone. NEVER buying again!!!" Suppose the cleaning code lowercases everything, removes all punctuation, and then deletes common words, including "not". What reaches the sorter is close to "happy phone buying again". The shouting capitals and the exclamation marks are gone, and so is the word that reverses the meaning. An angry review ends up in the happy pile. The cleaning looked tidy, but it removed exactly the parts that carried the opinion. This example is my own illustration. Part B of this report shows the same effect on real output.

### 2. Text normalization

Text normalization means making different written forms of the same thing look the same, so the model treats them as one. Common steps are lowercasing, cleaning whitespace, and Unicode normalization.

**Lowercasing.** Jurafsky and Martin say that whether "They" and "they" count as the same word "depends on the task" [1]. The Camacho-Collados and Pilehvar study gives numbers for both sides [3]:

* Where it helps: on Ohsumed, a set of medical documents, a CNN+LSTM model scored 30.8 percent on plain tokenized text and 37.5 percent with lowercasing. The authors explain that lowercasing and lemmatizing mainly reduce sparsity, meaning the same word is no longer split into several rare forms.
* Where it removes useful information: on the IMDB movie reviews, a CNN model dropped from 87.5 percent to 84.2 percent after lowercasing, and the authors mark this drop as statistically significant. A general weakness of lowercasing, which may or may not explain this drop, is that it cannot tell Apple the company from apple the fruit.

**Whitespace.** A pattern that turns runs of spaces into one space is almost always safe for classification. It is not safe when layout carries meaning, for example in poems, tables, or code.

**Unicode normalization.** One character can be stored in more than one way. The Unicode standard calls such forms canonically equivalent, and says they should look and behave the same when displayed correctly [7]. Take the letter Å. It can be one code point (U+00C5), or the letter A (U+0041) followed by a separate ring mark (U+030A). The angstrom sign (U+212B) also maps to U+00C5 [7]. I checked this in Python. As raw strings they are not equal, so a search for one does not find the other. After Normalization Form C (NFC) they are equal. This is why the W3C recommends NFC for all content [7].

Here is a case where normalization removes useful information. The stronger forms NFKC and NFKD replace compatibility characters. The standard gives the ffi ligature (U+FB03), which becomes the three letters "ffi", and a superscript five, which becomes a plain five. The standard warns that these forms "must not be blindly applied to arbitrary text", because they can erase distinctions, for example in mathematical writing [7].

### 3. Word level, subword and character level tokenization

A tokenizer decides what one unit of text is. The three kinds differ in the size of that unit.

**Word level.** Each word, and usually each punctuation mark, is one token. It is simple and easy to read. The Penn Treebank rules even split "doesn't" into "does" and "n't" [1]. Its weakness is unknown words. In the words of Jurafsky and Martin, "no matter how big our vocabulary, we will never have a vocabulary that captures all the possible words" [1].

**Subword.** Words are split into smaller pieces that the tokenizer learned from data. Byte pair encoding (BPE) starts from single characters and repeatedly merges the pair that appears most often next to each other [1]. Sennrich, Haddow and Birch adapted this idea for machine translation, so that rare words, names and compounds can be built from known parts and the model has an open vocabulary without a dictionary fallback [4]. Common words stay whole, and rare words are cut into pieces.

**Character level.** Each character is one token. Jurafsky and Martin note that real systems often run BPE on the bytes of UTF 8 text. There are only 256 possible byte values, so no token is ever unknown [1]. A character tokenizer has no unknown words either, but it needs very long sequences to say anything.

**Vocabulary size** is the number of different entries a tokenizer knows. Jurafsky and Martin write it as |V|, the number of word types, and they separate it from the number of running tokens, which they call N [1]. A text with 884 thousand tokens can have a vocabulary of only 31 thousand.

The tokenizer controls the vocabulary size in different ways:

| | Word level | Subword | Character or byte level |
|:---|:---|:---|:---|
| Unit | A whole word | A piece of a word | One character or byte |
| Who sets the vocabulary size | The data. It grows with text length, a little faster than the square root [1] | The designer, through the number of merges. Real systems use 50,000, 100,000 or even 200,000 tokens [1] | Fixed and small. A byte tokenizer has 256 entries [1] |
| Unknown words | Many | Rare. Unknown words are built from known pieces | None for bytes |
| Tokens per sentence | Fewest | A few more than words | Most |
| Main trade off | Huge vocabulary, many unseen words | A good middle, but needs training on a large corpus | Tiny vocabulary, but very long sequences |

Vocabulary size also shows how much the tokenizer fits a language. Jurafsky and Martin show an English sentence of 14 words that becomes 18 tokens, while a Spanish sentence of 16 words becomes 33 tokens with the same tokenizer [1]. My reading of this is that a tokenizer trained mostly on one language cuts other languages into more pieces.

### 4. Stopwords

Stopwords are very common words that seem to add little for finding or sorting documents, for example "the", "of" and "to". A stop list is often built by counting how often each word appears in a collection, taking the most frequent ones, and then checking the list by hand [2].

**Benefits.** The index or the feature set gets smaller, and a model can focus on the rarer words. Manning and his co authors say a stop list "significantly" reduces the number of postings a system must store [2].

**Risks.** Removing common words can break meaning:

* In search, "President of the United States" is more exact than "President" and "United States" apart. Searching for flights to London loses its meaning if "to" is dropped. Some song titles and famous lines are made only of words that appear on stop lists [2].
* In opinion tasks, a word such as "not" often sits on a stop list, but it reverses the meaning of the next word. I show this in Part B.
* Modern search engines mostly stopped using stop lists. Compression made storing common words cheap, and weighting schemes such as idf already give them little influence [2].

**Why the right list depends on the task and the language.** First, a list is only a choice about which differences to ignore, and each task needs different information. A topic sorter may do well without "the", while an opinion classifier needs "not". Second, lists are not neutral. Nothman, Qin and Yurchak compared 52 English stop lists and found that they differ from each other and are not always consistent inside. One list has "hasn't" but not "hadn't", and some lists include words like "computer" that users do not expect. They also note that some lists do not match the tokenizer that is used with them [5]. Third, a list is written for one language, because the most frequent words are different in each language. For a language with few resources, a list may not exist at all, and the counting method above is a way to build one.

### 5. Stemming and lemmatization

Both try to bring different forms of a word to one base form, for example "am, are, is" to "be" [2]. A stemmer is "a crude heuristic process that chops off the ends of words". A lemmatizer uses a vocabulary and a study of word forms to return the dictionary form, called the lemma [2].

**A stemming algorithm: Porter.** Martin Porter published his suffix stripping algorithm in 1980 [6]. It works in about five phases of rules. A rule says that if a word ends in a certain suffix and the rest of the word passes a test, the suffix is replaced or removed. The test uses a measure of the stem that roughly counts syllables, and inside each group only the rule with the longest matching suffix is used [2][6]. One example from Manning and his co authors: the rule for "ement" applies only when the stem is long enough, so "replacement" becomes "replac" but "cement" stays whole. The same book shows that Porter turns "operate", "operating", "operates", "operation", "operative", "operatives" and "operational" all into "oper". That groups related words, and it also merges some words that a user would want to keep apart [2].

**A lemmatization approach: the WordNet lemmatizer.** NLTK's lemmatizer looks the word up in WordNet with its built in morphy function. It needs a part of speech, and the default is noun. It collects the candidate lemmas, returns the shortest one, and returns the word unchanged if WordNet does not know it [8]. This is why the part of speech matters. I test it in Part B.

**When to prefer each:**

* Stemming is quick, needs little language knowledge, and raises recall, but it can harm precision [2]. It fits search tasks, and it fits languages with rich word forms. Manning and his co authors report quite large gains from stemmers for Spanish, German and Finnish, and only small gains for English [2].
* Lemmatization gives real words, which is better when people read the output, for example topic labels. It also avoids some wrong merges. The cost is that it needs a lexicon and a part of speech tagger.
* Neither is a sure win. In the study I summarize in question 7, lemmatizing raised one score (PL04 with CNN, 58.7 to 64.2 percent) and lowered another (IMDB with CNN, 87.5 to 86.8 percent) [3]. The safest way is to test on validation data.

### 6. Regular expressions in text cleaning

A regular expression is a short pattern that describes a set of strings. A cleaning script uses it to find a pattern and then replace it. Jurafsky and Martin use regular expressions as a basic tool for text processing [1]. I ran the examples below in Python's `re` module.

**Pattern 1: `<[^>]+>`**
It matches an opening angle bracket, then one or more characters that are not a closing bracket, then a closing bracket. In other words, it matches an HTML tag such as `<b>` or `</i>`. On `<b>great</b> phone, <i>fast</i> delivery`, deleting the matches gives `great phone, fast delivery`.

**Pattern 2: `\s{2,}`**
It matches two or more whitespace characters in a row. Replacing each match with one space turns `too   many    spaces` into `too many spaces`.

**How an overly broad pattern damages text.** The pattern `<.*>` looks similar to pattern 1, but `.*` matches any characters and it is greedy. Jurafsky and Martin explain that patterns "are greedy, expanding to cover as much of a string as they can" [1]. So it matches from the first `<` to the last `>` in the text. On the same string, my test returned only `' delivery'`. It deleted the words "great phone, fast" together with the tags. The lazy version `<.*?>` gives the correct result again. A second broad pattern is `[^\w\s]`, which deletes all punctuation. On `don't call us, it costs 5.50!` it returned `dont call us it costs 550`. The apostrophe is gone, and the price changed from 5.50 to 550. Jurafsky and Martin call these two mistakes false positives, which are strings matched by accident, and false negatives, which are strings missed [1].

### 7. Case study: preprocessing choices for text classification

**Paper.** Camacho-Collados, J. and Pilehvar, M. T. (2018). *On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis.* Proceedings of the 2018 EMNLP Workshop BlackboxNLP, pages 40 to 46 [3].

**Task.** Two kinds of text classification: topic categorization and sentiment (polarity) detection. There are nine datasets. The topic sets are BBC, 20News, Reuters and Ohsumed (medical). The sentiment sets are RTC, IMDB, PL05, PL04 and Stanford phrases (SF). The scores are accuracy in percent. This is the first task type from my Assignment 1, text classification.

**Models.** A CNN similar to Kim's model, and a CNN with an LSTM layer added. Both start from 300 dimension word2vec embeddings trained on the 3 billion word UMBC corpus [3].

**Preprocessing options compared.**

1. Vanilla: tokenization only, with Stanford CoreNLP.
2. Lowercasing every token.
3. Lemmatizing every token, with CoreNLP.
4. Multiword grouping: joining words such as "United States" into one token, using the phrases from the Google News word2vec vectors.

In experiment 1, the same option is applied to the datasets and to the text used to train the embeddings. In experiment 2, the datasets stay vanilla and only the embedding training text changes [3].

**Results.** A few rows of the authors' experiment 1 table (accuracy in percent):

| Model and dataset | Vanilla | Lowercased | Lemmatized | Multiword |
|:---|:---:|:---:|:---:|:---:|
| CNN, IMDB | 87.5 | 84.2 | 86.8 | 87.9 |
| CNN, PL04 | 58.7 | 59.6 | 64.2 | 59.1 |
| CNN+LSTM, Ohsumed | 30.8 | 37.5 | 37.1 | 29.0 |
| CNN+LSTM, PL04 | 71.4 | 73.3 | 72.6 | 67.0 |

What the authors found [3]:

* Results change with the preprocessing choice. For the CNN+LSTM model the variation was about 2.4 percent, with a significant gap on seven of the nine datasets.
* Plain tokenization was "in the same ballpark as the best result" on 8 of 9 datasets for both models. Ohsumed was the exception, and there lowercasing and lemmatizing beat vanilla by over six points with CNN+LSTM.
* Lemmatization and multiword grouping did not help in general. Multiword grouping hurt Ohsumed (29.0) and PL04 (67.0) with CNN+LSTM.
* In experiment 2, embeddings built from multiword text, used on vanilla datasets, gave the best CNN result on seven of the nine datasets.

**Conclusion of the authors.** Simple tokenization works equally well or better than more complex steps, except for domain specific data, and the choice can change results a lot, so researchers should choose carefully and report it [3].

**My comment.** The study is limited. It uses English only, two model types, and four options, and it does not test stopword removal. The authors say more work is needed. The lesson for me is the same as in Part B. A step that helps one dataset can hurt another, so each step should be tested on the task.

## Part B: Practical

All results below come from my script `preprocessing_practice.py`. I ran it in Google Colab and saved the print out in `output.txt`.

### Tasks 1 and 2: lowercasing and whitespace

T1 became `urgent!!! you have won a £1,000 gift card. call 0712345678 now!`. T4 changed from a line with many extra spaces to `'The delivery was fast, and the packaging was great.'`.

**Question for task 1: what could a spam detector lose?** The script found three words written in capitals in T1: URGENT, WON and NOW. After lowercasing they look like normal words. Shouting in capitals is a common style in spam, so the detector loses that signal. Counting the capital words before lowercasing would keep it.

### Task 3: regex cleaning

The pattern `[^a-z\s]` on the lowercased T1 gave `'urgent    you have won a        gift card  call            now '`. It deleted 21 characters, and the output shows 13 different ones:

* four exclamation marks (three after "urgent" and one at the end)
* the pound sign, the comma, and the digits 1, 0, 0, 0 of the amount £1,000
* the full stop after "card"
* all ten digits of the phone number 0712345678

My own rules replace instead of delete. They are `[£$€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?` for money, `\b\d{10}\b` for the phone number, and `!{2,}` for repeated exclamation marks. The result was `URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!`.

**Short answer.** The deleting pattern left only plain letters, so the amount and the phone number vanished together with the punctuation. Those are two strong spam clues. My replace rules kept the fact that a money amount and a phone number were present, without storing the exact values, and they shortened the repeated marks to a single one.

### Task 4: tokenization

On T2, `split()` gave 14 tokens and `word_tokenize` gave 16. The two extra tokens come from `word_tokenize` splitting "don't" into "do" and "n't", and "price." into "price" and ".". With `split()` the full stop stays stuck to the word. On T4, `word_tokenize` gave 11 tokens and a vocabulary of 10 different tokens. After lowercasing the vocabulary fell to 9, because "The" and "the" became one.

### Task 5: stopwords and negation

| List | Tokens left from T2 |
|:---|:---|
| NLTK default | n't, think, jacket, worth, price, ".", happy, and the emoji |
| My list that keeps negation | n't, think, jacket, worth, price, ".", Not, happy, and the emoji |

**Short answer.** The only word removed by the default list and kept by mine was "Not". Without it, the review looks like a customer who is happy with the jacket. With it, "Not happy" survives and the complaint is still visible. One more detail: the tokenizer splits "don't" into "do" and "n't". The default list removed "do" but not "n't", so "n't" is left alone.

### Task 6: stemming

| Word | Porter | Lancaster | Snowball |
|:---|:---|:---|:---|
| connection | connect | connect | connect |
| connected | connect | connect | connect |
| connecting | connect | connect | connect |
| organization | organ | org | organ |
| organized | organ | org | organ |
| university | univers | univers | univers |
| universal | univers | univers | univers |
| flies | fli | fli | fli |
| wolves | wolv | wolv | wolv |
| better | better | bet | better |
| ran | ran | ran | ran |

**Short answer.** All three stemmers merge two different words into one stem: university and universal both become "univers", although they mean different things. Porter and Snowball also turn organization and organized into "organ", a real word but with another meaning, while Lancaster gives "org". Several stems are not real words. The script's WordNet check listed "fli", "wolv" and "univers" for all three stemmers, and "org" for Lancaster. Two more points: Lancaster turns "better" into "bet", which changes the meaning, and no stemmer links "ran" to "run".

### Task 7: lemmatization

| Word | Default (noun) | With the right hint |
|:---|:---|:---|
| connection | connection | connection (n) |
| connected | connected | connect (v) |
| connecting | connecting | connect (v) |
| organization | organization | organization (n) |
| organized | organized | organize (v) |
| university | university | university (n) |
| universal | universal | universal (a) |
| flies | fly | fly (v) |
| wolves | wolf | wolf (n) |
| better | better | good (a) |
| ran | ran | run (v) |

**Short answer.** Five words changed only when I gave the hint: connected, connecting, organized, better and ran. The lemmatizer treats every word as a noun unless told otherwise, so verb and adjective forms stay as they are. The words that were already nouns did not change, and "flies" gave "fly" in both cases. Unlike stemming, lemmatization kept university and universal apart.

### Task 8: sentence splitting

`sent_tokenize` gave 2 sentences: `Mr. Smith arrived in London on Friday.` and `He met the Apple team at 9 a.m.`

**Short answer.** "Mr." did not break the first sentence, so that case worked. The "a.m." sits at the very end of the text, and the second sentence ended there correctly, but this is a weak test because nothing follows it. Task 9 showed another small problem: `word_tokenize` split "a.m." into "a.m" and a separate full stop.

### Task 9: POS tagging and named entities

The tagger labelled Mr., Smith, London, Friday and Apple as proper nouns (NNP), "team" as NN, "9" as CD and "a.m" as NN. The entity table from `ne_chunk`:

| Text | Label from the model | Correct? |
|:---|:---|:---|
| Mr. | PERSON | The label is right, but it should be one entity with Smith |
| Smith | PERSON | Yes |
| London | GPE (place) | Yes |

**What the model missed.** The script listed two proper nouns that were not grouped into any entity: Friday and Apple. The person and the place were found. The organization was not: Apple here is a company, but it was left as a plain proper noun. The day, Friday, was not recognized either, and the time "9 a.m." was not marked as an entity.

### Task 10: full pipeline on T2

My regex keeps letters, spaces and apostrophes (`[^a-z\s']`), so "don't" is not broken.

| Stage | Run A: default stopwords | Run B: my negation list |
|:---|:---|:---|
| Lowercased | i don't think this jacket is worth the price. not happy at all (and the emoji) | same |
| After regex | `"i don't think this jacket is worth the price  not happy at all  "` | same |
| Tokens | i, do, n't, think, this, jacket, is, worth, the, price, not, happy, at, all | same |
| Stopwords removed | n't, think, jacket, worth, price, happy | n't, think, jacket, worth, price, not, happy |
| Lemmatized | n't, think, jacket, worth, price, happy | n't, think, jacket, worth, price, not, happy |

**Short answer.** The negation keeping version keeps the complaint, because "not happy" survives. In the default version "not" is gone, so only "happy" is left. Both versions lost the emoji, since the regex deleted it, and neither version has "do". I also ran a side check with the strict pattern `[^a-z\s]`. It deleted the apostrophe, so "don't" became "don" and "t", and my list removed both. The result was think, jacket, worth, price, not, happy, so the "n't" was lost as well.

### Which preprocessing steps for a spam detector?

For a spam detector I would use whitespace cleaning, because task 2 showed it only tidies the text. I would use `word_tokenize`, because it separates punctuation from words (task 4). I would replace money amounts, phone numbers and repeated exclamation marks with markers, as in task 3, and I would count the capital words before lowercasing, because lowercasing lost URGENT, WON and NOW (task 1). If I remove stopwords at all, I would use my shorter list that keeps negation words, because the default list deleted "Not" (task 5). I would avoid the strict pattern `[^a-z\s]`. It deleted 21 characters from T1, including the whole amount and the whole phone number, which are the best clues for spam (task 3). I would avoid aggressive stemming. Lancaster turned "better" into "bet", and all three stemmers merged university with universal (task 6). Lemmatization is safer, but it needs the right part of speech, since five of the eleven words changed only with a hint (task 7). I would skip sentence splitting, tagging and entity recognition for short messages, because they add little, and the entity model missed both Apple and Friday (tasks 8 and 9). As the study in question 7 shows, I would test each step on validation data before keeping it.

## References

[1] Jurafsky, D. and Martin, J. H. (2025). *Speech and Language Processing*, 3rd edition draft, chapter 2 (August 2025 draft). https://web.stanford.edu/~jurafsky/slp3/

[2] Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Online edition, sections "Dropping common terms: stop words" and "Stemming and lemmatization", from the Stanford NLP group website.

[3] Camacho-Collados, J. and Pilehvar, M. T. (2018). On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis. Proceedings of the 2018 EMNLP Workshop BlackboxNLP, pages 40 to 46. arXiv:1707.01780.

[4] Sennrich, R., Haddow, B., and Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL 2016. arXiv:1508.07909.

[5] Nothman, J., Qin, H., and Yurchak, R. (2018). Stop Word Lists in Free Open source Software Packages. Proceedings of the Workshop for NLP Open Source Software, pages 7 to 12. Association for Computational Linguistics. https://aclanthology.org/W18-2502/

[6] Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130 to 137.

[7] Unicode Consortium. Unicode Standard Annex 15: Unicode Normalization Forms. https://unicode.org/reports/tr15/

[8] NLTK Project. Documentation of `nltk.stem.wordnet.WordNetLemmatizer`, NLTK 3.9.2. https://www.nltk.org/_modules/nltk/stem/wordnet.html

Examples marked as my own test were run by me with Python's `re` and `unicodedata` modules. All Part B results come from `preprocessing_practice.py` and `output.txt`.
