# Assignment 2 – Preprocessing

**Goal:** Understand how raw text is prepared for NLP systems, including normalization, tokenization, stopwords, stemming, lemmatization and regular expressions, and why preprocessing choices depend on the task.

## Part A – Research

### Instructions
- **Do not use** the explanations or examples we provided earlier (e.g., classroom notes).
- You must conduct **deep research** using books, scholarly articles, and reliable sources.
- **You may use AI tools**, but **do not rely blindly**: do not submit answers you don’t fully understand or verify.
- The purpose is for you to **learn through research** — therefore, explain ideas clearly, restate key concepts in your own words, and reference the sources you used.
- Write in **English**, in a research-oriented style (with headings, paragraphs, and tables/figures where appropriate), and include **references** at the end.

1. Define **text preprocessing**. Why do NLP systems need it, and why can more cleaning sometimes make results worse? Use a real-life example.

2. Explain **text normalization** (such as lowercasing, whitespace and Unicode normalization). Give one situation where a normalization step helps and one where it removes useful information.

3. Compare **word-level**, **subword** and **character-level** tokenization. Explain what **vocabulary size** means and how the choice of tokenizer affects it.

4. What are **stopwords**? Discuss the benefits and risks of removing them, and explain why the right stopword list depends on the task and the language.

5. Compare **stemming** and **lemmatization**. Research at least one well-known stemming algorithm and one lemmatization approach, and explain when each might be preferred.

6. Explain how **regular expressions** are used in text cleaning. Give two example patterns, describe what each matches, and explain one way an overly broad pattern can damage text.

7. Find one **case study** (research paper or article) that compares preprocessing choices for an NLP task. Summarize the task, the preprocessing options compared, and how those choices affected the results.

## Part B – Practical

Now apply what you researched. Write your own Python script, `preprocessing_practice.py`, that processes the texts and words below and **prints** the result of every step (like the class teaching script: one heading per step, then the before and after).

### Setup
- Install the course requirements and run `python code/download-nltk-data.py` once.
- Use **NLTK** and Python's `re` module.
- Save the full printed output to `output.txt` (for example: `python preprocessing_practice.py > output.txt`).
- **Do not type outputs by hand.** Every result in your report must come from your own script.

### Texts to process

```text
T1: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
T2: I don't think this jacket is worth the price. Not happy at all 😞
T3: Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m.
T4:    The   delivery   was   fast,   and the  packaging   was   great.
```

### Words to process

```text
connection, connected, connecting, organization, organized,
university, universal, flies, wolves, better, ran
```

### Tasks

1. **Lowercasing:** lowercase **T1**. Print before and after. What information could a spam detector lose?

2. **Whitespace:** normalize the spaces in **T4**. Print before and after (use `repr()` so the spaces are visible).

3. **Regex cleaning:**
   - Apply the pattern `[^a-z\s]` to the lowercased **T1** and print the result. List everything that was deleted.
   - Write **your own** regex rules for T1 that **replace instead of delete**: the phone number becomes `PHONE_NUMBER`, the money amount becomes `MONEY`, and `!!!` becomes a single `!`. Print the result.

4. **Tokenization:** tokenize **T2** with `split()` and with `word_tokenize`. Print both and their token counts. Then tokenize **T4** and print its token count and its vocabulary size (number of unique tokens).

5. **Stopwords and negation:** remove NLTK's English stopwords from the **T2** tokens and print the result. Then build your own stopword list that keeps negation words (`not`, `n't`, `no`, `never`) and print the result again. Explain how the meaning changed between the two.

6. **Stemming:** stem every word in the word list with **Porter**, **Lancaster** and **Snowball**. Print a comparison for each word. Find at least **one pair of different words** that a stemmer merges into the same stem, and one word where a stem is not a real word.

7. **Lemmatization:** lemmatize every word in the word list with `WordNetLemmatizer`, first with the default and then with the correct part-of-speech hint (`pos="v"`, `"n"` or `"a"`). Print both. Which words changed only when you gave the hint?

8. **Sentence splitting:** split **T3** with `sent_tokenize`. How many sentences did you get? Did `Mr.` and `a.m.` break the sentences in the wrong places?

9. **POS tagging and NER:** run `pos_tag` and `ne_chunk` on **T3**. Print the tags and the entities. Make a table of each entity with its label, and mark whether the label is correct. Then check what the model **missed**: was every person, place, organization and day recognized? (For example, `Apple` here is a company, not a fruit.)

10. **Full pipeline:** process **T2** through lowercase → regex → tokenize → stopwords → lemmatize. Print the text after **every** stage. Run it twice: once with NLTK's default stopwords and once with your negation-keeping list from task 5. Which version keeps the customer's complaint?

### Report

Add a **Practical** section to `answers.md` with:
- the table from task 6 and the entity table from task 9,
- a short answer (2–4 sentences) for each question in tasks 1, 3, 5, 6, 7, 8 and 10,
- one paragraph: **which preprocessing steps would you use for a spam detector, and which would you avoid?** Use your own outputs as evidence.

## Submission

Submit everything in `submissions/<github-username>/assignment-02/` when submissions open:

- `answers.md` – Part A research answers and the Part B practical report
- `preprocessing_practice.py` – your script
- `output.txt` – the full printed output of your script
