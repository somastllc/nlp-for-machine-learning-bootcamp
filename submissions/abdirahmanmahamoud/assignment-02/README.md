# Assignment 2 – Text Preprocessing

**Subject:** Natural Language Processing (NLP)  
**Assignment:** 02 – Text Preprocessing  
**Submitted by:** Abdirahman Mohamoud

---

## Overview

This assignment explores the theory and practice of text preprocessing for NLP. It covers normalization, tokenization, stopwords, stemming, lemmatization, regular expressions, and a case study on the effectiveness of preprocessing in sentiment analysis.

## Files

- `answers.md` – Written answers to the research questions and case study.
- `preprocessing_practice.py` – Python script demonstrating preprocessing techniques with NLTK.
- `output.txt` – Captured output from running the script.

## Running the Script

The script requires Python 3 and the NLTK library. The following NLTK resources must be downloaded before running it:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
```

Then run:

```bash
python preprocessing_practice.py
```
