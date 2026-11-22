# Lesson 2: Text Preprocessing

This assignment explores practical ways to turn unstructured text into input that NLP tools can handle. It combines a short research review with experiments on sample messages and word forms.

## Topics in This Assignment

- Normalizing case, spaces, and Unicode text
- Splitting text into words and sentences
- Comparing word, subword, and character tokenization
- Using regex to find, replace, or remove text patterns
- Evaluating stopword removal while preserving negation
- Comparing Porter, Lancaster, and Snowball stemming
- Testing WordNet lemmatization with part-of-speech hints
- Inspecting POS tags and named entities, including model errors
- Choosing preprocessing steps to suit sentiment analysis or spam detection

## Practical Files

`preprocessing_practice.py` runs the experiments with Python and NLTK. It prints the transformations and model predictions so they can be reviewed in `output.txt`. The practical report and research discussion are collected in `answers.md`.

To regenerate the captured results from this folder, run:


The script needs NLTK and its English tokenizer, stopword, WordNet, tagging, and named-entity resources. The course download helper was not present in the checkout used for this run; if the resources are missing, install NLTK and download the required data before running the script.

## Learning Objective

The experiments show that preprocessing is a set of modeling decisions, not a universal cleanup recipe. Removing punctuation, capitalization, numbers, or negation may simplify text but can also erase signals needed to recognize a complaint, identify entities, or flag a suspicious message.