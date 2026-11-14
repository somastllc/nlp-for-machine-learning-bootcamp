# Stone Age — Welcome to the NLP for Machine Learning Bootcamp

**Led by Sharafdin Yusuf & Omar Tood · Sponsored by Dugsiiye**

This bootcamp follows the [DS/ML bootcamp](https://github.com/somastllc/Jun-ds-ml-bootcamp-2026). You already know the basic ML journey: load data, prepare features, train and evaluate. Here the raw input is **language**. We will work through that journey again, one text problem at a time.

## Where every NLP problem starts

Imagine a computer receives this message:

> “WIN a free prize now!”

To us, it may look like spam. To a classifier, it starts as a string of characters. It cannot use those characters as ordinary numeric features. Before we can ask a model to predict *spam* or *ham*, we must decide what counts as a **document**, split text into **tokens**, build a **vocabulary**, and represent a message as numbers. That is our starting point.

Now imagine the same kind of message in Somali:

> “Ku guulayso abaalmarin bilaash ah!”

A Somali reader recognizes the same promotional pattern. A computer faces the same problem as before, plus new ones: an English stopword list knows none of these words, and Somali attaches meaning to word endings in ways English tools do not expect. Most NLP tools and datasets were built for English first. Working with Somali text means checking every step instead of trusting the defaults.

That gap is why we are building **[SomLTK](https://github.com/somastllc/SomLTK)**, the Somali Language Toolkit: an NLTK-style toolkit designed for Somali text, so that Somali stopwords, word endings and spelling are handled by tools made for the language rather than borrowed from English. SomLTK is still in development. In this course we first learn how the English tools work and where they break on Somali, which is exactly the knowledge needed to use, test and contribute to a Somali toolkit.
