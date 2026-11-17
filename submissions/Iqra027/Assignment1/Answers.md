1.*NLP* Natural Language Processing (NLP) is simply the branch of technology that teaches computers how to read, understand, and respond to human language—whether that's typed text or spoken words.

*Artificial Intelligence (AI)* is the biggest circle. It’s the broad goal of getting machines to do things that normally require human smarts—like recognizing faces, driving a car, or holding a conversation.

*Machine Learning (ML)* sits inside AI. Instead of a programmer sitting down and writing millions of rigid "if-this-then-that" rules, ML lets the computer study thousands of examples and figure out the patterns on its own.

*Natural Language Processing (NLP)* is the specialized field focused specifically on language. Because human language is messy, full of slang, and relies heavily on context, modern NLP leans on Machine Learning to actually learn how people talk and write—all in service of that bigger AI goal. (Though older or simpler NLP tools still use basic grammar rules and dictionaries alongside ML.)

*An example will be siri*
**NLP** handles the language part. It turns your spoken voice into text, breaks the sentence apart, and figures out what you actually mean—identifying the action ("call my mom"), the date ("tomorrow"), and the time ("5 PM").

**Machine Learning** is what makes it accurate over time. The system has been trained on millions of voice recordings from different people, accents, and backgrounds, so it learns how to recognize your specific voice and phrasing without you needing to speak like a robot.

**AI** is the overall smart experience: your phone understanding a casual request and automatically setting the reminder on your calendar just like a helpful human assistant would.

2.

|  Review ID |  Customer Review (Input) |  Sentiment (Target) |
| --- | --- | --- |
| 1 | "The coffee is very fresh." | Positive |
| 2 | "The service is very slow." | Negative |
| 3 | "The coffee is not fresh."  | Negative |

**Corpus:** The entire collection of text you are working with. Here, our corpus is all three customer reviews grouped together.

**Document:** One single piece of text inside that collection. Each individual review in the table (Review 1, Review 2, or Review 3) counts as one document.

**Token:** The individual pieces you get when you chop a document up—usually single words. Breaking down Review 1 gives us five word tokens: The, coffee, is, very, and fresh.

**Vocabulary:** The master list of every unique token across the entire corpus (usually after converting everything to lowercase and removing punctuation). Even though words like the, coffee, is, very, and fresh show up more than once across the reviews, we only count each unique word once. That gives us a vocabulary of 8 words: the, coffee, is, very, fresh, service, slow, and not.

**Feature:** Any measurable clue pulled from the text that helps a computer model make a decision. For example, a model might count how many times the word slow appears in a review, or check whether the word not is present. Each of those word counts or checks acts as a feature.

**Label:** The final answer or category attached to each document that we want the model to learn to predict. In our table, the labels are Positive and Negative.

3.
1.**Text Classification**
This task sorts an entire chunk of text into predefined categories. Instead of labeling single words, the system reads the whole message and decides what overall bucket it belongs to.

Input: A complete piece of text, such as an email, a product review, or a news article.

Output: One or more predefined category labels for the whole text—for example, tagging an email as Spam vs. Not Spam, or sorting news into Sports, Politics, or Health.

2.**Sequence Labeling**
While text classification assigns one label to an entire text, sequence labeling zooms in to label every individual word (token) in order. Common examples include identifying parts of speech (nouns, verbs, adjectives) or spotting names of people, companies, and places (Named Entity Recognition).

Input: An ordered sequence of tokens, usually a sentence.

Output: A matching sequence of tags—one label per token. For example, given ["Amina", "lives", "in", "Nairobi"], the system outputs ["Person", "Other", "Other", "Location"].

3.**Search and Similarity**
This task measures how closely two pieces of text match in meaning or shared words so a system can find the most useful information. It powers search engines, FAQ finders, and duplicate-question detectors.

Input: A search query (or a reference document) along with a collection of documents to search through.

Output: Either a numerical similarity score (showing how closely two texts match) or a ranked list of the most relevant documents ordered from best match to worst.

4.**Text Generation**
Text generation produces brand-new, natural-sounding text rather than sorting or tagging existing words. It powers machine translation, automatic summarization, smart autocomplete, and AI chatbots.

Input: A prompt, a question, a long article to summarize, or a sentence in one language to translate.

Output: Newly written text created by the model—such as a translated sentence, a short summary, or a conversational reply.

4.
*Supervised Learning:* Learning with an Answer Key
In supervised learning, you train the computer using text that already has the right answers (called target labels) attached to it. The computer reads the examples, sees the correct labels, and figures out the patterns so it can guess the right label when it sees new text later on.

*Simple Example:* Sorting customer reviews. Imagine giving a computer 1,000 phone reviews where a person has already tagged every single one as Good or Bad. By studying those examples, the computer learns that words like "fast" and "love it" usually mean Good, while "broken" and "terrible" mean Bad. Once trained, it can sort new reviews all by itself.

*Unsupervised Learning: Finding Patterns on Its Own*
In unsupervised learning, you give the computer a big pile of plain text with no labels or answers at all. Nobody tells the computer what to look for. Instead, it reads through everything and groups similar pieces of text together based on the words they share.

*Simple Example:* Grouping news stories. Suppose you hand a computer 10,000 random news articles without telling it what any of them are about. The computer notices that one group of articles uses words like "goals," "match," and "stadium," while another group uses words like "president," "vote," and "parliament." It naturally separates them into piles—and later, a person can look at the piles and label them Sports and Politics.

*Why One Needs Labels and the Other Doesn't*
Why supervised learning needs labels: Because you are asking the computer to make a specific prediction. Without labels acting as a teacher or answer key during practice, the computer wouldn't know if its guesses were right or wrong, so it couldn't learn from its mistakes.

*Why unsupervised learning doesn't need labels:* Because the goal is simply exploration and grouping. The computer isn't trying to guess a specific right answer—it is just comparing documents to each other and putting look-alikes in the same bucket. Since there is no "right" or "wrong" bucket beforehand, no labels are needed.

5.
Computers and Machine Learning models are essentially giant calculators—they only understand numbers, not raw letters or words. Before a model can find patterns, measure similarities, or make predictions from human language, every piece of text has to be translated into lists of numbers (vectors) so the computer can do math on them.

One Simple Method: Bag of Words (BoW)
One of the easiest ways to turn text into numbers is called Bag of Words.

Here is how it works:

1.Build a word list (vocabulary): First, the computer collects every unique word from all the sentences you give it.

2.Count the words: Next, for each sentence, it simply counts how many times each word from that master list shows up, putting 0 if a word isn't there.

For example, imagine our vocabulary is just four words: [coffee, is, good, bad].

The sentence "coffee is good" turns into [1, 1, 1, 0].

The sentence "coffee is bad" turns into [1, 1, 0, 1].

It gets its name because it treats a sentence like a messy bag of Scrabble tiles—it cares about which words are inside and how often they appear, not where they sit in the sentence.

**The Main Limitation of Bag of Words**
The biggest flaw with Bag of Words is that it completely ignores word order and context. Because it throws away the grammar and sequence of words, two sentences with opposite meanings can end up looking almost identical to the computer.

For example, look at these two sentences:

"The service is good, not bad."

"The service is bad, not good."

Both sentences contain the exact same words (the, service, is, good, not, bad) the exact same number of times. To a Bag of Words model, those two sentences produce the exact same numbers, even though one is a compliment and the other is a complaint.

6.
Building NLP tools for languages like Somali is tough mostly because the technology was originally designed around English, leaving several major hurdles for developers:

**Not enough training data:** Computers learn by studying millions of labeled examples, and Somali simply doesn't have as many public datasets as English. Because most available text comes from formal news sites, AI models often get confused when people use casual slang or everyday social media language.

**Inconsistent spelling:** Online, people spell Somali words in all kinds of ways—often dropping double vowels or using shortcuts (like writing manta instead of maanta). To a computer, two different spellings look like two completely unrelated words, which hurts search accuracy.

**Complex word endings:** A single Somali root word (like adeeg for "service") can change its endings in dozens of ways (adeegga, adeegyada, adeegaaga). Without tools built specifically for Somali grammar, a computer doesn't realize all those words share the same core meaning.

**Dialects and mixing languages:** People naturally speak different regional dialects and mix English or Arabic words right into Somali sentences (like "update gareeyo"). An AI trained only on standard textbook Somali easily gets tripped up when languages blend together like that.

7.
A great real-world example of NLP applied to an African language is the 2023 research paper "Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources" by Abdisalam Badel, Ting Zhong, Wenxin Tai, and Fan Zhou (published at the EMNLP 2023 conference).

*The Real-World Problem*
When Somali speakers search for information online, most search systems struggle because there are almost no dedicated Somali search datasets. Previously, systems often had to translate a user's Somali search query into English first and then translate the results back—which frequently loses cultural context, local names, and accurate meaning.

*Summary of the Case Study*
**Task:** Building a direct Somali search engine (information retrieval) system that takes a user's question or keywords in Somali and finds the most relevant Somali documents without relying on English translation.

**Data:** The researchers created the first dedicated Somali Information Retrieval corpus containing 2,335 Somali documents (16,450 sentences) gathered from Somali news websites and poetry books. They also interviewed native Somali speakers to create realistic search queries and manually labeled which documents matched each query.

**Method:** They built a custom Python-based search indexing system, created a specialized list of Somali stopwords (common filler words to ignore), and used a query-expansion technique called Pseudo-Relevance Feedback (PRF)—where the system looks at the top initial search results, picks out key related Somali words, and automatically refines the search to find even better matches.

**Findings:** The study proved that building dedicated Somali language resources and using Somali-specific query expansion significantly improves search accuracy compared to relying on cross-language translation. It also gave future researchers a benchmark dataset to test and improve Somali search tools.

*Which NLP Task from Question 3 Does It Cover?*
This case study directly covers Search and Similarity—the system takes a user's search query and a collection of documents as input, compares their relevance, and outputs a ranked list of the best-matching documents.