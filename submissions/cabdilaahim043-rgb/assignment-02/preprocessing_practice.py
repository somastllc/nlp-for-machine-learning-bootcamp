"""Assignment 2: reproducible NLTK preprocessing exercises.
Run after installing NLTK data: python preprocessing_practice.py > output.txt
"""
import re
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tree import Tree

T1 = 'URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!'
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = 'Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m.'
T4 = '   The   delivery   was   fast,   and the  packaging   was   great.   '
WORDS = 'connection connected connecting organization organized university universal flies wolves better ran'.split()

def section(name, before=None, after=None):
    print('\n' + '=' * 70 + '\n' + name + '\n' + '=' * 70)
    if before is not None: print('BEFORE:', before)
    if after is not None: print('AFTER: ', after)

def entities_from_tree(tree):
    return [(' '.join(w for w, _ in item.leaves()), item.label())
            for item in tree if isinstance(item, Tree)]

def wordnet_pos(tag):
    return {'J':'a', 'V':'v', 'N':'n', 'R':'r'}.get(tag[0], 'n')

def main():
    lower = T1.lower()
    section('1. Lowercase T1', repr(T1), repr(lower))
    normalized = ' '.join(T4.split())
    section('2. Normalize T4 whitespace', repr(T4), repr(normalized))
    deleted = re.findall(r'[^a-z\s]', lower)
    cleaned = re.sub(r'[^a-z\s]', '', lower)
    section('3a. Regex deletion [^a-z\\s]', repr(lower), repr(cleaned))
    print('DELETED CHARACTERS:', deleted)
    print('DELETED UNIQUE CHARACTERS:', sorted(set(deleted)))
    replaced = re.sub(r'(?<!\w)0\d{9}(?!\d)', 'PHONE_NUMBER', T1)
    replaced = re.sub(r'£\s?\d[\d,]*(?:\.\d+)?', 'MONEY', replaced)
    replaced = re.sub(r'!{3}', '!', replaced)
    section('3b. Regex replacement (preserve meaning)', repr(T1), repr(replaced))
    split_tokens, nltk_tokens = T2.split(), word_tokenize(T2)
    section('4. T2 tokenization', repr(T2))
    print('split():', split_tokens, 'COUNT:', len(split_tokens))
    print('word_tokenize():', nltk_tokens, 'COUNT:', len(nltk_tokens))
    t4_tokens = word_tokenize(normalized)
    print('T4 TOKENS:', t4_tokens)
    print('T4 TOKEN COUNT:', len(t4_tokens))
    print('T4 VOCABULARY SIZE (case-sensitive):', len(set(t4_tokens)))
    defaults = set(stopwords.words('english'))
    keep_negation = {'not', "n't", 'no', 'never'}
    custom = defaults - keep_negation
    default_filtered = [w for w in nltk_tokens if w.lower() not in defaults]
    custom_filtered = [w for w in nltk_tokens if w.lower() not in custom]
    section('5. Stopwords and negation', nltk_tokens)
    print('NLTK DEFAULT:', default_filtered)
    print('NEGATION-KEEPING:', custom_filtered)
    stemmers = {'Porter': PorterStemmer(), 'Lancaster': LancasterStemmer(), 'Snowball': SnowballStemmer('english')}
    section('6. Stemming comparison')
    print('WORD | PORTER | LANCASTER | SNOWBALL')
    for w in WORDS:
        print(' | '.join([w] + [stemmer.stem(w) for stemmer in stemmers.values()]))
    for name, stemmer in stemmers.items():
        groups = {}
        for w in WORDS: groups.setdefault(stemmer.stem(w), []).append(w)
        print(name, 'MERGED GROUPS:', {s: ws for s, ws in groups.items() if len(ws) > 1})
    lemma = WordNetLemmatizer()
    # Explicit, contextual POS interpretations: adjective better; verb ran; plural nouns flies/wolves.
    hints = ['n','v','v','n','v','n','a','n','n','a','v']
    section('7. WordNet lemmatization')
    print('WORD | DEFAULT | POS HINT | WITH POS')
    for w, hint in zip(WORDS, hints):
        print(f'{w} | {lemma.lemmatize(w)} | {hint} | {lemma.lemmatize(w, pos=hint)}')
    sentences = sent_tokenize(T3)
    section('8. Sentence splitting', repr(T3), sentences)
    print('SENTENCE COUNT:', len(sentences))
    tokens3 = word_tokenize(T3)
    tags3 = pos_tag(tokens3)
    tree = ne_chunk(tags3)
    entities = entities_from_tree(tree)
    section('9. POS tagging and NER', repr(T3))
    print('TOKENS:', tokens3)
    print('POS TAGS:', tags3)
    print('NER TREE:', tree)
    print('ENTITIES:', entities)
    print('ENTITY TABLE (manually evaluate correctness in answers.md):')
    for entity, label in entities: print(f'{entity} | {label}')
    print('CHECK EXPECTED MENTIONS: Smith (person), London (place), Friday (day), Apple (organization).')
    # Keep apostrophes to preserve n't; remove other punctuation, but keep emoji as text until regex stage.
    for name, stopset in [('NLTK DEFAULT', defaults), ('NEGATION-KEEPING', custom)]:
        section('10. Full pipeline: ' + name)
        text = T2
        print('ORIGINAL:', repr(text))
        text = text.lower()
        print('LOWERCASE:', repr(text))
        text = re.sub(r"[^a-z\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        print('REGEX:', repr(text))
        tokens = word_tokenize(text)
        print('TOKENIZE:', tokens)
        filtered = [w for w in tokens if w not in stopset]
        print('STOPWORDS:', filtered)
        tagged = pos_tag(filtered)
        lemmas = [lemma.lemmatize(w, wordnet_pos(tag)) for w, tag in tagged]
        print('LEMMATIZE:', lemmas)
        print('FINAL:', ' '.join(lemmas))

if __name__ == '__main__':
    try:
        main()
    except LookupError as exc:
        print('MISSING NLTK DATA. Run: python code/download-nltk-data.py', file=__import__('sys').stderr)
        raise
