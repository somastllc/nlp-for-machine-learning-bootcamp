import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

D = ['Win a FREE phone now! Reply WIN to claim.',
     'Free entry: win cash prizes today. Call now.',
     'Are we still meeting for lunch today?',
     'Please send me the lecture notes before class.',
     'The lecture was not good, but the notes were great.']
N = ['Claim your free prize now', 'Is the lunch meeting moved to tomorrow?']
R = ['the food was tasty', 'the food was delicious', 'the food was terrible']
np.set_printoptions(linewidth=200, threshold=np.inf)

def heading(number, title):
    print(f'\n{number}. {title}\n' + '-' * 60)

def words(v):
    return v.get_feature_names_out().tolist()

def one_hot(word, vocabulary):
    return [int(word == item) for item in vocabulary]

def run():
    print('Vectorization practice | scikit-learn', sklearn.__version__, '| NumPy', np.__version__)
    heading(1, 'One-hot encoding')
    vocab = ['cash', 'class', 'free', 'lunch', 'notes', 'win']
    print('Vocabulary:', vocab)
    for sentence in ['free cash', 'win free cash']:
        print(sentence)
        for word in sentence.split():
            print(word, one_hot(word, vocab))
        print('Number of vectors:', len(sentence.split()))
    heading(2, 'Bag of Words')
    cv = CountVectorizer()
    X = cv.fit_transform(D)
    A = X.toarray()
    terms = words(cv)
    print('Vocabulary:', cv.vocabulary_)
    print('Column order:', terms)
    print('Shape:', X.shape)
    print(A)
    print('D1 win count:', A[0, cv.vocabulary_['win']])
    reordered = cv.transform(['win free phone', 'phone free win']).toarray()
    print('Reordered sentences:\n', reordered)
    print('Equal:', np.array_equal(*reordered))
    heading(3, 'Tokens and vocabulary')
    df = np.count_nonzero(A, axis=0)
    print('Total tokens:', A.sum(), '| Vocabulary size:', len(terms))
    print('Words in multiple documents:', [w for w, f in zip(terms, df) if f > 1])
    heading(4, 'Preprocessing')
    variants = [CountVectorizer(), CountVectorizer(lowercase=False), CountVectorizer(stop_words='english')]
    for label, v in zip(['Default', 'Case sensitive', 'English stopwords'], variants):
        v.fit(D[:1])
        print(label, words(v))
    print('Removed stopwords:', sorted(set(words(variants[0])) - set(words(variants[2]))))
    stop = CountVectorizer(stop_words='english').fit(D)
    print('Full stopword vocabulary:', words(stop))
    print('Is not removed?', 'not' not in stop.vocabulary_)
    heading(5, 'Vocabulary limits')
    print('Full size:', len(terms))
    for label, v in [('min_df=2', CountVectorizer(min_df=2)), ('max_features=5', CountVectorizer(max_features=5))]:
        v.fit(D)
        print(label, len(words(v)), words(v))
    heading(6, 'N-grams')
    for n in [1, 2]:
        v = CountVectorizer(ngram_range=(1, n)).fit(D[4:])
        print('D5 range', (1, n), words(v))
    for n in [1, 2, 3]:
        v = CountVectorizer(ngram_range=(1, n)).fit(D)
        print('Full corpus range', (1, n), 'size:', len(words(v)))
    heading(7, 'TF-IDF')
    tv = TfidfVectorizer()
    T = tv.fit_transform(D).toarray()
    assert words(tv) == terms
    print('Column order:', terms)
    print(np.round(T, 2))
    for i in [0, 1]:
        print(f'D{i+1}: word | count | weight')
        for j, w in enumerate(terms):
            if A[i, j]:
                print(f'{w} | {A[i,j]} | {T[i,j]:.6f}')
    heading(8, 'Textbook TF-IDF calculation')
    length = int(A[1].sum())
    print('N:', len(D), '| D2 tokens:', length)
    for w in ['cash', 'today']:
        j = cv.vocabulary_[w]
        count = A[1,j]
        tf = count / length
        idf = np.log(len(D) / df[j])
        print(f'{w}: count={count}, df={df[j]}, TF={tf:.6f}, IDF={idf:.6f}, TF-IDF={tf*idf:.6f}, raw-count TF-IDF={count*idf:.6f}')
    heading(9, 'Sparse matrix')
    total = A.size
    print('Total cells:', total, '| Nonzero:', X.nnz, '| Zero:', total-X.nnz)
    print(f'Percentage zeros: {100*(total-X.nnz)/total:.2f}%')
    heading(10, 'Fit and transform')
    train = CountVectorizer().fit(D[:4])
    print('Training column order:', words(train))
    print('Test vectors:\n', train.transform(N).toarray())
    for i, text in enumerate(N, 1):
        print(f'N{i} OOV:', sorted(set(train.build_analyzer()(text)) - set(words(train))))
    heading(11, 'Data leakage')
    leaked = CountVectorizer().fit(D[:4] + N)
    print('Training only:', words(train))
    print('Training plus test:', words(leaked))
    print('Leaked words:', sorted(set(words(leaked)) - set(words(train))))
    heading(12, 'No meaning')
    review = CountVectorizer()
    B = review.fit_transform(R)
    print('Vocabulary:', words(review))
    print(B.toarray())
    print('Cosine similarity:\n', np.round(cosine_similarity(B), 4))
    return terms, A, T

if __name__ == '__main__':
    run()
