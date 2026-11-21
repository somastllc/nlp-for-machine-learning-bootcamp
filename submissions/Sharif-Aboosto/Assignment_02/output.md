## TASK 1: LOWERCASING (T1)

Before: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
After:  urgent!!! you have won a £1,000 gift card. call 0712345678 now!
All-caps words lost: ['URGENT', 'WON', 'NOW']
Uppercase ratio before: 0.44
Uppercase ratio after: 0.0
 
## TASK 2: WHITESPACE (T4)
Before: '   The   delivery   was   fast,   and the  packaging   was   great.'
After:  'The delivery was fast, and the packaging was great.'
 
## TASK 3: REGEX CLEANING (T1)
 
--- 3a. Delete with [^a-z\s] ---
Before: urgent!!! you have won a £1,000 gift card. call 0712345678 now!
After:  urgent you have won a  gift card call  now
Deleted characters: ['!', '!', '!', '£', '1', ',', '0', '0', '0', '.', '0', '7', '1', '2', '3', '4', '5', '6', '7', '8', '!']
Number deleted: 21
 
--- 3b. Replace instead of delete ---
Before: URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!
After:  URGENT! You have WON a MONEY gift card. Call PHONE_NUMBER NOW!
 
## TASK 4: TOKENIZATION
T2 split():        ['I', "don't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price.', 'Not', 'happy', 'at', 'all', '😞']
Count: 14
T2 word_tokenize:  ['I', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', '.', 'Not', 'happy', 'at', 'all', '😞']
Count: 16
T4 tokens: ['The', 'delivery', 'was', 'fast', ',', 'and', 'the', 'packaging', 'was', 'great', '.']
T4 token count: 11
T4 vocabulary size: 10
 
## TASK 5: STOPWORDS AND NEGATION (T2)
Tokens: ['i', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', '.', 'not', 'happy', 'at', 'all', '😞']
With NLTK stopwords:      ["n't", 'think', 'jacket', 'worth', 'price', '.', 'happy', '😞']
With my negation list:    ["n't", 'think', 'jacket', 'worth', 'price', '.', 'not', 'happy', '😞']
 
## TASK 6: STEMMING
Word          Porter      Lancaster   Snowball    
connection    connect     connect     connect     
connected     connect     connect     connect     
connecting    connect     connect     connect     
organization  organ       org         organ       
organized     organ       org         organ       
university    univers     univers     univers     
universal     univers     univers     univers     
flies         fli         fli         fli         
wolves        wolv        wolv        wolv        
better        better      bet         better      
ran           ran         ran         ran         
 
Merged pair: university -> univers | universal -> univers
Porter stems that are not real words: ['fli', 'univers', 'wolv']
 
## TASK 7: LEMMATIZATION
Word          Default       POS  With POS      
connection    connection    n    connection    
connected     connected     v    connect       
connecting    connecting    v    connect       
organization  organization  n    organization  
organized     organized     v    organize      
university    university    n    university    
universal     universal     a    universal     
flies         fly           v    fly           
wolves        wolf          n    wolf          
better        better        a    good          
ran           ran           v    run           
Changed only with POS hint: ['connected', 'connecting', 'organized', 'better', 'ran']
 
## TASK 8: SENTENCE SPLITTING (T3)
Sentence 1: Mr. Smith arrived in London on Friday.
Sentence 2: He met the Apple team at 9 a.m.
Number of sentences: 2
 
## TASK 9: POS TAGGING AND NER (T3)
POS tags: [('Mr.', 'NNP'), ('Smith', 'NNP'), ('arrived', 'VBD'), ('in', 'IN'), ('London', 'NNP'), ('on', 'IN'), ('Friday', 'NNP'), ('.', '.'), ('He', 'PRP'), ('met', 'VBD'), ('the', 'DT'), ('Apple', 'NNP'), ('team', 'NN'), ('at', 'IN'), ('9', 'CD'), ('a.m', 'NN'), ('.', '.')]
 
Entities:
  Mr.        PERSON
  Smith      PERSON
  London     GPE
 
## TASK 10: FULL PIPELINE (T2)
 
--- NLTK stopwords ---
1. Lowercase: i don't think this jacket is worth the price. not happy at all 😞
2. Regex:     i don't think this jacket is worth the price not happy at all
3. Tokenize:  ['i', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', 'not', 'happy', 'at', 'all']
4. Stopwords: ["n't", 'think', 'jacket', 'worth', 'price', 'happy']
5. Lemmatize: ["n't", 'think', 'jacket', 'worth', 'price', 'happy']
 
--- My negation list ---
1. Lowercase: i don't think this jacket is worth the price. not happy at all 😞
2. Regex:     i don't think this jacket is worth the price not happy at all
3. Tokenize:  ['i', 'do', "n't", 'think', 'this', 'jacket', 'is', 'worth', 'the', 'price', 'not', 'happy', 'at', 'all']
4. Stopwords: ["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']
5. Lemmatize: ["n't", 'think', 'jacket', 'worth', 'price', 'not', 'happy']