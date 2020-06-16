
# Module 2 (Python 3)

## Basic NLP Tasks with NLTK


```python
import nltk
try:
    from nltk.book import *
except:
    #nltk.download('gutenberg')
    #nltk.download('genesis')
    #nltk.download('inaugural')
    #nltk.download('nps_chat')
    #nltk.download('webtext')
    #nltk.download('treebank')
    nltk.download('all')
    from nltk.book import *
```

### Counting vocabulary of words


```python
text7
```




    <Text: Wall Street Journal>




```python
sent7
```




    ['Pierre',
     'Vinken',
     ',',
     '61',
     'years',
     'old',
     ',',
     'will',
     'join',
     'the',
     'board',
     'as',
     'a',
     'nonexecutive',
     'director',
     'Nov.',
     '29',
     '.']




```python
len(sent7)
```




    18




```python
len(text7)
```




    100676




```python
len(set(text7))
```




    12408




```python
list(set(text7))[:10]
```




    ['PAP',
     'lap-shoulder',
     'INS',
     '73',
     'price',
     'delisted',
     'Calif',
     'elaborate',
     'pre-emptive',
     'Shrum']



### Frequency of words


```python
dist = FreqDist(text7)
len(dist)
```




    12408




```python
vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]
```




    ['Pierre', 'Vinken', ',', '61', 'years', 'old', 'will', 'join', 'the', 'board']




```python
# How many times a particular word occured --> in this case word 'four'
dist['four']
```




    20




```python
# Frequency of a word which has length of at-least 5 and occured at-least 100 times
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords
```




    ['billion',
     'company',
     'president',
     'because',
     'market',
     'million',
     'shares',
     'trading',
     'program']



### Normalization and stemming


```python
# This is an example of Normalization
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ') # To bring all words to same case, so that we can compare then easily
words1
```




    ['list', 'listed', 'lists', 'listing', 'listings']




```python
# This is an example of Stemming
porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]
```




    ['list', 'list', 'list', 'list', 'list']



### Lemmatization


```python
udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]
```




    ['Universal',
     'Declaration',
     'of',
     'Human',
     'Rights',
     'Preamble',
     'Whereas',
     'recognition',
     'of',
     'the',
     'inherent',
     'dignity',
     'and',
     'of',
     'the',
     'equal',
     'and',
     'inalienable',
     'rights',
     'of']




```python
[porter.stem(t) for t in udhr[:20]] # Still Lemmatization
```




    ['univers',
     'declar',
     'of',
     'human',
     'right',
     'preambl',
     'wherea',
     'recognit',
     'of',
     'the',
     'inher',
     'digniti',
     'and',
     'of',
     'the',
     'equal',
     'and',
     'inalien',
     'right',
     'of']




```python
#  you will also notice that the fifth word here, universal declaration of human rights is not lemmatized because that is with a capital R, it's a different word that was not lemmatized to right
# Second last word "RIGHTS" is lemmatized to "RIGHT"
WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]
```




    ['Universal',
     'Declaration',
     'of',
     'Human',
     'Rights',
     'Preamble',
     'Whereas',
     'recognition',
     'of',
     'the',
     'inherent',
     'dignity',
     'and',
     'of',
     'the',
     'equal',
     'and',
     'inalienable',
     'right',
     'of']



### Tokenization


```python
text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')
```




    ['Children', "shouldn't", 'drink', 'a', 'sugary', 'drink', 'before', 'bed.']




```python
nltk.word_tokenize(text11)
```




    ['Children',
     'should',
     "n't",
     'drink',
     'a',
     'sugary',
     'drink',
     'before',
     'bed',
     '.']




```python
text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
len(sentences)
```




    4




```python
sentences
```




    ['This is the first sentence.',
     'A gallon of milk in the U.S. costs $2.99.',
     'Is this the third sentence?',
     'Yes, it is!']



## Advanced NLP Tasks with NLTK

### POS tagging


```python
nltk.help.upenn_tagset('MD')
```

    MD: modal auxiliary
        can cannot could couldn't dare may might must need ought shall should
        shouldn't will would



```python
text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)
```




    [('Children', 'NNP'),
     ('should', 'MD'),
     ("n't", 'RB'),
     ('drink', 'VB'),
     ('a', 'DT'),
     ('sugary', 'JJ'),
     ('drink', 'NN'),
     ('before', 'IN'),
     ('bed', 'NN'),
     ('.', '.')]




```python
text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)
```




    [('Visiting', 'VBG'),
     ('aunts', 'NNS'),
     ('can', 'MD'),
     ('be', 'VB'),
     ('a', 'DT'),
     ('nuisance', 'NN')]




```python
# Parsing sentence structure
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)
```

    (S (NP Alice) (VP (V loves) (NP Bob)))



```python
text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1
```




    <Grammar with 13 productions>




```python
parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)
```

    (S
      (NP I)
      (VP
        (VP (V saw) (NP (Det the) (N man)))
        (PP (P with) (NP (Det a) (N telescope)))))
    (S
      (NP I)
      (VP
        (V saw)
        (NP (Det the) (N man) (PP (P with) (NP (Det a) (N telescope))))))



```python
from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)
```

    (S
      (NP-SBJ
        (NP (NNP Pierre) (NNP Vinken))
        (, ,)
        (ADJP (NP (CD 61) (NNS years)) (JJ old))
        (, ,))
      (VP
        (MD will)
        (VP
          (VB join)
          (NP (DT the) (NN board))
          (PP-CLR (IN as) (NP (DT a) (JJ nonexecutive) (NN director)))
          (NP-TMP (NNP Nov.) (CD 29))))
      (. .))


### POS tagging and parsing ambiguity


```python
text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)
```




    [('The', 'DT'), ('old', 'JJ'), ('man', 'NN'), ('the', 'DT'), ('boat', 'NN')]




```python
text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)
```




    [('Colorless', 'NNP'),
     ('green', 'JJ'),
     ('ideas', 'NNS'),
     ('sleep', 'VBP'),
     ('furiously', 'RB')]


