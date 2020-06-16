
# coding: utf-8

# # Module 2 (Python 3)

# ## Basic NLP Tasks with NLTK

# In[10]:


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


# ### Counting vocabulary of words

# In[19]:


text7


# In[20]:


sent7


# In[21]:


len(sent7)


# In[22]:


len(text7)


# In[23]:


len(set(text7))


# In[24]:


list(set(text7))[:10]


# ### Frequency of words

# In[25]:


dist = FreqDist(text7)
len(dist)


# In[26]:


vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]


# In[27]:


# How many times a particular word occured --> in this case word 'four'
dist['four']


# In[28]:


# Frequency of a word which has length of at-least 5 and occured at-least 100 times
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords


# ### Normalization and stemming

# In[29]:


# This is an example of Normalization
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ') # To bring all words to same case, so that we can compare then easily
words1


# In[30]:


# This is an example of Stemming
porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# ### Lemmatization

# In[31]:


udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]


# In[24]:


[porter.stem(t) for t in udhr[:20]] # Still Lemmatization


# In[25]:


#  you will also notice that the fifth word here, universal declaration of human rights is not lemmatized because that is with a capital R, it's a different word that was not lemmatized to right
# Second last word "RIGHTS" is lemmatized to "RIGHT"
WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]


# ### Tokenization

# In[28]:


text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')


# In[29]:


nltk.word_tokenize(text11)


# In[30]:


text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
len(sentences)


# In[31]:


sentences


# ## Advanced NLP Tasks with NLTK

# ### POS tagging

# In[33]:


nltk.help.upenn_tagset('MD')


# In[34]:


text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)


# In[35]:


text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)


# In[37]:


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


# In[40]:


text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1


# In[41]:


parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)


# In[42]:


from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# ### POS tagging and parsing ambiguity

# In[43]:


text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)


# In[44]:


text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)

