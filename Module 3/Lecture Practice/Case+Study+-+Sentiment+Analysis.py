
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# *Note: Some of the cells in this notebook are computationally expensive. To reduce runtime, this notebook is using a subset of the data.*

# # Case Study: Sentiment Analysis

# ### Data Prep

# In[26]:


import pandas as pd
import numpy as np

# Read in the data
df = pd.read_csv('Amazon_Unlocked_Mobile.csv')

# Sample the data to speed up computation
# Comment out this line to match with lecture
df = df.sample(frac=0.1, random_state=10)

df.head()


# In[27]:


# Drop missing values
df.dropna(inplace=True)

# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)


# In[28]:


# Most ratings are positive
# Looking at the mean of the positively rated column, we can see that we have imbalanced classes.
df['Positively Rated'].mean()


# In[29]:


from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positively Rated'], 
                                                    random_state=0)


# In[33]:


print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)


# # CountVectorizer

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer allows us to use the bag-of-words approach by converting a collection of text documents into a matrix of token counts.
# Fitting the CountVectorizer consists of the tokenization of the trained data and building of the vocabulary.
# Fitting the CountVectorizer tokenizes each document by finding all sequences of characters of at least two letters or numbers separated by word boundaries. Converts everything to lowercase and builds a vocabulary using these tokens.
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
print(vect)


# In[38]:


# We can get the vocabulary by using the get_feature_names method.
# This vocabulary is built on any tokens that occurred in the training data.
vect.get_feature_names()[::2000]


# In[39]:


len(vect.get_feature_names())


# In[41]:


# transform the documents in the training data to a document-term matrix
# giving us the bag-of-word representation of X_train.
# This representation is stored in a SciPy sparse matrix, where each row corresponds to a document and each column a word from our training vocabulary.
# The entries in this matrix are the number of times each word appears in each document.
# Because the number of words in the vocabulary is so much larger than the number of words that might appear in a single review, most entries of this matrix are zero
X_train_vectorized = vect.transform(X_train)

X_train_vectorized


# In[42]:


from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[44]:


from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
# Note that any words in X_test that didn't appear in X_train will just be ignored
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[45]:


# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# # Tfidf

# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Tf–idf, or Term frequency-inverse document frequency, allows us to weight terms based on how important they are to a document.
# High weight is given to terms that appear often in a particular document, but don't appear often in the corpus
# Features with low tf–idf are either commonly used across all documents or rarely used and only occur in long documents
# Features with high tf–idf are frequently used within specific documents, but rarely used across all documents.
# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())


# In[47]:


X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[48]:


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[49]:


sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[50]:


# One problem with our previous bag-of-words approach is word order is disregarded. So, not an issue, phone is working is seen the same as an issue, phone is not working.
# Our current model sees both of these reviews as negative reviews.
# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# # n-grams

# In[51]:


# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[52]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[54]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[55]:


# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

