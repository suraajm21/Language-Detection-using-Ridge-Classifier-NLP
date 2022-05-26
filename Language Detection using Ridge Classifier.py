#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Reading the dataset

# In[2]:


df = pd.read_csv(r'C:\Users\User\Documents\SSN Documentation\Suraaj\NLP Projects\Language Detection.csv')
df


# ## Value Counts of Language Label  

# In[3]:


df["language"].value_counts()


# ## Value Counts of Text Feature 

# In[4]:


df["Text"].value_counts()


# ## Data Preprocessing  

# ### Train Test Split 

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Text, 
                                                    df.language,
                                                    test_size=0.325000000000000001,
                                                    random_state=2551,
                                                    shuffle=True)


# ## Training the model using Ridge Classifier Technique 

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier


# ### NLP system entegration to Data    

# In[7]:


X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(X_train)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)


# ### Model Creating 

# In[8]:


model = RidgeClassifier()

model.fit(X_train_tfidf, y_train)


# ##  Model Accuracy Score

# In[9]:


model.score(X_CountVectorizer.transform(X_test),y_test)


# # Prediction

# ## 1.English

# ### Data of Prediction 

# In[10]:


text = """I quite like him. 
I'm so in love with him and my heart flutters when I see him."""
text = [text]
text_counts = X_CountVectorizer.transform(text)


# ### Prediction Processing 

# In[11]:


prediction = model.predict(text_counts)
f"Prediction is {prediction[0]}"


# # 2.Turkish

# ### Data of Prediction 

# In[12]:


text = """Türkiye Cumhuriyeti güçlüdür ve 
ilelebet baki kalacaktır."""
text = [text]
text_counts = X_CountVectorizer.transform(text)


# ### Prediction Processing 

# In[13]:


prediction = model.predict(text_counts)

f"Prediction is {prediction[0]}"

