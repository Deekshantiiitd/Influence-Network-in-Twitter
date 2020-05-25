#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Use textblob to find the sentiment
import csv
import pandas
from textblob import TextBlob
from tqdm import tqdm_notebook as tqdm
from translate import Translator


# In[4]:


dataframe = pandas.read_csv('Retweet-Data+Sentiment.csv') # Load csv file


# In[3]:


sentiment = {} # Get sentiment for tweet with the tweet - id
for index, row in tqdm((dataframe[["Tweet-Id","Tweet","Language"]].drop_duplicates()).iterrows()): # Remove duplicate rows
    if row["Language"] != "en": # If language is non english translate to english and find the sentiment
        textb = TextBlob(Translator(from_lang=row["Language"],to_lang="en").translate(row["Tweet"])).sentiment[0]
        if (textb!=0):
            sentiment[row["Tweet-Id"]] = 1 if textb > 0 else -1
        else:
            sentiment[row["Tweet-Id"]] = 0
    else:# If language is english just find the sentiment
        textb = TextBlob(row["Tweet"]).sentiment[0]
        if (textb!=0):
            sentiment[row["Tweet-Id"]] = 1 if textb > 0 else -1
        else:
            sentiment[row["Tweet-Id"]] = 0


# In[4]:


answerList = []
for index, row in tqdm(dataframe.iterrows()):
    answerList += [sentiment[row["Tweet-Id"]]]


# In[5]:


dataframe["Sentiment"] = answerList
dataframe.to_csv("Retweet-Data+Sentiment.csv")


# In[ ]:





# In[ ]:




