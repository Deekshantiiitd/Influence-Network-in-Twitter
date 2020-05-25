#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Code to find link prediction such that given a user and it followers and previous tweets what it the next probable retweet

import csv # Impot csv to read data from csv
import pandas as pd # Use pandas to read data from csv
import numpy as np # Create a numpy array
import networkx as nx # Use networkx to create graphs
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm # Use tqdm to see the progress
from math import log10 as log # Use log to get the log of value
from sklearn.metrics import accuracy_score # Get the metrices
import time # 


# In[34]:


dataframe = pd.read_csv("Retweet-Data+Sentiment.csv") # Load data from csv
edgeData = dataframe[["FollowerId","FolloweeId"]] # Create a graph of follower and followee
edgeData2 = dataframe[["FollowerId","Retweet-Time","Tweet-Id"]] # Create a graph of neigbour and tweet
tweetData = dataframe[["Tweet-Id","FolloweeId","No-Of-Followers","No-Of-Friends","Retweet-Count","Tweet-Time","Likes","Language","Sentiment"]] # Data about a tweet


# In[35]:


edgeList = [(row["FollowerId"],row["FolloweeId"]) for index,row in edgeData.iterrows()] # Create a edgelist for graph follower and followee
edgeList2 = [(row["FollowerId"],row["Retweet-Time"],row["Tweet-Id"]) for index,row in edgeData2.iterrows()] # Create a edgelist for graph followee and tweet


# In[36]:


nodes = list(set(list((np.array(edgeList)).flatten()))) # Create nodes of graph 1
nodes2 = list(set(list(((np.array(edgeList2))[:,[0,2]]).astype(np.int64).flatten()))) # Create nodes of graph 2

edgesList2 = np.array(edgeList2) # Convert edge list data to numpy array to be split
x_train, x_test, y_train, y_test = train_test_split(edgesList2[:,0:2], edgesList2[:,2].astype(np.int64), test_size=0.01) # Get user tweet time and current correct tweet

edgeTrain = np.array(list(zip(x_train[:,0].astype(np.int64),y_train))) # Get training edges only


# In[37]:


tweetData = tweetData.groupby(["Tweet-Id"]).max() # Tweet data take max to get the latest data related to the tweet


# In[38]:


directedGraph = nx.DiGraph() # Create a follower followee di graph
directedGraph.add_nodes_from(nodes) # Add nodes to the graph
directedGraph.add_edges_from(edgeList) # Add edges to the graph

directedGraph2 = nx.DiGraph() # Create a followee tweet di graph
directedGraph2.add_nodes_from(nodes2) # Add nodes to the graph -2
directedGraph2.add_edges_from(edgeTrain)  # Add edges to the graph -2


# In[39]:


def retweetPrediction(x_test,tweetData,graph,graph2): # Code to predict next tweet for the followee
    answerTest = [] # Predicited answer are stored here
    for test in tqdm(x_test): # For test in the data set
        testSample = int(test[0]) # Get test sample and extract followee id
        neighbour = list(graph2.neighbors(testSample)) # Get follower the current user is following
        neighbourTweet = list(graph.neighbors(testSample)) # Get previous tweets of the user
        neighbourData = tweetData.loc[list(neighbourTweet)] # Get data of the previous tweet
        languages = neighbourData["Language"].tolist() # Get previous languages of the user
        sentiment = neighbourData["Sentiment"].sum() # Get user sentiment by summing over the tweets
        sentiment = 0 if sentiment==0 else (1 if sentiment > 0 else -1) # Get sentiment of the user
        maxScore = 0 # Get max score tweet 
        t2 = time.strptime(test[1],'%a %b %d %H:%M:%S +0000 %Y') # Convert user tweet time to get next possible retweet
        maxScoreId = 0 # Get score id of the tweet
        for index,row in tweetData.iterrows(): # Iterate over the all possible tweets
            score = 1 # score for current tweet
            if index not in neighbourTweet: # If not previous tweet of the user
                if row["Language"] in languages: # If one the previous languages of user tweet
                    score += 10

                if row["Sentiment"] == sentiment: # If sentiment of user matches
                    score += 10

                if row["FolloweeId"] in neighbour: # If user follow the writer of tweet
                    t1 = time.strptime(row["Tweet-Time"],'%a %b %d %H:%M:%S +0000 %Y') # Get tweet time
                    temp = ((time.mktime(t2)-time.mktime(t1))/(60*60)) # Convert tweet time to hour
                    if temp<= 0: # If tweet time > retweet time
                        score += 0
                    else: # Else divide it by the user time
                        score += 3000/temp
                    

                score += (log(row["Likes"]+1)+log(row["No-Of-Followers"]+1)/log(row["No-Of-Friends"]+2)) # Add followee data
                score += log(row["Retweet-Count"]+1)*5 # Give more weightage to retweet time
                if score>maxScore: # If score > previous heighest score
                    maxScoreId = index # Store the index
                    maxScore = score # Previous higher score
        answerTest.append(maxScoreId) # Append the score id for the user
    return answerTest # Return the answer


# In[40]:


answer = retweetPrediction(x_test,tweetData,directedGraph2,directedGraph) # Get the predicted next tweet id 


# In[41]:


print("Accuracy of the correct next tweet prediciton : ",accuracy_score(answer,y_test)) # Get the accuracy score

