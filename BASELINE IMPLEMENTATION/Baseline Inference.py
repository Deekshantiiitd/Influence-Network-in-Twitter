#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import files
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm 
import sys
import itertools
import networkx as nx
import time
import plotly.express as px
from math import log10 as log
import matplotlib.pyplot as plt
from matplotlib import style


# In[2]:


data=pd.read_csv("Retweet-Data.csv")

screenIdDict = {}
screenId = data.iloc[:,1:3]
screenId = screenId.values.tolist() 
for i in screenId:
    if i[0] not in screenIdDict:
        screenIdDict[i[0]] = i[1]


# In[3]:


edges = data.iloc[:,0:2]
edges = edges.values.tolist()
graph = nx.MultiDiGraph()
graph.add_edges_from(edges)
print()

edges2 = [tuple(edge) for edge in edges]
graph2 = nx.DiGraph()
graph2.add_edges_from(edges2)
print()


# In[4]:


nx.write_gexf(graph,"Project_Retweet.gexf")


# ***Baseline - 1***

# In[5]:


def follower_followee(data): # Calculate follower followee ratio 
    data1=data[["FolloweeId","No-Of-Followers","No-Of-Friends"]]
    follower_followe_dict={}
    for i in range(len(data1)):
        if data1.iloc[i]["No-Of-Friends"]!=0:
            follower_followe_dict[data1.iloc[i]["FolloweeId"]]=data1.iloc[i]["No-Of-Followers"]/data1.iloc[i]["No-Of-Friends"]
        else:
            follower_followe_dict[data1.iloc[i]["FolloweeId"]]=data1.iloc[i]["No-Of-Followers"]
    #         print(data1.iloc[i]["FollowerId"])
    df1 = pd.DataFrame()
    df1['FolloweeId'] = follower_followe_dict.keys()
    df1['ratio'] = follower_followe_dict.values() 
    return df1
    


# In[6]:


df1 = follower_followee(data)


# In[7]:


def tweet_retweet(data): # Calculate tweet retweet ratio 
    data2=data[["FolloweeId","Statuses-Count","Retweet-Count"]]
    tweet_retweet_dict={}
    for i in range(len(data2)):
        if data2.iloc[i]["FolloweeId"] in tweet_retweet_dict.keys():
            x=tweet_retweet_dict[data2.iloc[i]["FolloweeId"]]
            x=max(x,data2.iloc[i]["Retweet-Count"]/data2.iloc[i]["Statuses-Count"])
            tweet_retweet_dict[data2.iloc[i]["FolloweeId"]]=x

        else:
            tweet_retweet_dict[data2.iloc[i]["FolloweeId"]]=data2.iloc[i]["Retweet-Count"]/data2.iloc[i]["Statuses-Count"]
    df2 = pd.DataFrame()
    df2['FolloweeId'] = tweet_retweet_dict.keys()
    df2["ratio"] = tweet_retweet_dict.values()
    return df2


# In[8]:


df2 = tweet_retweet(data)


# In[9]:


def retweet_follower(data):  
    data3 = (data[["FolloweeId","No-Of-Followers","Tweet-Id","Retweet-Count"]]).values.tolist()
    data3 = sorted(data3,key = lambda x:x[2])
    
    newData = []
    for key,group in itertools.groupby(data3,key = lambda x:x[2]):
        newData += [max(group,key=lambda x: x[3])]

    newData = sorted(newData,key = lambda x:x[0])
    finalData = []
    for key,group in itertools.groupby(newData,key = lambda x:x[0]):
        groupData = list(group)
        sumList = sum([i[3] for i in groupData])
        if(groupData[0][1] != 0):
            finalData.append((key,sumList/groupData[0][1]))
        else:
            finalData.append((key,0))
            
    df3 = pd.DataFrame()
    df3["FolloweeId"] = [i[0] for i in finalData]
    df3["ratio"] = [i[1] for i in finalData]
    return df3


# In[10]:


df3 = retweet_follower(data)


# In[11]:


def Tweet_Retweet(data):
    data4 = (data[["FolloweeId","Tweet-Time","Retweet-Time","Retweet-Count"]]).values.tolist()
    ts = time.strptime(data4[1][1],'%a %b %d %H:%M:%S +0000 %Y')
    ts2 = time.strptime(data4[1][2],'%a %b %d %H:%M:%S +0000 %Y')
    
    newData = [[data[0],(time.mktime(time.strptime(data[2],'%a %b %d %H:%M:%S +0000 %Y')) - time.mktime(time.strptime(data[1],'%a %b %d %H:%M:%S +0000 %Y'))),data[3]] for data in data4]
    finalData = []
    newData = sorted(newData,key = lambda x:x[0])
    for key,group in itertools.groupby(newData,key = lambda x:x[0]):
        finalData += [max(group,key=lambda x: x[1])]
     
    df4 = pd.DataFrame()
    df4["FolloweeId"] = [i[0] for i in finalData]
    df4["retweetTime"] = [i[1] for i in finalData]
    df4["retweetCount"] = [i[2] for i in finalData]
    return df4


# In[12]:


df4 = Tweet_Retweet(data)


# In[13]:


def communityStructure(data):
    data5 = (data[["FolloweeId","No-Of-Followers","Tweet-Id","Retweet-Count"]]).values.tolist()
    data5 = sorted(data5,key = lambda x:x[2])
    
    newData = []
    for key,group in itertools.groupby(data5,key = lambda x:x[2]):
        newData += [max(group,key=lambda x: x[3])]

    newData = sorted(newData,key = lambda x:x[0])
    finalData = []
    for key,group in itertools.groupby(newData,key = lambda x:x[0]):
        groupData = list(group)
        sumList = sum([i[3] for i in groupData])
        if(groupData[0][1] != 0):
            finalData.append((key,sumList))
        else:
            finalData.append((key,0))
    
    finalData = sorted(finalData,key = lambda x:x[1])
    
    followeeId = [i[0] for i in finalData[:10]+finalData[-50:]]
    
    dataList = data.values.tolist()
    finalCommunityData = []
    
    for d in dataList:
        if(d[1] in followeeId):
            finalCommunityData += [d]
    
    df5 = pd.DataFrame()
    for i in range(len(data.columns)):  
        df5[data.columns[i]] = [j[i] for j in finalCommunityData] 
    
    return df5


# In[14]:


df5 = communityStructure(data)


# In[15]:


df4_c = (Tweet_Retweet(df5)).sort_values(by=['FolloweeId'])
df1_c = (follower_followee(df5)).sort_values(by=['FolloweeId'])
df2_c = (tweet_retweet(df5)).sort_values(by=['FolloweeId'])


# In[16]:


df4 = df4.sort_values(by=['FolloweeId'])
df3 = df3.sort_values(by=['FolloweeId'])
df1 = df1.sort_values(by=['FolloweeId'])
df2 = df2.sort_values(by=['FolloweeId'])


# In[17]:


def retransmission_graph(df4,df1):
    deg_ratio=[]
    ret_count=[]
    for i in range(len(df1)):
        ret_count.append(df4["retweetCount"][i])
        deg_ratio.append(df1["ratio"][i])
    return  ret_count,deg_ratio


# In[18]:


def plotRetransmissionGraph(df4,df1):
    y_axis,x_axis=retransmission_graph(df4,df1)
    px.scatter(x=x_axis,y=y_axis,labels={'y':"Rin",'x':'kin/kout'},title = "Rin transmission vs Kin/Kout").show()


# In[19]:


plotRetransmissionGraph(df4,df1)


# In[20]:


def enthusiasm_graph(df1,df4):
    user_duration=[]
    deg_ratio=[]
    for i in range(len(df1)):
        deg_ratio.append(df1["ratio"][i])
        if df4["retweetCount"][i]!=0:
            user_duration.append(df4["retweetTime"][i]/(df4["retweetCount"][i]*3600))
        else:
            user_duration.append(0)
        
    return user_duration,deg_ratio


# In[21]:


def plotEnthusiasmGraph(df1,df4):
    y1_axis,x1_axis=enthusiasm_graph(df1,df4)
    px.scatter(x=x1_axis,y=y1_axis,labels={'y':"User Enthusiasm",'x':'kin/kout'},title="User Enthusiasm vs Kin/Kout").show()


# In[22]:


plotEnthusiasmGraph(df1,df4)


# In[23]:


def popularityVsRetweet_graph(df2,df1):
    deg_ratio=[]
    ret_count=[]
    for i in range(len(df1)):
        ret_count.append(df2["ratio"][i])
        deg_ratio.append(df1["ratio"][i])
    return  ret_count,deg_ratio


# In[24]:


def plotPopularityVsRetweet_graph(df1,df2):
    y1_axis,x1_axis=popularityVsRetweet_graph(df2,df1)
    px.scatter(x=x1_axis,y=y1_axis,labels={'y':"Retweet/Followers",'x':'kin/kout'},title="User Engine vs kin/kout").show()


# In[25]:


plotPopularityVsRetweet_graph(df1,df2)


# ****Plots for small community graph****

# In[26]:


plotRetransmissionGraph(df4_c,df1_c)


# In[27]:


plotEnthusiasmGraph(df1_c,df4_c)


# In[28]:


plotPopularityVsRetweet_graph(df1_c,df2_c)


# ***Baseline - 2***

# In[29]:


def follower_followeeVsPageRank(df,graph):
    dflist = df.values.tolist()
    pageRank = nx.pagerank(graph)
    x_axis = []
    y_axis = []
    for d in dflist:
        if d[0] in pageRank:
            x_axis += [log(d[1]+1)] 
            y_axis += [log(pageRank[d[0]])]
    px.scatter(x=x_axis,y=y_axis,labels={'y':"Page Rank",'x':'Follower/Followee'},title="Follower/Followee vs Page Rank").show()


# In[30]:


follower_followeeVsPageRank(df1,graph2)


# In[31]:


def SNPRatio(df3,df2):
    values = []
    for index in range(len(df3)):
        values.append([df3.iloc[index,0],(df3.iloc[index,1] + df2.iloc[index,1])/2])  
    ids = [i[0] for i in sorted(values,key= lambda x:x[1],reverse=True)]
    
    return [screenIdDict[i] for i in ids[:10]]


# In[32]:


SNPRatio(df3,df2)


# In[33]:


def findtop(df):
    temp = ((df2.sort_values(by=["ratio"]).tail(5))["FolloweeId"]).values.tolist()
    
    return [screenIdDict[i] for i in temp]
    


# In[34]:


findtop(df2)

