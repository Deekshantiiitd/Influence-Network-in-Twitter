import csv
import pandas as pd
import numpy as np
import math
from collections import Counter
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import regularizers
from sklearn.metrics import roc_auc_score
from keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, log_loss
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings("ignore")


Retweet_data=pd.read_csv("Retweet-Data (with No of Followers and likes).csv")

Retweet_data.head()


# #### Make Follower-Followee Directed Graph


Twitter_Graph=nx.DiGraph()


for i in tqdm(range(len(Retweet_data))):
    Twitter_Graph.add_edge(Retweet_data.iloc[i]['FollowerId'],Retweet_data.iloc[i]['FolloweeId'])


Degree_central=nx.degree_centrality(Twitter_Graph)


# #### Make Data for the Followee 

# Make followee ids list
Followee_Id=list(Retweet_data['FolloweeId'].unique())


new_columns=[ 'No-Of-Followers','No-Of-Friends', 'Statuses-Count', 'Retweet-Count','Likes' ,'Followe-Friend']

Data=pd.DataFrame(index=Followee_Id,columns=new_columns)

for i in tqdm(range(len(Retweet_data))):
    Data.loc[Retweet_data.iloc[i]['FolloweeId']]=Retweet_data.iloc[i][new_columns]


# ### Make Final Features DataFrame

Results=pd.DataFrame(index=Followee_Id)
N=len(Retweet_data)


# #### User Activity Measure 


### General Activity Measure
Results['General_Activity_Measure']=Data['Statuses-Count']+Data['Retweet-Count']+Data['Likes']  


### Topical signal
Results['Topical signal']=(Data['Statuses-Count']+Data['Retweet-Count'])/N


### signal Strength
Results['Signal Strength']=Data['Statuses-Count']/(Data['Statuses-Count']+Data['Retweet-Count'])


# #### Popularity Measure


#### Follower Rank
Results['Follower Rank']=Data['No-Of-Followers']/(Data['No-Of-Followers']+Data['No-Of-Friends'])


#### Popularity 
Results['Popularity']=1-(Data['No-Of-Followers']**2.718)


#### Degree Centralirty
degree_central_score=[]
for node in Followee_Id:
    degree_central_score.append(Degree_central[node])
    
for node in tqdm(Followee_Id):
    Results['degree_centrality']=degree_central_score


print(Results['Label']=0)

print(Results.shape)


# #### Feature Threshold 

Results.loc[(Results['Follower Rank'] > 0.05 ) & (Results['Topical signal'] > 0.05) 
            &(Results['Signal Strength'] > 0.50 ) & (Results['Popularity'] < -1.000000000e+6)
            & (Results['General_Activity_Measure'] > 7000 ) ,'Label']=

sns.pairplot(Results, hue='Label')


# ### Making Training and Testing data


Y=pd.DataFrame()
Y['Label']=Results['Label']

df_train, df_test,label_train,label_test= train_test_split(Results,Y,stratify=Y,test_size=0.3,shuffle=True,)

df_train=df_train.drop(labels='Label',axis=1)
df_test=df_test.drop(labels='Label',axis=1)

df_train['Follower Rank'].fillna(0.0,inplace=True)
df_test['Follower Rank'].fillna(0.0,inplace=True)



# ### Data Normalization and preprocessing

mm_scaler = preprocessing.MinMaxScaler()
df_train_normalize = mm_scaler.fit_transform(df_train)
df_test_normalize=mm_scaler.transform(df_test)


# ### Apply Multi layer perceptron Model

Y_train = to_categorical(label_train)
Y_test = to_categorical(label_test)


model1 = Sequential()
model1.add(Dense(8, input_dim=len(df_train_normalize[0]), activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(4, activation='relu'))
model1.add(Dropout(0.1))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history1=model1.fit(df_train_normalize, Y_train,epochs=200,batch_size=32,validation_split=0.3, shuffle=False)
score = model1.evaluate(df_train_normalize, Y_train, batch_size=64)


loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
plt.plot(loss1)
plt.plot(val_loss1)
plt.legend(['loss', 'val_loss'])
plt.grid()
plt.show()

acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']
plt.plot(acc1)
plt.plot(val_acc1)
plt.legend(['accuracy', 'val_accuracy'])
plt.grid()
plt.show()

predictions_train = model1.predict(df_train_normalize)
scores=model1.evaluate(df_train_normalize,Y_train,verbose=10)

print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

prediction_test1= model1.predict(df_test_normalize)
scores2 = model1.evaluate(df_test_normalize, Y_test, verbose=4)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1])) 

print("Precision score micro: ",precision_score(prediction_test1.argmax(axis=-1),Y_test.argmax(axis=-1),average="micro")) 

print("Precision score macro: ",precision_score(prediction_test1.argmax(axis=-1),Y_test.argmax(axis=-1),average="macro")) # Print macro precision for final class value


print("f1_score macro: ",f1_score(prediction_test1.argmax(axis=-1),Y_test.argmax(axis=-1),average="micro")) # Print macro precision for final class value

print(classification_report(prediction_test1.argmax(axis=-1),Y_test.argmax(axis=-1)))


test_followee_id=list(df_test.index)
Y_Predict=[]
for i in tqdm(range(len(test_followee_id))):
    if prediction_test1[i][0]>prediction_test1[i][1]:
        Y_Predict.append(1)
        df_predicted_label.loc[test_followee_id[i]]['Label']='Influencer'
    else:
        df_predicted_label.loc[test_followee_id[i]]['Label']='Not Influencer'
        Y_Predict.append(0)
print(df_predicted_label)


# ### Apply Sequential neural network

model2 = tf.keras.Sequential()
model2.add(layers.Dense(500, input_dim=len(df_train_normalize[0]), activation='relu',kernel_initializer='random_normal')) ### Add input layer
model2.add(layers.Dense(100, activation='relu',kernel_initializer='random_normal'))  ### Add one hidden layer
model2.add(layers.Dense(2, activation='sigmoid',kernel_initializer='random_normal'))   ### Add output layer
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model on training data
history2=model2.fit(df_train_normalize, Y_train,epochs=20,batch_size=32,validation_split=0.3, shuffle=False)

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
plt.plot(loss2)
plt.plot(val_loss2)
plt.legend(['loss', 'val_loss'])
plt.grid()
plt.show()

acc2 = history2.history['acc']
val_acc2 = history2.history['val_acc']
plt.plot(acc2)
plt.plot(val_acc2)
plt.legend(['accuracy', 'val_accuracy'])
plt.grid()
plt.show()

# prediction and evaluation on training set
predictions_train = model2.predict(df_train_normalize)
scores=model2.evaluate(df_train_normalize,Y_train,verbose=0)


print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

# prediction and evaluation on testing set
prediction_test2= model2.predict(df_test_normalize)
scores2 = model2.evaluate(df_test_normalize, Y_test, verbose=0)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1])) 



print("Precision score macro: ",precision_score(prediction_test2.argmax(axis=-1),Y_test.argmax(axis=-1),average="macro")) # Print macro precision for final class value
print("Precision score micro: ",precision_score(prediction_test2.argmax(axis=-1),Y_test.argmax(axis=-1),average="micro")) 


print("f1_score macro: ",f1_score(prediction_test2.argmax(axis=-1),Y_test.argmax(axis=-1),average="micro")) # Print macro precision for final class value



test_followee_id=list(df_test.index)
df_predicted_label=pd.DataFrame(columns=['Label'],index=test_followee_id)
for i in tqdm(range(len(test_followee_id))):
    if prediction_test[i][0]>prediction_test[i][1]:
        df_predicted_label.loc[test_followee_id[i]]['Label']='Influencer'
    else:
        df_predicted_label.loc[test_followee_id[i]]['Label']='Not Influencer'
print(df_predicted_label)


# ## Random Forest Model


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier


model3 = RandomForestClassifier(n_estimators=1,max_depth=8, random_state=80)
model3.fit(df_train, label_train)
prediction_test=model3.predict(df_test)


f1Score_micro=f1_score(list(label_test['Label']),prediction_test,average='micro')
f1Score_macro=f1_score(list(label_test['Label']),prediction_test,average='macro')
acc_score=accuracy_score(label_test['Label'],prediction_test)
print(f"Average micro f1_score is :{f1Score_micro}\n")
print(f"Average macro f1_score is :{f1Score_macro}\n")
print(f"Accuracy VALUE is :{acc_score}\n")    

precision_recall_fscore_support(list(label_test['Label']),prediction_test,average='macro')


test_followee_id=list(df_test.index)
df_predicted_label=pd.DataFrame(columns=['Label'],index=test_followee_id)
for i in tqdm(range(len(test_followee_id))):
    if prediction_test[i]==1:
        df_predicted_label.loc[test_followee_id[i]]['Label']='Influencer'
    else:
        df_predicted_label.loc[test_followee_id[i]]['Label']='Not Influencer'
print(df_predicted_label)

