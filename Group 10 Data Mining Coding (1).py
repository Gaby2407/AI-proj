#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

# In[211]:


import pandas as pd
import sklearn
import warnings
warnings.filterwarnings("ignore")


# In[212]:


df=pd.read_csv("/home/gaby/Downloads/Bank_Campaign_CIA.csv",sep=';',header=[0],na_values=['unknown'])
print(df.head())


# In[213]:


newdf = df[~df.isnull().any(axis=1)]
newdf2=newdf[['age','loan','marital','job','education','default','housing','poutcome','subscribed']]


# In[214]:


newdfy=newdf[newdf['subscribed']=="yes"].sample(2000)
newdfn=newdf[newdf['subscribed']=="no"].sample(2000)


# In[215]:


newdfy2=newdf2[newdf2['subscribed']=="yes"].sample(2000)
newdfn2=newdf2[newdf2['subscribed']=="no"].sample(2000)       


# In[216]:


frames=[newdfy,newdfn]
frames2=[newdfy2,newdfn2]


# In[217]:


newdf= pd.concat(frames)
newdf2=pd.concat(frames2)


# In[218]:


newdf['subscribed']=[1 if i=="yes" else 0 for i in newdf['subscribed']]
newdf2['subscribed']=[1 if i=="yes" else 0 for i in newdf2['subscribed']]     


# In[219]:


newdf=pd.get_dummies(newdf)
newdf2=pd.get_dummies(newdf2)


# In[220]:


print("dataset after one-hot-encoding and balancing classes")
print(newdf.head())


# In[221]:


inputs = newdf.drop('subscribed', axis = 1)
target = newdf['subscribed']
inputs.head()


# In[222]:


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs, target)


# In[223]:


model.score(inputs, target)


# # Training and Testing for logistic regression

# In[224]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


# In[225]:


newdf.head()


# In[226]:


x = inputs.values
y = target.values


# In[227]:


#print("MODEL -- Training and Test set shape" 80% and 20%)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("Logistic Regression with all 20 columns of original dataset with one-hot-encoding")
print("train_x size", train_x.shape)
print("train_y size", train_y.shape)
print("test_x size", test_x.shape)
print("test_y size", test_y.shape)


# In[228]:


logreg = LogisticRegression(max_iter = 1000)
logreg.fit(train_x, train_y)
pred_y = logreg.predict(test_x)


# In[229]:


#Accuracy - Ratio of cases classified correctly
print("Accuracy : ", metrics.accuracy_score(test_y, pred_y))


# In[230]:


#F1 Score - A combined metric, the harmonic mean of Precision and Recall.
print("F1 Score : ", metrics.f1_score(test_y, pred_y))


# In[231]:


#Precision - Accuracy of a predicted positive outcome
print("Precision Score : ", metrics.precision_score(test_y, pred_y))


# In[232]:


#Recall - Measures model’s ability to predict a positive outcome
print("Recall Score : ", metrics.recall_score(test_y,pred_y))


# In[233]:


rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print("Rmse score :",rmse)


# ## Training and Testing first iteration

# In[234]:


inputs = newdf2.drop('subscribed', axis = 1)
target = newdf2['subscribed']


# In[235]:


x = inputs.values
y = target.values


# In[236]:


#print("MODEL -- Training and Test set shape" 80% and 20%)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)
print("Logistic Regression with 8 significant columns one-hot-encoded")
print("train_x size", train_x.shape)
print("train_y size", train_y.shape)
print("test_x size", test_x.shape)
print("test_y size", test_y.shape)


# In[237]:


logreg = LogisticRegression()
logreg.fit(train_x, train_y)
pred_y = logreg.predict(test_x)


# In[238]:


#Accuracy - Ratio of cases classified correctly
print("Accuracy : ", metrics.accuracy_score(test_y, pred_y))


# In[239]:


#F1 Score - A combined metric, the harmonic mean of Precision and Recall.
print("F1 score:",metrics.f1_score(test_y, pred_y))


# In[240]:


#Precision - Accuracy of a predicted positive outcome
print("Precision score:",metrics.precision_score(test_y, pred_y))


# In[241]:


#Recall - Measures model’s ability to predict a positive outcome
print("Recall score:",metrics.recall_score(test_y, pred_y))


# In[242]:


rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print("Rmse score:",rmse)

