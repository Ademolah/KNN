#!/usr/bin/env python
# coding: utf-8

# In[15]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[16]:


#importing the dataset
df = pd.read_csv('Social_Network_Ads.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:, -1].values


# In[17]:


#splitting the df
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.25 , random_state= 0)


# In[18]:


#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[19]:



#training the KNN model on the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier (n_neighbors = 5)
classifier.fit(x_train, y_train)


# In[20]:


#predicting new result
print(classifier.predict(sc.transform([[30, 87000]])))


# In[21]:


#predicting the test set result
y_pred = classifier.predict (x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))


# In[22]:


#composing the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




