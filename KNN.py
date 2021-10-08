#!/usr/bin/env python
# coding: utf-8

# # Predict whether a patient is diabetic or not

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# In[3]:


filename = 'diabetes'
path = 'E:/desktop/ML/KNN Algorithm/{}.csv'.format(filename)
data = pd.read_csv(path)
data.head()


# In[4]:


main_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']


# In[5]:


for i in main_columns:
    data[i] = data[i].replace(0,np.NaN)
    mean = int(data[i].mean(skipna=True))
    data[i] = data[i].replace(np.NaN,mean)


# In[6]:


data.head()


# ### Spilt the Data

# In[8]:


X = data.iloc[:,0:8]
Y = data.iloc[:,8]


# In[9]:


X.head()


# In[10]:


Y.head()


# In[11]:


X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,random_state=0,test_size=0.2)


# In[20]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[21]:


import math
n = math.sqrt(len(Y_test))
n


# ### Define the model : Init the KNN where n_neighbors=n-1

# In[ ]:


classifier = KNeighborsClassifier(n_neighbors = 11,p=2,metric='euclidean')
classifier.fit(X_train,Y_train)


# In[23]:


Y_pred  = classifier.predict(X_test)
Y_pred


# In[24]:


cm = confusion_matrix(Y_test,Y_pred)
cm


# In[25]:


f1_score(Y_test,Y_pred)


# In[27]:


accuracy_score(Y_test,Y_pred)


# # Accuracy of the fitted model is 80 %
