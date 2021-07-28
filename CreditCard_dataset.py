#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("Documents/creditcard.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df["Class"].value_counts()


# In[7]:


sn.countplot(df["Class"])


# In[8]:


df.describe()


# In[9]:


df.info


# In[10]:


corr=df.corr()


# In[11]:


corr


# In[12]:


######


# In[13]:


x=df.drop(["Class"],axis=1)
y=df["Class"]


# In[14]:


######convert data into balanced data


# In[17]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_res, y_res =ros.fit_resample(x,y)


# In[24]:


x_res.shape, y_res.shape


# In[19]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,random_state=1212121)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=641,random_state=0)


# In[22]:


rf.fit(x_train,y_train)


# In[23]:


rfpred=rf.predict(x_test)


# In[24]:


n_errors=(rfpred!=y_test)


# In[26]:


n_errors


# In[25]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score


# In[27]:


cf=confusion_matrix(rfpred,y_test)


# In[28]:


cf


# In[29]:


accuracy_score(rfpred,y_test)


# In[30]:


recall_score(rfpred,y_test)


# In[31]:


precision_score(rfpred,y_test)


# In[32]:


f1_score(rfpred,y_test)


# In[ ]:




