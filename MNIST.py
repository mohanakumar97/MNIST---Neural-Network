#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import keras as ke
import pandas_profiling as pp
from keras.datasets import mnist


# In[2]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[3]:


x_train.shape


# In[4]:


plt.matshow(x_train[0])


# In[5]:


y_train


# In[6]:


x_train.max()


# In[7]:


x_train=x_train/255
x_test=x_test/255


# In[8]:


x_train[1]


# In[9]:


from keras.models import Sequential


# In[10]:


from keras.layers import Dense,Activation,Flatten


# In[11]:


model=Sequential()


# In[12]:


model.add(Flatten(input_shape=[28,28]))


# In[14]:


model.add(Dense(70,activation='relu'))


# In[15]:


model.add(Dense(20,activation='softmax'))


# In[16]:


model.summary()


# In[17]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[18]:


model.fit(x_train,y_train)


# In[19]:


model.fit(x_train,y_train,epochs=10)


# In[20]:


plt.matshow(x_test[0])


# In[21]:


pred=model.predict(x_test)


# In[22]:


pred


# In[23]:


pred[0]
np.argmax(pred[0])


# In[24]:


model.evaluate(x_test,y_test)


# In[25]:


predictions = model.predict(x_test)


# In[26]:


predictions


# In[ ]:




