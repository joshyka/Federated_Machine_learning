#!/usr/bin/env python
# coding: utf-8

# In[10]:


from numpy.random import seed
seed(42)

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# In[11]:


import os;
p = r"/home/oem/Scaleout/TestML/Temperature_Prediction_ML_Model"
os.chdir(p)
os.getcwd()


# In[12]:


data = pd.read_csv(r"/home/oem/Scaleout/TestML/Temperature_Prediction_ML_Model/weatherHistory.csv", header=None)
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header
data.drop(['Formatted Date','Summary','Precip Type','Apparent Temperature (C)','Pressure (millibars)','Daily Summary','Loud Cover','Wind Bearing (degrees)','Visibility (km)'], axis=1, inplace=True)
data = data.astype(float)
data



# In[13]:


sns.pairplot(data)


# In[14]:


X=data[['Humidity','Wind Speed (km/h)']]
y=data['Temperature (C)']


# In[15]:


#X = np.array(y).reshape((-1,1))
y = np.array(y).reshape((-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X))
xscale=scaler_x.transform(X)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


# In[17]:


model = Sequential()
model.add(Dense(16, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[18]:


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


# In[19]:


history = model.fit(X_train, y_train, epochs=100, batch_size=50,  verbose=1, validation_split=0.2)


# In[21]:


from keras.models import Sequential
accuracy = model.evaluate(X_train, y_train, verbose=1)


# In[22]:


print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[23]:


Xn = data[['Humidity','Wind Speed (km/h)']]
Xnew = Xn[20:40]
Yn = data[['Temperature (C)']]
Yactual = Yn[20:40]

print(Xn)


# In[24]:


Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)

ynew = scaler_y.inverse_transform(ynew) 
Xnew = scaler_x.inverse_transform(Xnew)
print([Yactual, ynew])
