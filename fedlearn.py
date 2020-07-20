# -*- coding: utf-8 -*-
# In[1]:


import sys
import warnings
warnings.simplefilter(action='ignore')

from keras.datasets import fashion_mnist
import numpy
from keras.utils import to_categorical
from random import sample
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os;

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


# In[2]: Loading the dataset and Normalizing

#def normalize(data):
#    data_norm = data.astype('float32')
#    data_norm = data_norm/255.0
#    return data_norm

def load_data():
    #(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    #trainX = trainX.reshape((trainX.shape[0], 28*28))
    #testX = testX.reshape((testX.shape[0], 28*28))

    #trainY = to_categorical(trainY)
    #testY = to_categorical(testY)

    #trainX, testX = normalize(trainX), normalize(testX)
        
    data = pd.read_csv(r"C:\Users\JOSH\Downloads\weatherHistory.csv", header=None)
    new_header = data.iloc[0]
    data = data[1:]
    data.columns = new_header
    data.drop(['Formatted Date','Summary','Precip Type','Apparent Temperature (C)','Pressure (millibars)','Daily Summary','Loud Cover','Wind Bearing (degrees)','Visibility (km)'], axis=1, inplace=True)
    data = data.astype(float)
    
    X=data[['Humidity','Wind Speed (km/h)']]
    y=data['Temperature (C)']
    
    #X = np.array(y).reshape((-1,1))
    y = np.array(y).reshape((-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    print(scaler_x.fit(X))
    xscale=scaler_x.transform(X)
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

    return X_train, y_train, X_test, y_test

# In[3]:

def partition_data(trainX, trainY, num):
    ran_order = sample(range(0, trainX.shape[0]), trainX.shape[0])
    local_size=int(trainX.shape[0]/num)
    partitionedX=[]
    partitionedY=[]
    for i in range(0,num):
        partitionedX.append(trainX[ran_order[i*local_size:(i+1)*local_size]])
        partitionedY.append(trainY[ran_order[i*local_size:(i+1)*local_size]])

    return numpy.array(partitionedX), numpy.array(partitionedY)



# In[4]:
    
def save_data(partitionedX, partitionedY, testX, testY):
    for i in range(0, partitionedX.shape[0]):
        numpy.savetxt(r"C:\Users\JOSH\Downloads\localX_"+str(i)+".csv", partitionedX[i], fmt="%i", delimiter=",")
        numpy.savetxt(r"C:\Users\JOSH\Downloads\localY_"+str(i)+".csv", partitionedY[i], fmt="%i", delimiter=',')
    numpy.savetxt(r"C:\Users\JOSH\Downloads\testX.csv", testX, fmt="%i", delimiter=",")
    numpy.savetxt(r"C:\Users\JOSH\Downloads\testY.csv", testY, fmt="%i", delimiter=",")


# In[5]:

def define_model():
        model = Sequential()
        model.add(Dense(16, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))

        #opt = SGD(lr = 0.01, momentum = 0.9)
        #model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

        return model


# In[6]:



def fed_learn():
    num_clients = 5
    num_run = 10
    num_epoch = 5
    bs = 10

    trainX, trainY, testX, testY = load_data()
    X, Y = partition_data(trainX, trainY, num_clients)

    ## build global model
    global_model = define_model()
    global_weights = global_model.get_weights()

    for run_round in range(0, num_run):
        print("This is run ", run_round)
        local_weights_list = list()

        for client in range(0, num_clients):
            ##print("For client ", client)
            local_model = define_model()
            local_model.set_weights(global_weights)
            
            #local_model.fit(X_train, y_train, epochs=100, batch_size=50,  verbose=1, validation_split=0.2)

            local_model.fit(X[client], Y[client], epochs=num_epoch, batch_size=bs, verbose=1, validation_split=0.2)
            ##_, accuracy = local_model.evaluate(X[client], Y[client], verbose=0)
            ##print("Accuracy: %.2f"%(accuracy*100))

            local_weights_list.append(local_model.get_weights())

        global_weights = numpy.mean(local_weights_list, axis=0)
        global_model.set_weights(global_weights)
        _, accuracy = global_model.evaluate(testX, testY, verbose=0)
        print("Global model accuracy: %.2f" % (accuracy * 100))

fed_learn()


# In[7]:


