# -*- coding: utf-8 -*-

# In[1]:

import numpy
from random import sample
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# In[2]: 

def load_data():
   
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
            local_model = define_model()
            local_model.set_weights(global_weights)
            local_model.fit(X[client], Y[client], epochs=num_epoch, batch_size=bs, verbose=1, validation_split=0.2)
            local_weights_list.append(local_model.get_weights())

        global_weights = numpy.mean(local_weights_list, axis=0)
        global_model.set_weights(global_weights)
        _, _, accuracy = global_model.evaluate(testX, testY, verbose=0)
        print("Global model accuracy: %.2f" % (accuracy * 100))

fed_learn()
