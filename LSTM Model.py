#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:59:00 2021

@author: Elite Traders
@Motivation: There has been signficant correlation between the dotcom bubble and current market rally. We can potentially 
             potentially predict nasdaq direction with high confidence if we train the model dating back to dotcom era.
@Goal: Intention of this model is to predict the next few weeks of Nasdaq index to better equip on hedging the portoflio
       incase of a big anticipated drawdown.
@Algorithm: LSTM
@Research Paper reference: https://ieeexplore.ieee.org/document/8489208

"""

import pandas as pd
import numpy as np
import yfinance as yf
import copy
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
import sklearn.metrics as sm

def calculate_techincal_indicators(df):
    df_with_ti = copy.deepcopy(df)
    #Add 8-day, 20-day and 200-day EMAs
    df_with_ti["ema_8"] = df_with_ti.Close.ewm(span=8).mean().fillna(0)
    df_with_ti["ema_20"] = df_with_ti.Close.ewm(span=20).mean().fillna(0)
    df_with_ti["ema_200"] = df_with_ti.Close.ewm(span=200).mean().fillna(0)
    
    #Add MACD - a Trend following trailing indicators.
    df_with_ti["ema_12"] = df_with_ti.Close.ewm(span=12).mean().fillna(0)
    df_with_ti["ema_26"] = df_with_ti.Close.ewm(span=26).mean().fillna(0)
    df_with_ti["macd"] = df_with_ti["ema_12"]-df_with_ti["ema_26"]
    df_with_ti.drop(['ema_12', 'ema_26'], axis =1) #drop the transient column used for macd as they are not part of the feature
    
    #Add RSI a momentum indicator to identify if index is overbought or oversold.
    df_with_ti["rsa_14"] = talib.RSI(df_with_ti.Close.values, timeperiod = 14)
    df_with_ti["rsa_9"] = talib.RSI(df_with_ti.Close.values, timeperiod = 9)
    
    #Add OBV a leading momentum indicator that gives volume queues combined with the other indicators
    df_with_ti["obv"] = talib.OBV(df_with_ti.Close, df_with_ti.Volume)
    
    #BB a momentun indicator over 21 days moving average and 2 standard deviation
    upperband, middleband, lowerband = talib.BBANDS(df_with_ti.Close, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
    df_with_ti["bb_upper"] = upperband
    df_with_ti["bb_lower"] = lowerband
    
    #plot the index values
    plt.figure(figsize=(12,8))
    plt.plot(df_with_ti['Close'], label= 'Actual')
    plt.plot(df_with_ti['ema_8'], label= '8 days ema')
    plt.plot(df_with_ti['ema_20'], label= '20 days ema')
    plt.plot(df_with_ti['ema_200'], label= '200 days ema')
    plt.legend(loc='best')
    plt.plot()
    
    df_with_ti = df_with_ti.fillna(0)
    df_with_ti.info()
    
    print("The dataset has {} samples and {} features".format(df_with_ti.shape[0],df_with_ti.shape[1]))
    return df_with_ti


#import the Nasdaq composite data during dotcome bubble from 3/1/1997 till most recent date 9/30/2018
df = yf.download(tickers="^IXIC",interval='1wk', start='1991-1-1', end='2018-9-30')
df.head()

#check of null data and outliers
df.isnull().sum() #Since no invalid data, no need to clean it up.

print("Total number of weeks for training: {}".format(df.shape[0]))

df_with_ti = calculate_techincal_indicators(df)

values = df_with_ti.values
values = values.astype('float32')
values
print("Min: ", np.min(values))
print("Max: ", np.max(values))

#final_df = prepare_data(df_with_ti, 60, 3)
final_df = df_with_ti
final_df.head(4)

#split into train and test dataset
val = final_df.values
train_sample = int(len(val)*0.8)
train = val[:train_sample,:]
test = val[train_sample:,:]

print(train.shape, test.shape)

X,y = train, test
sc = MinMaxScaler()
X = sc.fit_transform(X)
X

X_train = []
y_train = []
for i in range(60, X.shape[0]):
    X_train.append(X[i-60:i])
    y_train.append(X[i,3])
    if i<=61:
        print(X_train)
        print('\n')
        print(y_train)
        print()


X_train,y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)

#create LSTM model
#model = Sequential()
#model.add(LSTM(75,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
#model.add(LSTM(30,return_sequences=True))
#model.add(LSTM(30,return_sequences=True))
#model.add(Dense(1,kernel_initializer="normal",activation="linear"))
#model.compile(loss="mae",optimizer ="adam",metrics=["accuracy"])
#model.summary()

NUM_NEURONS_FirstLayer = 1000
NUM_NEURONS_SecondLayer = 500


opt = SGD(lr=0.001, momentum=0.9)

#Build the model
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=opt)

X_test = []
y_test = []
y= sc.transform(y)
for i in range(60, y.shape[0]):
    X_test.append(y[i-60:i])
    y_test.append(y[i,3])
    if i<=61:
        print(X_test)
        print('\n')
        print(y_test)
        print()

X_test, y_test = np.array(X_test), np.array(y_test)
fitted_model = model.fit(X_train,y_train,epochs=50,batch_size=10, validation_data= (X_test, y_test), shuffle=False)

#plot the index values
plt.figure(figsize=(12,8))
plt.plot(fitted_model.history["loss"], label= 'train')
plt.plot(fitted_model.history["val_loss"], label= 'test')
plt.legend(loc='best')
plt.plot()


y_predict = model.predict(X_test)

sc.scale_
normal_scale = 1/sc.scale_[3]
y_test = y_test*normal_scale
y_predict = y_predict*normal_scale

#mean_y_test = y_test.mean()
#mean_y_predict = y_predict.mean()
#print(mean_y_test,mean_y_predict)

#accuracy = round((mean_y_predict/mean_y_test)*100,2)
#accuracy
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_predict), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_predict), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_predict), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_predict), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_predict), 2))

#predict for the year 2020 till date.
new_df = yf.download(tickers="^IXIC",interval='1wk', start='2019-1-1', end='2021-3-06')
new_df.head()

#check of null data and outliers
new_df.isnull().sum() #Since no invalid data, no need to clean it up.

print("Total number of weeks for training: {}".format(new_df.shape[0]))

new_df_with_ti = calculate_techincal_indicators(new_df)
new_val = new_df_with_ti.values
new_X = new_val
new_X = sc.transform(new_X)

new_X_predict = []
new_y = []
for i in range(60, new_X.shape[0]):
    new_X_predict.append(new_X[i-60:i])
    new_y.append(new_X[i,3])
    if i<=61:
        print(new_X_predict)
        print('\n')
        print(new_y)
        print()


new_X_predict,new_y = np.array(new_X_predict), np.array(new_y)
new_y_predict = model.predict(new_X_predict)

new_y = new_y*normal_scale
new_y_predict = new_y_predict*normal_scale

#plot the index values
plt.figure(figsize=(12,8))
plt.plot(new_y, label= 'Actual')
plt.plot(new_y_predict, label= 'Predicted')
plt.legend(loc='best')
plt.plot()










