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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD
import sklearn.metrics as sm
from pickle import load
from pickle import dump


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


def train_model(ticker, no_historic_steps=60, skip_forward=0, y_index=3, start_date='2001-1-1',end_date='2019-12-30'):
    
    df = yf.download(tickers=ticker,interval='1d', start=start_date, end=end_date)
    df_with_ti = calculate_techincal_indicators(df)
    
    #split into train and test dataset
    val = df_with_ti.values
    train_sample = int(len(val)*0.8)
    train = val[:train_sample,:]
    test = val[train_sample:,:]
    
    print(train.shape, test.shape)
    
    X,y = train, test
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    y= sc.transform(y)
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(no_historic_steps, X.shape[0]-skip_forward):
        X_train.append(X[i-no_historic_steps:i])
        y_train.append(X[i+skip_forward,y_index])
        if i<=61:
            print(X_train)
            print('\n')
            print(X_test)
            print()
            
    for i in range(no_historic_steps, y.shape[0]-skip_forward):
        X_test.append(y[i-no_historic_steps:i])
        y_test.append(y[i+skip_forward,y_index])
        if i<=61:
            print(y_train)
            print('\n')
            print(y_test)
            print()   
            
    
    
    X_train,y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape, y_train.shape)
    
    NUM_NEURONS_FirstLayer = 1000
    NUM_NEURONS_SecondLayer = 500
    opt = SGD(lr=0.01, momentum=0.9)
    
    #Build the model
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=opt) 
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    fitted_model = model.fit(X_train,y_train,epochs=50,batch_size=10, validation_data= (X_test, y_test), shuffle=False)
    
    #plot the index values
    plt.figure(figsize=(12,8))
    plt.plot(fitted_model.history["loss"], label= 'train')
    plt.plot(fitted_model.history["val_loss"], label= 'test')
    plt.legend(loc='best')
    plt.plot()
    
    
    y_predict = model.predict(X_test)
    #y_predict = y_predict.reshape(y_predict.shape[0],number_of_steps)
    
    min = sc.data_min_[y_index]
    max = sc.data_max_[y_index]
    y_test = (y_test*(max-min))+min
    y_predict = (y_predict*(max-min))+min
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_predict), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_predict), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_predict), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_predict), 2)) 
    print("R2 score =", round(sm.r2_score(y_test, y_predict), 2))
    
    model.save('nasdaq_gene')
    dump(sc, open('scaler', 'wb'))

train_model("^IXIC", skip_forward=4)

#Load the model
model = tf.keras.models.load_model('nasdaq_gene')
sc = load(open('scaler', 'rb'))

#predict for the year 2020 till date.
new_df = yf.download(tickers="^IXIC",interval='1d', start='2020-01-1', end='2021-3-08')
new_df.head()

#check of null data and outliers
new_df.isnull().sum() #Since no invalid data, no need to clean it up.

print("Total number of days for training: {}".format(new_df.shape[0]))

new_df_with_ti = calculate_techincal_indicators(new_df)
new_val = new_df_with_ti.values
new_X = new_val
new_X = sc.transform(new_X)

new_X_predict = []
new_y = []
fast_forward = 4
for i in range(60, new_X.shape[0]-fast_forward):
    new_X_predict.append(new_X[i-60:i])
    new_y.append(new_X[i+fast_forward,3])
    if i<=61:
        print(new_X_predict)
        print('\n')
        print(new_y)
        print()


new_X_predict,new_y = np.array(new_X_predict), np.array(new_y)
new_y_predict = model.predict(new_X_predict)
new_y_predict=new_y_predict.reshape(new_y_predict.shape[0])

max=sc.data_max_[3]
min=sc.data_min_[3]
new_y = (new_y*(max-min))+min
new_y_predict = (new_y_predict*(max-min))+min

#calculate actual vs predicted stats and their petcentage draw down
df_actual = pd.DataFrame(columns =["Index_Value", "percentage_difference"])
df_predicted = pd.DataFrame(columns=["Index_Value", "percentage_difference"])
df_merged = pd.DataFrame(columns=["Actual", "predicted"])

df_actual["Index_Value"] = new_df_with_ti.iloc[60:len(new_df_with_ti)-fast_forward,3]
df_predicted["Index_Value"] = new_df_with_ti.iloc[60:len(new_df_with_ti)-fast_forward,3]
df_actual["percentage_difference"]=(1-(df_actual["Index_Value"].shift()/df_actual["Index_Value"]))*100

df_predicted["Index_Value"] = new_y_predict
df_predicted["percentage_difference"]=(1-(df_predicted["Index_Value"].shift()/df_predicted["Index_Value"]))*100
df_merged["Actual"] = df_actual["percentage_difference"]
df_merged["predicted"] = df_predicted["percentage_difference"]


#plot the index values
plt.figure(figsize=(12,8))
plt.plot(new_y, label= 'Actual')
plt.plot(new_y_predict, label= 'Predicted')
plt.legend(loc='best')
plt.plot()










