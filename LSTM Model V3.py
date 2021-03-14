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

import numpy as np
import yfinance as yf
import copy
import talib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm1
import sklearn.metrics as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD
from pickle import load
from pickle import dump
import datetime


def calculate_techincal_indicators(df):
    df_with_ti = copy.deepcopy(df)
    #Add 8-day, 20-day and 200-day EMAs
    #df_with_ti["ema_8"] = df_with_ti.Close.ewm(span=8).mean().fillna(0)
    #df_with_ti["ema_20"] = df_with_ti.Close.ewm(span=20).mean().fillna(0)
    #df_with_ti["ema_200"] = df_with_ti.Close.ewm(span=200).mean().fillna(0)
    
    #Add MACD - a Trend following trailing indicators.
    #df_with_ti["ema_12"] = df_with_ti.Close.ewm(span=12).mean().fillna(0)
    #df_with_ti["ema_26"] = df_with_ti.Close.ewm(span=26).mean().fillna(0)
    #df_with_ti["macd"] = df_with_ti["ema_12"]-df_with_ti["ema_26"]
    #df_with_ti=df_with_ti.drop(['ema_12', 'ema_26'], axis =1) #drop the transient column used for macd as they are not part of the feature
    
    #Add RSI a momentum indicator to identify if index is overbought or oversold.
    df_with_ti["rsa_14"] = talib.RSI(df_with_ti.Close.values, timeperiod = 14)
    df_with_ti["rsa_9"] = talib.RSI(df_with_ti.Close.values, timeperiod = 9)
    
    #Add OBV a leading momentum indicator that gives volume queues combined with the other indicators
    df_with_ti["obv"] = talib.OBV(df_with_ti.Close, df_with_ti.Volume)
    
    #BB a momentun indicator over 21 days moving average and 2 standard deviation
    #upperband, middleband, lowerband = talib.BBANDS(df_with_ti.Close, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
    #df_with_ti["bb_upper"] = upperband
    #df_with_ti["bb_lower"] = lowerband
 
    
    df_with_ti = df_with_ti.fillna(0)
    df_with_ti=df_with_ti.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis =1)
    print("The dataset has {} samples and {} features".format(df_with_ti.shape[0],df_with_ti.shape[1]))
    return df_with_ti

def train_model(ticker, no_historic_steps=60, skip_forward=1, y_index=0, start_date='2001-1-1',end_date='2020-12-31', is_build_mode=True):
    
    '''
    skip_forward=5
    start_date='2016-1-1'
    end_date='2020-12-31'
    is_build_mode=True
    ticker = "^IXIC"
    no_historic_steps=60
    y_index=1
    '''
    
    
    df = yf.download(tickers=ticker,interval='1d', start=start_date, end=end_date)
    df_with_ti = calculate_techincal_indicators(df)
    
    if is_build_mode:
        train_percent = 0.8
    else:
        train_percent = 1
    #split into train and test dataset
    val = df_with_ti.values
    sc = MinMaxScaler()
    sc.fit(val)
    train_sample = int(len(val)*train_percent)
    train = val[:train_sample,:]
    test = val[train_sample:,:]
    
    print(train.shape, test.shape)
    
    X = train
    X = sc.transform(X)
    
    X_train = []
    y_train = []
    for i in range(no_historic_steps, X.shape[0]-skip_forward):
        X_train.append(X[i-no_historic_steps:i])
        y_train.append(X[i+skip_forward,y_index])
        if i<=61:
            print(X_train)
            print('\n')
            print(y_train)
            print()
      
    if is_build_mode:
        y=test
        y= sc.transform(y)
        X_test = []
        y_test = []
        for i in range(no_historic_steps, y.shape[0]-skip_forward):
            X_test.append(y[i-no_historic_steps:i])
            y_test.append(y[i+skip_forward,y_index])
            if i<=61:
                print(X_test)
                print('\n')
                print(y_test)
                print()   
                
    
    
    X_train,y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape, y_train.shape)
    
    NUM_NEURONS_FirstLayer = 500
    NUM_NEURONS_SecondLayer = 250
    opt = SGD(lr=0.001, momentum=0.9)
    
    #Build the model
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=opt) 
    
    if is_build_mode:
        X_test, y_test = np.array(X_test), np.array(y_test)
        fitted_model = model.fit(X_train,y_train,epochs=20,batch_size=10, validation_data= (X_test, y_test), shuffle=False)
    else:
        fitted_model = model.fit(X_train,y_train,epochs=20,batch_size=10, shuffle=False)
    
    model.save('nasdaq_gene')
    dump(sc, open('scaler', 'wb'))
    
    #plot the index values
    plt.figure(figsize=(12,8))
    plt.plot(fitted_model.history["loss"], label= 'train')
    if is_build_mode:
        plt.plot(fitted_model.history["val_loss"], label= 'test')
    plt.legend(loc='best')
    plt.plot()
    
    if is_build_mode:
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
    
def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    #ser = predicted
    #n=5
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm1.add_constant(x_scaled)
        model = sm1.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def predict(current_date, model, sb):
    #current_date = datetime.datetime.now().date()
    #next_day_model = tf.keras.models.load_model('Trained_Models/next/nasdaq_gene')
    #next_day_sc = load(open('Trained_Models/next/scaler', 'rb'))
    
    #five_days_model = tf.keras.models.load_model('Trained_Models/5/nasdaq_gene')
    #five_days_sc = load(open('Trained_Models/5/scaler', 'rb'))
    
    end_date = current_date + datetime.timedelta(days=1)
    start_date = (current_date - datetime.timedelta(weeks=52))
    #predict for the year 2020 till date.
    df = yf.download(tickers="^IXIC",interval='1d', start=str(start_date), end=str(end_date))
    df = calculate_techincal_indicators(df)
    
    #Next day prediction
    new_val = df.values
    new_X = new_val
    new_X = five_days_sc.transform(new_X)
    
    new_X_predict = []
    new_X_predict.append(new_X[new_X.shape[0]-60:new_X.shape[0]])
    new_X_predict = np.array(new_X_predict)
    
    '''
    predicted = next_day_model.predict(new_X_predict)
    next_day = predicted[0][0]
    max=next_day_sc.data_max_[3]
    min=next_day_sc.data_min_[3]
    next_day = (next_day*(max-min))+min
    '''
    
    predicted = five_days_model.predict(new_X_predict)
    five_days = predicted[0][0]
    max=five_days_sc.data_max_[0]
    min=five_days_sc.data_min_[0]
    five_days = (five_days*(max-min))+min
    
    return five_days


fast_forward = 5
train_model("^IXIC", skip_forward=fast_forward, start_date='2016-1-1',end_date='2021-3-1', is_build_mode=False)
#current_date = datetime.datetime.now().date()
    
five_days_model = tf.keras.models.load_model('model/nasdaq_gene')
five_days_sc = load(open('model/scaler', 'rb'))


df = yf.download(tickers="^IXIC",interval='1d', start=datetime.date(2021,3,1), end=datetime.date(2021,3,13))
df_results = df[["Close"]]
predicted = []
slope_list=[]
for index, row in df_results.iterrows():
    current = index.date()
    #earlier = index.date() - datetime.timedelta(days=5)
    predicted.append((predict(current, five_days_model, five_days_sc)))
 
predicted=np.array(predicted)
slope_predicted = slope(predicted,5)
for i in range(0, len(slope_predicted)):
    slope_list.append(slope_predicted[i])

df_results["predicted"] = predicted
df_results["slope"] = slope_list
df_results["actual_change"] = ((df_results.Close.shift(-5)/df_results.Close)-1)*100
df_results["predicted_change"] = ((df_results.predicted/df_results.predicted.shift(5))-1)*100
#df_results["predicted_change"] = df_results.predicted_change.shift(-5)
#ådf_results["five_days_predicted"] = five_d
df_results["Close"] = df_results.Close.shift(-5)
df_results[["Close","predicted"]].plot()

print(df_results[["predicted","slope"]])


#print((predict(current)/predict(earlier)-1)*100)













