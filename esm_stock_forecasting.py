# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:45:32 2019

@author: tommy
"""

# =============================================================================
# The purpose of this script is to make 1 step ahead forecasts of stock returns
# using Exponential Smoothing Models and compute MASE for evaluating accuarcy 
# =============================================================================

#%%
# load libraries

from pandas_datareader import data # get stock data
import pandas as pd
import numpy as np
import statsmodels as sm # esm models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, \
ExponentialSmoothing
from sklearn.metrics import mean_absolute_error


#%%
# download stock data, convert to returns and subset 150 trading days

tickers = ['^GSPC', 'AAPL', 'GOOGL', 'FIT']
panel_data = data.DataReader(tickers, 'yahoo', '2018-01-25', '2019-01-25')

close_prices = panel_data['Close']

returns = close_prices.pct_change(periods = 1).tail(150)


#%%
# =============================================================================
# esm_mase function takes a series as input. It creates 60 1 step ahead 
# forecasts using a moving window and 3 types of ESM models. It outputs the MASE 
# for the 3 models and the MAE for naive forecast as a list
# =============================================================================

def esm_mase(ts):

    test_n = 60
    ses = []
    trend = []
    dtrend = []
    j=0
    
    for i in range(test_n,0,-1): #(60,59,58...3,2,1)
        # moving window, walk foward 1 step 
        train = np.asarray(ts[j:len(ts)-i])
        j= j+1
        
        # 3 different types of ES models. Each one makes 1 step ahead predictions
        ses.append(SimpleExpSmoothing(train).fit(optimized = True).\
                   forecast(1)[0])
    
        trend.append(ExponentialSmoothing(train, 
                                    trend='add',
                                    damped=False,
                                    seasonal='None').fit(optimized = True).\
                                     forecast(1)[0])
        
        dtrend.append(ExponentialSmoothing(train, 
                                    trend='add',
                                    damped=True,
                                    seasonal='None').fit(optimized = True).\
                                     forecast(1)[0])
        
    print('done with step: ', j-1)
    
    test = ts.tail(test_n) # test set
    # naive forecast predicts no change in price aka return = 0
    naive_mae = mean_absolute_error([0] *test_n, test)
    
    # calculate mase
    mase_ses = mean_absolute_error(ses, test) / naive_mae
    mase_trend = mean_absolute_error(trend, test) / naive_mae
    mase_dtrend = mean_absolute_error(dtrend, test) / naive_mae
    
    metrics = [mase_ses, mase_trend, mase_dtrend, naive_mae]
    return(metrics)
#%%
# loop through each column in returns and apply esm_mase function
    
mase_list = []
for column in returns:
    mase_list.append(esm_mase(returns[column]))

#%%
# create a df for the mase values for stock forecasts

ts_names = returns.columns.values
mase_df = pd.DataFrame(mase_list, columns=['ses','trend','dtrend','naive_mae'],
                       index = ts_names)

# subset any mase values that indicate the forecasts are better than baseline
mase_df.loc[(mase_df.ses < 1) | (mase_df.trend < 1) | (mase_df.dtrend < 1)]




#%%


def esm_forecast(ts):
    
    ts = ts.to_frame()
    # create a column for log returns
    ts['log_return'] = np.log(ts.iloc[:,0]) - np.log(ts.iloc[:,0].shift(1))
    
    # list to store 1-step ahead forecasts. 60 days used for training and first 
    # row of returns is NA so the first 61st obs are not forecasted. 
    # 0s used as placeholder
    forecast = [0]*61 
    start = 1
    end = 61

    for i in range(len(ts)-61):
        # rolling window of 60 obs used for train set
        train = np.asarray(ts.log_return[start:end])
        start = start+1
        end = end+1
        
        # append the 1 step ahead prediction to forecast
        forecast.append(SimpleExpSmoothing(train).fit(optimized = True).\
                        forecast(1)[0])
    
    # create a col for forecasted prices. 
    ts['forecast'] = np.exp(forecast) * ts.iloc[:,0].shift(1)
    
    # return a series of forecasted prices that is the same length as orginal 
    # ts but the rows 0-60 aren't valid forecasts. they were used for training
    return(ts['forecast'])

#%%
ts = aapl.AAPL

esm_forecast(ts)










