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
        train = np.asarray(ts[0+j:len(ts)-i])
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



















