# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:32:20 2019

@author: tommy
"""

# =============================================================================
# Functions for computing forecast error (MAE and MASE) of ESM models
# and Auto Arima model using walk foward cross validation.
# =============================================================================

#%%
# load libraries

import pandas as pd
import numpy as np
import statsmodels as sm # esm models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, \
ExponentialSmoothing
#The 'pyramid' package will be migrating to a new namespace beginning in 
#version 1.0.0: 'pmdarima'
from pyramid.arima import auto_arima
from sklearn.metrics import mean_absolute_error

#%%

def esm_arima(ts):
    """
    Create 60 1 day ahead forecast using a moving window training set of 60 days 
    for a single time series. Forecasts are created using 3 types of exponential 
    smoothing models and auto arima models. Calculates forecast error on the 60 
    out of sample forecasts.
    
    Parameters
    ----------
    ts : series
        a time series of returns (percent change from 1 day to the next)        
        
    Returns
    -------
    list
        mean absolute error and mean absolute scaled error for each model 
    
    """

    test_n = 60
    ses = []
    trend = []
    dtrend = []
    arima = []
    j=0
        
    for i in range(test_n,0,-1): #(60,59,58...3,2,1)
        # moving window, walk foward 1 step 
        train = np.asarray(ts[j:len(ts)-i])
        j= j+1
        
        # 3 different types of ESM models. Each 1 makes 1 step ahead predictions
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
        
        # Auto arima model makes 1 step ahead prediction.
        model = auto_arima(train, trace=False, error_action='ignore', 
                           suppress_warnings=True, max_p=15, max_q=15,
                           d=0, D=0, max_order=20, seasonal = False)
        model.fit(train)
        forecast = model.predict(n_periods=1)
        
        arima.append(forecast)
        
        print('done with step: ', j)
    
    test = ts.tail(test_n) # test set
    
    # naive forecast predicts no change in price aka return = 0
    naive_mae = mean_absolute_error([0] * test_n, test)
    
    # calculate MAE for all 4 model types
    ses_mae = mean_absolute_error(ses, test)
    trend_mae = mean_absolute_error(trend, test)
    dtrend_mae = mean_absolute_error(dtrend, test)
    arima_mae = mean_absolute_error(arima, test)
    
    # calculate MASE for all 4 model types
    ses_mase = ses_mae / naive_mae
    trend_mase = trend_mae / naive_mae
    dtrend_mase = dtrend_mae / naive_mae
    arima_mase = arima_mae / naive_mae
    
    # create list of all metrics
    metrics = [naive_mae, ses_mae, trend_mae, dtrend_mae, arima_mae,
               ses_mase, trend_mase, dtrend_mase, arima_mase]
    
    return(metrics)
#%%
    
def esm_arima_df(df):
    """
    Create 60 1 day ahead forecast using a moving window training set of 60 days 
    for a set of time series. Forecasts are created using 3 types of exponential 
    smoothing models and auto arima models. Calculates forecast error on the 60 
    out of sample forecasts.
    
    Parameters
    ----------
    df : dataframe
        each column of the df is a time series for prices of a financial asset         
        
    Returns
    -------
    dataframe
        mean absolute error and mean absolute scaled error for each model as 
        columns and asset name as row index.
    
    """
    # convert prices to returns and subset the last 120 days
    returns = df.pct_change(periods =1).tail(120)
    # replace inf values from division by zero to zero
    returns = returns.replace(np.inf, 0)
    returns = returns.replace(np.NINF, 0) # negative infinity
    
    i = 1 
    ncol = returns.shape[1]
    mae_mase_list = []
    
    # for each col in returns df, apply get_mae_mase function and save results 
    # to mae_mase_list
    for column in returns:
        mae_mase_list.append(esm_arima(returns[column]))
        print("done with column: " + column + ", # ", i, '/', ncol)
        i = i +1  

    ts_names = returns.columns.values
    
    # create a df with MAE and MASE values for each model and asset
    mae_mase_df = pd.DataFrame(mae_mase_list, 
                  columns = ['naive_mae', 'ses_mae', 'trend_mae', 'dtrend_mae', 
                             'arima_mae', 'ses_mase', 'trend_mase', 
                             'dtrend_mase', 'arima_mase'] ,
                  index = ts_names)
    
    return(mae_mase_df)