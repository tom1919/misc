# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:39:33 2019

@author: tommy
"""

# =============================================================================
# Make one day ahead predictions for stocks using ESM and ARIMA models and test
# if predictions are any better than naive forecasts of predicting the next day
# price is the current days price 
# =============================================================================

#%%
# load libraries and functions
import pandas as pd
from pandas_datareader import data # get stock data
import os
os.chdir('C:\\Users\\tommy\\Google Drive\\misc')
import esm_arima_functions as eaf

#%%
# download stock data, and subset for clossing prices

tickers = ['^GSPC', 'AAPL', 'GOOGL', 'FIT']
panel_data = data.DataReader(tickers, 'yahoo', '2018-01-25', '2019-01-25')

close_prices = panel_data['Close']
    
#%%
# compute MAE and MASE from ESM and ARIMA models that make 1 step ahead pred.
mae_mase_df = eaf.esm_arima_df(close_prices)

#%%
# subset for any stocks where esm or arima could make better predictions
# than a random walk. If MASE is less than 1 then the preidctions are better
# than a naive / random walk model 

mae_mase_df.loc[(mae_mase_df.ses_mase < 1) 
                | (mae_mase_df.trend_mase < 1) 
                | (mae_mase_df.dtrend_mase < 1)
                | (mae_mase_df.arima_mase < 1)]