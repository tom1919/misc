# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 07:29:07 2019

@author: tommy
"""


#%%
# load libraries

from pandas_datareader import data # get stock data
import pandas as pd
import numpy as np
import statsmodels as sm # esm models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, \
ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from arch import arch_model
from random import gauss


#%%
# download stock data, convert to returns and subset 150 trading days

tickers = ['^GSPC', 'MSFT', 'GOOGL', 'FIT']
panel_data = data.DataReader(tickers, 'yahoo', '2015-01-25', '2019-01-25')

close_prices = panel_data['Close']

returns = close_prices.pct_change(periods = 1).tail(500)


#%%

foo2 = returns['MSFT'].values.tolist() 
foo3 = [x * 100 for x in foo2]

am = arch_model(foo3, p=1, q=1, o=1, mean='Constant', dist='StudentsT')
fit = am.fit(update_freq=15)
print(fit.summary())


am = arch_model(foo2,  p=1, q=1)

pred = fit.forecast(horizon = 10)
pred.variance
