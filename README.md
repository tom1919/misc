# misc

- esm_arima_functions.py
	- Functions for computing forecast error (MAE and MASE) of Exponential 
	  Smoothing and Arima model using walk foward cross validation.
	  
- stock_forecasting_error.py
	- Make one day ahead predictions for stocks using ESM and ARIMA models and test
	  if predictions are any better than naive forecasts of predicting the next day
	  price is the current days price
	- download stock data and apply the functions in esm_arima_functions.py

- volatility_forecasting.py
	- forecast variance of stock returns
	
- power_ttest.Rmd
	- Example code for A/B test for difference between means using a t-test

