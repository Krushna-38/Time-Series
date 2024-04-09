# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:11:01 2024

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('dark_background')

#load the dataset
df=pd.read_csv('AirPassengers.csv')
df.columns
df=df.rename({'#Passengers':'Passengers'}, axis=1)
print(df.dtypes)

#Month is text and passengers in int
#Now let us convert into date and time
df['Month']=pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month', inplace=True)

plt.plot(df.Passengers)
#There is increasing trend and it has got seasonality
#Is the data stationary
#Dickey-Fuller test

from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_=adfuller(df)
print("pvalue= ", pvalue, "if above 0.05, data is not stationary")
#Since the data is not stationary we may need SARIMA and not just ARIMA
#Now let us extract the year and month from the date and time column

df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years = df['year'].unique()

#Plot yearly and monthly values as boxplot
sns.boxplot(x='year', y='Passengers', data=df)
#No. of passengers are going up year by year
sns.boxplot(x='month',y='Passengers', data=df)
#Over all there is higher trend in july and August compared to rest of the

#Extract and plot trend, seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df['Passengers'], model='additive')

#Additive time series:
#Value= base level + Trend + Seasonality +Error
#Multiplicative Time Series:
    #Value = Baselevel x Trend x Seasonality x Error
    
trend= decomposed.trend
seasonal = decomposed.seasonal #Cyclic behaviour may not be seasonal
residual =decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passengers'],label='original',color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plo(residual,label='Residual', color='yellow')
plt.legend(loc='upper left')
plt.show()
"""
Trend us going upfrom 1950s to 60s
It is highly seasonal showing peaks at particular interval
This helps to select specific prediction model
"""

#AUTOCORRELATION
#Values are not correlated with x-axis but with its lag
#Meaning yesterdays value is depend on day before yesterday so on so  forth
#Autocorrelation is simply thecorrelation of a series with its own lags.
#Plot lag on x axis and correlation on y axis
#Any correlation above confidence lnes are statistically significan


from statsmodels.tsa.stattols import acf

acf_144=acf(df.Passengers, nlags=144)
plt.plot(acf_144)
#Auto correlation above zero means positive correlation and below as negative
#Obtain the same but with single line and more info..
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)
#Any lag before 40 has positive correlation
#Horizontal bands indicate 95% and 99% (dashed) confidence bands

