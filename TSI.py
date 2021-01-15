#!/usr/bin/env python
# coding: utf-8

# # Total Solar Irradiance Data from NASA's Source Mission - Climate Science
# 
# Data characterisation and exploration
# 
# For more information regarding the raw data format:
# 
# http://lasp.colorado.edu/data/sorce/tsi_data/daily/sorce_tsi_L3_c24h_latest.txt 

# ## 1) Import relevant libraries

# In[75]:


import pandas as pd
import numpy as np
import matplotlib as mplt


# ## 2) Load data 
# 
# Took the data - http://lasp.colorado.edu/data/sorce/tsi_data/daily/sorce_tsi_L3_c24h_latest.txt
# 
# Through preprocessing on excel from a txt file to this:

# In[76]:


df = pd.read_csv("/Users/bakerkagimu/desktop/TSI.csv", );
df.head()


# In[72]:


TSIstats = df.copy()


# In[74]:


TSIstats.tail()


# ## 3) Source Empirical Daily Total Solar Irradiance (TSI) at 1-AU

# In[90]:


TSIdaily = df[['5']].copy()
TSIdaily


# ## 4) Itterative ARIMA estimation for TSI

# In[92]:


import pmdarima as pmd
def arimamodel(TSIdaily):
    autoarima_model = pmd.auto_arima(TSIdaily, 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              trace=True)
    return autoarima_model
arima_model = arimamodel(TSIdaily)
arima_model.summary()


# ## 5) Plot daily TSI

# In[139]:


from matplotlib import pyplot
TSIdaily.plot()
pyplot.show()


# ## 6) Values below 800 should be taken out, average TSI = 1365 w/m^2 at 1 AU 

# In[102]:


TSI = df[800<df['5']]


# In[103]:


print(TSI)


# In[95]:


TSI['5'].plot()


# ## 7 Violin plot provides:
# 
# - Median
# - Interquartile ranges
# - Distribution
# - Spread

# In[106]:


var = TSI['5']


# In[107]:


sns.violinplot(var)


# ## 8) Check distribution 

# In[108]:


import seaborn as sns
sns.distplot(TSI['5'])


# Would have hoped for normally distributed at 1365 w/m^2

# ## 9) Check for stationariy: Augmented Dickey-Fuller test
# 
# Null hypothesis that the time series is non-stationary, ie either of the mean, variance and covariance vary with time
# 
# Ideally we can reject the null hypothesis at the 1 - 10% significant level(s)

# In[110]:


from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
result = adfuller(var)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# ## 9a) ADF Test
# 
# We can reject the null hypothesis at the 1-10% confidence interval.
# 
# Low p-value suggests it is highly unlikely that the observed time series came about by chance

# ## 10) Source Standard deviation

# In[115]:


TSI_standard_deviation = df["8"]


# In[116]:


max(TSI_standard_deviation)


# In[120]:


min(TSI_standard_deviation)


# ## 11) Check Distribution of TSI without zero values

# In[143]:


var.hist()


# In[144]:


min(var)


# In[145]:


max(var)


# ## 12) Plot standard deviation

# In[122]:


standard_deviation.plot()


# ## 13) Total Uncertainty in TSI at Earth 

# In[146]:


Ucern = df['14']


# In[147]:


Ucern.plot()


# Need to remove zeros
