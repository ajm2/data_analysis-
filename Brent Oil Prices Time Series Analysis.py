#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


##Importing the data 
df = pd.read_csv('BrentOilPrices.csv', index_col=0, parse_dates=True)
df.head()


# In[3]:


df.tail()


# In[95]:


df.shape


# In[4]:


##Descriptive Statistics
df.describe().transpose()


# In[5]:


##Looking for any null values 
df.info()


# In[6]:


##Add columns with year, month, and weekday 
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.weekday_name

df.sample(5, random_state = 0)


# In[45]:


##Visualizing the data overtime 
import seaborn as sns
##Use seaborn style deafaults and set the default figure size
sns.set(rc={'figure.figsize':(12,6)})
df['Price'].plot(linewidth=1)


# In[8]:


##Lets visualize the data from a single year 2019
seas = df.loc['2019', 'Price'].plot()
seas.set_ylabel('Price($)')


# In[9]:


from statsmodels.tsa.arima_model import ARIMA
##Lets try forecasting with an ARIMA model
model = ARIMA(df['Price'], order=(1,1,0))
model_fit = (model.fit(disp=False))
print(model_fit.summary())
##AR(1) model appears to perform the best 


# In[10]:


prediction = model_fit.predict(len(df), len(df))
print(prediction)


# In[19]:


##Taking log of the data
df_log = np.log(df['Price'])
plt.plot(df_log)


# In[14]:


rolling_mean = df['Price'].rolling(window = 12).mean()
rolling_std = df['Price'].rolling(window = 12).std()

plt.plot(df['Price'], color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'orange', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()


# In[15]:


result = adfuller(df['Price'])

print('ADF Statistic: {}'.format(result[0]))
print('P-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
##Series is not stationary with a p-value >.05


# In[39]:


def get_stationarity(timeseries):
    #rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    #rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(timeseries, color = 'green', label='Rolling Mean')
    std = plt.plot(timeseries, color = 'red', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Std')
    plt.show(block=False)
    
    #DickeyFuller Test
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critcal Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


# In[40]:


rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)

get_stationarity(df_log_minus_mean)
##After subtracting the rolling mean we can see the process is stationary


# In[34]:


##Exponential decay is used for transforming timeseries that is stationary
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0,
                                   adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)

get_stationarity(df_log_exp_decay)


# In[69]:


df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)

get_stationarity(df_log_shift)


# In[74]:


decomposition = seasonal_decompose(df_log, freq=100)
model = ARIMA(df_log, order=(2,1,2))
results = model.fit(disp=-1)
plt.plot(df_log_shift)
plt.plot(results.fittedvalues, color='red')


# In[78]:


predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log.iloc[0],
                                 index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,
                         fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df['Price'])
plt.plot(predictions_ARIMA)


# In[98]:


##With this model we can say with 95% confidence that 
results.plot_predict(1, 8246)

