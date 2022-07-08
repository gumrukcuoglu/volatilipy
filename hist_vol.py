#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:08:28 2022

Simple code that imports data for four stocks from Yahoo Finance, estimates
actual historical volatility and compares different approaches.

- Time dependent: (annually computed)
 Close-to-close
 Parkinson
 Garman-Klass
 Rogers-Satchell

- Constant:
 Moving window (window size 30 days, can be changed)
 Exponentially weighted moving average

This is basically my solution to an exercise in Wilmott's Qualitative Finance.

@author: emirg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# The period for data, in human readable format.
# 'YYYY-MM-DD'
date_in = '1980-03-04'
date_out = '2020-07-05'

def human_to_posix_time(date:str):
    return str(int(datetime.datetime.strptime(date, '%Y-%m-%d').timestamp()))

def get_yahoo_data(period1:str, period2:str, stock:str):
    """
    Scrape Yahoo for historical stock data

    Parameters
    ----------
    period1 : str
        YYYY-MM-DD   -First date in the data.
    period2 : str
        YYYY-MM-DD   -Last date in the data.
    stock : str
        ticker.

    Returns
    -------
    pandas DataFrame
    
    """
    url = 'https://query1.finance.yahoo.com/v7/finance/download/'\
        + stock+'?period1=' + human_to_posix_time(period1)\
            + '&period2=' + human_to_posix_time(period2)\
        + '&interval=1d&events=history&includeAdjustedClose=true'
    return pd.read_csv(url, index_col=0, parse_dates=True)

def vol_acc(data):
    "Given daily stock data, estimates the close-to-close volatility"
    # The 252 factor annualises the volatility
    return np.sqrt( 252* np.log(data['Adj Close']).diff() .var() )
    
def vol_p(data):
    "Given daily stock data, estimates the Parkinson volatility"
    return np.sqrt( 252 / np.log(2)\
                   * (np.log(data['High'] / data['Low'])**2).mean() )

def vol_gk(data):
    "Given daily stock data, estimates the Garman-Klass volatility"
    return np.sqrt( 252*(
                  0.511 * np.log(data['High'] / data['Low'])**2
                  - 0.019 * np.log(data['Close'] / data['Open'])
                  * np.log(data['High']*data['Low'] / data['Open']**2)
                  -2 * np.log(data['High'] / data['Open'])
                  * np.log(data['Low'] / data['Open'])
                  ).mean())

def vol_rs(data):
    "Given daily stock data, estimates the Rogers-Satchell volatility"
    return np.sqrt( 252*(
                  np.log(data['High'] / data['Close'])
                  * np.log(data['High'] / data['Open'])
                  + np.log(data['Low'] / data['Close'])
                  * np.log(data['Low'] / data['Open'])
                  ).mean())

def vol_gk_alt(data):
    "Given daily stock data, estimates the Garman-Klass volatility (alternative)"
    # Not including it. I found it on the web but it looks like GK corrected
    # for drift, so it gives the exact same result as RS.
    return np.sqrt( 252*(
                  0.5 * np.log(data['High'] / data['Low'])**2
                  - (2 * np.log(2)-1) * np.log(data['Close'] / data['Open'])**2 
                  ).mean())

def daily_returns(data):
    "Calculate daily returns (from Adj Close value) given some data"
    # Daily Returns
    return (-data['Adj Close'].diff(-1) / data['Adj Close']).rename('Returns')

def vol_moving_window(R, N):
    "Estimate vol. using a moving window of N days, given daily returns data"
    
    # initialise the volatility Series
    vol = pd.Series(index=R.index, name='volatility', dtype=float)
    
    # Calculate variance for the first window
    vol[-2] = (R[-N:]**2).mean()
    
    # Then the remaining windows (going backwards):
    for i in range(2,len(R)-N+1):
        vol[-1-i] = vol[-i] + (R[-i-N]**2 - R[-i]**2)/N
 
    # Annualise, take the square root and remove the first N days
    vol = np.sqrt(252*vol[vol.notna()])
    return vol

    
def vol_exp_weighted(R, N, weight):
    "Estimate using exp. weighted moving average, given returns, using min N data"
    
    # initialise the volatility Series
    vol = pd.Series(index=R.index, name='volatility', dtype=float)
    
    # First point using the minimum days of N
    # The powers of the weights for the first N days: (lambda**(N-i))
    pows = weight**(N- pd.Series(index= R.index[:N], data = range(N)))
    # Calculate the variance at point N-1
    vol[N-1] = (1-weight)*(R[:N]**2*pows).mean()
    # Then use the recursion relation:
    for i in range(N, len(R)):
        vol[i] = weight * vol[i-1] + (1-weight)*R[i]**2
    # Annualise, take the square root and remove the first N days
    vol = np.sqrt(252*vol[vol.notna()])
    return vol

# The tickers to be analysed:
# !!! Note that code assumes 4 stocks for the plots
stocks = ['NOC', 'VRTX', 'CNC', 'BYD']

# These are the methods of estimation that I will use, and corresponding functions
method = {'acc': vol_acc, 'p': vol_p, 'gk': vol_gk, 'rs': vol_rs}

# In the second part, I will use 2 constant volatility estimation methods
method2={'exp weighted': vol_exp_weighted, 'moving window': vol_moving_window}
# Parameters for these methods:
# Method 1 (moving window): Size of the moving window in days
Nmov = 252

# Method 2 (exp. weighted moving average): Minimum number of points to average
Nexp = 252

 # Weight of earlier points
lam = 0.8
# The parameters for calling each method function
param={'moving window': (Nmov,), 'exp weighted': (Nexp, lam)}



# Let's start with initialising two figures
fig = plt.figure(figsize=(10,10))
fig2 = plt.figure(figsize=(20,10))


# For each stock, we first divide the data into yearly intervals
# and use the different methods to estimate the volatility.
for pl, stock in enumerate(stocks):
    if '.' in stock: # Let's not consider any foreign stocks
        continue
    # Read data
    data_stock = get_yahoo_data(date_in, date_out, stock)

    # For the second plot, we also calculate the returns
    returns = daily_returns(data_stock)
    
    # Years for which data available
    years = np.array(sorted(list(set(data_stock.index.year))))

    # Initialise annual volatility array, Nmethods x Nyears
    vol_annual = np.empty((len(method), len(years)), dtype=float)
    
    # Loop over years
    for i, year in enumerate(years):
        data_year = data_stock[data_stock.index.year==year]
        # Loop over methods
        for j, key in enumerate(method):
            vol_annual[j,i] = method[key](data_year)
    # Plot estimated annual volatility
    ax = fig.add_subplot(221+pl)
    # and the constant sigma estimates
    ax2 = fig2.add_subplot(221+pl)
    
    # For the first plot
    for j, key in enumerate(method):
        ax.plot(years, vol_annual[j], label=key, marker='+')
    # For the second plot
    for key in method2:
        method2[key](returns, *param[key]).plot(ax=ax2, label=key)

    ax.set_ylabel(r'$\sigma$')
    ax.set_title(stock)
    ax.legend()

    ax2.set_ylabel(r'$\sigma$')
    ax2.set_xlabel('')
    ax2.set_title(stock)
    ax2.legend()

fig.suptitle('Estimates of annual volatility for various stocks', fontsize=28)
fig2.suptitle('Estimates of constant volatility for various stocks', fontsize=28)
plt.show()
