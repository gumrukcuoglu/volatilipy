#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:30:32 2022

Given Eurex product id for a CALL option and risk-free interest rate,
this code downloads all available option data from Eurex, computes
the implied volatility from Black-Scholes model, using Newton-Raphson
method and produces some plots

@author: emirg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import calendar
import datetime
from urllib.request import urlopen


def d1(E, T, S, r, sig):
    return (np.log(S/E) + (r + 0.5 * sig**2)*T)/ (sig * np.sqrt(T))

def black_scholes(E, T, S, r, sig):
    "Standard Black-Scholes formula for European options"
    D1 = d1(E, T, S, r, sig)
    D2 = D1 - sig * np.sqrt(T)
    return S * norm.cdf(D1) - E * norm.cdf(D2) * np.exp(-r * T)

def vega(E, T, S, r, sig):
    "Calculates the vega for European options"
    D1 = d1(E, T, S, r, sig)
    return S * np.sqrt(T) * norm.pdf(D1)
    
def implied_volatility(V, E, T, S, r, tol):
    """
    Calculates implied volatility using Newton-Raphson method
    
    :V: Option market value
    :E: Strike
    :T: time to expiry
    :S0: asset value today
    :r: risk-free rate
    :tol: tolerance for convergence
    :ref_sig: estimated vol to start the iteration with
    """
    ref_sig=0.5
    sig = ref_sig # Start with this value first
    N_it = 1000 # maximum iterations
    
    for i in range(N_it):
        V_bs = black_scholes(E, T, S, r, sig)
        dV = V_bs - V
        dVds = vega(E, T, S, r, sig)
        # Stop if vega is zero
        if np.abs(dVds) < 1.e-15:
            print('Zero vega at V={}, E={}, T={}'.format(V, E, T))
            # ..but keep the point if we're close to the market value
            if abs(dV) < tol:
                imp_sig = sig
                break
            else:
                imp_sig = np.nan
                break
        ds = dV / dVds
        sig -= ds
        V_new = black_scholes(E, T, S, r, sig)
        # Keep the point if required accuracy is reached
        if abs(ds) < tol or abs(V_new - V) < tol:
            imp_sig = sig
            break
    # Drop the point if no convergence at the end
    if i == N_it -1:
        print('Maximum iterations reached')
        imp_sig = np.nan
    # Drop negative sigma values
    if imp_sig < 0:
        imp_sig = np.nan
        print('Negative vol at V={}, E={}, T={}'.format(V, E, T))
    return imp_sig


def get_expiry_day(cal, date):
    "Given an int (YYYYMM), finds the Saturday after the third Friday of the month"
    year = date//100
    month = date - 100 * year
    # Prepare a matrix of datetime objects for the month
    monthcal = cal.monthdatescalendar(year, month)
    # Determine the 3rd Friday of the month
    expiry_day = [day for week in monthcal for day in week if\
                  day.weekday() == calendar.FRIDAY and \
                  day.month == month][2]
    # Return the next Saturday
    return expiry_day + datetime.timedelta(days=1)


def download_eurex(productid:str, date:str, cal, other_info=False):
    """"
    Download and parse data from Eurex.
    :productid: To be manually found from Eurex.com
    :date: string format DDYY
    :cal: a calendar file to compute the last trading day of option
    :other_info: bool. If true also get the list of available expiry dates
    
    """
    url = 'https://www.eurex.com/ex-en/markets/equ/opt/'\
          + productid\
          + '!quotesSingleViewOption?callPut=Call&maturityDate='\
          + date
    if other_info:
        df = pd.read_html(url) # Read everything
        # Get the available expiry dates and product name
        date_df = df[0].dropna(subset=["Product name", "Expiry"])
        all_months = sorted([
            datetime.datetime.strptime(x, '%b %y').strftime('%Y%m') 
            for x in date_df["Expiry"]])
        name = date_df["Product name"][0]
        df = df[1] # Only keep the option data
        # Also get the last closing price 
        S0 = get_asset_price(url)
    else:
        df = pd.read_html(url, match="Last price")[0]
        
    # Calculate the number of days to expiry
    t = get_expiry_day(cal, int(date)) - \
        datetime.datetime.strptime(df["Date"][0], '%m/%d/%Y').date()
    
    # Keep only strike and last price columns
    df = df[["Strike price", "Last price"]]
    df.columns = ["Strike", "Value"] # Rename for convenience
        
    # Remove any nan values (For some reason, total Value appears as NaN)
    # but missing values are 'n.a.' (str)
    df = df[(df["Value"].notna()) & (df["Value"]!= 'n.a.')].astype(float)
    # Add the expirty date as a third column, annualised
    df["Expiry"] = t.days/252
    if other_info:
        df = (S0, name, all_months, df)
    return df

def last_trading_day(today:datetime.date):
    # Day of the week
    wday = today.isoweekday() 
    # days since last workday (assuming trading continues today)
    diff = (1 + wday) * max(2 - wday, 0) + 1
    return today - datetime.timedelta(days = diff) 
        
def get_asset_price(url:str):
    html = urlopen(url).read().decode("utf-8")
    i_price = html.find(
        '<dt class="productProperty">Underlying closing price:</dt>\n<dd>'
        )+63  # location of the asset price
    # leave enough tolerance for the number of digits just in case
    asset_price = float(html[i_price:i_price+9].rstrip('</d>'))
    print('Last closing price for underlying is {}'.format(asset_price))
    return asset_price

# Set up a calendar
mycal = calendar.Calendar(firstweekday=calendar.SUNDAY)

# This should be any date that is available. The remaining ones are downloaded.
last_trade = last_trading_day(datetime.date.today())

# Use the next month for the first scraping.
# This is to avoid issues if yesterday was an expiry day.
first_call = (last_trade + datetime.timedelta(days=31)).strftime('%Y%m')

prod_id='48364' # BAYE European call
#prod_id='55024' # Siemens European call

# According to ycharts.com, this is the rate in Germany since May 2022.
r0 = 0.95e-2
tolerance = 1.e-5



# Get underlying closing price, product name, list of available expiries 
# and option data for the first trial.
S0, prod_name, expiries, option_data = \
    download_eurex(prod_id, first_call, mycal, True)

# Remove the one we've already downloaded:
expiries.remove(first_call)

# Read the options data with the remaining expiries
counter=0
for T in expiries:
    # Append this to the DataFrame
    option_data = pd.concat( [option_data, 
                              download_eurex(prod_id, T, mycal)],
                            ignore_index = True)
    counter+=1
    print(('\rDownloading data {:2d}/{:2d}').
           format(counter, len(expiries)),end='\r')
else:
    print("Finally finished!        ") 
    
# Calculate the implied volatilities
option_data['Vol'] = option_data.apply(lambda row: 
                                       implied_volatility(
                                           row["Value"],
                                           row["Strike"],
                                           row["Expiry"],
                                           S0, r0, tolerance)
                                       , axis=1)
# Calculate the implied value from those volatilities
option_data['Value_inf'] = option_data.apply(lambda row: 
                                             black_scholes(
                                                 row["Strike"],
                                                 row["Expiry"],
                                                 S0, r0,
                                                 row["Vol"]), 
                                             axis=1)
# Some of the market values are incompatible with BS. I remove those
print('{:3d}% of values dropped'.
      format(int(option_data['Vol'].isna().sum()/len(option_data)*100))
      ,flush=True)
option_data = option_data.dropna(subset=['Vol'])

# Check that the implied vols really imply the market price
print("Maximum DeltaV = ({:1.3f})*tolerance"
      .format(np.abs(option_data["Value_inf"]-option_data["Value"])
              .max()/tolerance))

fig = plt.figure(figsize=(20,10))

fig2 = plt.figure(figsize=(10,10))

# Fixed expiry:
expiries = sorted(list(set(option_data.Expiry)))
ax = fig.add_subplot(121)
ax.set_ylabel(r'Implied $\sigma$')
ax.set_title('{} - fixed expiry'.format(prod_name))
for expiry in expiries:
    option_data[option_data['Expiry'] == expiry][['Strike', 'Vol']].plot(
        x='Strike', y='Vol', ax=ax, marker='+',
        label='{} days'.format(expiry*252))

# Fixed Strike:
strikes = sorted(list(set(option_data['Strike'])))
ax2 = fig.add_subplot(122)
ax2.set_ylabel(r'Implied $\sigma$')
ax2.set_title('{} - fixed strike'.format(prod_name))
for strike in strikes:
    option_data[option_data['Strike'] == strike][['Expiry', 'Vol']].plot(
        x='Expiry', y='Vol', ax=ax2, marker='+', legend=False)

ax2.set_xlabel('Expiry [Trading years]')

# 3D plot for implied volatility
X, Y = np.meshgrid(np.array(expiries), np.array(strikes))

Z = np.empty_like(X, dtype=float)
Z[:] = np.nan # Initialise the volatility matrix with nan

for i in range(len(strikes)):
    for j in range(len(expiries)):
        temp = option_data[(option_data.Expiry.round(3)==X[i,j].round(3)) &
                    (option_data.Strike.round(3)==Y[i,j].round(3))].Vol
        if len(temp):
            Z[i,j] = temp.iloc[0]

ax3 = fig2.add_subplot(projection='3d')
ax3.scatter(252*X,Y,Z)
ax3.view_init(20, 40)

ax3.set_xlabel('Expiry [days]')
ax3.set_ylabel('Strike')
ax3.set_zlabel(r'Implied $\sigma$')
plt.title('{} Call'.format(prod_name))
plt.show()
# Horrible plot. Don't have enough points for this.

# ax3.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.view_init(20, -30)

