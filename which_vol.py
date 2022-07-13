#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:59:23 2022

Demonstration for different ways of delta-hedging when the market estimation 
of volatility (i.e. implied) doesn't match the actual volatility.

This is a very basic code: instead of evolving by time-steps, I just 
generate a time-series for the asset that follows log-normal dist, 
then use analytic solutions of BS to evaluate profit and loss given the 
evolution of the asset.

The goal is to reproduce various plots in
* Ahmad, R. and Willmott, P. (2005) Which Free Lunch Would You Like Today Sir?
    Delta-Hedging, Volatility Arbitrage and Optimal Portfolios.
    Willmott Magazine, 2005, 64-79.

@author: emirg
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import warnings

#!!!
# d1 and d2 evaluate to infinity at expiry, but their CDF gives 1, so it 
# doesn't cause any harm. For the moment, I'm turning off warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)
    
def simulate_asset(S0, T, mu, sig, Nstep):
    """
    Generates log-normal distribution

    Parameters
    ----------
    S0 : float
        Asset price today.
    T : float
        Expiry.
    mu : float
        Growth rate.
    sig : float
        Volatility.
    Nstep : int
        Number of timesteps.

    Returns
    -------
    list of times
    timeseries of asset prices

    """
    dt = T / Nstep
    S=np.empty(Nstep+1)
    S[0] = S0
    for i in range(Nstep):
        S[i+1] = S[i] + mu * S[i] * dt \
            + sig * S[i] * random.normalvariate(0,np.sqrt(dt))
    return np.linspace(0, T, Nstep+1), S

def d1(E, Dt, S, r, sig):
    """
    Parameters
    ----------
    E : strike
    Dt : T - t
    S : asset
    r : interest rate
    sig : volatility
    """
    return (np.log(S/E) + (r + 0.5 * sig**2)*Dt)/ (sig * np.sqrt(Dt))

def d2(E, Dt, S, r, sig):
    return d1(E, Dt, S, r, sig) - sig * np.sqrt(Dt)

def Delta(E, Dt, S, r, sig):
    return norm.cdf(d1(E, Dt, S, r, sig))

def compute_Profit_and_Loss(E, S, r, s_i, s_h, t):
    """
    Given a time-series of asset values, it produces a time-series
    of net P&L resulting from delta-hedging with a given volatility.
    
    Parameters
    ----------
    E : float
        Strike.
    S : np.array(dtype=float)
        Asset timeseries.
    r : float
        Interest rate.
    s_i : float
        Implied volatility.
    s_h : float
        Hedging volatility.
    t : np.array(dtype=float)
        array of discrete times


    Returns
    -------
    np.array(float)
        M2M profit.

    """
    Nsteps = len(t)-1
    T = t[-1]
    
    # implied value of call option
    V_i = S * norm.cdf(d1(E, T-t, S, r, s_i)) - \
        E0 * np.exp(-r*(T-t)) * norm.cdf(d2(E, T-t, S, r, s_i))
    # Delta after each hedging:
    Delta_h_post = Delta(E, T-t, S, r, s_h)
    Delta_h_pre = np.insert(Delta_h_post[:-1], 0, 0, axis=0)
    
    # Portfolio before hedging
    portfolio_pre = V_i - Delta_h_pre * S
    portfolio_post = V_i - Delta_h_post * S
    
    # Cashflow (equivalent to (Delta_h_post - Delta_h_pre)*S )
    cashflow = portfolio_pre - portfolio_post
    # Note that the _pre values mean nothing at t=0, so let me remove them.
    portfolio_pre[0] = cashflow[0] = 0
    # Cash balance
    balance = np.empty(Nsteps+1)
    balance[0] = S[0] * Delta_h_post[0] - V_i[0]
    
    for j in range(timesteps):
        balance[j+1] = balance[j] * np.exp(r * (t[j+1] - t[j])) + cashflow[j+1]
        
    # Return mark-to-market net position
    print(Delta_h_post[-20:])
    return balance + portfolio_post
    
# strike of call option
E0 = 100
# value of asset at t=0
asset = 100
# volatilities (actual, implied, hedging)
sig_a, sig_i, sig_h = 0.30, 0.20, 0.20
# growth rate
mu0 = -0.9
# risk-free rate
r0 = 0.05
# expiry
T0 = 1
# total timesteps
timesteps = 100

# We first plot the evolution of profit for 10 different tries.
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot()
ax.set_xlabel('Time')
ax.set_ylabel('Mark-to-market P&L')
ax.set_xlim(0,1)

for _ in range(8):
    t, S = simulate_asset(asset, T0, mu0, sig_a, timesteps)
    PnL = compute_Profit_and_Loss(E0, S, r0, sig_i, sig_h, t)
    ax.plot(t, PnL)

plt.show()    


# Now do a number of simulations and look at the distribution of net profit
# Number of trials:
Ntry = 1000

# Initialise array for net profit and loss
net_PnL= np.empty(Ntry)

for i in range(Ntry):
    t, S = simulate_asset(asset, T0, mu0, sig_a, timesteps)
    net_PnL[i] = compute_Profit_and_Loss(E0, S, r0, sig_i, sig_h,t)[-1]

print("After {} tries, the distribution of the net mark-to-market P&L is:"
      .format(Ntry))
print("mean = {}, std = {}".format(net_PnL.mean(), net_PnL.std(ddof=1)))