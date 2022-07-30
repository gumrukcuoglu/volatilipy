#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:59:23 2022

Demonstration for different ways of delta-hedging when the market estimation 
of volatility (i.e. implied) doesn't match the actual volatility. Assumes 
implied volatility stays the same until expiry, as well as a constant
actual vol.

Updates: 
    (30 July 2022)
    added dividend yield 
    added bid-offer spread - transaction costs (30 July 2022)
    changed the way asset is simulated. 

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
    
def simulate_asset(S0, T, mu, D, sig, Nstep):
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
    D : float
        Dividend yield.
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
        S[i+1] = S[i] + (mu - D) * S[i] * dt \
            + sig * S[i] * random.normalvariate(0,np.sqrt(dt))
    return np.linspace(0, T, Nstep+1), S

def d1(E, Dt, S, r, D, sig):
    """
    Parameters
    ----------
    E : strike
    Dt : T - t
    S : asset
    r : interest rate
    D : dividend yield
    sig : volatility
    """
    return (np.log(S/E) + (r - D + 0.5 * sig**2)*Dt)/ (sig * np.sqrt(Dt))

def d2(E, Dt, S, r, D, sig):
    return d1(E, Dt, S, r, D, sig) - sig * np.sqrt(Dt)

def Delta(E, Dt, S, r, D, sig):
    delt = np.exp(-D * Dt) * norm.cdf(d1(E, Dt, S, r, D, sig))
    # But at Dt=0, Delta is always 1 but python assign nan. Let's fix it.
    # delt[-1]=1. 
    return delt

def compute_Profit_and_Loss(E, S, r, D, s_i, s_h, t, kappa, last=True):
    """
    Given a time-series of asset values, it produces a time-series
    of net P&L resulting from delta-hedging with a given volatility.
    Also output time series of hedging-errors defined as:
        forward(portfolio_post(i-1), dt) -portfolio_pre(i)
    
    
    Parameters
    ----------
    E : float
        Strike.
    S : np.array(dtype=float)
        Asset timeseries.
    r : float
        Interest rate.
    D : float
        Dividend yield.
    s_i : float
        Implied volatility.
    s_h : float
        Hedging volatility.
    t : np.array(dtype=float)
        array of discrete times
    kappa : float
        bid-ask spread parameter. bid - ask = 2 * kappa * (asset value)    
    last : bool
        If true, only return the last PnL. Otherwise return the time series


    Returns
    -------
    np.array(float)
        M2M profit.
        Hedging error.
    """
    Nsteps = len(t)-1
    T = t[-1]
    deltat = T/Nsteps
    
    # implied value of call option
    V_i = S * np.exp(-D*(T-t)) * norm.cdf(d1(E, T-t, S, r, D, s_i)) - \
        E0 * np.exp(-r*(T-t)) * norm.cdf(d2(E, T-t, S, r, D, s_i))
    
    Delta_h = Delta(E, T-t, S, r, D, s_h)
    
    # Initialise portfolio before and after hedging
    portfolio_pre = np.empty(Nsteps+1)
    portfolio_post = np.empty(Nsteps+1)
    
    # Initialise cash balance
    balance = np.empty(Nsteps+1)

    # pre values have no meaning at t=0
    portfolio_pre[0] = 0 
    # We start with the hedged portfolio:
    portfolio_post[0] = V_i[0] - Delta_h[0] * S[0]
    
    # At the first step, we sold some assets (if Delta>0), so added the
    # transaction costs there too
    balance[0] = S[0] * Delta_h[0] * (1 - np.abs(Delta_h[0]) * kappa) - V_i[0]

    for i in range(Nsteps):
        # Value of portfolio at step i+1, before rehedging
        portfolio_pre[i+1] = V_i[i+1] - Delta_h[i]*S[i+1]
        
        # New amount of asset to be hedged at step i+1
        diff = Delta_h[i+1]-Delta_h[i]
        # include effect of transaction costs.
        diff -= np.abs(diff)*kappa
        
        # Value of portfolio after re-hedging
        portfolio_post[i+1] = portfolio_pre[i+1] - diff*S[i+1]

        # Cashflow at this step
        cashflow = portfolio_pre[i+1] - portfolio_post[i+1]

        # add cashflows to the balance (with appropriate forward valuing)
        balance[i+1] = balance[i] * np.exp(r * deltat) + cashflow
    
    # Hedge error:
    hedge_error = (portfolio_post[:-1]*np.exp(r*T/Nsteps)
                    -portfolio_pre[1:])
    
    # Mark-to-market net position        
    M2Mnet = balance + portfolio_post
    # Return mark-to-market net position and (mean,std) of hedge error
    if last:
        return M2Mnet[-1], [hedge_error.mean(), hedge_error.std()]
    else:
        return M2Mnet, [hedge_error.mean(), hedge_error.std()]
    
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
# Dividend yield
D0 = 0
# Kappa (transation cost fraction - Leland model)
kappa0 = 0.001
# expiry
T0 = 1
# total timesteps
timesteps = 365

tstep = T0/timesteps
 
# We first plot the evolution of profit for 10 different tries.
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot()
ax.set_xlabel('Time')
ax.set_ylabel('Mark-to-market P&L')
ax.set_xlim(0,1)

for _ in range(8):
    t, S = simulate_asset(asset, T0, mu0, D0, sig_a, timesteps)
    PnL = compute_Profit_and_Loss(E0, S, r0, D0, sig_i, sig_h, t, kappa0, 
                                  False)[0]
    ax.plot(t, PnL)

plt.show()    


# Now do a number of simulations and look at the distribution of net profit
# Number of trials:
Ntry = 1000

# Initialise array for net profit and loss
net_PnL = np.empty(Ntry)
hedge_err = np.empty((Ntry,2))
for i in range(Ntry):
    t, S = simulate_asset(asset, T0, mu0, D0, sig_a, timesteps)
    net_PnL[i], hedge_err[i] = \
        compute_Profit_and_Loss(E0, S, r0, D0, sig_i, sig_h, t, kappa0)
    

print("After {} tries, with re-hedging every {:.2f} days,"\
      " the distribution of the net mark-to-market P&L is:"
      .format(Ntry, tstep*365))
print("mean = {}, std = {}".format(net_PnL.mean(), net_PnL.std(ddof=1)))
