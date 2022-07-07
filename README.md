# volatilipy
volatility codes

This repo contains some small codes I've written (and continuing to write) while studying quantitative finance. They are written in python3 (hence the anti-English repo title). They do not contain any ground-breaking methods or techniques, more of an opportunity for me to try out the theoretical stuff I'm learning.

So far, the repo contains the following:
===
> imp_vol: which is a self-contained code that scrapes eurex.com for options data for a chosen product and calculates the volatility implied by the Black-Scholes model. The code uses the ubiquitous Netwon-Raphson method for iteration. In the end, it produces some cute plots showing the dependence of the implied volatility on the strike and expiry.
