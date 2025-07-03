#!/usr/bin/env python
# coding: utf-8

# # Sharpe Style Analysis
# 
# Sharpe Style Analysis is an elegant and simple decomposition exercise similar to what we did in the previous lab session, with the added constraint that the coefficients are all positive and add to 1.
# 
# Therefore, the coefficients of performing style analysis on the observed return of a manager can be interpreted as weights in a portfolio of building blocks which together, _mimic_ that return series. The exercise can reveal drifts in a manager's style as well as provide insight into what the manager is likely doing to obtain the returns.
# 
# 
# # Performing Sharpe Style Analysis
# 
# The key to obtaining the weights is our old friend, the quadriatic optimizer. We are asking the optimizer to find the weights that minimizes the square of the difference between the observed series and the returns of a benchmark portfolio that holds the explanatory building blocks in those same weights. This is equivalent to minimizing the _tracking error_ between the two return series.
# 
# The code to implement this is a very slight modification of the `minimize_vol` we have previously implemented:
# 
# ```python                         
# def style_analysis(dependent_variable, explanatory_variables):
#     """
#     Returns the optimal weights that minimizes the Tracking error between
#     a portfolio of the explanatory variables and the dependent variable
#     """
#     n = explanatory_variables.shape[1]
#     init_guess = np.repeat(1/n, n)
#     bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
#     # construct the constraints
#     weights_sum_to_1 = {'type': 'eq',
#                         'fun': lambda weights: np.sum(weights) - 1
#     }
#     solution = minimize(portfolio_tracking_error, init_guess,
#                        args=(dependent_variable, explanatory_variables,), method='SLSQP',
#                        options={'disp': False},
#                        constraints=(weights_sum_to_1,),
#                        bounds=bounds)
#     weights = pd.Series(solution.x, index=explanatory_variables.columns)
#     return weights
# ```
# 
# The Objective function is a very simple one-liner
# 
# ```python
# def portfolio_tracking_error(weights, ref_r, bb_r):
#     """
#     returns the tracking error between the reference returns
#     and a portfolio of building block returns held with given weights
#     """
#     return tracking_error(ref_r, (weights*bb_r).sum(axis=1))
# ```

# In[1]:


import numpy as np
import pandas as pd

import edhec_risk_kit_202 as erk

%load_ext autoreload
%autoreload 2


# In[2]:


ind = erk.get_ind_returns()["2000":]

# Let's construct a manager that invests in 30% Beer, 50% in Smoke and 20% in other things that have an average return of 0% and an annualized vol of 15%

# In[3]:


mgr_r = 0.3*ind["Beer"] + .5*ind["Smoke"] + 0.2*np.random.normal(scale=0.15/(12**.5), size=ind.shape[0])

# Now, assume we knew absolutely nothing about this manager and all we observed was the returns. How could we tell what she was invested in?

# In[4]:


weights = erk.style_analysis(mgr_r, ind)*100

# In[5]:


weights.sort_values(ascending=False).head(6).plot.bar()

# Contrast this to the results of a regression. Because the model is in fact very true (i.e. we really did construct the manager's returns out of the building blocks), the results are remarkably accurate. However, the negative coefficients are hard to intepret and in real-life data, those will be much larger. However when it works well, such as in this artificial example here, the results can be very accurate.

# In[6]:


coeffs = erk.regress(mgr_r, ind).params*100
coeffs.sort_values().head()

# Negative 4.5% in Household?

# In[7]:


coeffs.sort_values(ascending=False).head(6).plot.bar()

# ## Style Drift: Time Varying Exposures using Style Anaylsis
# 
# One of the most common ways in which Sharpe Style Analysis can be used is to measure style drift. If you run the style analysis function over a rolling window of 1 to 5 years, you can extract changes in the style exposures of a manager.
# 
# We'll look at Rolling Windows in the next lab session.
# 
# As an exercise to the student, download a set of returns from Yahoo Finance, and try and measure the style drift in your favorite fund manager. Use reliable Value and Growth ETFs such as "SPYG" and "SPYV" along with a SmallCap ETF such as "SLY" and LargeCap ETF such as "OEF".
# 
# Alternately, the Fama-French research factors and use the Top and Bottom portfolios by Value (HML) and Size (SMB) to categorize mutual funds into categories. This is very similar to the "Style Box" methodology employed by Morningstar and displayed on their website. Compare your results with their results to see if they agree!

# # Warning: Potential Misuse of Style Analysis
# 
# Style Analysis works best when the explanatory indices are in fact a good specification of what is happening. For instance, it usually gives you very useful and revealing insight if you use a stock market index (such as SPY) and other broad indices, ETFs or mutual funds (such as a Value Fund, a Growth Fund, an International Fund, a Bond Fund etc).
# 
# Part of the skill in extracting meaningful results is to pick the right set of explanatory variables.
# 
# However, a part of the challenge with Style Analysis is that it will _always_ return a portfolio. Although it is possible to develop a figure of merit of fit quality similar to an $R^2$, it will still always give you an answer, however unreasonable it might be, and it's not always obvious how much one can rely on the result.
# 
# For instance, we can try and extract the major industries that Buffer invested in since 2000 as follows:

# In[8]:


brka_m = pd.read_csv("brka_m.csv", index_col=0, parse_dates=True).to_period('M')

# In[9]:


mgr_r_b = brka_m["2000":]["BRKA"]
weights_b = erk.style_analysis(mgr_r_b, ind)
weights_b.sort_values(ascending=False).head(6).round(4)*100

# If we want to look at the last decade (2009-2018):

# In[10]:


brk2009 = brka_m["2009":]["BRKA"]
ind2009 = ind["2009":]
erk.style_analysis(brk2009, ind2009).sort_values(ascending=False).head(6).round(4)*100

# Should you believe the analysis? Probably not. However, when the specification is in fact accurate (as we saw in the articially generated series) the results can be very revealing

# In[ ]:



