#!/usr/bin/env python
# coding: utf-8

# # Factor Analysis using the CAPM and Fama-French Factor models
# 
# The main idea in Factor Analysis is to take a set of observed returns and decompose it into a set of explanatory returns.
# 
# We'll follow _Asset Management_ (Ang 2014, Oxford University Press) Chapter 10 and analyze the returns of Berkshire Hathaway.
# 
# First, we'll need the returns of Berkshire Hathaway which are contained in `data/brka_d_rets.csv`. Read it in as follows:

#In_aux[3]:


import pandas as pd

brka_d = pd.read_csv("data/brka_d_ret.csv", parse_dates=True, index_col=0)
brka_d

#In_aux[2]:


brka_d.tail()

# Next, we need to convert these to monthly returns. The simplest way to do so is by using the `.resample` method, which allows you to run an aggregation function on each group of returns in a time series. We'll give it the grouping rule of 'M' which means _monthly_ (consult the `pandas`) documentation for other codes)
# 
# We want to compound the returns, and we already have the `compound` function in our toolkit, so let's load that up now, and then apply it to the daily returns.

#In_aux[4]:


import edhec_risk_kit_201 as erk

%load_ext autoreload
%autoreload 2


# def compound(r):
#     """
#     returns the result of compounding the set of returns in r
#     """
#     return np.expm1(np.log1p(r).sum())

def compound_prod_direct_test(r):
    """
    returns the result of compounding the set of returns in r
    """
    return (1+r).prod() - 1

brka_m_prod_direct_test = brka_d.resample('M').apply(compound_prod_direct_test).to_period('M')
brka_m_prod_direct_test.head()
brka_m = brka_d.resample('M').apply(erk.compound).to_period('M')
brka_m.head()

#In_aux[5]:


brka_m.to_csv("brka_m.csv") # for possible future use!

# Next, we need to load the explanatory variables, which is the Fama-French monthly returns data set. Load that as follows:

#In_aux[6]:


fff = erk.get_fff_returns()
fff.head()

# Next, we need to decompose the observed BRKA 1990-May 2012 as in Ang(2014) into the portion that's due to the market and the rest that is not due to the market, using the CAPM as the explanatory model.
# 
# i.e.
# 
# $$ R_{brka,t} - R_{f,t} = \alpha + \beta(R_{mkt,t} - R_{f,t}) + \epsilon_t $$
# 
# We can use the `stats.api` for the linear regression as follows:

#In_aux[6]:


import statsmodels.api as sm
import numpy as np
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
mkt_excess = fff.loc["1990":"2012-05",['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()

#In_aux[7]:


lm.summary()

# ### The CAPM benchmark interpretation
# 
# This implies that the CAPM benchmark consists of 46 cents in T-Bills and 54 cents in the market. i.e. each dollar in the Berkshire Hathaway portfolio is equivalent to 46 cents in T-Bills and 54 cents in the market. Relative to this, the Berkshire Hathaway is adding (i.e. has $\alpha$ of) 0.61% _(per month!)_ although the degree of statistica significance is not very high.
# 
# Now, let's add in some additional explanatory variables, namely Value and Size.

#In_aux[8]:


exp_var["Value"] = fff.loc["1990":"2012-05",['HML']]
exp_var["Size"] = fff.loc["1990":"2012-05",['SMB']]
exp_var.head()

#In_aux[9]:


lm = sm.OLS(brka_excess, exp_var).fit()
lm.summary()

# ### The Fama-French Benchmark Interpretation
# 
# The alpha has fallen from .61% to about 0.55% per month. The loading on the market has moved up from 0.54 to 0.67, which means that adding these new explanatory factors did change things. If we had added irrelevant variables, the loading on the market would be unaffected.
# 
# We can interpret the loadings on Value being positive as saying that Hathaway has a significant Value tilt - which should not be a shock to anyone that follows Buffet. Additionally, the negative tilt on size suggests that Hathaway tends to invest in large companies, not small companies.
# 
# In other words, Hathaway appears to be a Large Value investor. Of course, you knew this if you followed the company, but the point here is that numbers reveal it!
# 
# The new way to interpret each dollar invested in Hathaway is: 67 cents in the market, 33 cents in Bills, 38 cents in Value stocks and short 38 cents in Growth stocks, short 50 cents in SmallCap stocks and long 50 cents in LargeCap stocks. If you did all this, you would still end up underperforming Hathaway by about 55 basis points per month.
# 
# We can now add the following code to the toolkit:
# 
# ```python
# import statsmodels.api as sm
# def regress(dependent_variable, explanatory_variables, alpha=True):
#     """
#     Runs a linear regression to decompose the dependent variable into the explanatory variables
#     returns an object of type statsmodel's RegressionResults on which you can call
#        .summary() to print a full summary
#        .params for the coefficients
#        .tvalues and .pvalues for the significance levels
#        .rsquared_adj and .rsquared for quality of fit
#     """
#     if alpha:
#         explanatory_variables = explanatory_variables.copy()
#         explanatory_variables["Alpha"] = 1
#     
#     lm = sm.OLS(dependent_variable, explanatory_variables).fit()
#     return lm
# ```
# 
# 
# ## Exercise to the Student
# 
# I used this particular period because of the example in Ang (2014). However, I have provided data going up to 2018. Have the results held up? Are Buffet's tilts consistent over time?

#In_aux[10]:


result = erk.regress(brka_excess, mkt_excess)

#In_aux[11]:


result.params

#In_aux[12]:


result.tvalues

#In_aux[13]:


result.pvalues

#In_aux[14]:


result.rsquared_adj

#In_aux[15]:


exp_var.head()

#In_aux[16]:


erk.regress(brka_excess, exp_var, alpha=False).summary()

#In_aux[ ]:




#In_aux[ ]:



