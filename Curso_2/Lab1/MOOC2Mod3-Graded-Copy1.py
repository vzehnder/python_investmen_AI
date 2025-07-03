#!/usr/bin/env python
# coding: utf-8

# # Solutions for Week 3 - Follow along in the same order as the lab_23.ipynb file

# In[1]:


import numpy as np
import pandas as pd

def as_colvec(x):
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)

def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir


# In[2]:


# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)


# In[3]:


from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
# Computes the posterior expected returns based on 
# the original black litterman reference model
#
# W.prior must be an N x 1 vector of weights, a Series
# Sigma.prior is an N x N covariance matrix, a DataFrame
# P must be a K x N matrix linking Q and the Assets, a DataFrame
# Q must be an K x 1 vector of views, a Series
# Omega must be a K x K matrix a DataFrame, or None
# if Omega is None, we assume it is
#    proportional to variance of the prior
# delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


# In[4]:


# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w


# In[5]:


import edhec_risk_kit_206 as erk

ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2013":]
ind49_mcap = erk.get_ind_market_caps(49, weights=True)["2013":]
inds = ['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food']
rho_ = ind49_rets[inds].corr()
vols_ = ind49_rets[inds].std()*np.sqrt(12)
w_eq_ = ind49_mcap[inds].iloc[0]
w_eq_ = w_eq_/w_eq_.sum()
# Compute the Covariance Matrix


 
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
                                                                                 
        ##  I HAVE CHANGED THE COMMENTED LINE WITH THE NON COMMENTED LINE           
    
#sigma_prior_ =  (vols_.T).dot(vols_) * rho_
sigma_prior_ = ind49_rets[inds].cov()*np.sqrt(12)

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###


# Compute Pi and compare:


# Q1  Which industry sector has the highest capweight?
print(ind49_mcap.iloc[[-1]][inds])

print()
# Q2 Use the same data as the previous question, which industry sector has the highest implied return?
pi_ = implied_returns(delta=2.5, sigma=sigma_prior_, w=w_eq_)
print(pi_.sort_values())

print()
# Q3 Use the same data and assumptions as the previous question. Which industry sector has the lowest implied return?
pi_ = implied_returns(delta=2.5, sigma=sigma_prior_, w=w_eq_)
print(pi_.sort_values())
 
sigma_prior_

# In[64]:


# Hlth will outperform other Rtail and Whlsl by 5%
q_ = pd.Series([.03]) # just one view
# start with a single view, all zeros and overwrite the specific view
p_ = pd.DataFrame([0.]*len(inds), index=inds).T
# find the relative market caps of Rtail and Whlsl to split the
# relative outperformance of Hlth ...
w_rtail =  w_eq_.loc["Rtail"]/(w_eq_.loc["Rtail"]+w_eq_.loc["Whlsl"])
w_whlsl =  w_eq_.loc["Whlsl"]/(w_eq_.loc["Rtail"]+w_eq_.loc["Whlsl"])
p_.iloc[0]['Hlth'] = 1.
p_.iloc[0]['Rtail'] = -w_rtail
p_.iloc[0]['Whlsl'] = -w_whlsl

# Q4 Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  
#  (Hint: Use the same logic as View 1 in the He-Litterman paper)
#  What is the entry you will use for the Pick Matrix P for Whlsl. (Hint: Remember to use the correct sign)


# Q5 Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  
#  (Hint: Use the same logic as View 1 in the He-Litterman paper)
#  What is the entry you will use for the Pick Matrix P for Rtail. (Hint: Remember to use the correct sign)

p_

# In[65]:


delta = 2.5
tau = 0.05 # from Footnote 8
# Find the Black Litterman Expected Returns
bl_mu_, bl_sigma_ = bl(w_eq_, sigma_prior_, p_, q_, tau = tau)

#implied_returns(delta, sigma_prior_, w_eq_)

#  Q 6 Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  
#  (Hint: Use the same logic as View 1 in the He-Litterman paper) Once you impose this view (use delta = 2.5 and tau = 0.05 as in the paper), 
#  which sector has the lowest implied return?

bl_mu_.sort_values()

# In[67]:


def w_star(delta, sigma, mu):
    return (inverse(sigma).dot(mu))/delta

wstar_ = w_star(delta=2.5, sigma=bl_sigma_, mu=bl_mu_)

# Q7 Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  
# (Hint: Use the same logic as View 1 in the He-Litterman paper) 
# Which sector now has the highest weight in the MSR portfolio using the Black-Litterman model?

# Q8 Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  
# (Hint: Use the same logic as View 1 in the He-Litterman paper) 
# Which sector now has the lowest weight in the MSR portfolio using the Black-Litterman model?

# display w*
wstar_.sort_values()

# In[71]:


# Q 10 Now, let’s assume you change the relative view. 
# You still think that it Hlth will outperform Rtail and Whlsl but you think that the outperformance will be 5% not the 3% you originally anticipated. 
# Under this new view which sector has the highest expected return? 

q_[0] = .05
bl_mu_, bl_sigma_ = bl(w_eq_, sigma_prior_, p_, q_, tau = tau)
bl_mu_.sort_values()

# In[72]:


# Q 11 Now, let’s assume you change the relative view. You still think that it Hlth will outperform Rtail and Whlsl but you think that
# the outperformance will be 5% not the 3% you originally anticipated. 
# Under this new view which sector does the Black-Litterman model assign the highest weight?

wstar = w_star(delta=2.5, sigma=bl_sigma_, mu=bl_mu_)
# display w*
wstar.sort_values()
