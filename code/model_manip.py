#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:53:00 2022

@author: yl254115
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from regressors import stats as statsreg

def train_model(X, y):
    model = RidgeCV(alphas=np.logspace(-5, 5))
    model.fit(X, y)
    
    results = {}
    results['model'] = model
    results['r2'] = model.score(X, y)
    results['coefs'] = model.coef_ # n_features
    results['pvals'] = statsreg.coef_pval(model, X, y) # n_features + 1 (for intercept)
    results['r2_adj'] = statsreg.adj_r2_score(model, X, y)
    
    
    return results

#import statsmodels.api as sm
# X = sm.add_constant(X)
# model = sm.OLS(y, X)
# results = model.fit()
# coefs = results.params
# pvals = results.pvalues
# r2 = results.rsquared
