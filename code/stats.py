#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:58:39 2022

@author: yl254115
"""

import numpy as np
from mne.stats import fdr_correction

def add_FDR_correction(df):
    alpha = 0.05
    pvals = df['pvals'].values # n_ROIs X n_times
    shapes = [pvals[i].shape for i in range(951)]
    shapes_cumsum = np.cumsum(shapes)
    pvals_cat = np.concatenate(pvals)
    reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
                                            alpha=alpha,
                                            method='indep')
    pvals_fdr_whole_brain = np.empty(len(shapes))
    for i in range(len(shapes)):
        st = shapes_cumsum[i-1] if i>0 else 0
        ed = shapes_cumsum[i]
        pvals_fdr_whole_brain[i] = pvals_fdr[st:ed]
    df['pvals_fdr_whole_brain'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()
    df['reject_fdr_whole_brain'] = reject_fdr.reshape((pvals.shape[0], -1)).tolist()
    return df