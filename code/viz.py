#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:52:52 2022

@author: yl254115
"""

import os
import matplotlib.pyplot as plt

def plot_weight_of_encoding_model(results, X_names,
                                  data, session, unit,
                                  fn_fig, alpha=0.05):
    fig, ax = plt.subplots(figsize=(25, 10))
    if data.feature_type == 'semantic_categories':
        X_names = [f'{X_name}  ({len(data.neural_data[session]["dict_cat2object"][X_name])})'
                   for X_name in X_names if X_name not in ['word_frequency']]
        # xlabels = [f'{category} ({len(dict_cat2object[category])})'
        #             for category in dict_cat2object.keys()]
    # xlabels = [xlabel.replace('_', ' ') for xlabel in xlabels]
    ax.bar(X_names, results['coefs'])
    # for i_cat, reject in enumerate(row['reject_fdr_whole_brain'][1:]):
    for i_cat, pval in enumerate(results['pvals'][1:]):
        if pval<alpha:
            ax.annotate('*', (i_cat, results['coefs'][i_cat] + 0.05))
            
    # Cosmetics
    ax.set_xticklabels(X_names, rotation=45)
    ax.set_ylabel('Coefficient size', fontsize=18)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Semantic category', fontsize=24)
    ax.set_title(f"{session}, Unit {unit}, {'unit name'} ({'kind'}); Model R2, R2-adjusted = {results['r2']:1.2f}, {results['r2_adj']:1.2f}", fontsize=20)
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join('../figures', fn_fig))
    plt.close(fig)