#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:44:11 2021

@author: yl254115
"""

import os
import numpy as np
import pandas as pd
import utils
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from regressors import stats as statsreg
from mne.stats import fdr_correction

tmin, tmax = 300, 800
add_freq = True
concept_source = 'Top-down Category (WordNet)'
# concept_source = 'Bottom-up Category (Human Raters)'

#########
# PATHS #
#########
path2wordembeddings = '../data/THINGS/sensevec_augmented_with_wordvec.csv'
path2metadata = '../data/THINGS/things_concepts.tsv'

#############
# LOAD DATA #
#############
path2data = f'../data/aos_after_manual_clustering/'
data = utils.get_data(path2data)

###################
# LOAD EMBEDDINGS #
###################
df_metadata = pd.read_csv(path2metadata, delimiter='\t')
df_word_embeddings = pd.read_csv(path2wordembeddings, delimiter=',')

min_freq = np.nanmin(df_metadata['SUBTLEX freq'].values)

############
# ENCODING #
############


def get_semantic_categories(objects):
    categories = []
    for obj in objects:
        df_object = df_metadata.loc[df_metadata['Word'] == obj]
        if df_object[concept_source].size>0:
            concept_category = df_object[concept_source].values[0]
            if concept_category is np.nan:
                concept_category = 'other'
        else:
            concept_category = 'other'
        categories.append(concept_category)
    return categories



list_dicts = []
sessions = list(data.keys())
for session in sessions:
    objectnames = data[session]['objectname']
    objectnames_unique = list(set(data[session]['objectname']))
    categories = get_semantic_categories(objectnames_unique)
    
    category_list = list(set(categories))
    if np.nan in category_list:
        IX_nan = category_list.index(np.nan)
        category_list[IX_nan] = 'other' # replace nan
    category_list = list(set(category_list))
    
    category_list = sorted(category_list)
    n_concepts = len(category_list)
    
    units = list(set(data[session].keys()) - set(['objectname']))
    unit_names = [data[session][unit]['channel_name'] for unit in units]
    
    for unit, unit_name in zip(units, unit_names):
        spike_trains = data[session][unit]['trial']
        kind = data[session][unit]['kind']
        print(f'{session}, Unit {unit}/({len(units)}), {unit_name} ({kind})')
        # Compute spike count in [tmin, tmax] for each trail
        X, y, logfreqs = [], [], []
        dict_cat2object = {}
        for spike_train, objectname in zip(spike_trains, objectnames):
            # Neural data
            spike_train = spike_train.squeeze()
            IXs = np.logical_and(spike_train>=tmin, spike_train<=tmax)
            spike_train_crop = spike_train[IXs]
            IXs_baseline = spike_train<0
            spike_train_baseline = spike_train[IXs_baseline]
            spike_count_rel = spike_train_crop.size - spike_train_baseline.size
            y.append(spike_count_rel)
            
            # Object features (frequency + semantic category)
            df_object = df_metadata.loc[df_metadata['Word'] == objectname]
            if df_object['SUBTLEX freq'].size>0: # word frequency
                freq = df_object['SUBTLEX freq'].values[0] 
            else:
                freq = min_freq
            freq = np.max([freq, 1]) # avoid zero because of log below
            logfreqs.append(np.log10(freq))
            
            if df_object[concept_source].size>0:
                concept_category = df_object[concept_source].values[0]
                if concept_category is np.nan:
                    concept_category = 'other'
            else:
                concept_category = 'other'
                
            if concept_category in dict_cat2object.keys():
                dict_cat2object[concept_category].append(objectname)
            else:
                dict_cat2object[concept_category] = [objectname]
                
            IX_category = category_list.index(concept_category)
            category_onehot = np.zeros(n_concepts)
            category_onehot[IX_category] = 1 # one-hot encoding of category
            X.append(category_onehot)
        for key in dict_cat2object.keys():
            dict_cat2object[key] = list(set(dict_cat2object[key]))
        # to numpy
        logfreqs = stats.zscore(logfreqs)
        X = np.asarray(X)
        if add_freq:
            X = np.hstack((X, logfreqs[:, np.newaxis]))
        y = np.asarray(y)
        
        # Model
        model = RidgeCV(alphas=np.logspace(-5, 5))
        model.fit(X, y)
        r2 = model.score(X, y)
        coefs = model.coef_ # n_features
        pvals = statsreg.coef_pval(model, X, y) # n_features + 1 (for intercept)
        r2_adj = statsreg.adj_r2_score(model, X, y)
        # X = sm.add_constant(X)
        # model = sm.OLS(y, X)
        # results = model.fit()
        # coefs = results.params
        # pvals = results.pvalues
        # r2 = results.rsquared
        
        d = {'session':session,
             'unit':unit,
             'unit name':unit_name,
             'kind':kind,
             'dict_cat2object':dict_cat2object,
              'coefs':coefs,
              'pvals':pvals,
              'r2':r2,
              'r2_adj':r2_adj}

        list_dicts.append(d)
df = pd.DataFrame(list_dicts)           

# FDR
alpha = 0.05
# pvals = df['pvals'].values # n_ROIs X n_times
# shapes = [pvals[i].shape for i in range(951)]
# shapes_cumsum = np.cumsum(shapes)
# pvals_cat = np.concatenate(pvals)
# reject_fdr, pvals_fdr = fdr_correction(pvals_cat,
#                                        alpha=alpha,
#                                        method='indep')
# pvals_fdr_whole_brain = np.empty(len(shapes))
# for i in rangeW(len(shapes)):
#     st = shapes_cumsum[i-1] if i>0 else 0
#     ed = shapes_cumsum[i]
#     pvals_fdr_whole_brain[i] = pvals_fdr[st:ed]
# df['pvals_fdr_whole_brain'] = pvals_fdr.reshape((pvals.shape[0], -1)).tolist()
# df['reject_fdr_whole_brain'] = reject_fdr.reshape((pvals.shape[0], -1)).tolist()


fn = '../results/encoding.json'
df.to_json(fn)

# Plot
for i, row in df.iterrows():
    fig, ax = plt.subplots(figsize=(25, 10))
    xlabels = [f'{category} ({len(dict_cat2object[category])})'
                for category in row['dict_cat2object'].keys()]
    if add_freq:
        xlabels += ['log-frequency']
    xlabels = [xlabel.replace('_', ' ') for xlabel in xlabels]
    ax.bar(xlabels, row['coefs'])
    # for i_cat, reject in enumerate(row['reject_fdr_whole_brain'][1:]):
    for i_cat, pval in enumerate(row['pvals'][1:]):
        if pval<alpha:
            ax.annotate('*', (i_cat, row['coefs'][i_cat] + 0.05))
            
    # Cosmetics
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_ylabel('Coefficient size', fontsize=18)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Semantic category', fontsize=24)
    ax.set_title(f"{row['session']}, Unit {row['unit']}, {row['unit name']} ({row['kind']}); Model R2-adjusted = {row['r2_adj']:1.2f}", fontsize=20)
    plt.subplots_adjust(bottom=0.25)
    fn = f"{row['session']}_{row['unit']}_{row['unit name']}.png"
    fig.savefig(os.path.join('../figures', fn))
    plt.close(fig)
            
            
            
            