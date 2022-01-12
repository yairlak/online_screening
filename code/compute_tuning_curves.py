# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:44:11 2021

@author: yl254115
"""

import os
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# utilility modules
from data_manip import DataHandler
from data_manip import prepare_features, prepare_neural_data
from model_manip import train_model
from viz import plot_weight_of_encoding_model
from stats import add_FDR_correction

parser = argparse.ArgumentParser()
# GENERAL
parser.add_argument('--concept_source', default='Top-down Category (manual selection)',
                    help='Field name from THINGS for semantic categories',
                    choices = ['Bottom-up Category (Human Raters)',
                               'Top-down Category (WordNet)',
                               'Top-down Category (manual selection)'])
parser.add_argument('--response-measure', default='spike_count',
                    help='Which measure is used to summarize the neural response',
                    choices = ['spike_count', 'zscore'])
parser.add_argument('--feature_type', default='word_embeddings',
                    help='Which measure is used to summarize the neural response',
                    choices = ['semantic_categories', 'word_embeddings'])
# STATS
parser.add_argument('--alpha', type=float, default=0.05,
                    help='alpha for stats')

# FLAGS
parser.add_argument('--add-freq', action='store_true', default=False,
                    help='If True, adds word frequency as another feature')
parser.add_argument('--dont-plot', action='store_true', default=False,
                    help='If True, plotting to figures folder is supressed')
# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/')

args=parser.parse_args()

#############
# LOAD DATA #
#############
data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_neural_data() # -> data.neural_data
data.load_word_embeddings() # -> data.df_word_embeddings

############
# ENCODING #
############
list_dicts = []
sessions = list(data.neural_data.keys())
for session in sessions:
    units = list(set(data.neural_data[session]['units'].keys()))
    unit_names = [data.neural_data[session]['units'][unit]['channel_name'] for unit in units]
    
    for unit, unit_name in zip(units, unit_names):
        print(f'{session}, Unit {unit}/({len(units)}), {unit_name} ({data.neural_data[session]["units"][unit]["kind"]})')
        print('n_objects in "other"', len(data.neural_data[session]['dict_cat2object']['other']))
        # PREPARE DATA FOR ENCODING MODEL
        X, X_names, remove = prepare_features(data, session, unit)
        y = prepare_neural_data(data, session, unit)
        
        # REMOVE trials with no feature-data (e.g., nan in word-embedding)
        if remove:
            print('{len(remove)} trials were removed.')
            X = np.delete(X, remove, axis=0)
            y = np.delete(y, remove, axis=0)
        
        # FIT AN ENCODING MODEL
        results = train_model(X, y) 
        # Results is a dict with keys: model, r2, r2_adj, coefs, pvals
        
        # PLOT WEIGHTS OF ENCODING MODEL
        if not args.dont_plot:
            fn_fig = f'weights_{args.feature_type}_{session}_{unit}_{unit_name}.png'
            plot_weight_of_encoding_model(results, X_names,
                                          data,
                                          session, unit,
                                          fn_fig, alpha=args.alpha)
        
        # COLLECT RESULTS INTO A DATAFRAME
        d = {'session':session,
             'unit':unit,
             'unit name':unit_name,
             'kind':data.neural_data[session]['units'][unit]['kind'],
             'dict_cat2object':data.neural_data[session]['dict_cat2object'],
             'model':results['model'],
             'coefs':results['coefs'],
             'pvals':results['pvals'],
             'r2':results['r2'],
             'r2_adj':results['r2_adj']}

        list_dicts.append(d)
        
# SAVE RESULTS TO JSON
df = pd.DataFrame(list_dicts)
df = add_FDR_correction(df)         
fn_df = '../results/encoding.json'
df.to_json(fn_df)

    
    
    