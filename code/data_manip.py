#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:46:08 2022

@author: yl254115
"""
import os
import glob
import statistics
import mat73
import numpy as np
from scipy import stats
import scipy.io as sio
import pandas as pd


class DataHandler(object):
    def __init__(self, args):
        for arg in args.__dict__.keys():
            setattr(self, arg, args.__dict__[arg])
            
    
    
    def load_neural_data(self):
        neural_data = {}
    
        # Get all session files in target folder
        fn_cherries = glob.glob(os.path.join(self.path2data, '*cherries.mat'))
        
        # load trialInfo + cherries for each session
        for fn_cherrie in fn_cherries:

            subject_session_name = fn_cherrie.split(os.sep)[-1].split("_")[0]
            subject_session_path = self.path2data + os.sep + subject_session_name
            paradigm = subject_session_name[6:9]

            # Load cherries (contains also condition info), zscores and pvalues
            cherries = load_matlab_file(fn_cherrie)# sio.loadmat(fn_cherrie)
            zscores = load_matlab_file(subject_session_path + '_zscores.mat')["zscores_rs"] 
            pvals = load_matlab_file(subject_session_path + '_os_responses.mat')["pvals_rs"]
            stimlookup = load_matlab_file(subject_session_path + '_stimlookup.mat')["stimlookup"][0]

            # Get subject and session numbers
            subject = int(cherries['conditions']['subject'][0][0][0][0])
            session = int(cherries['conditions']['session'][0][0][0][0])
            if subject_session_name == '090e13aos2' : # There seems to be the wrong session number stored
                session = 2
            subject_session_key = f'{subject}_{session}_{paradigm}'

            neural_data[subject_session_key] = {}
            objectnames = [e[0] for e in cherries['conditions']['objectname'][0][0][0]]
            objectnumbers = [int(e) for e in cherries['conditions']['objectnumber'][0][0][0]]
            neural_data[subject_session_key]['objectnames'] = objectnames
            neural_data[subject_session_key]['objectnumbers'] = objectnumbers
            neural_data[subject_session_key]['objectindices_session'] = [np.where(stimlookup == objectname)[0][0] for objectname in objectnames]
            neural_data[subject_session_key]['stimlookup'] = [stim[0] for stim in stimlookup]
            
            if self.load_cat2object : 
                neural_data[subject_session_key]['dict_cat2object'] = \
                    get_dict_cat2object(objectnames,
                                        self.df_metadata, self.concept_source)
            
            
            neural_data[subject_session_key]['units'] = {}
            for unit_num in range(cherries['cherries'].shape[1]):
                
                site = cherries['cherries'][0, unit_num]['site'][0]
                if site == "RAH" or site == "RMH" :
                    site = "RH"
                if site == "LAH" or site == "LMH" :
                    site = "LH"

                site = site[1:]

                neural_data[subject_session_key]['units'][unit_num + 1] = {}
                neural_data[subject_session_key]['units'][unit_num + 1]['site'] = site
                neural_data[subject_session_key]['units'][unit_num + 1]['trial'] = cherries['cherries'][0, unit_num]['trial'][0, :]
                neural_data[subject_session_key]['units'][unit_num + 1]['class_num'] = cherries['cherries'][0, unit_num]['classno'][0, 0]
                neural_data[subject_session_key]['units'][unit_num + 1]['channel_num'] = cherries['cherries'][0, unit_num]['channr'][0, 0]
                neural_data[subject_session_key]['units'][unit_num + 1]['channel_name'] = cherries['cherries'][0, unit_num]['chnname'][0]
                neural_data[subject_session_key]['units'][unit_num + 1]['kind'] = cherries['cherries'][0, unit_num]['kind'][0]
                neural_data[subject_session_key]['units'][unit_num + 1]['zscores'] = zscores[unit_num]
                neural_data[subject_session_key]['units'][unit_num + 1]['p_vals'] = pvals[unit_num]
    
        self.neural_data = neural_data

    def calculate_responses(self, alpha, alpha_unit, start_time_fr, end_time_fr, include_self = False) :
        self.response_data = []

        for session in self.neural_data : 
            session_data = self.neural_data[session]
            stimuli_indices = session_data['objectindices_session']
            
            things_indices = np.asarray(self.get_THINGS_indices(session_data['stimlookup']))
            for unit in session_data['units'] : 
                unit_data = session_data['units'][unit]

                if not np.any(unit_data['p_vals'] < alpha) : 
                    continue

                response_indices = np.where(unit_data['p_vals'] < alpha_unit)[0]
                num_responses = len(response_indices)

                if num_responses > 0 : # should never happen as this should be >= than alpha
                    response_data_unit = {}
                    response_data_unit['site'] = unit_data['site']
                    response_data_unit['channel_num'] = unit_data['channel_num']
                    response_data_unit['channel_name'] = unit_data['channel_name']
                    response_data_unit['class_num'] = unit_data['class_num']
                    response_data_unit['kind'] = unit_data['kind']
                    response_data_unit['subject'] = int(session.split("_")[0])
                    response_data_unit['session'] = int(session.split("_")[1])
                    response_data_unit['paradigm'] = session.split("_")[2]
                    
                    firing_rates_all = get_mean_firing_rate_normalized(unit_data['trial'], stimuli_indices, start_time_fr, end_time_fr)
                    response_data_unit['firing_rates'] = firing_rates_all[response_indices]
                    response_data_unit['p_vals'] = unit_data['p_vals'][response_indices]
                    response_data_unit['zscores'] = unit_data['zscores'][response_indices]
                    response_data_unit['things_indices'] = things_indices[response_indices]

                    response_data_unit['similarities'] = []
                    response_data_unit['firing_rates_dist'] = []

                    for r1 in range(num_responses) :
                        for r2 in range(r1) :
                            if r2 == r1 and not include_self: 
                                continue
                            response_data_unit['firing_rates_dist'].append(abs(response_data_unit['firing_rates'][r1] - response_data_unit['firing_rates'][r2]))
                            response_data_unit['similarities'].append(self.similarity_matrix[response_data_unit['things_indices'][r1]][response_data_unit['things_indices'][r2]])
                    
                    spearman_rho, spearman_p = stats.spearmanr(response_data_unit['similarities'], response_data_unit['firing_rates_dist']).correlation
                    response_data_unit['spearman'] = spearman_rho
                    response_data_unit['spearman_p'] = spearman_p
                    
                    self.response_data.append(response_data_unit)

    
    def load_metadata(self):
        self.df_metadata = pd.read_csv(self.path2metadata,
                                       delimiter='\t')
        
    def load_word_embeddings(self):
        self.df_word_embeddings = pd.read_csv(self.path2wordembeddings,
                                              delimiter=',',
                                              header=None)
    def load_word_embeddings_tsne(self):
        self.df_word_embeddings_tsne = pd.read_csv(self.path2wordembeddingsTSNE,
                                              delimiter=';',
                                              header=None)

    def load_similarity_matrix(self): 
        self.similarity_matrix = pd.read_csv(self.path2semanticdata + 'similarityMatrix_' + self.metric + '.csv',
                                              delimiter=self.similarity_matrix_delimiter,
                                              header=None)
        
    def load_categories(self):
        self.df_categories = pd.read_csv(self.path2categories,
                                       delimiter='\t')

    def get_THINGS_indices(self, objects) : 
        return [np.where(object == self.df_metadata.uniqueID)[0][0] for object in objects]



def load_matlab_file(file) :  
    try : 
        return sio.loadmat(file)
    except : 
        return mat73.loadmat(file)


def get_dict_cat2object(objectnames, df_metadata, concept_source):
    dict_cat2object = {}
    for objectname in objectnames:
        object_category = object2category(objectname, df_metadata, concept_source)
        
        if object_category in dict_cat2object.keys():
            dict_cat2object[object_category].append(objectname)
        else:
            dict_cat2object[object_category] = [objectname]
    
    for key in dict_cat2object.keys():
        dict_cat2object[key] = list(set(dict_cat2object[key]))
    
    return dict_cat2object

def get_mean_firing_rate_normalized(all_trials, objectnumbers, min_t = 100, max_t = 1000, min_ratio_active_trials = 0.5, min_firing_rate = 0.6) : 

    consider = np.ones(max(objectnumbers) + 1)
    firing_rates = np.zeros(max(objectnumbers) + 1) 
    median_firing_rates = np.zeros(max(objectnumbers) + 1) 
    stimuli = np.unique(objectnumbers)
    factor = 1000 / (max_t - min_t)

    for stim in stimuli : 
        stim_trials = all_trials[np.where(objectnumbers == stim)]
        num_active = 0

        firing_rates_for_median = []

        for trial_spikes in stim_trials : 
            trial_spikes = trial_spikes[0]
            trial_spikes = trial_spikes[np.where((trial_spikes >= min_t) & (trial_spikes < max_t))]
            #trial_spikes = trial_spikes[]
            firing_rates[stim] += len(trial_spikes)
            firing_rates_for_median.append(len(trial_spikes) * factor)
            if len(trial_spikes) > 0 : 
                num_active += 1

        total_firing_rate = 1000 * firing_rates[stim] / (max_t - min_t) #1000 * statistics.median(firing_rates_for_median) / (max_t - min_t)
        #firing_rates_for_median = 1000 * firing_rates_for_median / (max_t - min_t) #1000 * statistics.median(firing_rates_for_median) / (max_t - min_t)
        if total_firing_rate < min_firing_rate or num_active / len(stim_trials) < min_ratio_active_trials or len(stim_trials) < 6: 
            consider[stim] = 0

        firing_rates[stim] = total_firing_rate
        median_firing_rates[stim] = statistics.median(firing_rates_for_median)

        #object_trials = object_trials[np.where(object_trials >= min_t)]
        #object_trials = object_trials[np.where(object_trials <= max_t)]
        #firing_rates[object - 1] = np.count_nonzero(object_trials)

    return firing_rates / max(firing_rates), consider, median_firing_rates


def object2category(objectname, df_metadata, concept_source):
    df_object = df_metadata.loc[df_metadata['uniqueID'] == objectname]
    if df_object[concept_source].size>0:
        object_category = df_object[concept_source].values[0]
        if object_category is np.nan:
            object_category = 'other'
    else:
        object_category = 'other'
    return object_category


def get_category_list(objectnames, df_metadata, concept_source):
    category_list = []
    for objectname in list(set(objectnames)):
        category_list.append(object2category(objectname,
                                             df_metadata, concept_source))
    
    if np.nan in category_list:
        IX_nan = category_list.index(np.nan)
        category_list[IX_nan] = 'other' # replace nan
    category_list = list(set(category_list))
    category_list = sorted(category_list)
    return category_list


    
def prepare_features(data, session, unit):
    min_freq = np.nanmin(data.df_metadata['SUBTLEX freq'].values)

    
    if data.feature_type == 'semantic_categories':
        # Get a sorted list of semantic categories for all objects
        feature_names = get_category_list(data.neural_data[session]['objectnames'],
                                          data.df_metadata,
                                          data.concept_source)
    elif data.feature_type == 'word_embeddings':
        n_dim = data.df_word_embeddings.values.shape[1]
        feature_names = [f'dim_{i}' for i in range(1, n_dim+1)]
        
    n_concepts = len(feature_names)
  
    X, logfreqs, remove = [], [], []
    for i_obj, objectname in enumerate(data.neural_data[session]['objectnames']):
        if data.feature_type == 'semantic_categories':
            object_category = object2category(objectname,
                                              data.df_metadata,
                                              data.concept_source)
            IX_category = feature_names.index(object_category)
            feature_vector = np.zeros(n_concepts)
            feature_vector[IX_category] = 1 # one-hot encoding of category
        elif data.feature_type == 'word_embeddings':
            IX_object_in_metadata = np.where(data.df_metadata['uniqueID'] == objectname)
            assert len(IX_object_in_metadata[0]) == 1, f'{objectname}, {IX_object_in_metadata}'
            feature_vector = data.df_word_embeddings.iloc[IX_object_in_metadata].values[0,:]
            if any(np.isnan(feature_vector)):
                remove.append(i_obj)
        X.append(feature_vector)
    
    # to numpy
    logfreqs = stats.zscore(logfreqs)
    X = np.asarray(X)
    if data.add_freq:
        X = np.hstack((X, logfreqs[:, np.newaxis]))
        feature_names.append('word_frequency')
    
    return X, feature_names, remove

def prepare_neural_data(data, session, unit):
    tmin, tmax = 300, 800 # To be replaced with effect size
    spike_trains = data.neural_data[session]['units'][unit]['trial']
    y = []
    for spike_train in spike_trains:
        # Neural data
        spike_train = spike_train.squeeze()
        IXs = np.logical_and(spike_train>=tmin, spike_train<=tmax)
        spike_train_crop = spike_train[IXs]
        IXs_baseline = spike_train<0
        spike_train_baseline = spike_train[IXs_baseline]
        spike_count_rel = spike_train_crop.size - spike_train_baseline.size
        y.append(spike_count_rel)
        
    return np.asarray(y)