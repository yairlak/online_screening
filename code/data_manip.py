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
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import fdrcorrection


class DataHandler(object):
    def __init__(self, args):
        for arg in args.__dict__.keys():
            setattr(self, arg, args.__dict__[arg])
            
    
    
    def load_neural_data(self, alpha=0.001, min_t = 0, max_t = 1000, min_ratio_active_trials = 0.5, min_firing_rate = 1, min_active_trials = 5, min_t_baseline = -500):
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
            zstatistics = load_matlab_file(subject_session_path + '_zscores.mat')["zscores_rs"] 
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
            objectindices = [np.where(stimlookup == objectname)[0][0] for objectname in objectnames]
            neural_data[subject_session_key]['objectnames'] = objectnames
            neural_data[subject_session_key]['objectnumbers'] = objectnumbers
            neural_data[subject_session_key]['objectindices_session'] = objectindices
            neural_data[subject_session_key]['stimlookup'] = [stim[0] for stim in stimlookup]
            
            if self.load_cat2object : 
                neural_data[subject_session_key]['dict_cat2object'] = \
                    get_dict_cat2object(objectnames,
                                        self.df_metadata, self.concept_source)
            

            neural_data[subject_session_key]['units'] = {}
            for unit_num in range(cherries['cherries'].shape[1]):
                trial_data = cherries['cherries'][0, unit_num]['trial'][0, :]
                firing_rates, zscores, consider, num_spikes = get_mean_firing_rate_normalized(
                    trial_data, objectindices, min_t, max_t, min_ratio_active_trials, min_firing_rate, min_active_trials, min_t_baseline)
                #fdr_corrected = fdrcorrection(pvals[unit_num], alpha=alpha) # 0: pass alpha, 1 : alpha values

                neural_data[subject_session_key]['units'][unit_num + 1] = {}
                neural_data[subject_session_key]['units'][unit_num + 1]['site'] = cherries['cherries'][0, unit_num]['site'][0]
                neural_data[subject_session_key]['units'][unit_num + 1]['trial'] = trial_data
                neural_data[subject_session_key]['units'][unit_num + 1]['class_num'] = cherries['cherries'][0, unit_num]['classno'][0, 0]
                neural_data[subject_session_key]['units'][unit_num + 1]['channel_num'] = cherries['cherries'][0, unit_num]['channr'][0, 0]
                neural_data[subject_session_key]['units'][unit_num + 1]['channel_name'] = cherries['cherries'][0, unit_num]['chnname'][0]
                neural_data[subject_session_key]['units'][unit_num + 1]['kind'] = cherries['cherries'][0, unit_num]['kind'][0]
                neural_data[subject_session_key]['units'][unit_num + 1]['p_vals'] = pvals[unit_num]
                neural_data[subject_session_key]['units'][unit_num + 1]['zscores'] = zscores
                neural_data[subject_session_key]['units'][unit_num + 1]['zstatistics'] = zstatistics[unit_num]
                neural_data[subject_session_key]['units'][unit_num + 1]['consider'] = consider
                neural_data[subject_session_key]['units'][unit_num + 1]['firing_rates'] = firing_rates
                #neural_data[subject_session_key]['units'][unit_num + 1]['responses'] = np.where((fdr_corrected[1] < alpha) & (consider > 0))[0]
                neural_data[subject_session_key]['units'][unit_num + 1]['responses'] = np.where((pvals[unit_num] < alpha) & (consider > 0))[0]
                neural_data[subject_session_key]['units'][unit_num + 1]['num_spikes'] = num_spikes
                
   
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
                    
                    firing_rates_all = get_mean_firing_rate_normalized(unit_data['trial'], stimuli_indices, start_time_fr, end_time_fr)[0]
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
        
    def load_wordnet_ids(self):
        self.df_wordnet_ids = pd.read_csv(self.path2worndetids, skip_blank_lines=False,
                                              #delimiter=',',
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
    
    def get_category_similarities(self) : 
        categories = self.df_categories
        num_categories = len(categories.columns)

        center_categories = []
        binary_distances = []

        for c1 in categories.columns : 
            embeddings = self.df_word_embeddings[categories[c1] == 1].values
            embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
            center_categories.append(embeddings.mean(axis=0))

        dist = pairwise_distances(center_categories)
        #dist -= dist.min()
        dist /= dist.max()

        binary_distances = np.zeros((num_categories, num_categories))
        for c1 in range(num_categories) : 
            category_distances = dist[c1,:]
            distance_indices = np.argsort(category_distances)
            for c2 in range(num_categories) : 
                binary_distances[c1][distance_indices[c2]] = c2
        
        return 1-dist, num_categories - binary_distances


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

def get_mean_firing_rate_normalized(all_trials, objectnumbers, min_t = 0, max_t = 1000, min_ratio_active_trials = 0.5, min_firing_rate = 1, min_active_trials = 5, min_t_baseline = -500) : 

    num_objects = max(objectnumbers) + 1
    consider = np.ones(num_objects)
    firing_rates = np.zeros(num_objects) 
    zscores = np.zeros(num_objects) 
    stimuli = np.unique(objectnumbers)
    factor = 1000 / (max_t - min_t)
    factor_baselines = 1000 / (0 - min_t_baseline)
    baseline_firing_rates = np.zeros(num_objects) 
    num_spikes = np.zeros(num_objects)

    for stim in stimuli : 
        stim_trials = all_trials[np.where(objectnumbers == stim)]
        num_active = 0

        #firing_rates_for_median = []

        for trial_spikes in stim_trials : 
            trial_spikes_all = trial_spikes[0]
            trial_spikes_stim = trial_spikes_all[np.where((trial_spikes_all >= min_t) & (trial_spikes_all < max_t))]
            baseline_spikes = trial_spikes_all[np.where((trial_spikes_all >= min_t_baseline) & (trial_spikes_all < 0))]
            baseline_firing_rates[stim] += len(baseline_spikes) * factor_baselines
            firing_rates[stim] += len(trial_spikes_stim) * factor
            num_spikes[stim] += len(trial_spikes_stim)
            #firing_rates_for_median.append(firing_rate)
            if len(trial_spikes_all) > 0 : 
                num_active += 1

        firing_rates[stim] /= float(len(stim_trials))
        baseline_firing_rates[stim] /= float(len(stim_trials))

        if firing_rates[stim] < min_firing_rate or num_active / len(stim_trials) < min_ratio_active_trials or len(stim_trials) < min_active_trials: 
            consider[stim] = 0
    

    mean_baselines = statistics.mean(baseline_firing_rates)
    mean_firing_rates = statistics.mean(firing_rates)
    stddev_firing_rates = statistics.stdev(firing_rates)

    if mean_baselines == 0 : 
        print("WARNING! Baseline is 0 for this unit. Mean firing rate after onset: " + str(statistics.mean(firing_rates)))
        stddev_baselines = 1.0
    else : 
        stddev_baselines = statistics.stdev(baseline_firing_rates)
        #firing_rates = firing_rates / mean_baselines

    zscores = (firing_rates - mean_baselines) / stddev_baselines
    #mean_firing_rates = statistics.mean(firing_rates)
    #zscores = (firing_rates - mean_firing_rates) / stddev / mean_baselines
    #zscores = (firing_rates - mean_baselines) / stddev
    #zscores = stats.zscore(zscores)
    #zscores = (zscores - statistics.mean(zscores)) / statistics.stdev(zscores)
    normalized_firing_rates = (firing_rates - min(firing_rates)) # (firing_rates - mean_baselines)
    normalized_firing_rates = normalized_firing_rates / max(normalized_firing_rates)

    #firing_rates = firing_rates - min(firing_rates)
    #firing_rates = firing_rates / max(firing_rates)
    normalized_zscores = zscores - min(zscores)
    normalized_zscores = normalized_zscores / max(normalized_zscores)
    #for stim in stimuli : 
    #    zscores[stim] = (firing_rates[stim] - mean_baselines) / stddev
    consider[np.where(zscores<2)] = 0

    return normalized_firing_rates, zscores, consider, num_spikes


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


def create_category_map(categories, concepts, path) : 
    #suggested_mapping = {
    #    "concepts" : concepts.columns
    #}

    preferred_categories = ["dessert", "fruit", "vegetable", "bird", "clothing accessory", "insect", "vegetables", "weapon", "kitchen appliance", "kitchen tool", "furniture", "medical equipment", "tool", "animal", "toy", "sports equipment", "home decor"]

    concept_to_category_map = defaultdict(lambda:[])
    concept_categories = categories.dot(categories.columns + ',').str.rstrip(',')
    concept_categories_list_array = concept_categories.str.split(',').to_numpy()

    suggested_mapping = []
    index_distinct_mapping = []
    i = -1
    for category_list in concept_categories_list_array : 
        i += 1

        if len(category_list) == 1 and len(category_list[0]) != 0: 
            suggested_mapping.append(category_list[0])
            index_distinct_mapping.append(i)
            continue

        preferred_category_found = ""
        for preferred_category in preferred_categories : 
            if preferred_category in category_list : 
                preferred_category_found = preferred_category
                break
        suggested_mapping.append(preferred_category_found)
        
    #concept_to_category_map["id"] = concepts.index
    concept_to_category_map["concepts"] = concepts.values
    concept_to_category_map["categories"] = concept_categories
    concept_to_category_map["most specific"] = suggested_mapping
    concept_to_category_map["suggested mapping"] = suggested_mapping

    stimuli_to_category_map_df = pd.DataFrame(concept_to_category_map)
    #stimuli_to_category_map_df.index.name = "id"
    stimuli_to_category_map_df.to_csv(path + "concept_category_map_all.csv", sep=';', index_label="id")

    ##stimuli_to_category_map_df = stimuli_to_category_map_df.drop(index_distinct_mapping)
    ##stimuli_to_category_map_df.to_csv(args.path2semanticdata + "concept_category_map.csv", sep=';', index_label="id")

    return stimuli_to_category_map_df #print("done") # TODO: add categories from wordnet (?)

    
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