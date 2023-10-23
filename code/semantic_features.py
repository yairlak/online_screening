
import os
import math
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.multitest 
import statsmodels.regression.linear_model as lm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from mne.stats import fdr_correction
import time
import argparse

from utils import *
from plot_helper import *
from data_manip import *

from sklearn import linear_model

spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
mod = sm.OLS(spector_data.endog, spector_data.exog)


parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str, #"90_1_aos" / None ; 90_3_aos, channel 68 cluster 1
                    #"88_1_aos", "88_3_aos", .. 89_3
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--consider_only_responses', default=False, 
                    help='If True, only stimuli eliciting responses are analyzed')
parser.add_argument('--only_SU', default=False, 
                    help='If True, only single units are considered')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for significance') 
parser.add_argument('--alpha_categories', type=float, default=0.01,
                    help='Alpha for significance for categories (only color)') 
parser.add_argument('--threshold_p_value', type=float, default=0.05,
                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_categories', type=float, default=0.5,
#                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_pca', type=float, default=0.25,
#                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_embedding', type=float, default=0.25,
#                    help='Threshold to only keep units where model makes sense') 
parser.add_argument('--analyze', type=str, default="PCA", #"categories", "embedding", "PCA" --> use categories from things, all 300 features, or PCA
                    help='If True, categories are considered, if false word embedding')
parser.add_argument('--pca_components', type=int, default=27,  
                    help='Number of components for PCA')

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2categories',
                    default='../data/THINGS/category_mat_manual.tsv')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') 
parser.add_argument('--path2images', 
                    default='../figures/semantic_features') 



def save_plt(filename) : 

    file = args.path2images + os.sep + args.analyze + os.sep + filename 

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file + ".png", bbox_inches="tight")
        plt.clf()


print("\n--- START ---")
startLoadData = time.time()
args=parser.parse_args()

start_time_avg_firing_rate = 100 #100 #should fit response interval, otherwise spikes of best response can be outside of this interval and normalization fails
stop_time_avg_firing_rate = 800 #800 # 800 for rodrigo
min_ratio_active_trials = 0.5
min_firing_rate_consider = 1
paradigm = args.path2data.split(os.sep)[-1].split('/')[-2].split('_')[0]

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_neural_data() # -> data.neural_data
data.load_categories() # -> data.df_categories
data.load_word_embeddings() # -> data.df_word_embeddings

#num_pcs = np.append(np.array(range(10, 220, 10)), args.pca_components)
num_pcs = [args.pca_components]
categories_pca = []

# WHICH SESSION(S) TO RUN
if args.session is None:
    sessions = list(data.neural_data.keys())
else:
    sessions = [args.session]

if args.analyze == "categories" : 
    categories = data.df_categories
    #threshold_rsquared = args.threshold_r_squared_categories
else : 
    categories = data.df_word_embeddings
    #threshold_rsquared = args.threshold_r_squared_embedding
    if args.analyze == "PCA" : 
        categories_copy = categories.fillna(0)
        
        pca = PCA(n_components=300)
        principalComponents = pca.fit_transform(categories_copy)
        variance_ratio_fig = sns.barplot(x=np.asarray(range(300)), y=pca.explained_variance_ratio_, color='blue') 
        save_plt("explained_variance_ratio")

        for i in num_pcs : 
            n_components = i
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(categories_copy)
            categories = pd.DataFrame(data = principalComponents)
            #threshold_rsquared = args.threshold_r_squared_pca
            categories_pca.append(categories)

        variance_fig = sns.barplot(x=categories.keys(), y=pca.explained_variance_, color='blue') 
        save_plt("explained_variance" + os.sep + str(n_components) + "_pcs")

        sorted_names = []
        for i in range(args.pca_components) : 
            column = categories.iloc[:,i]
            sort_indices = np.argsort(column)[::-1]
            sorted_names_pc = data.df_metadata.uniqueID[sort_indices].values
            sorted_names.append( sorted_names_pc)

        #pd.DataFrame(sorted_names).transpose().to_csv()... # also switch booleans for index and header
        pd.DataFrame(sorted_names).to_csv(args.path2images + os.sep + args.analyze + os.sep + "stimuli_sorted_by_pcs.csv", index=True, header=False) # or without transpose and header=False
    #categories = data.df_word_embeddings.dropna()

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")

start_prepare_data_time = time.time()

unit_counter = 0
responsive_unit_counter = 0
session_counter = 0
r_squared_counter = 0
entropies = []
num_significant_weights = []
num_categories_spanned = []
num_responsive_stimuli = []
pvalues = []
#zscores = []
rsquaredSites = {}
rsquaredSites["all"] = []
pValueSites = {}
pValueSites["all"] = []
rsquaredPCA = []
sites_to_exclude = ["LFa", "LTSA", "LTSP", "Fa", "TSA", "TSP", "LTP", "LTB", "RMC", "RAI", "RAC", "RAT", "RFO", "RFa", "RFb", "RFc"] 
#"RT": pat 102 session 1: anteater unit
#LPL: pat 102 session 3, channel 36, cluster1: mug, teapot

for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    print("subject " + str(subject_num) + ", session " + str(session_num))
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stimuli_indices = data.neural_data[session]['objectindices_session']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))
    
    for unit in units:
        
        site = data.neural_data[session]['units'][unit]['site']
        unit_data = data.neural_data[session]['units'][unit]
        
        if (not unit_data['kind'] == 'SU' and args.only_SU) or site in sites_to_exclude :
            continue
        
        if site == "RAH" or site == "RMH" :
            site = "RH"
        if site == "LAH" or site == "LMH" :
            site = "LH"

        unit_counter += 1
        channel = unit_data['channel_num']
        cluster = unit_data['class_num']
        firing_rates = unit_data['firing_rates']
        response_stimuli_indices = consider = unit_data['responses'] 

        if len(response_stimuli_indices) > 0 :
            responsive_unit_counter += 1 

            consider_indices = np.array(range(len(firing_rates)))
            if args.consider_only_responses : 
                consider_indices = response_stimuli_indices

            consider_indices_THINGS = things_indices[consider_indices]        
            firing_rates_consider = firing_rates[consider_indices]
            #firing_rates_responses_df = pd.DataFrame(data=firing_rates_consider)

            if args.analyze == "PCA" : 
                #categories_responses_df = category_pca[:-1].iloc[things_indices[response_stimuli_indices]]
                count_cat = 0
                for categories in categories_pca :
                    categories_consider_df = categories.iloc[consider_indices_THINGS] ## categories_pca
                    regression_model = sm.OLS(firing_rates_consider, categories_consider_df.values, missing='drop')               
                    fitted_data = regression_model.fit() 
                    if len(rsquaredPCA) < count_cat + 1 :
                        rsquaredPCA.append([])
                    rsquaredPCA[count_cat].append(fitted_data.rsquared)
                    #print("Fitting model for " + str(num_pcs[count_cat]) + " pcs")
                    count_cat += 1

            categories_responses_df = categories.iloc[things_indices[response_stimuli_indices]]
            categories_consider_df = categories.iloc[consider_indices_THINGS] ## categories_pca

            regression_model = sm.OLS(firing_rates_consider, categories_consider_df.values, missing='drop')

            if args.analyze == "embedding" : 
                X = np.nan_to_num(categories_consider_df.values)
                y = firing_rates_consider
                ridge = Ridge()
                loo = LeaveOneOut()
                y_pred = np.zeros_like(y)

                for train_index, test_index in loo.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                
                    ridge.fit(X_train, y_train)
                    y_pred[test_index] = ridge.predict(X_test)

                corr = stats.spearmanr(y, y_pred)
                params = ridge.coef_
                pvalue = corr.pvalue
                rsquared = 0
                fdr_corrected = [[False for i in range(len(params))], np.ones(len(params))]
            else :                                                   
                fitted_data = regression_model.fit() 
                fdr_corrected = statsmodels.stats.multitest.fdrcorrection(fitted_data.pvalues, alpha=args.alpha)
                #fdr_corrected = fdr_correction(fitted_data.pvalues, alpha=args.alpha)
                
                pvalue = fitted_data.f_pvalue
                params = fitted_data.params
                rsquared = fitted_data.rsquared

                
            if site in rsquaredSites : 
                pValueSites[site].append(pvalue)
                rsquaredSites[site].append(rsquared)
            else :
                pValueSites[site] = [pvalue]
                rsquaredSites[site] = [rsquared]
            pValueSites["all"].append(pvalue)
            rsquaredSites["all"].append(rsquared)

            pvalues.append(pvalue)

            if pvalue < args.threshold_p_value : #args.analyze == "embedding" and (pvalue > args.threshold_p_value) or rsquared > threshold_rsquared : #(fitted_data.f_pvalue > args.threshold_p_value)
                r_squared_counter += 1
                ##zscores = np.concatenate(zscores, ((firing_rates_consider - mean_firing_rates) / stddev_firing_rates / mean_baseline))
                num_significant_weights.append(np.count_nonzero(fdr_corrected[0]))
                num_responsive_stimuli.append(len(response_stimuli_indices))
                #if len(response_stimuli_indices) > 1 : 
                responsive_categories = categories_responses_df.any(axis='rows')
                if not responsive_categories.value_counts().keys().any() : 
                    num_categories_spanned.append(0)
                else: 
                    num_categories_spanned.append(responsive_categories.value_counts()[True])

            entropy = stats.entropy(params)
            entropies.append(entropy)

            fileDescription = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num) + '_ch' + '{:02d}'.format(channel)  + '_cl' + str(cluster) + '_' + site 
            color_sequence = ['red' if fdr_corrected[1][i] < args.alpha_categories else 'blue' for i in range(len(fdr_corrected[1])) ]
            text_categories = np.array([str(categories_consider_df.keys()[i]) + ", p: " + str(round(fdr_corrected[1][i], 5)) for i in range(len(categories_consider_df.keys()))])
            coef_fig = sns.barplot(x=text_categories, y=params, palette=color_sequence)
            coef_fig.set_xticklabels(coef_fig.get_xticklabels(), rotation=270)
            plt.title("rsquared = " + str(rsquared) + ", pvalue: " + str(pvalue) + ", Entropy = " + str(entropy))
            save_plt("coef_regression" + os.sep + fileDescription)


semantic_fields_path = "semantic_fields" + os.sep 
sites = list(rsquaredSites.keys())
sites.remove('all')
sites = ['all'] + sorted(sites)

create2DhemispherePlt(rsquaredSites, sites)
save_plt(semantic_fields_path + "rsquared_hemispheres")

create2DhemispherePlt(pValueSites, sites)
save_plt(semantic_fields_path + "pvalue_hemispheres")

plt.figure(figsize=(10,4))
sitesTitles = [site + " (" + str(len(rsquaredSites[site])) + ")" for site in sites]
createStdErrorMeanPlt(sitesTitles, [rsquaredSites[site] for site in sites], "r squared of regression of unit activation based on category / feature", "r squared", [0,1])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "rsquared_sites")

plt.figure(figsize=(10,4)) 
sitesTitles = [site + " (" + str(len(pValueSites[site])) + ")" for site in sites]
createStdErrorMeanPlt(sitesTitles, [pValueSites[site] for site in sites], "pvalue of regression of unit activation based on category / feature", "p value", [-0.005,0.15])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "pvalues_sites")

if args.analyze == "PCA" : 
    createStdErrorMeanPlt([str(cat) for cat in num_pcs], rsquaredPCA, "r squared of regression for different pcs", "r squared")
    save_plt(semantic_fields_path + "rsquared_pcs")

sns.histplot(x=entropies)
save_plt(semantic_fields_path + "entropy_distibution")

createHistPlt(pvalues, [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1.0], 1.0, "Number of units with pvalue of model lower than x; total num: " + str(len(pvalues)), "", "blue", True)
save_plt(semantic_fields_path + "pvalues")

createHistPlt(num_significant_weights, range(0,28), 1.0, "Number of significant weights")
save_plt(semantic_fields_path + "num_significant_weights")

if len(num_categories_spanned) > 0 : 
    createHistPlt(num_categories_spanned, range(0,max(num_categories_spanned)))
    save_plt(semantic_fields_path + "num_responsive_categories")

plt.figure(figsize=(10,4)) 
if len(num_responsive_stimuli) == 0 : 
    num_responsive_stimuli = [0]
createHistPlt(num_responsive_stimuli, range(0,max(num_responsive_stimuli)))
save_plt(semantic_fields_path + "num_responsive_stimuli")

print("Time plotting data: " + str(time.time() - start_prepare_data_time) + " s\n")
print("Num sessions: " + str(session_counter))
print("Num units: " + str(unit_counter))
print("Num responsive units: " + str(responsive_unit_counter))
print("Num r squared: " + str(r_squared_counter))