
import os
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.stats.multitest 
import statsmodels.regression.linear_model as lm
#import mne.stats.fdr_correction
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
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.01,
                    help='Alpha for significance') 
parser.add_argument('--threshold_r_squared_categories', type=float, default=0.5,
                    help='Threshold to only keep units where model makes sense') 
parser.add_argument('--threshold_r_squared_pca', type=float, default=0.25,
                    help='Threshold to only keep units where model makes sense') 
parser.add_argument('--threshold_r_squared_embedding', type=float, default=0.25,
                    help='Threshold to only keep units where model makes sense') 
parser.add_argument('--analyze', type=str, default="categories", #"categories", "embedding", "PCA" --> use categories from things, all 300 features, or PCA
                    help='If True, categories are considered, if false word embedding')
parser.add_argument('--pca_components', type=int, default=20,  
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



def save_img(fig, filename) : 

    file = args.path2images + os.sep + args.analyze + os.sep + filename 

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fig.write_image(file + ".svg")
        fig.write_image(file + ".png")



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

# WHICH SESSION(S) TO RUN
if args.session is None:
    sessions = list(data.neural_data.keys())
else:
    sessions = [args.session]

if args.analyze == "categories" : 
    categories = data.df_categories
    threshold_rsquared = args.threshold_r_squared_categories
else : 
    categories = data.df_word_embeddings
    threshold_rsquared = args.threshold_r_squared_embedding
    if args.analyze == "PCA" : 
        categories = categories.fillna(0)
        pca = PCA(n_components=args.pca_components)
        principalComponents = pca.fit_transform(categories)
        categories = pd.DataFrame(data = principalComponents)
        threshold_rsquared = args.threshold_r_squared_pca
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
rsquaredSites = {}
rsquaredSites["all"] = []


for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stimuli_indices = data.neural_data[session]['objectindices_session']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))
    
    for unit in units:
        unit_counter += 1

        site = data.neural_data[session]['units'][unit]['site']
        unit_data = data.neural_data[session]['units'][unit]
        channel = unit_data['channel_num']
        cluster = unit_data['class_num']
        firing_rates, consider, median_firing_rates, stddevs = get_mean_firing_rate_normalized(unit_data['trial'], stimuli_indices, start_time_avg_firing_rate, stop_time_avg_firing_rate, min_ratio_active_trials, min_firing_rate_consider)
        
        response_stimuli_indices = np.where((unit_data['p_vals'] < args.alpha) & (consider > 0))[0]

        if len(response_stimuli_indices) > 0 :
            responsive_unit_counter += 1 

            consider_indices = np.array(range(len(firing_rates)))
            if args.consider_only_responses : 
                consider_indices = response_stimuli_indices

            consider_indices_THINGS = things_indices[consider_indices]        

            firing_rates_consider = firing_rates[consider_indices]
            #firing_rates_responses_df = pd.DataFrame(data=firing_rates_consider)
            categories_consider_df = categories.iloc[consider_indices_THINGS]
            categories_responses_df = categories.iloc[things_indices[response_stimuli_indices]]

            regression_model = sm.OLS(firing_rates_consider, categories_consider_df.values, missing='drop')

            fitted_data = regression_model.fit()
            entropy = stats.entropy(fitted_data.params)
            entropies.append(entropy)
            #fitted_data.pvalues
            fdr_corrected = statsmodels.stats.multitest.fdrcorrection(fitted_data.pvalues, alpha=args.alpha)
            #fdr_corrected_mne = mne.stats.fdr_correction(fitted_data.pvalues, alpha=args.alpha)
            #np.nan_to_num(fdr_corrected.pvalues, 100)

            #lm.RegressionResults(regression_model)
            if site in rsquaredSites : 
                rsquaredSites[site].append(fitted_data.rsquared)
            else :
                rsquaredSites[site] = [fitted_data.rsquared]
            rsquaredSites["all"].append(fitted_data.rsquared)

            if fitted_data.rsquared > threshold_rsquared: 
                r_squared_counter += 1
                num_significant_weights.append(np.count_nonzero(fdr_corrected[0]))
                num_responsive_stimuli.append(len(response_stimuli_indices))
                #if len(response_stimuli_indices) > 1 : 
                responsive_categories = categories_responses_df.any(axis='rows')
                if not responsive_categories.value_counts().keys().any() : 
                    num_categories_spanned.append(0)
                else: 
                    num_categories_spanned.append(responsive_categories.value_counts()[True])


            #regression_model = linear_model.LinearRegression()
            #regression_model.fit(categories_consider_df, firing_rates_responses_df)

            fileDescription = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num) + '_ch' + '{:02d}'.format(channel)  + '_cl' + str(cluster) + '_' + site 
            color_sequence = ['red' if fdr_corrected[1][i] < args.alpha else 'blue' for i in range(len(fdr_corrected[1])) ]
            text = [str(categories_consider_df.keys()[i]) + ", p: " + str(round(fdr_corrected[1][i], 5)) for i in range(len(categories_consider_df.keys()))]
            coef_fig = go.Figure(data=[go.Bar(
                x=text, 
                y=fitted_data.params, 
                marker_color=color_sequence, 
                )], layout={'title': "rsquared = " + str(fitted_data.rsquared) +  ", Entropy = " + str(entropy)})
            save_img(coef_fig, "coef_regression" + os.sep + fileDescription)

semantic_fields_path = "semantic_fields" + os.sep 

sites = list(rsquaredSites.keys())
sitesTitles = [site + " (" + str(len(rsquaredSites[site])) + ")" for site in sites]
rsquaredPlot = createStdErrorMeanPlot(sitesTitles, [rsquaredSites[site] for site in sites], "r squared of regression of unit activation based on category / feature", "r squared")
save_img(rsquaredPlot, semantic_fields_path + "rsquared")

entropy_plot = px.histogram(x=entropies)
save_img(entropy_plot, semantic_fields_path + "entropy_distibution")

num_significant_weights_plot = createHist(num_significant_weights, range(0,27), 1.0, "Number of significant weights", "", color="blue")
#px.histogram(x=range(0,27),y=num_significant_weights, nbins=27, title="Distribution of significant weights of linear regression")
num_significant_weights_plot.update_xaxes(dtick=1)
save_img(num_significant_weights_plot, semantic_fields_path + "num_significant_weights")

spanned_categories_responses_plot = px.histogram(x=num_categories_spanned)
spanned_categories_responses_plot.update_xaxes(dtick=1)
save_img(spanned_categories_responses_plot, semantic_fields_path + "num_responsive_categories")

num_responsive_stimuli_plot, bins = np.histogram(num_responsive_stimuli, bins = range(1,max(num_responsive_stimuli)))
fig = px.bar(x=bins[:-1], y=num_responsive_stimuli_plot)
fig.update_xaxes(dtick=1)
#num_responsive_stimuli_plot = px.histogram(x=num_responsive_stimuli)
#num_responsive_stimuli_plot.update_xaxes(dtick=1)
save_img(fig, semantic_fields_path + "num_responsive_stimuli")


print("Time plotting data: " + str(time.time() - start_prepare_data_time) + " s\n")
print("Num sessions: " + str(session_counter))
print("Num units: " + str(unit_counter))
print("Num responsive units: " + str(responsive_unit_counter))
print("Num r squared: " + str(r_squared_counter))