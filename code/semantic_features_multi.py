
import os
import math
import numpy as np
import pandas as pd
import time
import argparse

from utils import *
from plot_helper import *
from data_manip import *

from sklearn import linear_model


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
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for responses') 

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
                    default='../figures/semantic_features_multi') 



def save_img(fig, filename) : 

    file = args.path2images + "/" + filename 

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

# WHICH SESSION(S) TO RUN
if args.session is None:
    sessions = list(data.neural_data.keys())
else:
    sessions = [args.session]

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")

start_prepare_data_time = time.time()

unit_counter = 0
responsive_unit_counter = 0
session_counter = 0

for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stimuli_indices = data.neural_data[session]['objectindices_session']
    #object_names = data.neural_data[session]['objectnames']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))

    #dissimilarity_matrix_concepts = np.zeros((len(things_indices), len(things_indices)))
    session_categories = data.df_categories.iloc[things_indices]
    dissimilarity_matrix_concepts = scipy.spatial.distance_matrix(session_categories,session_categories)

    # for i in range(len(things_indices)) : 
    #     cat1 = np.where(data.df_categories.iloc[things_indices[i]] == 1)[0]
        
    #     for j in range(len(things_indices)) : 
    #         dissimilarity_matrix_concepts[i,j] = np.linalg.norm(cat1 - cat2)
    #         cat2 = np.where(data.df_categories.iloc[things_indices[j]] == 1)[0]
    #         any_equal = False
    #         for c in cat1 : 
    #             if np.any(cat2 == c) :
    #                 any_equal = True

    #         #if len(cat1) > 1 :
    #         #    print("More than one category possible: " + object_names[i])
    #         #if len(cat2) > 1 :
    #         #    print("More than one category possible: " + object_names[j])
    #         if not any_equal : 
    #             dissimilarity_matrix_concepts[i,j] = 1

    neural_responses = []
    
    for unit in units:
        unit_counter += 1

        unit_data = data.neural_data[session]['units'][unit]
        firing_rates, consider, median_firing_rates, stddevFiringRates, baselineFiringRates = get_mean_firing_rate_normalized(unit_data['trial'], stimuli_indices, start_time_avg_firing_rate, stop_time_avg_firing_rate, min_ratio_active_trials, min_firing_rate_consider)
        
        response_stimuli_indices = np.where((unit_data['p_vals'] < args.alpha) & (consider > 0))[0]

        if len(response_stimuli_indices) > 0 :
            responsive_unit_counter += 1 
            neural_responses.append(firing_rates)

    distance_matrix = scipy.spatial.distance_matrix(np.transpose(neural_responses), np.transpose(neural_responses))

    regression_model = linear_model.LinearRegression()
    regression_model.fit(pd.DataFrame(dissimilarity_matrix_concepts), pd.DataFrame(distance_matrix))
    #category_names = data.df_categories.keys()[things_indices]
        

    fileDescription = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num)  
    coef_fig = px.bar(x=data.neural_data[session]['stimlookup'], y=regression_model.coef_[0], labels={'x':"concepts", 'y':"coef"})
    coef_fig.update_xaxes(dtick=1)
    coef_fig.update_layout(width=int(2000))
    save_img(coef_fig, "coef_regression" + os.sep + fileDescription)

    


print("\nTime plotting data: " + str(time.time() - start_prepare_data_time) + " s\n")
print("\nNum sessions: " + str(session_counter) + " \n")
print("\nNum units: " + str(unit_counter) + " \n")
print("\nNum responsive units: " + str(responsive_unit_counter) + " \n")