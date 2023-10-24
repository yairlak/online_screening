
import os
import math
import numpy as np
import pandas as pd
import time
import argparse

import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.multitest 

from utils import *
from plot_helper import *
from data_manip import *

from sklearn import linear_model
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances


parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str, #"90_1_aos" / None ; 90_3_aos, channel 68 cluster 1
                    #"88_1_aos", "88_3_aos", .. 89_3
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# DATA AND MODEL
parser.add_argument('--metric', default='zscores', # zscores, or pvalues or firing_rates
                    help='Metric to rate responses') # best firing_rates = best zscore ?!
parser.add_argument('--region', default='All', # "All, "AM", "HC", "EC", "PHC"
                    help='Which region to consider') # 

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--consider_only_responses', default=False, 
                    help='If True, only units responding to at least one stimulus are analyzed')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for responses') 
parser.add_argument('--alpha_colors', type=float, default=0.01,
                    help='Alpha for coloring') 

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
                    default='../data/aos_after_manual_clustering/') #aos_after_manual_clustering or aos_one_session or aos_two_sessions
parser.add_argument('--path2images', 
                    default='../figures/semantic_features_multi') 



def save_img(filename) : 
    response_string = "_all_units"

    if args.consider_only_responses : 
        response_string = "_only_responses"

    file = args.path2images + os.sep + args.metric + response_string + os.sep + args.region + os.sep + filename 

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file + ".png", bbox_inches="tight")
        plt.clf()
        #fig.write_image(file + ".svg")
        #fig.write_image(file + ".png")


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
num_things = len(data.df_metadata.uniqueID)

regression_categories_p = {}
regression_categories_params = {}

region_dict = {
    "AM" : ["LA", "RA"],
    "HC" : ["LMH", "RMH", "LAH", "RAH"],
    "EC" : ["LEC", "REC"],
    "PHC" : ["LPHC", "RPHC"]
}

#map_categories = {
#    "animal; bird" : "animal", 
#    "animal; food" : "animal", 
#    "animal; insect" : "animal",
#    "clothing; clothing accessory" : "clothing", 
#    "container; home decor" : "container", 
#    "electronic device; tool" : "tool", 
#    "electronic device; kitchen appliance" : "electronic device", 
#    "food; vegetables" : "food", 
#    "office supply; tool" : "tool", 
#    "sports equipment; weapon" : "sports equipment", 
#    "tool; weapon" : "tool", 
#}

sites_to_exclude = ["LFa", "LTSA", "LTSP", "Fa", "TSA", "TSP", "LTP", "LTB", "RMC", "RAI", "RAC", "RAT", "RFO", "RFa", "RFb", "RFc"] 

for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stim_names = data.neural_data[session]['stimlookup']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))
    #stim_combination_matrix = np.array(np.meshgrid(object_names, object_names)).T.reshape(-1, 2)

    session_categories = data.df_categories.iloc[things_indices]
    num_categories = len(session_categories)
    num_stimuli = len(stim_names)
    print(session)

    neural_responses = []
    
    for unit in units:

        unit_data = data.neural_data[session]['units'][unit]
        site = unit_data["site"]

        if site in sites_to_exclude : 
            continue

        if not args.region == "All" and site not in region_dict[args.region] : 
            continue

        unit_counter += 1

        if len(unit_data['responses']) > 0 or not args.consider_only_responses: # TODO: also non-responsive?
            responsive_unit_counter += 1 

            if args.metric == "firing_rates" : 
                neural_responses.append(unit_data['firing_rates'] )
            else : 
                neural_responses.append(unit_data['zscores']) 

    if len(neural_responses) == 0 : 
        print("No units for session " + session + ", region: " + args.region)
        continue

    #neural_responses_df = pd.DataFrame(neural_responses, columns=stim_names)
    neural_responses = np.transpose(neural_responses)
    
    stim_category_table = []
    stimuli_per_category = {}
    category_list_rsa = []
    stim_names_list_rsa = []     
    stim_indices_list_rsa = []        
    dissimilarity_matrix_responses = 1.0 - np.corrcoef(neural_responses) #np.zeros((len(stim_names), len(stim_names)))
    #dissimilarity_matrix_responses = scipy.spatial.distance_matrix(np.transpose(neural_responses), np.transpose(neural_responses))
    #dissimilarity_matrix_responses = np.zeros((len(stim_names), len(stim_names)))

    #for i in range(num_stimuli) : 
    #    for j in range(i+1, num_stimuli) : 
    #        pearson = stats.pearsonr(neural_responses_transposed[i], neural_responses_transposed[j])
    #        dissimilarity_matrix_responses[i,j] = 1-pearson.correlation
    #        dissimilarity_matrix_responses[j,i] = 1-pearson.correlation
            #stim_names_combined.append([stim_names[i], stim_names[j]])

    #dissimilarity_matrix_responses = dissimilarity_matrix_responses + dissimilarity_matrix_responses.T - np.diag(np.diag(dissimilarity_matrix_responses)) ## fill lower triangle
    dissimilarity_matrix_responses_triangular = dissimilarity_matrix_responses[np.triu_indices(len(stim_names), k = 1)]

    for column in session_categories : #?--> 27
        stim_to_category_list = np.outer(session_categories[column], session_categories[column]) #np.zeros(num_categories*(num_categories-1) /2)
        stim_to_category_list_triangular = stim_to_category_list[np.triu_indices(num_stimuli, k = 1)]
        stim_category_table.append(stim_to_category_list_triangular)

        stim_indices_in_category = np.where(session_categories[column])[0]
        #stim_names_in_category = stim_names[stim_indices_in_category]

        category_list_rsa.extend(np.asarray([column] * len(stim_indices_in_category)))
        stim_names_list_rsa.extend(np.array(stim_names)[stim_indices_in_category])
        stim_indices_list_rsa.extend(stim_indices_in_category)

    dissimilarities_rsa = np.zeros((len(stim_indices_list_rsa), len(stim_indices_list_rsa)))  #dissimilarity_matrix_responses[stim_indices_list_rsa,stim_indices_list_rsa] 

    for i in range(len(stim_indices_list_rsa)) : 
        for j in range(len(stim_indices_list_rsa)) :
            dissimilarities_rsa[i,j] = dissimilarity_matrix_responses[stim_indices_list_rsa[i],stim_indices_list_rsa[j]]


    #regression_model = linear_model.LinearRegression()
    #regression_model.fit(pd.DataFrame(dissimilarity_matrix_concepts), pd.DataFrame(distance_matrix))
    #category_names = data.df_categories.keys()[things_indices]

    regression_model = sm.OLS(pd.DataFrame(dissimilarity_matrix_responses_triangular), pd.DataFrame(np.transpose(stim_category_table)))                #, missing='drop'     
    fitted_data = regression_model.fit() 
    #fdr_corrected = statsmodels.stats.multitest.fdrcorrection(np.nan_to_num(fitted_data.pvalues.values), alpha=args.alpha_colors)
    
    pvalue = fitted_data.f_pvalue
    params = fitted_data.params
    rsquared = fitted_data.rsquared 
    pvalues = fitted_data.pvalues.values 


    for i in range(len(session_categories.columns)):
        category = session_categories.columns[i]
        if(category in regression_categories_p) : 
            regression_categories_p[category].append(pvalues[i])
            regression_categories_params[category].append(params[i])
        else : 
            regression_categories_p[category] = [pvalues[i]]
            regression_categories_params[category] = [params[i]]

    fileDescription = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num)  
    color_sequence = ['red' if pvalues[i] < args.alpha_colors else 'blue' for i in range(len(pvalues)) ]
    text_categories = np.array([str(session_categories.columns[i]) + ", p: " + str(round(pvalues[i], 5)) for i in range(len(session_categories.columns))])
    coef_fig = sns.barplot(x=text_categories, y=params, palette=color_sequence)
    coef_fig.set_xticklabels(coef_fig.get_xticklabels(), rotation=270)
    plt.title("rsquared = " + str(rsquared) + ", pvalue: " + str(pvalue))
    save_img("coef_regression" + os.sep + fileDescription)

    f, ax = plt.subplots(1,1, figsize=(10, 8))
    plt.imshow(dissimilarities_rsa, cmap='jet', vmin=0.3,vmax=1)
    plt.colorbar()
    binsize = [ len([iterator for iterator in category_list_rsa if iterator == category]) for category in session_categories.columns]#np.histogram(category_list_rsa, num_categories)[0]
    edges = np.concatenate([np.asarray([0]), np.cumsum(binsize)])[:-1]
    diff_to_next = np.asarray(binsize).astype(float) / 2.0
    ax.set_xticks(list(np.array(edges)) + diff_to_next)
    ax.set_yticks(list(np.array(edges)) + diff_to_next)
    ax.set_xticklabels(session_categories.columns, rotation = 30)
    ax.set_yticklabels(session_categories.columns)
    ax.vlines(edges,0,len(category_list_rsa)-1)
    ax.hlines(edges,0,len(category_list_rsa)-1)
    ax.set_title('Stimuli sorted by categories')
    save_img("rsa" + os.sep + fileDescription)

    
    mds = manifold.MDS(n_components=2, random_state=1, dissimilarity='precomputed', normalized_stress='auto')
    pos = mds.fit(dissimilarity_matrix_responses).embedding_
    pos *= np.sqrt((neural_responses**2).sum()) / np.sqrt((pos**2).sum())
    distance_mds = euclidean_distances(pos)
    #stress = 0.5 * np.sum((distance_mds - dissimilarity_matrix_responses)**2)

    #clf = PCA(n_components=2)
    #X_mds = clf.fit_transform(neural_responses)
    #pos = clf.fit_transform(pos)

    categories_stimuli_list_array = session_categories.dot(session_categories.columns + ',').str.rstrip(',').str.split(',').to_numpy()
    categories_stimuli_list = [categories[0] for categories in categories_stimuli_list_array]
    #categories_stimuli_list = ['; '.join(categories) for categories in categories_stimuli_list_array] ##all categories separated by ;

    mds_df = pd.DataFrame(pos, columns=('x', 'y'))
    mds_df["categories"] = categories_stimuli_list

    mds_plot = sns.scatterplot(x="x", y="y", data=mds_df, hue="categories")
    sns.move_legend(mds_plot, "upper left", bbox_to_anchor=(1, 1))
    plt.title("MDS for " + session + "; stress: " + str(mds.stress_))
    save_img("mds" + os.sep + fileDescription)


plt.figure(figsize=(10,4)) 
categories = regression_categories_p.keys()
categories_text = [category + ", p: " + str(round(statistics.median(regression_categories_p[category]), 5)) for category in categories]
createStdErrorMeanPlt(categories_text, [regression_categories_p[category] for category in categories], "pvalue of linear regression", "p value", [-0.05, 1.0])
plt.xticks(rotation=45, ha='right')
plt.xlabel("p: median pvalue")
plt.ylabel("pvalue")
save_img("regression_p")

plt.figure(figsize=(10,4)) 
categories = regression_categories_params.keys()
createStdErrorMeanPlt(categories, [regression_categories_params[category] for category in categories], "params of linear regression", "coef")
plt.ylabel("param")
plt.xticks(rotation=45, ha='right')
save_img("regression_params")

print("\nTime plotting data: " + str(time.time() - start_prepare_data_time) + " s\n")
print("Num sessions: " + str(session_counter))
print("Num units: " + str(unit_counter))
print("Num responsive units: " + str(responsive_unit_counter))