
import os
import math
import numpy as np
import pandas as pd
import time
import argparse

from utils import *
from plot_helper import *
from data_manip import *

import seaborn as sns
import matplotlib as mpl
from pylab import cm
from matplotlib.lines import Line2D

import statsmodels.api as sm
import statsmodels.stats.multitest 

from sklearn import linear_model
from sklearn import manifold
from scipy.cluster import hierarchy 
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
parser.add_argument('--region', default='AM', # "All, "AM", "HC", "EC", "PHC"
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
                    default='../data/aos_after_manual_clustering/') #aos_after_manual_clustering or aos_one_session or aos_selected_sessions
parser.add_argument('--path2images', 
                    default='../figures/semantic_features_multi') 


def get_categories_for_stimuli(categories) : 
    categories_stimuli_list_array = categories.dot(categories.columns + ',').str.rstrip(',').str.split(',').to_numpy()
    return [categories[0] for categories in categories_stimuli_list_array]


def mds(dissimilarity_matrix, categories, file_description) :
    
    if np.any(np.isnan(dissimilarity_matrix)):
        print("WARNING: mds not possible due to nan for " + file_description)
        return
    #print("mds: " + file_description)
    
    if isinstance(categories, pd.DataFrame):
        categories_stimuli_list_array = get_categories_for_stimuli(categories)
    else : 
        categories_stimuli_list_array = categories
    #categories_stimuli_list = ['; '.join(categories) for categories in categories_stimuli_list_array] ##all categories separated by ;

    mds_model = manifold.MDS(n_components=2, random_state=0, dissimilarity='precomputed', normalized_stress='auto')
    mds = mds_model.fit_transform(np.round(dissimilarity_matrix,10).astype(np.float64))

    #clf = PCA(n_components=2)
    #mds = clf.fit_transform(mds)

    mds_df = pd.DataFrame(mds, columns=('x', 'y'))
    mds_df["categories"] = categories_stimuli_list_array

    mds_plot = sns.scatterplot(x="x", y="y", data=mds_df, hue="categories")
    sns.move_legend(mds_plot, "upper left", bbox_to_anchor=(1, 1))
    plt.title("MDS for " + file_description)
    save_img("mds" + os.sep + file_description)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dissimilarity_matrix)
    pca_df = pd.DataFrame(principalComponents, columns=('x', 'y'))
    pca_df["categories"] = categories_stimuli_list_array
    
    pca_plot = sns.scatterplot(x="x", y="y", data=pca_df, hue="categories")
    sns.move_legend(pca_plot, "upper left", bbox_to_anchor=(1, 1))
    plt.title("PCA for " + file_description)
    save_img("pca" + os.sep + file_description)


def TSNE(neural_responses, categories, file_description) :
    
    if np.any(np.isnan(neural_responses)):
        print("WARNING: TSNE not possible due to nan for " + file_description)
        return
    #("tsne: " + file_description)

    if isinstance(categories, pd.DataFrame):
        categories_stimuli_list_array = get_categories_for_stimuli(categories)
    else : 
        categories_stimuli_list_array = categories

    tsne_model = manifold.TSNE(n_components=2, random_state=0)
    tsne = tsne_model.fit_transform(np.transpose(np.asarray(neural_responses)))
    tsne_df = pd.DataFrame(tsne, columns=('x', 'y'))
    tsne_df["categories"] = categories_stimuli_list_array

    tsne_plot = sns.scatterplot(x="x", y="y", data=tsne_df, hue="categories")
    sns.move_legend(tsne_plot, "upper left", bbox_to_anchor=(1, 1))
    plt.title("TSNE for " + file_description)
    save_img("tsne" + os.sep + file_description)


def rsa_plot(dissimilarities, categories_sorted, categories, file_description, title, vmin=0.1, vmax = 1.0) : 

    #mds(dissimilarities, categories_sorted, file_description)
    
    categories_copy = categories.copy()
    empty_columns = categories.columns[np.where([category not in categories_sorted for category in categories])[0]]
    categories_copy = categories_copy.drop(columns=empty_columns)

    f, ax = plt.subplots(1,1, figsize=(10, 8))
    #plt.imshow(np.nan_to_num(dissimilarities, 1.0), cmap='jet', vmin=0.7*np.percentile(dissimilarities, 5), vmax=np.percentile(dissimilarities, 75))
    plt.imshow(np.nan_to_num(dissimilarities, 1.0), cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    binsize = [ len([iterator for iterator in categories_sorted if iterator == category]) for category in categories_copy]
    edges = np.cumsum(binsize)[:-1] 
    tick_position = list(np.concatenate([np.asarray([0]),np.array(edges)])) + np.asarray(binsize).astype(float) / 2.0 - 0.5
    ax.set_xticks(tick_position)
    ax.set_yticks(tick_position)
    ax.set_xticklabels(categories_copy.columns, rotation = 90)
    ax.set_yticklabels(categories_copy.columns)
    ax.vlines(edges-0.5,-0.4,len(dissimilarities)-0.5, linewidth=1.2)
    ax.hlines(edges-0.5,-0.4,len(dissimilarities)-0.5, linewidth=1.2)
    ax.set_title(title)
    save_img("rsa" + os.sep + file_description)


def rsa(dissimilarity_matrix, stimuli_names, categories, file_description, vmin = 0.0, vmax = 1.0) : 

    #try : 
    #    dissimilarity_matrix = 1.0-pd.DataFrame(neural_responses).corr(min_periods=4).to_numpy() #min_periods = min number of observations
    #except : 
    #    print("Error calculating dissimilarity matrix. File: " + file_description)
    #    return
    
    category_list_rsa = []
    stim_names_list_rsa = []     
    stim_indices_list_rsa = []   
    empty_columns = []

    for column in categories : 
        stim_indices_in_category = np.where(categories[column])[0]
        if(len(stim_indices_in_category)) < 2 : 
            empty_columns.append(column)
            continue
        category_list_rsa.extend(np.asarray([column] * len(stim_indices_in_category)))
        stim_names_list_rsa.extend(np.array(stimuli_names)[stim_indices_in_category])
        stim_indices_list_rsa.extend(stim_indices_in_category)

    categories_copy = categories.copy()
    categories_copy = categories_copy.drop(columns=empty_columns)
    
    dissimilarities_rsa_all  = [[dissimilarity_matrix[i,j] for j in stim_indices_list_rsa] for i in stim_indices_list_rsa]

    categories_for_stimuli = category_list_rsa
    dissimilarity_df = pd.DataFrame(dissimilarities_rsa_all, columns = categories_for_stimuli)
    np.fill_diagonal(dissimilarity_df.values, np.nan)
    dissimilarity_df["categories"] = categories_for_stimuli
    dissimilarity_group = dissimilarity_df.groupby(['categories']).mean().transpose().groupby(level=0)
    diss_mean = dissimilarity_group.mean()
    diss_sem = dissimilarity_group.sem()
    
    #mds_model_category = manifold.MDS(n_components=2, random_state=0, dissimilarity='precomputed', normalized_stress='auto')
    #mds_category = mds_model_category.fit_transform(dissimilarity_group)

    if "all" in file_description : 
        file_description_sem_diss = "sem_diss" + os.sep + file_description + "_categories" 
        #mds(diss_mean.values, diss_mean.columns, file_description_sem_diss)

        f = plt.figure()
        #no_nan = np.where(np.asarray([np.count_nonzero(np.isnan(diss_sem.values[:,i])) for i in range(diss_sem.values[0])]) == 0)[0]
        #diss_mds = 
        diss_mean.plot(kind="bar", yerr=diss_sem.values, ylabel=categories_copy.columns, figsize=(100,10), width=0.75)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        save_img("sem_diss" + os.sep + file_description)

        for category in categories_copy.columns: 
            f = plt.figure()
            diss_mean.sort_values(category)[category].plot(kind="bar", yerr=diss_sem[category].values)
            save_img(file_description_sem_diss + os.sep + category)

    
    rsa_plot(diss_mean, categories_copy.columns, categories_copy, "categories" + os.sep + file_description, "Mean correlation between stimuli from categories", vmin=vmin, vmax=vmax)
    rsa_plot(dissimilarities_rsa_all, category_list_rsa, categories_copy, file_description, 'Stimuli sorted by categories', vmin=vmin, vmax=vmax)

    return dissimilarity_matrix

def get_diss_matrix(neural_responses) :
    try : 
        dissimilarity_matrix = 1.0-pd.DataFrame(neural_responses).corr(min_periods=4).to_numpy() #min_periods = min number of observations
    except : 
        print("Error calculating dissimilarity matrix. File: " + file_description)
        return
    
    num_stim = dissimilarity_matrix.shape[0]
    dissimilarity_matrix_triangular = dissimilarity_matrix[np.triu_indices(num_stim, k = 1)]

    return dissimilarity_matrix, dissimilarity_matrix_triangular
    

def dendrogram(dissimilarity_matrix_triangular, stim_names, categories, file_description) : 
    
    if np.any(np.isnan(dissimilarity_matrix_triangular)):
        print("WARNING: dendrogram not possible due to nan for " + file_description)
        return
    
    category_names = get_categories_for_stimuli(categories)
    
    unique_categories = np.unique(category_names)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(unique_categories))
    legend_elements = [Line2D([0], [0], color=cm.brg(norm(i)) , lw=4, label=unique_categories[i]) for i in range(len(unique_categories))]
    legend_elements[0] = Line2D([0], [0], color='y' , lw=4, label=unique_categories[0])

    plt.figure(figsize=(20,4)) 
    Z = hierarchy.linkage(dissimilarity_matrix_triangular, 'ward')
    dn = hierarchy.dendrogram(Z, labels=np.asarray(stim_names), leaf_font_size = 10)
    
    ax = plt.gca()
    x_labels = ax.get_xmajorticklabels()
    for i in range(len(x_labels)):
        stim_index = np.where(stim_names == x_labels[i].get_text())[0][0]
        category_index = np.where(unique_categories == category_names[stim_index])[0][0]
        if category_index == 0 : 
            color = 'y'# [0,255,255,0]
        else :
            color = cm.brg(norm(category_index)) # category_color_map(category_index)
        x_labels[i].set_color(color)

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    save_img("dendrogram" + os.sep + file_description)
    

def analyze_indexed(neural_responses, stim_names, categories, file_description, indices, vmin=0.0, vmax=1.0) : 
    

    indices_int = indices.astype(int)
    neural_responses_reduced = np.array(neural_responses)[:,(indices_int)]
    stim_names_reduced = stim_names[indices_int]
    categories_reduced = categories.iloc[indices_int]

    dissimilarity_matrix, dissimilarity_matrix_triangular = get_diss_matrix(neural_responses_reduced)
    rsa(dissimilarity_matrix, stim_names_reduced, categories_reduced, file_description, vmin, vmax)
    mds(dissimilarity_matrix, categories_reduced, file_description) 
    linear_regression(dissimilarity_matrix_triangular, categories_reduced, dissimilarity_matrix.shape[0], file_description)


def analyze_no_nan(neural_responses, stim_names, categories, file_description, ratio=0.0, vmin=0.0, vmax=1.0) : 
    indices = get_indices_not_nan_ratio(neural_responses, ratio)
    analyze_indexed(neural_responses, stim_names, categories, file_description + "_no_nan" + str(ratio).replace(".",","), indices, vmin, vmax)


def analyze(neural_responses, stim_names, categories, file_description, vmin=0.0, vmax=1.0) : 
    
    dissimilarity_matrix, dissimilarity_matrix_triangular = get_diss_matrix(neural_responses)
    rsa(dissimilarity_matrix, stim_names, categories, file_description, vmin, vmax)
    TSNE(neural_responses, categories, file_description)
    dendrogram(dissimilarity_matrix_triangular, stim_names, categories, file_description)
    mds(dissimilarity_matrix, categories, file_description) 
    return linear_regression(dissimilarity_matrix_triangular, categories, dissimilarity_matrix.shape[0], file_description)


def linear_regression(dissimilarity_matrix_triangular, categories, num_stim, file_description) : 
    
    stim_category_table = []
    #num_stim = dissimilarity_matrix.shape[0]
    #dissimilarity_matrix_responses_triangular = dissimilarity_matrix[np.triu_indices(num_stim, k = 1)]

    for column in categories : #?--> 27
        stim_to_category_list = np.outer(categories[column], categories[column]) #np.zeros(num_categories*(num_categories-1) /2)
        stim_to_category_list_triangular = stim_to_category_list[np.triu_indices(num_stim, k = 1)]
        stim_category_table.append(stim_to_category_list_triangular)

    #regression_model = linear_model.LinearRegression()
    #regression_model.fit(pd.DataFrame(dissimilarity_matrix_concepts), pd.DataFrame(distance_matrix))
    #category_names = data.df_categories.keys()[things_indices]

    regression_model = sm.OLS(pd.DataFrame(dissimilarity_matrix_triangular), pd.DataFrame(np.transpose(stim_category_table)))                #, missing='drop'     
    fitted_data = regression_model.fit() 
    #fdr_corrected = statsmodels.stats.multitest.fdrcorrection(np.nan_to_num(fitted_data.pvalues.values), alpha=args.alpha_colors)
    
    pvalue = fitted_data.f_pvalue
    params = fitted_data.params
    rsquared = fitted_data.rsquared 
    pvalues = fitted_data.pvalues.values 

    color_sequence = ['red' if pvalues[i] < args.alpha_colors else 'blue' for i in range(len(pvalues)) ]
    text_categories = np.array([str(categories.columns[i]) + ", p: " + str(round(pvalues[i], 5)) for i in range(len(categories.columns))])
    coef_fig = sns.barplot(x=text_categories, y=params, palette=color_sequence)
    coef_fig.set_xticklabels(coef_fig.get_xticklabels(), rotation=270)
    plt.title("rsquared = " + str(rsquared) + ", pvalue: " + str(pvalue))
    save_img("coef_regression" + os.sep + file_description)

    return fitted_data


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

    plt.close()


def get_indices_not_nan_ratio(neural_responses_input, ratio = 0.0) : 
    neural_responses_input_np = np.asarray(neural_responses_input)
    num_stimuli = neural_responses_input_np.shape[1]
    return np.where(np.asarray([np.count_nonzero(np.isnan(neural_responses_input_np[:,i])) for i in range(num_stimuli)]) <= num_stimuli * ratio)[0]


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
data.load_categories() # -> data.df_categories
data.load_neural_data() # -> data.neural_data

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
session_counter_120 = 0
session_counter_150 = 0
num_things = len(data.df_metadata.uniqueID)
num_presented_things = np.zeros(len(data.df_metadata.uniqueID))

neural_responses_all = []
neural_responses_120 = []
neural_responses_150 = []
neural_responses_patients = {}
neural_responses_patients_start = {}
regression_categories_p = {}
regression_categories_params = {}
regression_categories_model_p = []
regression_categories_model_rsquared = []

dissimilarities_categories = {}
median_dissimilarity_categories = {}

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
start_stimuli_120 = []
start_stimuli_150 = []

for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    file_description = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num)  
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stim_names = data.neural_data[session]['stimlookup']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))
    num_presented_things[things_indices] += 1
    #stim_combination_matrix = np.array(np.meshgrid(object_names, object_names)).T.reshape(-1, 2)
    objectnames = data.neural_data[session]['objectnames']
    session_with_120_start_stimuli = objectnames[121] in objectnames[0:120]
    start_stimuli = objectnames[0:120] if session_with_120_start_stimuli else objectnames[0:150]
    first_session_120 = False
    first_session_150 = False

    if len(start_stimuli_120) == 0 and session_with_120_start_stimuli: 
        start_stimuli_120 = start_stimuli
        first_session_120 = True

    if len(start_stimuli_150) == 0 and not session_with_120_start_stimuli: 
        start_stimuli_150 = start_stimuli
        first_session_150 = True

    if session_with_120_start_stimuli : 
        start_indices = np.in1d(data.neural_data[session]['stimlookup'], start_stimuli_120).nonzero()[0] #np.where(data.neural_data[session]['stimlookup'] in objectnames)
        print("120 start stimuli")
        session_counter_120 += 1
    else : 
        start_indices = np.in1d(data.neural_data[session]['stimlookup'], start_stimuli_150).nonzero()[0]
        print("150 start stimuli")
        session_counter_150 += 1

    if first_session_120 : 
        print("first session with 120 start stimuli")
        start_indices_things_120 = things_indices[start_indices]
        categories_120 = pd.DataFrame(data.df_categories.iloc[start_indices_things_120])
        start_stimuli_120 = np.array(data.neural_data[session]['stimlookup'])[start_indices]
    
    if first_session_150 : 
        print("first session with 150 start stimuli")
        start_indices_things_150 = things_indices[start_indices]
        categories_150 = pd.DataFrame(data.df_categories.iloc[start_indices_things_150])
        start_stimuli_150 = np.array(data.neural_data[session]['stimlookup'])[start_indices]

    if session_with_120_start_stimuli and not set(start_stimuli).issubset(start_stimuli_120) :
        print("Warning! Session with different start stimuli found!")
    
    if not session_with_120_start_stimuli and not set(start_stimuli).issubset(start_stimuli_150) :
        print("Warning! Session with different start stimuli found!")

    session_categories = data.df_categories.iloc[things_indices]
    num_categories = len(session_categories)
    num_stimuli = len(stim_names)
    print(session)

    neural_responses = []
    #neural_responses_patient =[]
    
    for unit in units:

        unit_data = data.neural_data[session]['units'][unit]
        site = unit_data["site"]

        if site in sites_to_exclude : 
            continue

        if not args.region == "All" and site not in region_dict[args.region] : 
            continue

        unit_counter += 1
        
        score = unit_data[args.metric] 
        score_things = np.arange(num_things).astype(float)
        score_things[:] = np.nan
        score_things[things_indices] = score 
        score_start = score[start_indices]

        if len(unit_data['responses']) > 0 or not args.consider_only_responses: # TODO: also non-responsive?
            responsive_unit_counter += 1 
            neural_responses.append(score)
            neural_responses_all.append(score_things)

            if subject_num in neural_responses_patients : 
                neural_responses_patients[subject_num].append(score_things)
            else : 
                neural_responses_patients[subject_num] = [score_things]
            
            if subject_num in neural_responses_patients_start : 
                neural_responses_patients_start[subject_num].append(score_start)
            else : 
                neural_responses_patients_start[subject_num] = [score_start]

            if session_with_120_start_stimuli : 
                neural_responses_120.append(score_start)
            else : 
                neural_responses_150.append(score_start)


    if len(neural_responses) < 5 : 
        print("Not enough units for session " + session + ", region: " + args.region + ", units: " + str(len(neural_responses)))
        continue

    categories_df = pd.DataFrame(session_categories)
    stim_names_df = pd.DataFrame(stim_names)[0]
    fitted_data = analyze(neural_responses, stim_names_df, categories_df, file_description, vmin=0.0, vmax=1.0)

    params = fitted_data.params
    pvalues = fitted_data.pvalues.values 
    regression_categories_model_p.append(fitted_data.f_pvalue)
    regression_categories_model_rsquared.append(fitted_data.rsquared )
    
    for i in range(len(categories_df.columns)):
        category = categories_df.columns[i]
        if(category in regression_categories_p) : 
            regression_categories_p[category].append(pvalues[i])
            regression_categories_params[category].append(params[i])
        else : 
            regression_categories_p[category] = [pvalues[i]]
            regression_categories_params[category] = [params[i]]




print("ANALYZING PATIENT-WISE..")

for patient in neural_responses_patients : 
    print(str(patient))
    file_description_patient = "pat_" + str(patient)
    neural_responses_patient_np = np.asarray(neural_responses_patients[patient])

    if(len(neural_responses_patients_start[patient][0]) == 120) : 
        analyze(neural_responses_patients_start[patient], start_stimuli_120, categories_120, file_description_patient + "_120", vmin=0.5, vmax=1.0)
    else : 
        analyze(neural_responses_patients_start[patient], start_stimuli_150, categories_150, file_description_patient + "_150", vmin=0.5, vmax=1.0)
    
    ##analyze(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, vmin=0.0, vmax=1.0)
    analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.0, vmin=0.5, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.05, vmin=0.0, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.075, vmin=0.0, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.1, vmin=0.0, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.15, vmin=0.0, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.2, vmin=0.0, vmax=1.0) 
    #analyze_no_nan(neural_responses_patients[patient], data.df_metadata.uniqueID, data.df_categories, file_description_patient, ratio=0.95, vmin=0.0, vmax=1.0) 

print("ANALYZING 120 and 150..")

print("Stimuli in all sessions: ")
count_stim = 0
for stim in start_stimuli_120 : 
    if stim in start_stimuli_150 : 
        count_stim += 1 
        print(stim)
print(str(count_stim) + " stimuli in all sessions ")

if session_counter_120 > 0 : 
    file_description = "all_120"
    analyze(np.asarray(neural_responses_120), start_stimuli_120, categories_120, file_description, vmin=0.7, vmax=1.0)

if session_counter_150 > 0 : 
    file_description = "all_150"
    analyze(np.asarray(neural_responses_150), start_stimuli_150, categories_150, file_description, vmin=0.7, vmax=1.0)

print("ANALYZING ALL..")

file_description = "all"

thresh_frequently_presented = len(sessions) / 3.0
frequently_presented = np.where(num_presented_things >= thresh_frequently_presented)[0]
print("Frequently presented threshold: " + str(thresh_frequently_presented))

neural_responses_all_np = np.asarray(neural_responses_all)

vmin_all = 0.5
analyze(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, vmin=0.0, vmax=1.0)
analyze_indexed(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description + "_reduced", frequently_presented, vmin=vmin_all, vmax=1.0)
analyze_no_nan(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, ratio=0.9, vmin=vmin_all, vmax=1.0) 
#analyze_no_nan(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, ratio=0.85, vmin=vmin_all, vmax=1.0) 
#analyze_no_nan(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, ratio=0.8, vmin=vmin_all, vmax=1.0) 
analyze_no_nan(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, ratio=0.5, vmin=vmin_all, vmax=1.0) 
#analyze_no_nan(neural_responses_all_np, data.df_metadata.uniqueID, data.df_categories, file_description, ratio=0.2, vmin=vmin_all, vmax=1.0) 


neural_responses_reduced = np.array(neural_responses_all)[:,(frequently_presented.astype(int))]
stim_names_reduced = data.df_metadata.uniqueID[frequently_presented]

num_non_nan_units = [np.count_nonzero(~np.isnan(neural_responses_reduced[i,:])) for i in range(len(neural_responses_reduced))]
num_non_nan_stimuli = [np.count_nonzero(~np.isnan(neural_responses_reduced[:,i])) for i in range(len(stim_names_reduced))]

num_nan_units = [np.count_nonzero(np.isnan(neural_responses_reduced[i,:])) for i in range(len(neural_responses_reduced))]
num_nan_stimuli = [np.count_nonzero(np.isnan(neural_responses_reduced[:,i])) for i in range(len(stim_names_reduced))]


createHistPlt(num_non_nan_units, range(0,max(num_non_nan_units),10), factorY=1.0, labelX="Num units (not nan)", labelY="Num stimuli")
save_img("num_non_nan_units")

createHistPlt(num_non_nan_stimuli, range(0,max(num_non_nan_stimuli),50), factorY=1.0, labelX="Num stimuli (not nan)", labelY="Num units")
save_img("num_non_nan_stimuli")

createHistPlt(num_nan_units, range(0,max(num_nan_units),10), factorY=1.0, labelX="Num units (nan)", labelY="Num stimuli")
save_img("num_nan_units")

createHistPlt(num_nan_stimuli, range(0,max(num_nan_stimuli),50), factorY=1.0, labelX="Num stimuli (nan)", labelY="Num units")
save_img("num_nan_stimuli")

plt.figure(figsize=(10,4)) 
counts, bins = np.histogram(num_presented_things, bins=range(29))
sns.barplot(x=bins[:-1], y=counts, color='blue') 
plt.xlabel("Number of sessions a stimulus was presented in")
plt.ylabel("Number of stimuli")
save_img("num_presented_things")

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

counts, bins = np.histogram(regression_categories_model_rsquared, bins=[0.0,0.01,0.05,0.1,1.0])
sns.barplot(x=bins[1:], y=counts)
#sns.swarmplot(x=[0], y=np.asarray([regression_categories_model_rsquared]), color="0", alpha=.35)
save_img("regression_rsquared")

counts, bins = np.histogram(regression_categories_model_p, bins=[0.0,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0])
sns.barplot(x=bins[1:], y=counts)
#sns.swarmplot(x=["rsquared"], y=[regression_categories_model_p], color="0", alpha=.35)
save_img("regression_pvalue")


print("\nTime plotting data: " + str(time.time() - start_prepare_data_time) + " s\n")
print("Num sessions: " + str(session_counter))
print("Num sessions 120 : " + str(session_counter_120))
print("Num sessions 150 : " + str(session_counter_150))
print("Num units: " + str(unit_counter))
print("Num responsive units: " + str(responsive_unit_counter))