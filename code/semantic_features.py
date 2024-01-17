
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
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import mean_squared_error
from mne.stats import fdr_correction
import time
import argparse
from collections import defaultdict

from utils import *
from plot_helper import *
from data_manip import *

from sklearn import linear_model, datasets
from sklearn.datasets import fetch_california_housing

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
parser.add_argument('--only_SU', default=True, 
                    help='If True, only single units are considered')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')
parser.add_argument('--plot_regions', default='hemispheres',
                    help='"full"->all regions, "hemispheres"->split into hemispheres, "collapse_hemispheres"->regions of both hemispheres are collapsed')  
parser.add_argument('--analyze', type=str, default="categories", #"categories", "embedding", "PCA" --> use categories from things, all 300 features, or PCA
                    help='If True, categories are considered, if false word embedding')  

parser.add_argument('--response_metric', default='zscores', # zscores, or pvalues or firing_rates
                    help='Metric to rate responses') # best firing_rates = best zscore ?! 

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for significance') 
parser.add_argument('--alpha_categories', type=float, default=0.001,
                    help='Alpha for significance for categories (only color)') 
parser.add_argument('--threshold_p_value', type=float, default=0.05,
                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_categories', type=float, default=0.5,
#                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_pca', type=float, default=0.25,
#                    help='Threshold to only keep units where model makes sense') 
#parser.add_argument('--threshold_r_squared_embedding', type=float, default=0.25,
#                    help='Threshold to only keep units where model makes sense') 
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
                    default='../data/aos_after_manual_clustering/') #aos_after_manual_clustering aos_one_session
parser.add_argument('--path2images', 
                    default='../figures/semantic_features') 

def get_lofo_score(firing_rates_consider, categories_consider) : 
    
    regression_model_all = sm.OLS(firing_rates_consider, categories_consider.values, missing='drop')
    fitted_data_all = regression_model_all.fit()
    
    lofo_score = []

    for category in categories_consider.columns : 
        category_reduced = categories_consider.copy()
        category_reduced = category_reduced.drop(category, axis=1)
        regression_model_reduced = sm.OLS(firing_rates_consider, category_reduced.values, missing='drop')
        fitted_data_reduced = regression_model_reduced.fit()
        lofo_score.append(fitted_data_all.rsquared - fitted_data_reduced.rsquared)

    return lofo_score

def plot_lofo_score(regression_df, column_name, category_names="category_names", labelsize=40) : 
#def plot_lofo_score(regression_df, column_name="category_names", labelsize=40) : 
    
    regression_df = regression_df.sort_values(column_name, ascending=False)
    plt.figure(figsize=(barplotWidth, barplotHeight))
    lofo_fig = sns.barplot(y=category_names, x=column_name, data=regression_df, color='blue', orient = 'h')#, width=1.2)#, palette=regression_df["color"])
    #lofo_fig = sns.barplot(y=y, x=column_name, data=regression_df, color='blue', orient = 'h')#, width=1.2)#, palette=regression_df["color"])
    #lofo_fig.set_xticklabels(lofo_fig.get_xticklabels(), rotation=90)
    lofo_fig.set(xlabel=None, ylabel=None)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    #plt.title("lofo score")
    save_plt(column_name + os.sep + fileDescription)

def create_category_map(categories, concepts) : 
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
    stimuli_to_category_map_df.to_csv(args.path2semanticdata + "concept_category_map_all.csv", sep=';', index_label="id")

    stimuli_to_category_map_df = stimuli_to_category_map_df.drop(index_distinct_mapping)
    stimuli_to_category_map_df.to_csv(args.path2semanticdata + "concept_category_map.csv", sep=';', index_label="id")

    print("done") # TODO: add categories from wordnet (?)


def create_category_bar_graph(df, path, title="", xlabel="") : 
    df = df.sort_values(na_position='first')
    if len(df) > 0 : 
        plt.figure(figsize=(barplotWidth, barplotHeight / 2))
        adjustFontSize()
        plt.tick_params(labelsize=24)
        df.plot(kind='barh', stacked=True, color="blue")
        if len(xlabel) > 0 : 
            plt.xlabel(xlabel)
        if len(title) > 0 : 
            plt.title(title)
        save_plt(semantic_fields_path + os.sep + "category_counts" + os.sep + path)

def sort_by_sum_sites(df) : 
    new_df = df.copy()
    if len(new_df.columns) == 3 :   
        value_column = [column for column in new_df.columns if column not in ['category', 'site']][0]
        sort_df = new_df.copy().groupby("category").sum().sort_values(by=value_column, ascending=True).reset_index()
        new_df = pd.pivot_table(data=new_df, index=['category'], columns=['site'], values=value_column, fill_value=0).reset_index()
        new_df.set_index('category', inplace=True)
    else : 
        sort_df = new_df.sum(axis=1).sort_values().reset_index()
    indices = [sort_df.index[np.where(sort_df["category"] == new_df.index[i])[0][0]] for i in range(len(new_df.index))]
    new_df["indices"] = indices
    new_df = new_df.sort_values(by="indices")
    new_df = new_df.drop("indices", axis=1)
    return new_df

def create_category_plot_stacked(df, path) : 
    grouped_df = pd.DataFrame(df).reset_index() # category - site - presented
    grouped_df = sort_by_sum_sites(grouped_df)

    if len(grouped_df) > 0 : 
        plt.figure(figsize=(barplotWidth, barplotHeight))
        font = dict(weight='normal', size=14)
        plt.rc('font', **font)
        grouped_df.plot(kind='barh', stacked=True)
        plt.tick_params(labelsize=10)
        save_plt(semantic_fields_path + os.sep + "category_counts" + os.sep + path + os.sep + "all_stacked")

    return grouped_df


def create_category_plots(category_df, site) :
    
    category_site_counts_presented = category_df.groupby("category")["presented"].sum()#.sort_values(na_position='first')#[category_df["presented"] == 1]["category"].value_counts().sort_index()
    create_category_bar_graph(category_site_counts_presented, "presented_units" + os.sep + site)

    if site == "all" : 
        all_presented = create_category_plot_stacked(category_df.groupby(["category", "site"])["presented"].sum(), "presented_units")

    category_site_counts_responsive = category_df.loc[category_df["responsive"] == 1]["category"].value_counts()#.sort_values(na_position='first')
    create_category_bar_graph(category_site_counts_responsive, "responsive" + os.sep + site)
    if site == "all" : 
        all_responsive = create_category_plot_stacked(category_df.loc[category_df["responsive"] == 1][["category", "site"]].value_counts(), "responsive")

    category_site_counts_percentage = category_site_counts_responsive.divide(category_site_counts_presented, fill_value=0.0)#.sort_values(na_position='first')
    create_category_bar_graph(category_site_counts_percentage * 100, "responsive_percent" + os.sep + site, 
                            "Response probability of neurons to stimuli of respective category", "Response probability in %")
    if site == "all" : 
        all_responsive_percent = all_responsive.divide(all_presented, fill_value=0.0)
        all_responsive_percent = sort_by_sum_sites(all_responsive_percent)
        
        if len(all_responsive_percent) > 0 : 
            plt.figure(figsize=(barplotWidth, barplotHeight))
            font = dict(weight='normal', size=14)
            plt.rc('font', **font)
            all_responsive_percent.plot(kind='barh', stacked=True)
            plt.tick_params(labelsize=10)
            save_plt(semantic_fields_path + os.sep + "category_counts" + os.sep + "responsive_percent" + os.sep + "all_stacked")

            plt.figure(figsize=(barplotWidth, barplotHeight))
            #plt.tick_params(labelsize=18)
            all_responsive_percent.plot(kind='barh', stacked=False)
            plt.tick_params(labelsize=10)
            #plt.tick_params(axis='both', which='major', labelsize=10)
            save_plt(semantic_fields_path + os.sep + "category_counts" + os.sep + "responsive_percent" + os.sep + "all_grouped")

        first_unit_per_session = category_df.loc[category_df["first"] == 1]#.groupby(["category", "session", "presented"]).first()
        category_site_counts_session = first_unit_per_session.groupby("category")["presented"].sum()#.sort_values()
        create_category_bar_graph(category_site_counts_session, "presented_sessions" + os.sep + site)


def save_plt(filename) : 
    
    if args.only_SU : 
        unitPath = "SU"
    else : 
        unitPath = "MU_SU"

    file = args.path2images + os.sep + args.analyze + "_" + args.response_metric + os.sep + args.plot_regions + "_" + unitPath + os.sep + filename 

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file + ".png", bbox_inches="tight")
        plt.clf()
    plt.close()


print("\n--- START ---")
startLoadData = time.time()
args=parser.parse_args()

#start_time_avg_firing_rate = 100 #100 #should fit response interval, otherwise spikes of best response can be outside of this interval and normalization fails
#stop_time_avg_firing_rate = 800 #800 # 800 for rodrigo
#min_ratio_active_trials = 0.5
#min_firing_rate_consider = 1
paradigm = args.path2data.split(os.sep)[-1].split('/')[-2].split('_')[0]

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_categories() # -> data.df_categories

create_category_map(data.df_categories, data.df_metadata.uniqueID)

data.load_neural_data() # -> data.neural_data
data.load_word_embeddings() # -> data.df_word_embeddings

#num_pcs = np.append(np.array(range(10, 220, 10)), args.pca_components)
num_pcs = [args.pca_components]
categories_pca_all = []

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
            categories_pca_all.append(categories)

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


pca = PCA(n_components=args.pca_components)
principalComponents = pca.fit_transform(data.df_word_embeddings.copy().fillna(0))
categories_pca = pd.DataFrame(data = principalComponents)

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")

start_prepare_data_time = time.time()

unit_counter = 0
responsive_unit_counter = 0
session_counter = 0
r_squared_counter = 0
entropies = []
num_significant_weights = []
lofo_score_categories = []
lofo_score_pca = []
num_categories_spanned = []
num_responsive_stimuli = []
pvalues = []
#zscores = []
rsquaredCategoriesSites = defaultdict(lambda: [])
rsquaredPCASites = defaultdict(lambda: [])
pValueSites = defaultdict(lambda: [])
lofoCategoriesSites = defaultdict(lambda: [])
lofoPCASites = defaultdict(lambda: [])
rsquaredPCA = []
meanScoresSites = {}
meanScoresSites["all"] = []
categoryNames = data.df_categories.columns
categoriesSignificantCount = np.zeros((len(categoryNames)))
category_presented_counts = defaultdict()
category_responsive_counts = defaultdict()
category_counts = defaultdict(lambda: [])
numSignificant = []
#stddevScoresSites = []
sitesToConsider = ["LA", "RA", "LEC", "REC", "LAH", "RAH", "LMH", "RMH", "LPHC", "RPHC", "LPIC", "RPIC"]
#sitesToExclude = ["LFa", "LTSA", "LTSP", "Fa", "TSA", "TSP", "LPL", "LTP", "LTB", "RMC", "RAI", "RAC", "RAT", "RFO", "RFa", "RFb", "RFc", "RT", "RFI", "RFM", "RIN", "LFI", "LFM", "LIN"]
#"RT": pat 102 session 1: anteater unit
#LPL: pat 102 session 3, channel 36, cluster1: mug, teapot
barplotWidth = 5
barplotHeight = 30
labelsize = 40

for session in sessions:

    session_counter += 1
    subject_num = int(session.split("_")[0])
    session_num = int(session.split("_")[1])
    print("subject " + str(subject_num) + ", session " + str(session_num))
    
    units = list(set(data.neural_data[session]['units'].keys()))
    stimuli_indices = data.neural_data[session]['objectindices_session']
    things_indices = np.array(data.get_THINGS_indices(data.neural_data[session]['stimlookup']))
    categories_things = categories.iloc[things_indices]
    categories_presented = categories_things.sum()
    
    firstUnitInSession = 1
    for unit in units:
        
        site = data.neural_data[session]['units'][unit]['site']
        unit_data = data.neural_data[session]['units'][unit]
        
        if (not unit_data['kind'] == 'SU' and args.only_SU) or site not in sitesToConsider :
            continue
        
        if site == "RAH" or site == "RMH" :
            site = "RH"
        if site == "LAH" or site == "LMH" :
            site = "LH"
        
        if args.plot_regions == "collapse_hemispheres" : 
            site = site[1:]
        elif args.plot_regions == "hemispheres" : 
            site = site[0]

        unit_counter += 1
        channel = unit_data['channel_num']
        cluster = unit_data['class_num']
        firing_rates = unit_data[args.response_metric]
        response_stimuli_indices = unit_data['responses'] 

        response_categories = categories_things.iloc[response_stimuli_indices]
        #kitchen_responses = response_categories["kitchen appliance"].index[np.where(response_categories["kitchen appliance"])[0]]
        #for resp in kitchen_responses : 
        #    print("session: " + session + ", site: " + site + ", channel: " + str(channel) + ", cluster: " + str(cluster) 
        #        + ", concept: " +  data.df_metadata.uniqueID[resp] ) #data.df_metadata.uniqueID[resp]  data.neural_data[session]['stimlookup'][resp]

        for name in categoryNames :
            category_counts["session"].append(session)
            category_counts["site"].append(site)
            category_counts["category"].append(name)
            category_counts["presented"].append(categories_presented[name])
            category_counts["responsive"].append(response_categories[name].sum())
            category_counts["first"].append(firstUnitInSession)

            if name not in category_presented_counts : 
                category_presented_counts[name] = defaultdict()
            if site not in category_presented_counts[name] : 
                category_presented_counts[name][site] = defaultdict()
            if session not in category_presented_counts[name][site] : 
                category_presented_counts[name][site][session] = 0
            category_presented_counts[name][site][session] += 1
        firstUnitInSession = 0

        if len(response_stimuli_indices) > 0 :
            responsive_unit_counter += 1 
                
            for name in categoryNames :
                if name not in category_responsive_counts : 
                    category_responsive_counts[name] = defaultdict()
                if site not in category_responsive_counts[name] : 
                    category_responsive_counts[name][site] = 0
                category_responsive_counts[name][site] += 1

            all_presented_indices = np.array(range(len(firing_rates)))
            consider_indices = all_presented_indices.copy()
            if args.consider_only_responses : 
                consider_indices = response_stimuli_indices

            consider_indices_THINGS = things_indices[consider_indices]        
            firing_rates_consider = firing_rates[consider_indices]
            #firing_rates_responses_df = pd.DataFrame(data=firing_rates_consider)
            categories_consider_things = categories.iloc[consider_indices_THINGS] ## categories_pca
            categories_consider_pca = categories_pca.iloc[consider_indices_THINGS]

            categories_presented = categories_things.eq(1).sum()
            category_names_presented = categories_things.columns

            if args.analyze == "PCA" : 
                #categories_responses_df = category_pca[:-1].iloc[things_indices[response_stimuli_indices]]
                count_cat = 0
                for categories_pc in categories_pca_all :
                    regression_model = sm.OLS(firing_rates_consider, categories_pc.iloc[consider_indices_THINGS].values, missing='drop') 
                    #scores = cross_val_score(estimator = regression_model, X=firing_rates_consider, y=categories_consider_df.values, cv=10)    

                    fitted_data = regression_model.fit() 
                    if len(rsquaredPCA) < count_cat + 1 :
                        rsquaredPCA.append([])
                        #meanScoresPCA.append([])
                        #stddevScoresPCA.append([])
                    rsquaredPCA[count_cat].append(fitted_data.rsquared)
                    #meanScoresPCA[count_cat].append(scores.mean())
                    #stddevScoresPCA[count_cat].append(scores.std())

                    #print("Fitting model for " + str(num_pcs[count_cat]) + " pcs")
                    count_cat += 1

            #categories_responses_df = categories.iloc[things_indices[response_stimuli_indices]]
            #categories_session = categories_consider_things.loc[consider_indices_THINGS]

            regression_model = sm.OLS(firing_rates_consider, categories_consider_things.values, missing='drop')
            regression_model2 = linear_model.LinearRegression()

            if args.analyze == "embedding" : 
                X = np.nan_to_num(categories_consider_things.values)
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
                pvalues_fit = np.ones(len(params))
                #fdr_corrected = [[False for i in range(len(params))], np.ones(len(params))]
            else :     
                
                regression_model_pca = sm.OLS(firing_rates_consider, categories_consider_pca.values, missing='drop')
                scores = cross_val_score(estimator=regression_model2, X=categories_consider_things.values, y=firing_rates_consider, cv=5)                                              
                fitted_data = regression_model.fit() 
                fitted_data_pca = regression_model_pca.fit()
                #fdr_corrected = statsmodels.stats.multitest.fdrcorrection(fitted_data.pvalues, alpha=args.alpha)
                #fdr_corrected = fdr_correction(fitted_data.pvalues, alpha=args.alpha)
                
                scores = np.abs(np.asarray(scores))
                pvalue = fitted_data.f_pvalue
                params = fitted_data.params
                rsquared = fitted_data.rsquared
                pvalues_fit = fitted_data.pvalues
                meanScore = scores.mean()
                stddevScore = scores.std()

                lofo_categories_unit = get_lofo_score(firing_rates_consider, categories_consider_things)
                lofo_score_categories.append(lofo_categories_unit)
                lofo_pca_unit = get_lofo_score(firing_rates_consider, categories_consider_pca)
                lofo_score_pca.append(lofo_pca_unit)

                #print(str(meanScore))
                #meanScores.append(scores.mean())
                #stddevScores.append(scores.std())
                
                #regression_model_pca = sm.OLS(firing_rates_consider, categories_consider_pca.values, missing='drop')
                #fitted_data_pca = regression_model.fit() 

                #lofo_score_all.append(lofo_score)
                    

                
            #if site in rsquaredSites : 
            pValueSites[site].append(pvalue)
            rsquaredCategoriesSites[site].append(rsquared)
            rsquaredPCASites[site].append(fitted_data_pca.rsquared)
            lofoCategoriesSites[site].append(statistics.mean(lofo_categories_unit))
            lofoPCASites[site].append(statistics.mean(lofo_pca_unit))
                
            #else :
            #    pValueSites[site] = [pvalue]
            #    rsquaredSites[site] = [rsquared]

            if meanScore <= 1.0 : 
                if site in meanScoresSites : 
                    meanScoresSites[site].append(meanScore)
                else : 
                    meanScoresSites[site] = [meanScore]
            #else:
            #    print("WARNING! High cv score")


            pValueSites["all"].append(pvalue)
            rsquaredCategoriesSites["all"].append(rsquared)
            rsquaredPCASites["all"].append(fitted_data_pca.rsquared)
            meanScoresSites["all"].append(meanScore)
            lofoPCASites["all"].append(statistics.mean(lofo_pca_unit))
            lofoCategoriesSites["all"].append(statistics.mean(lofo_categories_unit))

            pvalues.append(pvalue)

            if pvalue < args.threshold_p_value : #args.analyze == "embedding" and (pvalue > args.threshold_p_value) or rsquared > threshold_rsquared : #(fitted_data.f_pvalue > args.threshold_p_value)
                r_squared_counter += 1
            if pvalue < args.threshold_p_value or True: 
                ##zscores = np.concatenate(zscores, ((firing_rates_consider - mean_firing_rates) / stddev_firing_rates / mean_baseline))
                num_responsive_stimuli.append(len(response_stimuli_indices))
                num_significant_weights.append(np.count_nonzero(np.where(pvalues_fit < args.alpha)[0]))
                #if len(response_stimuli_indices) > 1 : 
                responsive_categories = categories_consider_things.iloc[response_stimuli_indices].any(axis='rows')
                if not responsive_categories.value_counts().keys().any() : 
                    num_categories_spanned.append(0)
                else: 
                    num_categories_spanned.append(responsive_categories.value_counts()[True])

            entropy = stats.entropy(params)
            entropies.append(entropy)

            for c in range(len(pvalues_fit)) : 
                if pvalues_fit[c] < args.alpha_categories : 
                    name = str(categories_consider_things.keys()[c])
                    index = np.where(categoryNames == name)[0]
                    categoriesSignificantCount[index] += 1

            fileDescription = paradigm + '_pat' + str(subject_num) + '_s' + str(session_num) + '_ch' + '{:02d}'.format(channel)  + '_cl' + str(cluster) + '_' + site 
            color_sequence = ['red' if pvalues_fit[i] < args.alpha_categories else 'blue' for i in range(len(pvalues_fit)) ]
            text_categories = np.array([str(categories_consider_things.keys()[i]) + ", p: " + str(round(pvalues_fit[i], 5)) for i in range(len(categories_consider_things.keys()))])
            
            regression_df = pd.DataFrame()
            regression_df["categories"] = text_categories
            regression_df["params"] = params
            regression_df["color"] = color_sequence
            regression_df["lofo_score_categories"] = lofo_categories_unit
            regression_df["lofo_score_pca"] = lofo_pca_unit
            regression_df["category_names"] = categories.columns
            regression_df["category_pca"] = range(len(lofo_pca_unit))
            regression_df = regression_df.sort_values("params", ascending=False)

            plt.figure(figsize=(barplotWidth, barplotHeight))
            #coef_fig = sns.barplot(x=text_categories, y=params, palette=color_sequence)
            coef_fig = sns.barplot(y="category_names", x="params", data=regression_df, palette=regression_df["color"], orient = 'h')
            #coef_fig.set_xticklabels(coef_fig.get_xticklabels(), rotation=90)
            title = "rsquared = " + str(round(rsquared, 4)) + ", pvalue: " + str(pvalue) #+ ", Entropy = " + str(entropy)
            if args.analyze != "embedding" : 
                title += ", cv mean: " + str(round(meanScore, 4)) + ", stddev: " + str(round(stddevScore, 4))
            #plt.title(title)
            coef_fig.set(xlabel=None, ylabel=None)
            plt.tick_params(axis='both', which='major', labelsize=labelsize)
            save_plt("coef_regression" + os.sep + fileDescription)
            
            plot_lofo_score(regression_df, "lofo_score_categories")
            plot_lofo_score(regression_df, "lofo_score_pca", "category_pca")

            categories_sorted_indices = np.argsort(categories_presented)
            categories_presented = categories_presented[categories_sorted_indices]
            category_names_presented = category_names_presented[categories_sorted_indices]
            plt.figure(figsize=(barplotWidth, barplotHeight))
            plt.barh(category_names_presented, categories_presented)
            plt.tick_params(axis='both', which='major', labelsize=labelsize)
            save_plt("categories_presented" + os.sep + fileDescription)


semantic_fields_path = "semantic_fields" + os.sep 
sites = list(rsquaredCategoriesSites.keys())

category_counts_df = pd.DataFrame(category_counts)
create_category_plots(category_counts_df, "all")
sites.remove('all')

for site in sites : 
    category_site_counts = category_counts_df.loc[category_counts_df['site'] == site]
    create_category_plots(category_site_counts, site)


sites = ['all'] + sorted(sites)

adjustFontSize()

sortedSignificant = np.argsort(categoriesSignificantCount)
categoryNames = categoryNames[sortedSignificant]
categoriesSignificantCount = categoriesSignificantCount[sortedSignificant]
plt.figure(figsize=(barplotWidth, barplotHeight))
plt.barh(categoryNames, categoriesSignificantCount / responsive_unit_counter)
plt.xticks(rotation=90, ha='right')
plt.tick_params(labelsize=24)
save_plt(semantic_fields_path + "num_significant_categories")

create2DhemispherePlt(rsquaredCategoriesSites, sites)
save_plt(semantic_fields_path + "rsquared_hemispheres_categories")

create2DhemispherePlt(rsquaredPCASites, sites)
save_plt(semantic_fields_path + "rsquared_hemispheres_pca")

create2DhemispherePlt(pValueSites, sites)
save_plt(semantic_fields_path + "pvalue_hemispheres_categories")

plt.figure(figsize=(10,4))
sitesTitles = [site + " (" + str(len(rsquaredCategoriesSites[site])) + ")" for site in sites]
createStdErrorMeanPlt(sitesTitles, [rsquaredCategoriesSites[site] for site in sites], "r squared of regression of unit activation based on category / feature", "r squared", [0,1])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "rsquared_sites")

plt.figure(figsize=(10,4))
createStdErrorMeanPltCompare(sitesTitles, [rsquaredCategoriesSites[site] for site in sites], [rsquaredPCASites[site] for site in sites], "categories", "pca", "rsquared of regression based on category / feature", "", [0,1])
#plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "rsquared_compare")

plt.figure(figsize=(10,4))
createStdErrorMeanPltCompare(sitesTitles, [lofoCategoriesSites[site] for site in sites], [lofoPCASites[site] for site in sites], "categories", "pca", "mean lofo score for all categories", "", [0,1])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "lofo_compare")

plt.figure(figsize=(10,4)) 
sitesTitles = [site + " (" + str(len(pValueSites[site])) + ")" for site in sites]
createStdErrorMeanPlt(sitesTitles, [pValueSites[site] for site in sites], "pvalue of regression of unit activation based on category / feature", "p value", [-0.005,0.15])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "pvalues_sites")

plt.figure(figsize=(10,4)) 
sites_scores = list(meanScoresSites.keys())
sitesTitles = [site + " (" + str(len(meanScoresSites[site])) + ")" for site in sites_scores]
createStdErrorMeanPlt(sitesTitles, [meanScoresSites[site] for site in sites_scores], "mean of crossvalidation scores of regression of unit activation based on category / feature", "mean cv score", [0,1])
plt.xticks(rotation=45, ha='right')
save_plt(semantic_fields_path + "cv_scores_sites")

plt.figure(figsize=(20,20)) 
createStdErrorMeanPlt(categories.columns, np.transpose(lofo_score_categories), "", "", sort=True, horizontal=True)
#plt.xticks(rotation=90, ha='right')
plt.tick_params(axis='both', which='major', labelsize=24)
save_plt(semantic_fields_path + "lofo_score")


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
    adjustFontSize()

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