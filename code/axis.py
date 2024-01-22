

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib as plt

from collections import defaultdict
from sklearn.decomposition import PCA

from plot_helper import *
from utils import getSite, clear_folder
from data_manip import DataHandler

parser = argparse.ArgumentParser()

# SESSION/UNIT

# ANALYSIS
parser.add_argument('--min_t', default=0, type=int, #100
                    help="Relevant for calculating mean firing_rate. \
                        Min time offset of spike after stimulus onset.")
parser.add_argument('--max_t', default=1000, type=int, # 800
                    help="Relevant for calculating mean firing_rate. \
                        Max time offset of spike after stimulus onset.")

# FLAGS
parser.add_argument('--save_data', default=False,
                    help='If True, all heatmaps are saved')
parser.add_argument('--load_data', default=True,
                    help='If True, all heatmaps are loaded')
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--only_SU', default=True, 
                    help='If True, only single units are considered')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--pcs', default=27, 
                    help='Number of principal components for pca')

# PLOT
parser.add_argument('--plot_regions', default='hemispheres',
                    help='"full"->all regions, "hemispheres"->split into hemispheres, "collapse_hemispheres"->regions of both hemispheres are collapsed')  

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2wordembeddingsTSNE',
                    default='../data/THINGS/sensevec_TSNE.csv')
parser.add_argument('--path2data', 
                    default='../data/aos_one_session/') # also work with nos? aos_after_manual_clustering, aos_selected_sessions, aos_one_session
parser.add_argument('--path2neuraldata', 
                    default='../data/') 
parser.add_argument('--path2images', 
                    default='../figures/axis/') 

def get_data_path() : 
    return os.path.dirname(args.path2neuraldata + os.sep + "data" + os.sep)

def save_plt(filename) : 

    file = args.path2images + os.sep + filename 

    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()
    
def normalize(vector) : 
    norm = np.linalg.norm(vector)
    if norm == 0: 
       return vector
    return vector / norm

def random_point_on_plane(normal_vector, point_on_plane, num_dimensions):
    # Generate a random vector in the plane
    random_vector = np.random.randn(num_dimensions)

    # Project the random vector onto the plane by subtracting its component in the normal direction
    projection = np.dot(random_vector, normal_vector)
    random_point = random_vector - projection * normal_vector

    # Translate the point to be on the plane by adding the known point on the plane
    random_point_on_plane = random_point + point_on_plane

    return random_point_on_plane


def line_coefficients_2d(point1, point2):
    # Calculate the direction vector
    direction_vector = np.array(point2) - np.array(point1)

    # Calculate the normal vector
    normal_vector = np.roll(direction_vector, 1) #np.array([direction_vector[1], -direction_vector[0]])

    # Normalize the coefficients
    magnitude = np.linalg.norm(normal_vector)
    coefficients = normal_vector / magnitude

    return coefficients

def distance_to_line(point, line_coefficients):
    # point is a numpy array representing the coordinates of the point
    # line_coefficients is a numpy array representing the coefficients of the line equation

    #line_coefficients_new = line_coefficients[:-1]
    line_coefficients_new = line_coefficients
    # Calculate the numerator (signed distance)
    numerator = np.abs(np.dot(line_coefficients_new, point))
    #numerator = np.abs(np.dot(line_coefficients[:-1], point) + line_coefficients[-1])

    # Calculate the denominator (normalization factor)
    denominator = np.linalg.norm(line_coefficients_new)

    # Calculate the distance
    distance = numerator / denominator

    return distance

def prepare_data() : 
    
    startLoadData = time.time()
    args=parser.parse_args()
    data = DataHandler(args) # class for handling neural and feature data
    data.load_metadata() # -> data.df_metadata
    data.load_neural_data() # -> data.neural_data
    data.load_word_embeddings() # -> data.df_word_embeddings
    
    word_embeddings = data.df_word_embeddings.fillna(0)
    pca = PCA(n_components=args.pcs)
    principalComponents = pca.fit_transform(word_embeddings)

    heatmap_df = pd.read_pickle(args.path2neuraldata + os.sep + "heatmaps" + os.sep + "heatmaps_zscores_1000.pk1")
    #heatmap_df["sta"] = np.zeros((len(heatmap_df.index), 300))
    #heatmap_df["sta_pca"] = np.zeros((len(heatmap_df.index), len(principalComponents)))
    #heatmap_df["axis_dist_embedding"] = np.nan
    #heatmap_df["axis_dist_pca"] = np.nan

    print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")
    startPrepareSessionData = time.time()

    sta_embedding_all = []
    sta_pca_all = []
    dist_embedding = []
    dist_pca = []
    zscores = []
    names = []

    for session in list(data.neural_data.keys()) : 

        subjectNum = int(session.split("_")[0])
        sessionNum = int(session.split("_")[1])
    
        units = list(set(data.neural_data[session]['units'].keys()))
        thingsIndices = data.get_THINGS_indices(data.neural_data[session]['stimlookup'])

        for unit in units:
            site = data.neural_data[session]['units'][unit]['site']
            num_spikes = data.neural_data[session]['units'][unit]['num_spikes']
            channel = data.neural_data[session]['units'][unit]['channel_num']
            cluster = data.neural_data[session]['units'][unit]['class_num']
            kind = data.neural_data[session]['units'][unit]['kind']
            channelName = data.neural_data[session]['units'][unit]['channel_name']
            numStim = len(num_spikes)
            
            #if not site in sitesToConsider : 
            #    continue

            #site = getSite(site, args.plot_regions)

            normalizer = 0
            sta = np.zeros((len(data.df_word_embeddings.columns)), dtype=float)
            sta_pca = np.zeros((args.pcs), dtype=float)

            for i in range(numStim) :
                normalizer += num_spikes[i]
                sta += num_spikes[i] * np.array(data.df_word_embeddings.iloc[thingsIndices[i]]) 
                sta_pca += num_spikes[i] * principalComponents[thingsIndices[i]] 

            sta /= float(normalizer)
            sta_pca /= float(normalizer)

            axis_distances_embedding = np.zeros((numStim), dtype=float)
            axis_distances_pca = np.zeros((numStim), dtype=float)

            second_point_on_plane = random_point_on_plane(sta, sta, len(sta))
            second_point_on_plane_pca = random_point_on_plane(sta_pca, sta_pca, len(sta_pca))

            for stim in range(numStim) : 
                embedding = data.df_word_embeddings.iloc[thingsIndices[stim]] 
                embedding_pca = principalComponents[thingsIndices[stim]]
                line_coefficients_embedding = line_coefficients_2d(sta, np.zeros(len(sta)))
                line_coefficients_embedding = line_coefficients_2d(sta, second_point_on_plane)
                line_coefficients_pca = line_coefficients_2d(sta_pca, second_point_on_plane_pca)
                axis_distance_stim_embedding = distance_to_line(embedding, line_coefficients_embedding)
                axis_distance_stim_pca = distance_to_line(embedding_pca, line_coefficients_pca)
                axis_distances_embedding[stim] = axis_distance_stim_embedding
                axis_distances_pca[stim] = axis_distance_stim_pca

            name = "pat " + str(subjectNum) + ", session " + str(sessionNum) + ", " + channelName + ", channel " + str(channel) + ", cluster " + str(cluster) + ", " + kind
            df_index = np.where(heatmap_df["names"] == name)[0]
            if len(df_index) == 0 : 
                continue
            if len(df_index) > 1 : 
                print("ERROR")

            sta_embedding_all.append(sta)
            sta_pca_all.append(sta_pca)
            dist_embedding.append(axis_distances_embedding)
            dist_pca.append(axis_distances_pca)
            names.append(name)
            zscores.append(heatmap_df["zscores"][df_index[0]])
            #    unit_index = int(df_index[0])
            #heatmap_df.at[12, "sta"] = [1,2,3]
            #heatmap_df.at[int(df_index[0]), "sta"] = sta
            #heatmap_df.loc[heatmap_df["names"] == name, 'sta'] = sta
            #heatmap_df.loc[heatmap_df["names"] == name, 'sta_pca'] = sta_pca
            #heatmap_df.loc[heatmap_df["names"] == name, 'axis_dist_embedding'] = axis_distances_embedding
            #heatmap_df.loc[heatmap_df["names"] == name, 'axis_dist_pca'] = axis_distances_pca
                #heatmap_df.at[unit_index, "sta"] = sta
                #heatmap_df.at[unit_index, "sta_pca"] = sta_pca
                #heatmap_df.at[unit_index, "axis_dist_embedding"] = axis_distances_embedding
                #heatmap_df.at[unit_index, "axis_dist_pca"] = axis_distances_pca

    
    data_df = pd.DataFrame({"zscores" : zscores, 
                "sta_embedding" : sta_embedding_all, 
                "sta_pca" : sta_pca_all, 
                "dist_embedding" : dist_embedding, 
                "dist_pca" : dist_pca, 
                "name" : names})
    

    os.makedirs(os.path.dirname(get_data_path()), exist_ok=True)
    data_df.to_pickle(get_data_path() + os.sep + "axis.pk1")

    print("\nTime preparing data: " + str(time.time() - startPrepareSessionData) + " s\n")

    return data_df



args=parser.parse_args()
sitesToConsider = ["LA", "RA", "LEC", "REC", "LAH", "RAH", "LMH", "RMH", "LPHC", "RPHC", "LPIC", "RPIC"]

df = prepare_data()
df = pd.read_pickle(get_data_path() + os.sep + "axis.pk1")
clear_folder(args.path2images)

spearman_embedding = []
spearman_pca = []
spearman_embedding_p = []
spearman_pca_p = []

for index, unit in df.iterrows() :
    dist_embedding = unit["dist_embedding"]
    dist_pca = unit["dist_pca"]
    scores = unit["zscores"]

    if not isinstance(dist_pca, np.ndarray) : 
        continue

    spearman_embedding_value = 0.0
    spearman_embedding_unit = stats.spearmanr(dist_embedding, scores) 
    if not math.isnan(spearman_embedding_unit.correlation) : 
        spearman_embedding_value = spearman_embedding_unit.correlation
        spearman_embedding.append(spearman_embedding_unit.correlation)
        spearman_embedding_p.append(spearman_embedding_unit.pvalue)
        if spearman_embedding_unit.pvalue < 0.01 : 
            print("found good spearman for embedding: " + str(spearman_embedding_unit.pvalue) + ", " + unit["name"])

    spearman_pca_value = 0.0
    spearman_pca_unit = stats.spearmanr(dist_pca, scores) 
    if not math.isnan(spearman_pca_unit.correlation) : 
        spearman_pca_value = spearman_pca_unit.correlation
        spearman_pca.append(spearman_pca_unit.correlation)
        spearman_pca_p.append(spearman_pca_unit.pvalue)
        if spearman_pca_unit.pvalue < 0.01 : 
            print("found good spearman for pca: " + str(spearman_pca_unit.pvalue) + ", " + unit["name"])

    plt.scatter(dist_embedding, scores)
    save_plt("embedding" + os.sep + str(spearman_embedding_value) + '_' + unit["name"])
    #plt.savefig(args.path2images + "embedding" + os.sep + unit["name"])
    
    plt.scatter(dist_pca, scores)
    save_plt("pca" + os.sep + str(spearman_pca_value) + '_' + unit["name"])
    #plt.savefig(args.path2images + "pca" + os.sep + unit["name"])
    

print("spearman_embedding: " + str(spearman_embedding))
print("spearman_pca: " + str(spearman_pca))
print("spearman_embedding p: " + str(spearman_embedding_p))
print("spearman_pca p: " + str(spearman_pca_p))

print("DONE")