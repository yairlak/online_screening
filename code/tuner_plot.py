#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/07/22 16:46:37
@Author  :   Katharina Karkowski 
"""

import os
import math
import numpy as np
import time
import argparse
import itertools
import statistics 

from typing import List
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import dash
#import dash_table
#import dash_core_components as dcc
#import dash_html_components as html
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# utilility modules
from data_manip import DataHandler
#from data_manip import get_THINGS_indices
from data_manip import get_mean_firing_rate_normalized

parser = argparse.ArgumentParser()
                                  

# STATS
parser.add_argument('--alpha', default=0.001, type=float,
                    help='Alpha for selecting responding units')
parser.add_argument('--alpha_unit', default=0.1, type=float,
                    help='Alpha for selecting considered concepts for responding units')
parser.add_argument('--metric', default='cosine', type=str,
                    help='Distance metric')
parser.add_argument('--start_time_fr', default=100, type=int,
                    help='Start time for calculating firing rate')
parser.add_argument('--end_time_fr', default=800, type=int,
                    help='End time for calculating firing rate')

# FLAGS
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') 
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation/') 
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--similarity_matrix_delimiter', default=',', type=str,
                    help='Similarity metric delimiter')
   
args=parser.parse_args()

@dataclass
class Tuner:
    name: str
    subject: int
    session: int
    channel: int
    cluster: int
    paradigm: str
    sitename: str
    concepts: List = field (default_factory=lambda: [])
    similarities: List = field (default_factory=lambda: [])
    firingRates: List = field (default_factory=lambda: [])

def getTunerIndex(subject, session, channel, cluster, paradigm) : 

    for i in range(len(tuners)) : 
        if tuners[i].subject == subject and tuners[i].session == session and tuners[i].channel == channel and tuners[i].cluster == cluster and tuners[i].paradigm == paradigm: 
            return i

    return -1


def saveImg(fig, filename) : 

    file = args.path2images + filename + ".png"

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fig.write_image(file)


print("\n--- START ---")
startLoadData = time.time()

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_neural_data() # -> data.neural_data
data.load_word_embeddings() # -> data.df_word_embeddings
#data.load_word_embeddings_tsne() # -> data.df_word_embeddings_tsne
data.load_similarity_matrix() # -> data.similarity_matrix
sessions = list(data.neural_data.keys())

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")
startPrepareDataTime = time.time()

data.calculate_responses(args.alpha, args.alpha_unit, args.start_time_fr, args.end_time_fr, include_self = False)

print("\nTime calculating responses: " + str(time.time() - startPrepareDataTime) + " s\n")
startPrepareDataTime = time.time()

tuners = [
    Tuner("Engine", 88, 1, 77, 1, "aos", "", [], [], []),
    Tuner("Lizard", 88, 1, 75, 2, "aos", "", [], [], []),
    Tuner("Zucchini", 88, 1, 87, 2, "aos", "", [], [], []),
    Tuner("Terrarium", 88, 3, 73, 2, "aos", "", [], [], []),
    Tuner("Photograph", 88, 3, 92, 1, "aos", "", [], [], []),
    Tuner("Ambulance", 89, 1, 84, 1, "aos", "", [], [], []),
    #Tuner("89_2", 77, 2, [], "Machine Gun", "aos",  [], [], [], [], []),
    Tuner("Machine Gun", 89, 3, 62, 1, "aos", "", [], [], []),
    Tuner("Waffle1", 90, 1, 49, 1, "aos", "", [], [], []),
    Tuner("Waffle2", 90, 1, 49, 2, "aos", "", [], [], []),
    Tuner("Waffle3", 90, 1, 49, 3, "aos", "", [], [], []),
    Tuner("Ferry1", 90, 1, 60, 2, "aos", "", [], [], []),
    Tuner("Ferry2", 90, 1, 60, 3, "aos", "", [], [], []),
    Tuner("Hamburger1-1", 90, 2, 65, 3, "aos", "", [], [], []),
    Tuner("Hamburger1-2", 90, 2, 65, 4, "aos", "", [], [], []),
    Tuner("Pancake", 90, 2, 68, 3, "aos", "", [], [], []),
    Tuner("Lipstick", 90, 3, 49, 4, "aos", "", [], [], []),
    ##Tuner("90_3", 52, 1, [], "Onion1", "aos",  [], [], [], [], []),
    ##Tuner("90_3", 52, 2, [], "Onion2", "aos",  [], [], [], [], []),
    ##Tuner("90_3", 52, 3, [], "Onion2", "aos",  [], [], [], [], []),
    Tuner("Potato", 90, 4, 52, 1, "aos", "", [], [], []),
    Tuner("Coin", 90, 5, 52, 2, "aos", "", [], [], []),
    Tuner("Hamburger2-1", 90, 5, 56, 1, "aos", "", [], [], []),
    Tuner("Hamburger2-2", 90, 5, 56, 3, "aos", "", [], [], []),
    Tuner("Donkey - Petfood - Carrot", 90, 5, 67, 1, "aos", "", [], [], []),
]

spearman = []

for responding_cell in data.response_data : 
    
    print("Spearman correlation: " + str(responding_cell['spearman']))
    spearman.append(responding_cell['spearman'])

    tunerIndex = getTunerIndex(responding_cell["subject"], responding_cell["session"], 
        responding_cell["channel_num"], responding_cell["class_num"], responding_cell["paradigm"]) 

    if tunerIndex >= 0 :
        print("Found tuner " + tuners[tunerIndex].name)
        tuners[tunerIndex].firingRates = responding_cell["firing_rates_dist"]
        tuners[tunerIndex].similarities = responding_cell["similarities"]


for session in sessions:

    subjectNum = int(session.split("_")[0])
    sessionNum = int(session.split("_")[1])
    sessionParadigm = session.split("_")[2]
    objectNames = data.neural_data[session]['objectnames']
    stimuliIndices = data.neural_data[session]['objectindices_session']
    numStimuli = len(data.neural_data[session]['stimlookup'])
    thingsIndices = data.get_THINGS_indices(data.neural_data[session]['stimlookup'])
    units = list(set(data.neural_data[session]['units'].keys()))

    for unit in units:
        unitData = data.neural_data[session]['units'][unit]
        site = data.neural_data[session]['units'][unit]['site']
        trials = unitData['trial']
        channel = unitData['channel_num']
        cluster = unitData['class_num']

        firingRates = get_mean_firing_rate_normalized(trials, stimuliIndices, args.start_time_fr, args.end_time_fr)[0]

        tunerIndex = getTunerIndex(subjectNum, sessionNum, channel, cluster, sessionParadigm) 
        if tunerIndex >= 0 :

            print("Found tuner " + tuners[tunerIndex].name)

            bestResponse = np.argmax(firingRates) # best Response = highest z? highest response strength?
            indexBest = thingsIndices[bestResponse]

            for i in range(numStimuli) : # responseIndices
                index = thingsIndices[i]
                similarity = data.similarity_matrix[index][indexBest]
                tuners[tunerIndex].sitename = site
                tuners[tunerIndex].concepts.append(objectNames[i])
                tuners[tunerIndex].similarities.append(similarity)
                tuners[tunerIndex].firingRates.append(firingRates[i])

for tuner in tuners : 

    if len(tuner.similarities,) == 0 :
        print("WARNING: Did not find tuner " + tuner.name)
        continue

    help_fig = px.scatter(x=tuner.similarities, y=tuner.firingRates, trendline="ols") # lowess --> non linear
    # extract points as plain x and y
    x_trend = help_fig["data"][1]['x']
    y_trend = help_fig["data"][1]['y']

    tunerPlot = go.Figure()
    tunerPlot.add_trace(go.Scatter(
        x=tuner.similarities, 
        y=tuner.firingRates, 
        mode='markers', 
    ))

    tunerPlot.add_trace(
        go.Scatter(x=x_trend, y=y_trend, name='trend'))

    tunerPlot.update_layout(
        title=tuner.name + " -- subj " + str(tuner.subject) + ", sess " + str(tuner.session) + ", ch " + str(tuner.channel) + ", cl " + str(tuner.cluster),
        xaxis_title="Semantic similarity",
        yaxis_title="Firing rate dist", 
        showlegend=False
    )
    saveImg(tunerPlot, "tuners_only_responses" + os.sep + args.metric + '_' + tuner.name)


spearmanPlot = go.Figure()
spearmanPlot.add_trace(go.Scatter(
    x=np.arange(len(spearman)), 
    y=spearman, 
    mode='markers', 
))

spearmanPlot.update_layout(
    xaxis_title="neuron num",
    yaxis_title="spearman correlation", 
    showlegend=False
)
saveImg(spearmanPlot, "spearman_only_responses_" + args.metric)

print("\nTime plotting data: " + str(time.time() - startPrepareDataTime) + " s\n")

spearman = np.asarray(spearman)
spearman = spearman[~np.isnan(spearman)]
print("Mean spearman: " + str(np.mean(spearman)) + ", std: " + str(np.std(spearman)))

