#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/13 13:04:41
@Author  :   Katharina Karkowski 
"""

import os
import glob
import argparse
from pprint import pprint
import numpy as np
import pandas as pd

import time
from typing import List
from dataclasses import dataclass
import mat73
import scipy
import scipy.io
import scipy.interpolate 
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
#from dash import dash_table
#from dash import dcc
#from dash import html
from dash.dependencies import Input, Output

# utilility modules
from data_manip import DataHandler
from data_manip import prepare_features, prepare_neural_data

parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str,
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# DATA AND MODEL
parser.add_argument('--concept_source', default='Top-down Category (manual selection)',
                    help='Field name from THINGS for semantic categories',
                    choices = ['Bottom-up Category (Human Raters)',
                               'Top-down Category (WordNet)',
                               'Top-down Category (manual selection)'])

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=False, # TODO
                    help='If True, plotting to figures folder is supressed')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha for stats')

# PLOT
parser.add_argument('--interpolation_factor', type=float, default=100,
                    help='heatmap interpolation grid size')
parser.add_argument('--padding_factor', type=float, default=1.1,
                    help='padding around datapoints')

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2wordembeddingsTSNE',
                    default='../data/THINGS/sensevec_TSNE.csv')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') # also work with nos?
parser.add_argument('--path2images', 
                    default='../figures/heatmaps/') 

args=parser.parse_args()

## TODO: v7 os_responses


@dataclass
class Tuner:
    subjectsession: str
    channel: int
    cluster: int
    name: str
    paradigm: str
    stimuli: List[int] 
    stimuliX: List[float]
    stimuliY: List[float]
    stimuliNames: List[str]
    zscores: List[float]

def rescaleX(x) :
    xPadding = np.asarray([xOut / args.padding_factor for xOut in x])
    return args.interpolation_factor * (xPadding - xMinThings) / (xMaxThings - xMinThings) - 0.5

def rescaleY(y) :
    yPadding = np.asarray([yOut / args.padding_factor for yOut in y])
    return args.interpolation_factor * (yPadding - yMinThings) / (yMaxThings - yMinThings) - 0.5

def getInterpolatedMap(x, y, z) : 
    xMin = xMinThings * args.padding_factor
    xMax = xMaxThings * args.padding_factor
    yMin = yMinThings * args.padding_factor
    yMax = yMaxThings * args.padding_factor

    # Create regular grid
    xi, yi = np.linspace(xMin, xMax, args.interpolation_factor), np.linspace(yMin, yMax, args.interpolation_factor)
    xi, yi = np.meshgrid(xi, yi)

    xWithBorder = np.copy(x)
    yWithBorder = np.copy(y)
    zWithBorder = np.copy(z)

    xRange = np.arange(int(xMin), int(xMax))
    xWithBorder = np.append(xWithBorder, xRange) 
    xWithBorder = np.append(xWithBorder, xRange) 
    yWithBorder = np.append(yWithBorder, np.full(len(xRange), yMin))
    yWithBorder = np.append(yWithBorder, np.full(len(xRange), yMax))
    zWithBorder = np.append(zWithBorder, np.zeros(len(xRange) * 2))

    yRange = np.arange(int(yMin), int(yMax))
    xWithBorder = np.append(xWithBorder, np.full(len(yRange), xMin)) 
    xWithBorder = np.append(xWithBorder, np.full(len(yRange), xMax))
    yWithBorder = np.append(yWithBorder, yRange)
    yWithBorder = np.append(yWithBorder, yRange)
    zWithBorder = np.append(zWithBorder, np.zeros(len(yRange) * 2))

    # Interpolate missing data
    rbf = scipy.interpolate.Rbf(xWithBorder, yWithBorder, zWithBorder, function='linear')
    zi = rbf(xi, yi)

    return px.imshow(zi,aspect=0.8,color_continuous_scale='RdBu',origin='lower')

def createHeatMap(tuner) : 
    
    fig = getInterpolatedMap(np.array(tuner.stimuliX), np.array(tuner.stimuliY), np.array(tuner.zscores)) 
    
    zScores = np.copy(tuner.zscores)
    zScores -= min(zScores)
    zScores /= max(zScores)

    for stimulusNum in range(len(tuner.stimuliNames)) :
        opacityStim = zScores[stimulusNum]
        fig.add_trace(
            go.Scatter(
                mode='text',
                x=rescaleX([tuner.stimuliX[stimulusNum]]), y=rescaleY([tuner.stimuliY[stimulusNum]]),
                text=[tuner.stimuliNames[stimulusNum]],
                hovertext=[tuner.stimuliNames[stimulusNum] + ", z: " + str(round(tuner.zscores[stimulusNum], 2))],
                opacity=opacityStim,
                textfont=dict(
                    size=12,
                    color="black"
                ),
                name='zscore'
            )
        )

    fig.update_layout(graphLayout)

    return fig

#############
# LOAD DATA #
#############
startLoadData = time.time()

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_neural_data() # -> data.neural_data
#data.load_word_embeddings() # -> data.df_word_embeddings
data.load_word_embeddings_tsne() # -> data.df_word_embeddings_tsne

# WHICH SESSION(S) TO RUN
if args.session is None:
    sessions = list(data.neural_data.keys())
else:
    sessions = [args.session]

print("Time loading data: " + str(time.time() - startLoadData) + " s\n")


tuners = [ # clusters might not fit (manual clustering took place)
    #Tuner("088e03aos1", 17, 1, "Pacifier", "aos", [], [], [], [], []),
    Tuner("88_1", 77, 1, "Engine", "aos", [], [], [], [], []),
    Tuner("88_1", 75, 2, "Lizard", "aos", [], [], [], [], []), 
    Tuner("88_1", 87, 2, "Zucchini", "aos", [], [], [], [], []), 
    Tuner("88_3", 92, 1, "Photograph", "aos",  [], [], [], [], []),
    Tuner("89_1", 84, 1, "Ambulance", "aos",  [], [], [], [], []),
    Tuner("89_2", 77, 2, "Machine Gun", "aos",  [], [], [], [], []),
    Tuner("90_1", 49, 1, "Waffle1", "aos",  [], [], [], [], []),
    Tuner("90_1", 49, 2, "Waffle2", "aos",  [], [], [], [], []),
    Tuner("90_1", 49, 3, "Waffle3", "aos",  [], [], [], [], []),
    Tuner("90_1", 60, 2, "Ferry1", "aos",  [], [], [], [], []),
    Tuner("90_1", 60, 3, "Ferry2", "aos",  [], [], [], [], []),
    Tuner("90_2", 65, 3, "Hamburger1", "aos",  [], [], [], [], []),
    Tuner("90_2", 65, 4, "Hamburger2", "aos",  [], [], [], [], []),
    Tuner("90_2", 68, 3, "Pancake", "aos",  [], [], [], [], []),
    Tuner("90_3", 49, 4, "Lipstick", "aos",  [], [], [], [], []),
    #Tuner("90_3", 52, 1, "Onion1", "aos",  [], [], [], [], []),
    #Tuner("90_3", 52, 2, "Onion2", "aos",  [], [], [], [], []),
    #Tuner("90_3", 52, 3, "Onion2", "aos",  [], [], [], [], []),
    Tuner("90_4", 52, 1, "Potato", "aos",  [], [], [], [], []),
    Tuner("90_5", 52, 2, "Coin", "aos",  [], [], [], [], []),
    Tuner("90_5", 56, 1, "Hamburger1", "aos",  [], [], [], [], []),
    Tuner("90_5", 56, 3, "Hamburger2", "aos",  [], [], [], [], []),
    Tuner("90_5", 67, 1, "Donkey - Petfood - Carrot", "aos",  [], [], [], [], []),

]

nConcepts = data.df_word_embeddings_tsne.shape[0]
xThings = data.df_word_embeddings_tsne[:][1]
yThings = data.df_word_embeddings_tsne[:][0]

xMinThings = xThings.min()
xMaxThings = xThings.max()
yMinThings = yThings.min()
yMaxThings = yThings.max()

xThingsRescaled = rescaleX(xThings)
yThingsRescaled = rescaleY(yThings)

startPrepareData = time.time()
inputPath = args.path2data #+ tuner.paradigm + "_after_manual_clustering/"
allUnits = []

for session in sessions:
    if not hasattr(args, 'unit'):
        units = list(set(data.neural_data[session]['units'].keys()))
    else:
        units = [args.unit]

    patientNr = session.split("_")[0]
    sessionNr = session.split("_")[1]

    stimuliNames = np.unique(data.neural_data[session]['objectnames'])

    stimuliNums = []
    stimuliX = []
    stimuliY = []

    for stim in stimuliNames : 
        stimInThings = np.where(data.df_metadata.uniqueID == stim)[0][0]
        stimuliNums.append(stimInThings)
        stimuliX.append(xThings[stimInThings])
        stimuliY.append(yThings[stimInThings])
    
    for unit in units:
        pvals = data.neural_data[session]['units'][unit]['p_vals']
        zscores = data.neural_data[session]['units'][unit]['zscores']
        channelName = data.neural_data[session]['units'][unit]['channel_name']
        channel = data.neural_data[session]['units'][unit]['channel_num']
        cluster = data.neural_data[session]['units'][unit]['class_num']
        name = "pat " + str(patientNr) + ", session " + str(sessionNr) + ", " + channelName + ", channel " + str(channel) + ", cluster " + str(cluster)
        
        for tuner in tuners : 
            if tuner.subjectsession == session and tuner.channel == channel and tuner.cluster == cluster : 
                tuner.zscores = zscores
                tuner.stimuli = stimuliNums
                tuner.stimuliNames = stimuliNames
                tuner.stimuliX = stimuliX
                tuner.stimuliY = stimuliY

        if all(pval >= args.alpha for pval in pvals) : 
            #print("Skipping " + subjectSession + ", cell " + str(cellNum))
            continue

        allUnits.append(Tuner(session, channel, cluster, name, "aos", 
            stimuliNums, stimuliX, stimuliY, stimuliNames, zscores))

    print("Prepared session " + session)

print("Time preparing data: " + str(time.time() - startPrepareData) + " s")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

table_options = [{'tuners-column': tuners[i].name, 'id': i} for i in range(len(tuners))]
table_options_heatmap = [{'tuners-heatmap-column': tuners[i].name, 'id': i} for i in range(len(tuners))]

figureHeight = 900

graphLayout = go.Layout(
    xaxis=dict(ticks='', showticklabels=False),
    yaxis=dict(ticks='', showticklabels=False),
    showlegend=False, 
    autosize=False,
    height=600,
    width=900
)

graphLayoutBig = go.Layout(
    height=figureHeight,
    width=1100
)

heatmap = getInterpolatedMap(np.array(tuners[0].stimuliX), np.array(tuners[0].stimuliY), np.array(tuners[0].zscores))

allHeatmaps = []
for cell in allUnits : 
    heatMapFigure = createHeatMap(cell)
    allHeatmaps.append(
        html.Div([
            html.H3(children='Activation heatmap ' + cell.name),
            html.Div([
                dcc.Graph(id='heatmap-' + cell.name, figure=heatMapFigure)
            ], className="nine columns"),
        ], className="row"),
    )
    if not args.dont_plot : 
        heatMapFigure.write_image(args.path2images + "\\" + cell.subjectsession + "_ch" + str(cell.channel) + "_cl" + str(cell.cluster) + ".png")
    print("Created heatmap for " + cell.name)

print("Done loading all heatmaps!")

app.layout = html.Div(children=[
    html.H1(children='Tuners'),

    html.Div([
        html.H2(children='Activation heatmap'),
        
        html.Div([
            dcc.Graph(id='heatmap', figure=heatmap)
        ], className="nine columns"),

        html.Div([
            dash_table.DataTable(
                id='tuners-heatmap-table',
                style_cell={'textAlign': 'left'},
                columns=[{"name": "Tuners", "id": "tuners-heatmap-column", "deletable": False, "selectable": True}],
                data=table_options_heatmap, 
                editable=False,
                page_action='none',
                style_table={
                    'margin-top': 100,
                    'height': figureHeight - 180,
                    'overflowY': 'scroll'
                }
            ), 
        ], className="two columns"),
    ], className="row"),

    html.Div(children = allHeatmaps),
])


@app.callback(
    Output(component_id='heatmap', component_property='figure'), #src
    Input('tuners-heatmap-table', 'active_cell')
)
def update_output_div(active_cell):
    if(active_cell == None) :
        tuner = tuners[0]
    else : 
        tuner = tuners[active_cell['row']]

    fig = createHeatMap(tuner)
    fig.update_layout(graphLayoutBig)

    return fig 


if __name__ == '__main__':
    app.run_server(debug=False) # why ?
