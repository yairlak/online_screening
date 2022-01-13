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

# PATHS
parser.add_argument('--path2things', 
                    default='../data/THINGS/')
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') # also work with nos?
parser.add_argument('--path2images', 
                    default='../figures/heatmaps/') 

args=parser.parse_args()



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
    xPadding = [xOut / paddingFactor for xOut in x]
    return interpolateFactor * (xPadding- xMinThings) / (xMaxThings - xMinThings) - 0.5

def rescaleY(y) :
    yPadding = [yOut / paddingFactor for yOut in y]
    return interpolateFactor * (yPadding - yMinThings) / (yMaxThings - yMinThings) - 0.5

def getInterpolatedMap(x, y, z) : 
    xMin = xMinThings * paddingFactor
    xMax = xMaxThings * paddingFactor
    yMin = yMinThings * paddingFactor
    yMax = yMaxThings * paddingFactor

    # Create regular grid
    xi, yi = np.linspace(xMin, xMax, interpolateFactor), np.linspace(yMin, yMax, interpolateFactor)
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
#data = DataHandler(args) # class for handling neural and feature data
#data.load_metadata() # -> data.df_metadata
#data.load_neural_data() # -> data.neural_data
#data.load_word_embeddings() # -> data.df_word_embeddings

# WHICH SESSION(S) TO RUN
#if args.session is None:
#    sessions = list(data.neural_data.keys())
#else:
#    sessions = [args.session]

saveImages = False
interpolateFactor = 100
paddingFactor = 1.1
alpha = 0.001

thingsConceptsNames = pd.read_csv(args.path2things + "unique_id.csv", sep='\n', header=None).values[:,0]
senseVecs_embedded = pd.read_csv(args.path2things + "sensevec_TSNE.csv", sep=';', header=None).values

tuners = [ # clusters might not fit (manual clustering took place)
    #Tuner("088e03aos1", 17, 1, "Pacifier", "aos", [], [], [], [], []),
    Tuner("088e03aos1", 77, 1, "Engine", "aos", [], [], [], [], []),
    Tuner("088e03aos1", 75, 2, "Lizard", "aos", [], [], [], [], []), 
    Tuner("088e03aos1", 87, 2, "Zucchini", "aos", [], [], [], [], []), 
    Tuner("088e28aos3", 92, 1, "Photograph", "aos",  [], [], [], [], []),
    Tuner("089e02aos1", 84, 1, "Ambulance", "aos",  [], [], [], [], []),
    Tuner("089e24aos2", 77, 2, "Machine Gun", "aos",  [], [], [], [], []),
    Tuner("090e05aos1", 49, 1, "Waffle1", "aos",  [], [], [], [], []),
    Tuner("090e05aos1", 49, 2, "Waffle2", "aos",  [], [], [], [], []),
    Tuner("090e05aos1", 49, 3, "Waffle3", "aos",  [], [], [], [], []),
    Tuner("090e05aos1", 60, 2, "Ferry1", "aos",  [], [], [], [], []),
    Tuner("090e05aos1", 60, 3, "Ferry2", "aos",  [], [], [], [], []),
    Tuner("090e13aos2", 65, 3, "Hamburger1", "aos",  [], [], [], [], []),
    Tuner("090e13aos2", 65, 4, "Hamburger2", "aos",  [], [], [], [], []),
    Tuner("090e13aos2", 68, 3, "Pancake", "aos",  [], [], [], [], []),
    Tuner("090e23aos3", 49, 4, "Lipstick", "aos",  [], [], [], [], []),
    #Tuner("090e23aos3", 52, 1, "Onion1", "aos",  [], [], [], [], []),
    #Tuner("090e23aos3", 52, 2, "Onion2", "aos",  [], [], [], [], []),
    Tuner("090e37aos5", 52, 2, "Coin", "aos",  [], [], [], [], []),
    Tuner("090e37aos5", 56, 1, "Hamburger1", "aos",  [], [], [], [], []),
    Tuner("090e37aos5", 56, 3, "Hamburger2", "aos",  [], [], [], [], []),

]

nConcepts = senseVecs_embedded.shape[0]
xThings = np.zeros(nConcepts)
yThings = np.zeros(nConcepts)
for i in range(nConcepts) :
    xThings[i] = senseVecs_embedded[i, 1]
    yThings[i] = senseVecs_embedded[i, 0]

xMinThings = xThings.min()
xMaxThings = xThings.max()
yMinThings = yThings.min()
yMaxThings = yThings.max()

xThingsRescaled = rescaleX(xThings)
yThingsRescaled = rescaleY(yThings)

inputPath = args.path2data #+ tuner.paradigm + "_after_manual_clustering/"
allCells = []
allCherriesFiles = glob.glob(inputPath + "*_cherries.mat")
for cherryFile in allCherriesFiles : 
    subjectSession = cherryFile.split("\\")[-1].split("_")[0]

    cherries = scipy.io.loadmat(inputPath + subjectSession + "_cherries.mat")["cherries"]
    stimlookup = scipy.io.loadmat(inputPath + subjectSession + "_stimlookup.mat")["stimlookup"][0]
    zScores = scipy.io.loadmat(inputPath + subjectSession + "_zscores.mat")["zscores_rs"]
    pvals = mat73.loadmat(inputPath + subjectSession + "_os_responses.mat")["pvals_rs"]

    channels = cherries["channr"][0]
    clusters = cherries["classno"][0]

    stimuliNums = []
    stimuliNames = []
    stimuliX = []
    stimuliY = []

    for stim in stimlookup : 
        stimInThings = np.where(thingsConceptsNames == stim[0])[0][0]
        stimuliNums.append(stimInThings)
        stimuliNames.append(stim[0])
        stimuliX.append(xThings[stimInThings])
        stimuliY.append(yThings[stimInThings])
    
    for cellNum in range(len(cherries[0])) : 
        if all(pval >= alpha for pval in pvals[cellNum]) : 
            #print("Skipping " + subjectSession + ", cell " + str(cellNum))
            continue
        print("Loaded " + subjectSession + ", cell " + str(cellNum))
        channel = channels[cellNum][0][0]
        cluster = clusters[cellNum][0][0]
        name = subjectSession + " channel " + str(channel) + " cluster " + str(cluster)
        allCells.append(Tuner(subjectSession, channel, cluster, name, "aos", 
        stimuliNums, stimuliX, stimuliY, stimuliNames, zScores[cellNum]))

    print("Loaded " + subjectSession + " completely\n")

for tuner in tuners : 
    cherries = scipy.io.loadmat(inputPath + tuner.subjectsession + "_cherries.mat")["cherries"]
    channels = cherries["channr"][0]
    clusters = cherries["classno"][0]
    #channelNums = np.where(channels == [tuner.channel])
    #clusterNums = np.where(clusters == [tuner.cluster])
    clusterNum = np.intersect1d(np.where(channels == [tuner.channel]), np.where(clusters == [tuner.cluster]))[0]

    stimlookup = scipy.io.loadmat(inputPath + tuner.subjectsession + "_stimlookup.mat")["stimlookup"][0]
    tuner.zscores = scipy.io.loadmat(inputPath + tuner.subjectsession + "_zscores.mat")["zscores_rs"][clusterNum]#.clip(min = 0)
    
    for stim in stimlookup : 
        stimInThings = np.where(thingsConceptsNames == stim[0])[0][0]
        tuner.stimuli.append(stimInThings)
        tuner.stimuliNames.append(stim[0])
        tuner.stimuliX.append(xThings[stimInThings])
        tuner.stimuliY.append(yThings[stimInThings])


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

print("Loaded data!\n")

allHeatmaps = []
for cell in allCells : 
    heatMapFigure = createHeatMap(cell)
    allHeatmaps.append(
        html.Div([
            html.H3(children='Activation heatmap ' + cell.name),
            html.Div([
                dcc.Graph(id='heatmap-' + cell.name, figure=heatMapFigure)
            ], className="nine columns"),
        ], className="row"),
    )
    if saveImages : 
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
