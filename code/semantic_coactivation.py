#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/14 17:10:12
@Author  :   Katharina Karkowski 
"""

import os
import numpy as np
import time
import argparse
import itertools

from typing import List
from dataclasses import field
from dataclasses import dataclass
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
from data_manip import get_THINGS_indices

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
parser.add_argument('--metric', default='euclidean',
                    help='distance metric')

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=True, 
                    help='If True, plotting to figures folder is supressed')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha for stats')

# PLOT

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') # also work with nos?
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation/') 

args=parser.parse_args()


@dataclass
class Region:
    sitename: str
    coactivationMatrix: List 
    copresentationMatrix: List 
    coactivationMatrixFlattened: List
    copresentationMatrixFlattened: List 
    coactivationNormalizedFlattened: List 
    similarityMatrixFlattened: List

def createTableDiv(title, figureId, fig, tableId, columnName, columnId, columnData) : 
    
    return html.Div(children=[

        html.Div([
            html.H2(children=title),
            
            html.Div([
                dcc.Graph(id=figureId, figure=fig)
            ], className="nine columns"),

            html.Div([
                dash_table.DataTable(
                    id=tableId,
                    style_cell={'textAlign': 'left'},
                    columns=[{"name": columnName, "id": columnId, "deletable": False, "selectable": True}],
                    data=columnData, 
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
    ])


def createRegionsDiv(name, fig) : 
    figureId = name + "-overview"
    tableId = name + "-regions-table"
    columnId = name + "-regions-column"
    columnData = [{columnId: site, 'id': site} for site in allSiteNames]

    return createTableDiv(
        name, figureId, fig, tableId, "Regions", columnId, columnData), figureId, tableId


def createPlot(x, y, filename) :
    fig = px.scatter(x = x, y = y)
    #fig.update_layout(graphLayoutBig)

    if not args.dont_plot : 
        fig.write_image(args.path2images + os.sep + filename + ".png")

    return fig

#############
# LOAD DATA #
#############
startLoadData = time.time()

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
data.load_neural_data() # -> data.neural_data
data.load_word_embeddings() # -> data.df_word_embeddings
#data.load_word_embeddings_tsne() # -> data.df_word_embeddings_tsne
data.load_similarity_matrix() # -> data.similarity_matrix

# WHICH SESSION(S) TO RUN
if args.session is None:
    sessions = list(data.neural_data.keys())
else:
    sessions = [args.session]

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")


figureHeight = 900
figureWidth = 1100

startPrepareDataTime = time.time()
nTHINGS = len(data.df_metadata.uniqueID)
#zscoreMatrix = np.zeros((nTHINGS, nTHINGS))

trianglarIndices = np.triu_indices(nTHINGS, k = 0) # k = 1 --> without diagonal
similarityMatrixFlattened = data.similarity_matrix.to_numpy()[trianglarIndices]
uniqueSimilarities = np.unique(data.similarity_matrix.to_numpy())

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

regions[allRegionsName] = Region(allRegionsName, np.zeros((nTHINGS, nTHINGS)), np.zeros((nTHINGS, nTHINGS)), [], [], [], [])

for session in sessions:
    if not hasattr(args, 'unit'):
        units = list(set(data.neural_data[session]['units'].keys()))
    else:
        units = [args.unit]

    thingsIndices = get_THINGS_indices(data.df_metadata, data.neural_data[session]['stimlookup'])

    # do it before to make it faster
    allSitesSession = []
    for unit in units:
        site = data.neural_data[session]['units'][unit]['site']

        if site not in allSitesSession : 
            allSitesSession.append(site)

        if site not in allSiteNames : 
            allSiteNames.append(site)
            regions[site] = Region(site, np.zeros((nTHINGS, nTHINGS)), np.zeros((nTHINGS, nTHINGS)), [], [], [], [])

    for site in allSitesSession : 
        for i1, i2 in itertools.product(thingsIndices, thingsIndices) : 
            regions[allRegionsName].copresentationMatrix[i1, i2] += len(units)
            regions[site].copresentationMatrix[i1, i2] += len(units)

    for unit in units:
        pvals = data.neural_data[session]['units'][unit]['p_vals']
        #zscores = data.neural_data[session]['units'][unit]['zscores']
        site = data.neural_data[session]['units'][unit]['site']
        responses = [thingsIndices[i] for i in np.where(pvals < args.alpha)[0]]

        for i1, i2 in itertools.product(responses, responses) : 
            regions[allRegionsName].coactivationMatrix[i1, i2]
            regions[site].coactivationMatrix[i1, i2] += 1
        

print("Time preparing data: " + str(time.time() - startPrepareDataTime) + " s\n")


#regionsColumn = [{'regions-column': site, 'id': site} for site in allSiteNames]

allRegionCoactivationPlots = []
allRegionCopresentationPlots = []
allRegionCoactivationNormalizedPlots = []

figurePrepareTime = time.time()
for site in allSiteNames : 

    siteData = regions[site]

    siteData.coactivationMatrixFlattened = siteData.coactivationMatrix[trianglarIndices]
    siteData.copresentationMatrixFlattened = siteData.copresentationMatrix[trianglarIndices]

    relevantIndices = np.where(siteData.copresentationMatrixFlattened !=0 )
    siteData.similarityMatrixFlattened = similarityMatrixFlattened[relevantIndices]
    siteData.coactivationMatrixFlattened = siteData.coactivationMatrixFlattened[relevantIndices]
    siteData.copresentationMatrixFlattened = siteData.copresentationMatrixFlattened[relevantIndices]
    siteData.coactivationNormalizedFlattened = siteData.coactivationMatrixFlattened / siteData.copresentationMatrixFlattened

    allRegionCoactivationPlots.append(
        createPlot(siteData.similarityMatrixFlattened, siteData.coactivationMatrixFlattened, "coactivation_" + site))
    allRegionCopresentationPlots.append(
        createPlot(siteData.similarityMatrixFlattened, siteData.copresentationMatrixFlattened, "copresentation_" + site))
    allRegionCoactivationNormalizedPlots.append(
        createPlot(siteData.similarityMatrixFlattened, siteData.coactivationNormalizedFlattened, "coactivation_normalized_" + site))
    
    print("Created figures for " + site)

print("\nTime creating figures: " + str(time.time() - figurePrepareTime) + " s\n")

coactivationDiv, coactivationFigId, coactivationTableId = createRegionsDiv(
    "Coactivation", siteData.coactivationMatrixFlattened)
copresentationDiv, copresentationFigId, copresentationTableId = createRegionsDiv(
    "Copresentation", siteData.copresentationMatrixFlattened)
coactivationNormalizedDiv, coactivationNormalizedFigId, coactivationNormalizedTableId = createRegionsDiv(
    "Coactivation - Normalized", siteData.coactivationNormalizedFlattened)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Tuners'),
    coactivationDiv, 
    copresentationDiv, 
    coactivationNormalizedDiv
])

print("\n--- Ready! ---\n\n")


def getActivePlot(data, activeCell) : 
    if(activeCell == None) :
        return data[0]
    else : 
        return data[activeCell['row']]

@app.callback(
    Output(component_id=coactivationFigId, component_property='figure'), 
    Input(coactivationTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionCoactivationPlots, active_cell)

@app.callback(
    Output(component_id=copresentationFigId, component_property='figure'), 
    Input(copresentationTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionCopresentationPlots, active_cell)

@app.callback(
    Output(component_id=coactivationNormalizedFigId, component_property='figure'), 
    Input(coactivationNormalizedTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionCoactivationNormalizedPlots, active_cell)

if __name__ == '__main__':
    app.run_server(debug=False) # why ?