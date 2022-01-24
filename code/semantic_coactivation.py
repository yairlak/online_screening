#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/14 17:10:12
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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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
parser.add_argument('--metric', default='euclidean',
                    help='Distance metric')
parser.add_argument('--similarity_matrix_delimiter', default=',', type=str,
                    help='Similarity metric delimiter')

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=True, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for stats')

# PLOT
parser.add_argument('--step', type=float, default=0.01,
                    help='Plotting detail')
parser.add_argument('--max_stdev_outliers', type=float, default=5,
                    help='Limit for excluding outliers')   
parser.add_argument('--max_responses_unit', type=float, default=20,
                    help='Limit for counting responses per unit for histogram')               

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2data', 
                    default='../data/aosnos_after_manual_clustering/') # also work with nos?
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation/') 

args=parser.parse_args()


@dataclass
class NormArray:
    y : List = field(default_factory=lambda: np.zeros((nSimilarities)))
    normalizer : List = field(default_factory=lambda: np.zeros((nSimilarities)))
    similarity : List = field(default_factory=lambda: [])
    relevantIndices : List = field(default_factory=lambda: [])

    def normalize(self) : 
        self.relevantIndices = np.where(self.normalizer != 0)
        self.normalizer = self.normalizer[self.relevantIndices]
        self.similarity = uniqueSimilarities[self.relevantIndices]
        self.y = self.y[self.relevantIndices] / self.normalizer
        #return y, uniqueSimilarities[relevantIndices], relevantIndices

    def addValue(self, index, value) : 
        self.y[index] += value
        self.normalizer[index] += 1


@dataclass
class Region:
    sitename: str
    coactivationNorm : NormArray = field(default_factory=lambda: NormArray())
    zScoresNorm : NormArray = field(default_factory=lambda: NormArray())
    cohensD : NormArray = field(default_factory=lambda: NormArray())
    numResponsesHist : List = field (default_factory=lambda: [])

def createTableDiv(title, figureId, tableId, columnName, columnId, columnData) : 
    
    return html.Div(children=[

        html.Div([
            html.H2(children=title),
            
            html.Div([
                dcc.Graph(id=figureId)
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

def createRegionsDiv(name) : 
    figureId = name + "-overview"
    tableId = name + "-regions-table"
    columnId = name + "-regions-column"
    columnData = [{columnId: site, 'id': site} for site in allSiteNames]

    return createTableDiv(
        name, figureId, tableId, "Regions", columnId, columnData), figureId, tableId

def smooth(y, numPoints):
    if len(y) == 0 : 
        return y 
    else : 
        return np.convolve(y, np.ones(numPoints)/numPoints, mode='same')

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fitPartialGaussian(x, y) : 
    if len(y) <= 2 or not np.any(y > 0): 
        return x, y

    maxIndex = np.amax(np.where(y > 0)) + 1

    xGauss = x[:maxIndex]
    xGauss = np.append(xGauss, xGauss[-1]-xGauss[-2] + xGauss[-1])
    for i in range(1,maxIndex) : 
        xGauss = np.append(xGauss, x[maxIndex-i]-x[maxIndex-i-1] + xGauss[-1])

    yPart = y[:maxIndex]
    yGaussInput = np.concatenate((yPart, yPart[::-1]))

    xGauss, yGauss = fitGauss(xGauss, yGaussInput)

    xGaussPart = xGauss[:maxIndex]
    yGaussPart = yGauss[:maxIndex]

    return xGaussPart, yGaussPart 

def fitGauss(x, y) : 
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    try : 
        popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    except Exception: 
        print("WARNING: Error fitting gauss")
        return [], []

    yGauss = Gauss(x, *popt)

    return x, yGauss

def addPlot(fig, x, y, mode, name) : 
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name
        ))

def saveImg(fig, filename) : 
    if not args.dont_plot : 
        fig.write_image(args.path2images + filename + ".png")

def createPlot(x, y, yLabel, filename, plotHalfGaussian) :

    if len(y) == 0 : 
        meanY = 0
        stdevY = 1
    elif len(y) == 1 : 
        meanY = y[0]
        stdevY = 1
    else : 
        meanY = statistics.mean(y)
        stdevY = statistics.stdev(y)
    relevantIndices = np.where(abs(y - meanY) <= args.max_stdev_outliers * stdevY)
    xWithoutOutliers = x[relevantIndices]
    yWithoutOutliers = y[relevantIndices]

    if len(x) - len(xWithoutOutliers) > 0 : 
        print("-Excluded " + str(len(x) - len(xWithoutOutliers)) + " outliers-")

    if meanY == 0 : 
        xGauss = xWithoutOutliers
        yGauss = yWithoutOutliers
    else : 
        xGauss, yGauss = fitGauss(xWithoutOutliers, yWithoutOutliers)

    try : 
        yFitted = savgol_filter(yWithoutOutliers, 15, 3) # window size 51, polynomial order 3
    except Exception : 
        print("WARNING: Error applying filter")
        yFitted = yWithoutOutliers

    fig = go.Figure()

    addPlot(fig, xWithoutOutliers, yWithoutOutliers, 'markers', 'Data')
    #addPlot(fig, xWithoutOutliers, smooth(yWithoutOutliers, 5), 'lines', 'Smoothed 5 point avg')
    addPlot(fig, xGauss, yGauss, 'lines', 'Gaussian fit')
    addPlot(fig, xWithoutOutliers, yFitted, 'lines', 'Savgol filter')
    
    if plotHalfGaussian : 
        xPartialGauss, yPartialGauss = fitPartialGaussian(xWithoutOutliers, yWithoutOutliers)
        addPlot(fig, xPartialGauss, yPartialGauss, 'lines', 'Half gaussian fit')

    fig.update_layout(
        title_text=' '.join(filename.replace(os.sep, '_').split('_')),
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
    )

    saveImg(fig, filename)

    return fig

#############
# LOAD DATA #
#############
print("\n--- START ---")
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

startTimeAvgFiringRate = 100
stopTimeAvgFiringRate = 1000
firingRateFactor = (1000 / (stopTimeAvgFiringRate - startTimeAvgFiringRate))

paradigm = args.path2data.split(os.sep)[-1].split('/')[-2].split('_')[0]
includeSelfSimilarity = False
#minResponses = 3
startPrepareDataTime = time.time()
nTHINGS = len(data.df_metadata.uniqueID)

uniqueSimilarities = np.arange(0.0, 1.0 + (1.0 % args.step), args.step)
nSimilarities = len(uniqueSimilarities)
similarityMatrixToIndex = (data.similarity_matrix.to_numpy() / args.step).astype(int)

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

regions[allRegionsName] = Region(allRegionsName)

for session in sessions:
    
    startPrepareSessionData = time.time()
    
    if not hasattr(args, 'unit'):
        units = list(set(data.neural_data[session]['units'].keys()))
    else:
        units = [args.unit]

    stimlookup = data.neural_data[session]['stimlookup']
    objectNames = data.neural_data[session]['objectnames']
    thingsIndices = get_THINGS_indices(data.df_metadata, data.neural_data[session]['stimlookup'])

    # do it before to make it faster
    allSitesSession = []
    for unit in units:
        site = data.neural_data[session]['units'][unit]['site']

        if site not in allSitesSession : 
            allSitesSession.append(site)

        if site not in allSiteNames : 
            allSiteNames.append(site)
            regions[site] = Region(site)

    for site in allSitesSession : 
        for i1, i2 in itertools.product(thingsIndices, thingsIndices) : 
            if i1 == i2 and not includeSelfSimilarity :
                continue
            regions[allRegionsName].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units) # copresentation
            regions[site].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units)

    for unit in units:
        pvals = data.neural_data[session]['units'][unit]['p_vals']
        zscores = data.neural_data[session]['units'][unit]['zscores']
        site = data.neural_data[session]['units'][unit]['site']
        trials = data.neural_data[session]['units'][unit]['trial']
        responses = [thingsIndices[i] for i in np.where(pvals < args.alpha)[0]]

        for i1, i2 in itertools.product(responses, responses) :  
            if i1 == i2 and not includeSelfSimilarity :
                continue
            regions[allRegionsName].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1
            regions[site].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1

        if len(responses) > 0 :
            regions[allRegionsName].numResponsesHist.append(len(responses))
            regions[site].numResponsesHist.append(len(responses))

            bestZIndex = np.argmax(zscores)
            indexBest = thingsIndices[bestZIndex]

            for i in range(len(zscores)) : # responseIndices
                if i == bestZIndex and not includeSelfSimilarity : 
                    continue
                index = thingsIndices[i]
                similarityIndex = similarityMatrixToIndex[index, indexBest]
                regions[allRegionsName].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[site].zScoresNorm.addValue(similarityIndex, zscores[i])
                
            # Cohens d    
            numStimuliSession = len(thingsIndices)
            meanFiringRatesStimuli = np.zeros((numStimuliSession))
            stddevFiringRatesStimuli = np.zeros((numStimuliSession))
        
            numAllTrials = len(trials)
            allFiringRates = np.zeros((numAllTrials))
        
            for stimNum in range(numStimuliSession) :
                relevantTrials = np.where(np.asarray(objectNames) == stimlookup[stimNum])[0]
                responseFiringRates = []
                
                for t in relevantTrials :
                    relevantSpikes = trials[t]
                    relevantSpikes = relevantSpikes[np.where(relevantSpikes >= startTimeAvgFiringRate) and np.where(relevantSpikes < stopTimeAvgFiringRate)]
                    firingRate = float(len(relevantSpikes)) * firingRateFactor
                    responseFiringRates.append(firingRate)
                    allFiringRates[t] = firingRate

                meanFiringRatesStimuli[stimNum] = statistics.mean(responseFiringRates)
                stddevFiringRatesStimuli[stimNum] = statistics.stdev(responseFiringRates)

            meanAll = statistics.mean(allFiringRates)
            stddevAll = statistics.stdev(allFiringRates)

            #for response in responses : 
            for stimNum in range(numStimuliSession) : 

                index = thingsIndices[stimNum]
                if indexBest == index and not includeSelfSimilarity :
                    continue
        
                mean1 = meanFiringRatesStimuli[stimNum]
                s1 = stddevFiringRatesStimuli[stimNum]
                stddevNorm = math.sqrt(s1 * s1 + stddevAll * stddevAll)
                if stddevNorm == 0 : 
                    print('stddev is 0')
                    
                cohensDResult = (mean1 - meanAll) / stddevNorm
                similarityIndex = similarityMatrixToIndex[indexBest, index]
                regions[allRegionsName].cohensD.addValue(similarityIndex, cohensDResult)
                regions[site].cohensD.addValue(similarityIndex, cohensDResult)
        
    print("Prepared data of session " + session + ". Time: " + str(time.time() - startPrepareSessionData) + " s" )

print("\nTime preparing data: " + str(time.time() - startPrepareDataTime) + " s\n")

allRegionCoactivationPlots = []
allRegionCopresentationPlots = []
allRegionCoactivationNormalizedPlots = []
allRegionZScoresPlots = []
allRegionCohensDPlots = []
allRegionNumResponsesPlots = []

figurePrepareTime = time.time()
for site in allSiteNames : 

    siteData = regions[site]
    print("Create figures for " + site)

    coactivationBeforeNormalization = siteData.coactivationNorm.y
    siteData.coactivationNorm.normalize()
    siteData.zScoresNorm.normalize()
    siteData.cohensD.normalize()

    coactivationBeforeNormalization = coactivationBeforeNormalization[siteData.coactivationNorm.relevantIndices]

    fileDescription = paradigm + '_' + args.metric + '_' + site 
    allRegionCoactivationPlots.append(
        createPlot(siteData.coactivationNorm.similarity, coactivationBeforeNormalization, "Number of coactivations", "coactivation" + os.sep + fileDescription, False))
    allRegionCopresentationPlots.append(
        createPlot(siteData.coactivationNorm.similarity, siteData.coactivationNorm.normalizer, "Number of copresentations", "copresentation" + os.sep + fileDescription, False))
    allRegionCoactivationNormalizedPlots.append(
        createPlot(siteData.coactivationNorm.similarity, siteData.coactivationNorm.y * 100, "Normalized coactivation probability in %", "coactivation_normalized" + os.sep + fileDescription, True))
    allRegionZScoresPlots.append(
        createPlot(siteData.zScoresNorm.similarity, siteData.zScoresNorm.y, "Mean zscores", "zscores" + os.sep + fileDescription, True))
    allRegionCohensDPlots.append(
        createPlot(siteData.cohensD.similarity, siteData.cohensD.y, "Mean cohens d", "cohensd" + os.sep + fileDescription, True))

    counts, bins = np.histogram(siteData.numResponsesHist, bins=range(args.max_responses_unit + 1))
    numResponsesFig = px.bar(x=bins[:-1], y=counts, labels={'x':'Number of units', 'y':'Number of responses'})
    allRegionNumResponsesPlots.append(numResponsesFig)
    saveImg(numResponsesFig, "num_responses" + os.sep + paradigm + '_' + site)
    

print("\nTime creating figures: " + str(time.time() - figurePrepareTime) + " s\n")

coactivationDiv, coactivationFigId, coactivationTableId = createRegionsDiv("Coactivation")
copresentationDiv, copresentationFigId, copresentationTableId = createRegionsDiv("Copresentation")
coactivationNormalizedDiv, coactivationNormalizedFigId, coactivationNormalizedTableId = createRegionsDiv("Coactivation - Normalized")
zscoresDiv, zscoresFigId, zscoresTableId = createRegionsDiv("Mean zscores dependent on semantic similarity to best response")
cohensDDiv, cohensDFigId, cohensDTableId = createRegionsDiv("Mean cohens d dependent on semantic similarity to best response")
numRespDiv, numRespFigId, numRespTableId = createRegionsDiv("Number of units with respective response counts")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Tuners'),
    coactivationDiv, 
    copresentationDiv, 
    coactivationNormalizedDiv, 
    zscoresDiv, 
    cohensDDiv,
    numRespDiv
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

@app.callback(
    Output(component_id=zscoresFigId, component_property='figure'), 
    Input(zscoresTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionZScoresPlots, active_cell)

@app.callback(
    Output(component_id=cohensDFigId, component_property='figure'), 
    Input(cohensDTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionCohensDPlots, active_cell)
    
@app.callback(
    Output(component_id=numRespFigId, component_property='figure'), 
    Input(numRespTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionNumResponsesPlots, active_cell)
    
if __name__ == '__main__':
    app.run_server(debug=False) # why ?