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
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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
from data_manip import get_THINGS_indices
from data_manip import get_mean_firing_rate_normalized

parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str,
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# DATA AND MODEL
parser.add_argument('--metric', default='cosine',
                    help='Distance metric')
parser.add_argument('--similarity_matrix_delimiter', default=',', type=str,
                    help='Similarity metric delimiter')

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for responses')
parser.add_argument('--alpha_box', type=float, default=0.001,
                    help='Alpha for box significance')

# PLOT
parser.add_argument('--step', type=float, default=0.05,
                    help='Plotting detail')
parser.add_argument('--max_stdev_outliers', type=float, default=5,
                    help='Limit for excluding outliers')   
parser.add_argument('--max_responses_unit', type=float, default=20,
                    help='Limit for counting responses per unit for histogram')       
parser.add_argument('--minValuesBox', type=int, default=1,
                    help='min num values for box plot') 

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2data', 
                    default='../data/aosnos_after_manual_clustering/') 
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation/') 

args=parser.parse_args()


@dataclass
class NormArray:
    y : List = field(default_factory=lambda: np.zeros((nSimilarities)))
    normalizer : List = field(default_factory=lambda: np.zeros((nSimilarities)))
    similarity : List = field(default_factory=lambda: [])
    relevantIndices : List = field(default_factory=lambda: [])
    values : List = field(default_factory=lambda: [[] for i in range(nSimilarities)])
    mean : List = field(default_factory=lambda: [])
    stddev : List = field(default_factory=lambda: [])

    def normalize(self, name) : 
        #self.relevantIndices = np.where(self.normalizer != 0)
        #self.normalizer = self.normalizer[self.relevantIndices]
        #self.similarity = uniqueSimilarities[self.relevantIndices]
        #self.values = self.values[self.relevantIndices]
        #self.y = self.y[self.relevantIndices] / self.normalizer

        self.similarity = uniqueSimilarities
        #self.normalizer[np.where(self.normalizer) == 0] = 1.0
        for i in range(len(self.normalizer)) :
            if self.normalizer[i] > 0 : 
                self.values[i] /= self.normalizer[i] 
                self.y[i] = self.y[i] / self.normalizer[i]
        self.mean = [np.mean(self.values[i]) for i in range(len(self.values))]
        self.stddev = [np.std(self.values[i]) for i in range(len(self.values))]

        #for i in range(nSimilarities - 1) : 
        #    t_value, p_value = stats.ttest_ind(self.values[i], self.values[i+1]) #

        #    if p_value <= args.alpha_box : 
        #        print(name + ': p_value=%.8f' % p_value,
        #            'for similarity=%.2f ' % self.similarity[i])
        
        #return y, uniqueSimilarities[relevantIndices], relevantIndices

    def addValue(self, index, value) : 
        self.y[index] += value
        self.normalizer[index] += 1
        self.values[index].append(value)


@dataclass
class Region:
    sitename: str
    coactivationNorm : NormArray = field(default_factory=lambda: NormArray())
    zScoresNorm : NormArray = field(default_factory=lambda: NormArray())
    cohensD : NormArray = field(default_factory=lambda: NormArray())
    responseStrength : NormArray = field(default_factory=lambda: NormArray())
    firingRatesNorm : NormArray = field(default_factory=lambda: NormArray())
    numResponsesHist : List = field (default_factory=lambda: [])
    similaritiesArray : List = field (default_factory=lambda: [])
    spearmanCor : List = field (default_factory=lambda: [])
    spearmanP : List = field (default_factory=lambda: [])


def createTableDiv(title, figureId, tableId, columnName, columnId, columnData) : 
    
    return html.Div(children=[

        html.Div([
            html.Div([
                html.H3(title),
                dcc.Graph(id=figureId, style={'height': 500})
            ], 
            style={'width': '70%', 'display': 'inline-block'},
            #className="one columns"
            ),

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
                        #'height': 500,# figureHeight - 180,
                        'overflowY': 'scroll'
                    }
                ), 
            ], 
            style={'width': '15%', 'display': 'inline-block', 'margin-left': '5%'},
            #className="one columns"
            ),
        ]#, className="row"
        )

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

    file = args.path2images + filename + ".png"

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fig.write_image(file)

def createBoxPlot(normArray, title, yLabel, filename) :   
    
    fig = go.Figure()
    for i in range(len(normArray.values)) : 
        if(len(normArray.values[i]) >= args.minValuesBox) : 
            fig.add_trace(go.Box(
                y=normArray.values[i],
                name="{:.2f} ({})".format(uniqueSimilarities[i], len(normArray.values[i])),
                boxpoints=False,
                #boxpoints='all',
            ))
        else : 
            fig.add_trace(go.Box(
                y=[0.0],
                name="{:.2f} ({})".format(uniqueSimilarities[i], len(normArray.values[i])),
                boxpoints=False,
            ))

        if i < len(normArray.values)-1 : 
            t_value, p_value = stats.ttest_ind(normArray.values[i], normArray.values[i+1]) 

            if p_value <= args.alpha_box : 
                print(title + ': p_value=%.8f' % p_value,
                    'for similarity=%.2f ' % uniqueSimilarities[i])

    fig.update_layout(
        title_text=title,
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False
    )

    saveImg(fig, filename)

    return fig

def createPlot(x, y, yLabel, filename, plotHalfGaussian) :
    
    relevantIndices = np.where(y != 0)
    x = x[relevantIndices]
    y = y[relevantIndices]
    relevantIndices = np.where(np.isfinite(y))
    x = x[relevantIndices]
    y = y[relevantIndices]

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

onlyTwoSessions = False  # for testing purposes (first session has no responses)
figureHeight = 900
startBaseline = -500
startTimeAvgFiringRate = 100
stopTimeAvgFiringRate = 800 # 800 for rodrigo
firingRateFactor = (1000 / (stopTimeAvgFiringRate - startTimeAvgFiringRate))
firingRateFactorBaselines = (1000 / (0 - startBaseline))

paradigm = args.path2data.split(os.sep)[-1].split('/')[-2].split('_')[0]
includeSelfSimilarity = True
#minResponses = 3
startPrepareDataTime = time.time()
nTHINGS = len(data.df_metadata.uniqueID)

uniqueSimilarities = np.arange(0.0, 1.0 + (1.0 % args.step), args.step)
nSimilarities = len(uniqueSimilarities)
similarityMatrixToIndex = (data.similarity_matrix.to_numpy().round(decimals=4) / args.step).astype(int)

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

regions_df = {
    allRegionsName : {"spearmanCor" : [], "spearmanP" : []}
}

regions[allRegionsName] = Region(allRegionsName)

sessionCounter = 0
for session in sessions:

    subjectNum = int(session.split("_")[0])
    sessionNum = int(session.split("_")[1])
    sessionParadigm = session.split("_")[2]

    startPrepareSessionData = time.time()
    
    if not hasattr(args, 'unit'):
        units = list(set(data.neural_data[session]['units'].keys()))
    else:
        units = [args.unit]

    stimlookup = data.neural_data[session]['stimlookup']
    objectNames = data.neural_data[session]['objectnames']
    stimuliIndices = data.neural_data[session]['objectindices_session']
    thingsIndices = get_THINGS_indices(data.df_metadata, data.neural_data[session]['stimlookup'])
    numStimuli = len(stimlookup)

    # do it before to make it faster
    allSitesSession = []
    for unit in units:
        site = data.neural_data[session]['units'][unit]['site']

        if site not in allSitesSession : 
            allSitesSession.append(site)

        if site not in allSiteNames : 
            allSiteNames.append(site)
            regions[site] = Region(site)
            regions_df[site] = {"spearmanCor" : [], "spearmanP" : []}


    for site in allSitesSession : 
        for i1, i2 in itertools.product(thingsIndices, thingsIndices) : 
            if i1 == i2 and not includeSelfSimilarity :
                continue
            regions[allRegionsName].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units) # copresentation
            regions[site].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units)

    for unit in units:
        unitData = data.neural_data[session]['units'][unit]
        pvals = unitData['p_vals']
        zscores = unitData['zscores']
        site = unitData['site']
        trials = unitData['trial']
        channel = unitData['channel_num']
        cluster = unitData['class_num']
        responses = [thingsIndices[i] for i in np.where(pvals < args.alpha)[0]]
        firingRates = get_mean_firing_rate_normalized(trials, stimuliIndices, startTimeAvgFiringRate, stopTimeAvgFiringRate)
        similaritiesSpearman = []
        valuesSpearman = []

        zscores = zscores / max(zscores)
        
        if not len(firingRates) == numStimuli or not len(zscores) == numStimuli : 
            print("ERROR: array length does not fit for zscores or firing rates!")

        for i1, i2 in itertools.product(responses, responses) :  
            if i1 == i2 and not includeSelfSimilarity :
                continue
            regions[allRegionsName].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1
            regions[site].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1

        if len(responses) > 0 :
            regions[allRegionsName].numResponsesHist.append(len(responses))
            regions[site].numResponsesHist.append(len(responses))

            # zscores
            bestResponse = np.argmax(firingRates) # best Response = highest z? highest response strength?
            indexBest = thingsIndices[bestResponse]

            for i in range(numStimuli) : # responseIndices
                if i == bestResponse and not includeSelfSimilarity : 
                    continue
                index = thingsIndices[i]
                similarity = data.similarity_matrix[index][indexBest]
                similarityIndex = similarityMatrixToIndex[index, indexBest]
                regions[allRegionsName].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[site].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[allRegionsName].firingRatesNorm.addValue(similarityIndex, firingRates[i])
                regions[site].firingRatesNorm.addValue(similarityIndex, firingRates[i])

                regions[allRegionsName].similaritiesArray.append(similarity)
                regions[site].similaritiesArray.append(similarity)

                similaritiesSpearman.append(similarity)
                valuesSpearman.append(firingRates[i])

            # Baselines (for response strength)
            baselineFrequencies = np.zeros((len(trials)))
            for t in range(len(trials)): 
                baselineFrequencies[t] = float(len(np.where(trials[t] >= startBaseline) and np.where(trials[t] < 0)) ) * firingRateFactorBaselines
        
            meanBaseline = statistics.mean(baselineFrequencies)
                
            # Cohens d    
            numStimuliSession = len(thingsIndices)
            meanFiringRatesStimuli = np.zeros((numStimuliSession))
            stddevFiringRatesStimuli = np.zeros((numStimuliSession))
            medianFiringRatesStimuli = np.zeros((numStimuliSession))
        
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
                medianFiringRatesStimuli[stimNum] = statistics.median(responseFiringRates)


            meanAll = statistics.mean(allFiringRates)
            stddevAll = statistics.stdev(allFiringRates)
            maxMedianFiringRate = max(medianFiringRatesStimuli)

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

                # response strength
                responseStrengthUnit = (medianFiringRatesStimuli[stimNum] - meanBaseline) / maxMedianFiringRate
                regions[allRegionsName].responseStrength.addValue(similarityIndex, responseStrengthUnit)
                regions[site].responseStrength.addValue(similarityIndex, responseStrengthUnit)

        if len(valuesSpearman) > 0 : 
            spearman = stats.spearmanr(valuesSpearman, similaritiesSpearman)
            regions[site].spearmanCor.append(spearman.correlation)
            regions[site].spearmanP.append(spearman.pvalue)

            regions_df[site]["spearmanCor"].append(spearman.correlation)
            regions_df[site]["spearmanP"].append(spearman.pvalue)
            regions_df[allRegionsName]["spearmanCor"].append(spearman.correlation)
            regions_df[allRegionsName]["spearmanP"].append(spearman.pvalue)



    print("Prepared data of session " + session + ". Time: " + str(time.time() - startPrepareSessionData) + " s" )

    sessionCounter += 1
    if onlyTwoSessions and sessionCounter >= 2 : 
        break

print("\nTime preparing data: " + str(time.time() - startPrepareDataTime) + " s\n")


allRegionCoactivationPlots = []
allRegionCopresentationPlots = []
allRegionCoactivationNormalizedPlots = []
allRegionZScoresPlots = []
allRegionFiringRatesPlots = []
allRegionFiringRatesBarPlots = []
allRegionCohensDPlots = []
allRegionResponseStrengthPlots = []
allRegionNumResponsesPlots = []

figurePrepareTime = time.time()
spearmanPlot = go.Figure()
spearmanBarPlot = go.Figure()

for site in allSiteNames : 

    siteData = regions[site]
    print("Create figures for " + site)
    
    spearmanPlot.add_trace(
        go.Box(
            y=regions_df[site]["spearmanCor"], #siteData.spearmanCor,
            name=siteData.sitename,
            boxpoints='all',
        )
    )

    #spearmanBarPlot.add_trace(
    #    go.Bar(# name=siteData.sitename, 
    #        x=siteData.sitename, 
    #        y=siteData.spearmanCor.sort(),
    #    ),
    #)

    coactivationBeforeNormalization = siteData.coactivationNorm.y
    siteData.coactivationNorm.normalize("coactivation normalized")
    siteData.zScoresNorm.normalize("zScores")
    siteData.cohensD.normalize("cohensD")

    coactivationBeforeNormalization = coactivationBeforeNormalization[siteData.coactivationNorm.relevantIndices]

    fileDescription = paradigm + '_' + args.metric + '_' + site 
    
    allRegionCoactivationNormalizedPlots.append(
        createPlot(siteData.coactivationNorm.similarity, siteData.coactivationNorm.y * 100, "Normalized coactivation probability in %", "coactivation_normalized" + os.sep + fileDescription, True))
    allRegionZScoresPlots.append(
        createBoxPlot(siteData.zScoresNorm, "Mean zscores", "zscores", "zscores" + os.sep + fileDescription))
    allRegionFiringRatesPlots.append(
        createBoxPlot(siteData.firingRatesNorm, "Normalized firing rates", "firing_rates", "firing_rates" + os.sep + fileDescription))
    allRegionCohensDPlots.append(
        createBoxPlot(siteData.cohensD, "Mean cohens d", "cohensd", "cohensd" + os.sep + fileDescription))
    allRegionResponseStrengthPlots.append(
        createBoxPlot(siteData.responseStrength, "Mean response strength (median - meanBaseline) / maxMedian", "responseStrength", "responseStrength" + os.sep + fileDescription))
   

    counts, bins = np.histogram(siteData.numResponsesHist, bins=range(args.max_responses_unit + 1))
    numResponsesFig = px.bar(x=bins[:-1], y=counts, labels={'x':'Number of units', 'y':'Number of responses'})
    allRegionNumResponsesPlots.append(numResponsesFig)
    saveImg(numResponsesFig, "num_responses" + os.sep + paradigm + '_' + site)

saveImg(spearmanPlot, paradigm + "_spearmanPlot")

print("\nTime creating figures: " + str(time.time() - figurePrepareTime) + " s\n")

coactivationDiv, coactivationFigId, coactivationTableId = createRegionsDiv("Coactivation")
copresentationDiv, copresentationFigId, copresentationTableId = createRegionsDiv("Copresentation")
coactivationNormalizedDiv, coactivationNormalizedFigId, coactivationNormalizedTableId = createRegionsDiv("Coactivation - Normalized")
zscoresDiv, zscoresFigId, zscoresTableId = createRegionsDiv("Mean zscores dependent on semantic similarity to best response")
firingRatesDiv, firingRatesFigId, firingRatesTableId = createRegionsDiv("Normalized firing rates dependent on semantic similarity to best response")
firingRatesBarsDiv, firingRatesBarsFigId, firingRatesBarsTableId = createRegionsDiv("Error bars for normalized firing rates dependent on semantic similarity to best response")
cohensDDiv, cohensDFigId, cohensDTableId = createRegionsDiv("Mean cohens d dependent on semantic similarity to best response")
responseStrengthDiv, responseStrengthFigId, responseStrengthTableId = createRegionsDiv("Mean response strength dependent on semantic similarity to best response")
numRespDiv, numRespFigId, numRespTableId = createRegionsDiv("Number of units with respective response counts")
#tunersDiv = createTableDiv("Tuners", tunersFigId, tunersTableId, "Tuners", tunersColumnId, tableOptionsTuners)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Tuners'),
    dcc.Graph(id='spearman-plot', figure=spearmanPlot),
    
    #dcc.Graph(
    #    id='spearman-bar-plot',
    #    figure=spearmanBarPlot
    #),
    #coactivationDiv, 
    #copresentationDiv, 
    coactivationNormalizedDiv, 
    firingRatesDiv, 
    #firingRatesBarsDiv,
    responseStrengthDiv, 
    zscoresDiv, 
    cohensDDiv,
    #tunersDiv,
    numRespDiv, 
    
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
    Output(component_id=firingRatesFigId, component_property='figure'), 
    Input(firingRatesTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionFiringRatesPlots, active_cell)

@app.callback(
    Output(component_id=firingRatesBarsFigId, component_property='figure'), 
    Input(firingRatesBarsTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionFiringRatesBarPlots, active_cell)

@app.callback(
    Output(component_id=cohensDFigId, component_property='figure'), 
    Input(cohensDTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionCohensDPlots, active_cell)
    
@app.callback(
    Output(component_id=responseStrengthFigId, component_property='figure'), 
    Input(responseStrengthTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionResponseStrengthPlots, active_cell)
    
@app.callback(
    Output(component_id=numRespFigId, component_property='figure'), 
    Input(numRespTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionNumResponsesPlots, active_cell)

#@app.callback(
#    Output(component_id=tunersFigId, component_property='children'), 
#    Input(tunersTableId, 'active_cell')
#)
#def update_output_div(active_cell):
# 
    #if(active_cell == None) :
    #    tunerIndex = 0
    #else : 
    #    tunerIndex = active_cell['row']

#    return getActivePlot(tunerPlots, active_cell) #tunerPlots[tunerIndex] 

    
if __name__ == '__main__':
    app.run_server(debug=False) # why ?