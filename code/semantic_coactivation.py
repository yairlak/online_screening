#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/14 17:10:12
@Author  :   Katharina Karkowski 
"""

import os
import math
import numpy as np
import pandas as pd
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
#from data_manip import get_THINGS_indices
from data_manip import get_mean_firing_rate_normalized

parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str, #"90_1_aos" / None ; 90_3_aos, channel 68 cluster 1
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# DATA AND MODEL
parser.add_argument('--metric', default='cosine',
                    help='Distance metric')
parser.add_argument('--similarity_matrix_delimiter', default=',', type=str,
                    help='Similarity metric delimiter')

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=True, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--only_SU', default=True, 
                    help='If True, only single units are considered')
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
                    default='../data/aos_after_manual_clustering/') 
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
        self.similarity = uniqueSimilarities
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
        
    def addValue(self, index, value) : 
        self.y[index] += value
        self.normalizer[index] += 1
        self.values[index].append(value)


@dataclass
class Region:
    sitename: str
    coactivationNorm : NormArray = field(default_factory=lambda: NormArray())
    numResponsesPerConcept : List = field (default_factory=lambda: np.zeros(len(data.df_metadata['uniqueID'])))
    zScoresNorm : NormArray = field(default_factory=lambda: NormArray())
    cohensD : NormArray = field(default_factory=lambda: NormArray())
    responseStrength : NormArray = field(default_factory=lambda: NormArray())
    firingRatesNorm : NormArray = field(default_factory=lambda: NormArray())
    numResponsesHist : List = field (default_factory=lambda: [])
    similaritiesArray : List = field (default_factory=lambda: [])
    firingRatesScatterSimilarities : List = field (default_factory=lambda: [])
    firingRatesScatter : List = field (default_factory=lambda: [])
    responseStrengthHistResp : List = field (default_factory=lambda: [])
    responseStrengthHistNoResp : List = field (default_factory=lambda: [])
    #coactivationFull : List = field(default_factory=lambda: [[] for i in range(nSimilarities)])
    spearmanCor : List = field (default_factory=lambda: [])
    spearmanP : List = field (default_factory=lambda: [])
    spearmanCorSteps : List = field(default_factory=lambda: [[] for i in range(numCorSteps)])
    pearsonCor : List = field (default_factory=lambda: [])
    pearsonP : List = field (default_factory=lambda: [])
    pearsonCorSteps : List = field(default_factory=lambda: [[] for i in range(numCorSteps)])


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

def createCorrelationPlot(sitename, correlation) : 
    return go.Box(
        y=correlation, 
        name=sitename + " (" + str(len(correlation)) + ")",
        boxpoints='all',
    )

def createBoxPlot(x, values, title, yLabel, filename, boxpoints=False) :   
    
    fig = go.Figure()
    for i in range(len(values)) : 
        if(len(values[i]) >= args.minValuesBox) : 
            fig.add_trace(go.Box(
                y=values[i],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
                #boxpoints='all',
            ))
        else : 
            fig.add_trace(go.Box(
                y=[0.0],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
            ))

        if i < len(values)-1 : 
            t_value, p_value = stats.ttest_ind(values[i], values[i+1]) 

            if p_value <= args.alpha_box : 
                print(title + ': p_value=%.8f' % p_value,
                    'for value=%i ' % i)

    fig.update_layout(
        title_text=title,
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False
    )

    saveImg(fig, filename)

    return fig

def createPlot(x, y, yLabel, filename, plotHalfGaussian, ticktext=[]) :

    if len(y) == 0 : 
        meanY = 0
        stdevY = 1
    elif len(y) == 1 : 
        meanY = y[0]
        stdevY = 1
    else : 
        meanY = statistics.mean(y)
        stdevY = statistics.stdev(y)
    # relevantIndices = np.where(abs(y - meanY) <= args.max_stdev_outliers * stdevY)
    xWithoutOutliers = x#[relevantIndices]
    yWithoutOutliers = y#[relevantIndices]

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
        yaxis_title=yLabel
    )

    if len(ticktext) > 0 : 
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = xWithoutOutliers,
                ticktext = ticktext#[relevantIndices] 
            )
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
includeSelfSimilarity = False
startPrepareDataTime = time.time()
nTHINGS = len(data.df_metadata.uniqueID)

uniqueSimilarities = np.arange(0.0, 1.0 + (1.0 % args.step), args.step)
nSimilarities = len(uniqueSimilarities)
similarityMatrixToIndex = (data.similarity_matrix.to_numpy().round(decimals=4) / args.step).astype(int)
corStepSize = 0.1
numCorSteps = math.ceil(1.0 / corStepSize) + 1

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

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
    thingsIndices = data.get_THINGS_indices(data.neural_data[session]['stimlookup'])
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


    for site in allSitesSession :  
        for i1, i2 in itertools.product(thingsIndices, thingsIndices) : 
            if i2 > i1 or (i1 == i2 and not includeSelfSimilarity) :
                continue
            regions[allRegionsName].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units) # copresentation
            regions[site].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += len(units)

    for unit in units:
        unitData = data.neural_data[session]['units'][unit]
        if not unitData['kind'] == 'SU' :
            continue
        pvals = unitData['p_vals']
        zscores = unitData['zscores']
        site = unitData['site']
        trials = unitData['trial']
        channel = unitData['channel_num']
        cluster = unitData['class_num']
        responses = [thingsIndices[i] for i in np.where(pvals < args.alpha)[0]]
        firingRates = get_mean_firing_rate_normalized(trials, stimuliIndices, startTimeAvgFiringRate, stopTimeAvgFiringRate)
        similaritiesCor = []
        valuesCor = []
        similaritiesCorSteps = [[] for i in range(numCorSteps)]
        valuesCorSteps = [[] for i in range(numCorSteps)]

        zscores = zscores / max(zscores)
        
        if not len(firingRates) == numStimuli or not len(zscores) == numStimuli : 
            print("ERROR: array length does not fit for zscores or firing rates!")

        for i1 in responses :  
            regions[allRegionsName].numResponsesPerConcept[i1] += 1
            regions[site].numResponsesPerConcept[i1] += 1
            
            for i2 in responses : 
                if i2 > i1 or (i1 == i2 and not includeSelfSimilarity) :
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
                #if i == bestResponse and not includeSelfSimilarity : 
                #    continue
                index = thingsIndices[i]
                similarity = data.similarity_matrix[index][indexBest]
                similarityIndex = similarityMatrixToIndex[index, indexBest]
                regions[allRegionsName].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[site].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[allRegionsName].firingRatesNorm.addValue(similarityIndex, firingRates[i])
                regions[site].firingRatesNorm.addValue(similarityIndex, firingRates[i])

                regions[allRegionsName].firingRatesScatterSimilarities.append(similarity)
                regions[site].firingRatesScatterSimilarities.append(similarity)
                regions[allRegionsName].firingRatesScatter.append(firingRates[i])
                regions[site].firingRatesScatter.append(firingRates[i])

                regions[allRegionsName].similaritiesArray.append(similarity)
                regions[site].similaritiesArray.append(similarity)

                if index in responses : 
                    regions[allRegionsName].responseStrengthHistResp.append(firingRates[i])
                    regions[site].responseStrengthHistResp.append(firingRates[i])
                else : 
                    regions[allRegionsName].responseStrengthHistNoResp.append(firingRates[i])
                    regions[site].responseStrengthHistNoResp.append(firingRates[i])

                if not i == bestResponse :
                    corStep = int(similarity / corStepSize)
                    similaritiesCorSteps[corStep].append(similarity)
                    valuesCorSteps[corStep].append(firingRates[i])

                    similaritiesCor.append(similarity)
                    valuesCor.append(firingRates[i])


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
                #if indexBest == index and not includeSelfSimilarity :
                #    continue
        
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
  
        if len(valuesCor) > 0 : 
            spearman = stats.spearmanr(valuesCor, similaritiesCor)

            regions[site].spearmanCor.append(spearman.correlation)
            regions[site].spearmanP.append(spearman.pvalue)
            regions[allRegionsName].spearmanCor.append(spearman.correlation)
            regions[allRegionsName].spearmanP.append(spearman.pvalue)
            
            if len(valuesCor) >= 2 : 
                pearson = stats.pearsonr(valuesCor, similaritiesCor)
                regions[site].pearsonCor.append(pearson[0])
                regions[site].pearsonP.append(pearson[1])
                regions[allRegionsName].pearsonCor.append(pearson[0])
                regions[allRegionsName].pearsonP.append(pearson[1])

            for i in range(numCorSteps) : 
                if len(valuesCorSteps[i]) > 0 : 
                    spearman = stats.spearmanr(valuesCorSteps[i], similaritiesCorSteps[i]) 
                    if not math.isnan(spearman.correlation) : 
                        regions[site].spearmanCorSteps[i].append(spearman.correlation)
                        regions[allRegionsName].spearmanCorSteps[i].append(spearman.correlation)
                    
                if len(valuesCorSteps[i]) >= 2 : 
                    pearson = stats.pearsonr(valuesCorSteps[i], similaritiesCorSteps[i]) 
                    if not math.isnan(pearson[0]) and not math.isnan(pearson[1]) : 
                        regions[site].pearsonCorSteps[i].append(pearson[0])
                        regions[allRegionsName].pearsonCorSteps[i].append(pearson[0])


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
allRegionSpearmanPlots = []
allRegionSpearmanPPlots = []
allRegionPearsonPlots = []
allRegionPearsonPPlots = []
allRegionFiringRateScatterPlots = []
allRegionRespStrengthHistPlots = []
allRegionRespStrengthHistPlotsNo = []

figurePrepareTime = time.time()
spearmanPPlot = go.Figure()
spearmanPlot = go.Figure()
pearsonPPlot = go.Figure()
pearsonPlot = go.Figure()


for site in allSiteNames : 

    siteData = regions[site]
    print("Create figures for " + site)

    allRegionFiringRateScatterPlots.append(
        go.Figure(
            go.Scatter(
                x = siteData.firingRatesScatterSimilarities,
                y = siteData.firingRatesScatter,
                mode = 'markers'
            )
        ))
    
    spearmanPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanCor))
    spearmanPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanP))
    pearsonPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonCor))
    pearsonPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonP))

    coactivationBeforeNormalization = siteData.coactivationNorm.y.copy()
    siteData.coactivationNorm.normalize("coactivation normalized")

    ticktextCoactivation = np.asarray([str(round(siteData.coactivationNorm.similarity[i], 2))
        #+ ": " + str(siteData.coactivationNorm.y[i] * 100) 
        #+ ("%.2f" % coactivationBeforeNormalization[i])
        + " (" + str(round(siteData.coactivationNorm.y[i] * 100, 5)) 
        + " = " + str(coactivationBeforeNormalization[i])
        + "/" + str(siteData.coactivationNorm.normalizer[i]) + ")" for i in range(len(coactivationBeforeNormalization))])

    fileDescription = paradigm + '_' + args.metric + '_' + site 
    

    responseStrenthHistDf = pd.DataFrame({
        'firing rate': siteData.responseStrengthHistResp,
        #np.concatenate((
            #siteData.responseStrengthHistResp
            #siteData.responseStrengthHistNoResp
        #)),
        'responsiveness': ['Responses' for _ in range(len(siteData.responseStrengthHistResp))]
        #np.concatenate((
        #    np.array(['Responses' for _ in range(len(siteData.responseStrengthHistResp))])
            #np.array(['Non-Responses' for _ in range(len(siteData.responseStrengthHistNoResp))]) 
        #))
    })
    noResponseStrenthHistDf = pd.DataFrame({
        'firing rate': siteData.responseStrengthHistNoResp,
        'responsiveness': ['No responses' for _ in range(len(siteData.responseStrengthHistNoResp))]
    })
    responseStrenthHistFig = px.histogram(responseStrenthHistDf, x='firing rate', color="responsiveness")
    noResponseStrenthHistFig = px.histogram(noResponseStrenthHistDf, x='firing rate', color="responsiveness")
    saveImg(responseStrenthHistFig, "response_strength_hist" + os.sep + paradigm + '_' + site)
    saveImg(noResponseStrenthHistFig, "response_strength_hist_no" + os.sep + paradigm + '_' + site)
    allRegionRespStrengthHistPlots.append(responseStrenthHistFig)
    allRegionRespStrengthHistPlotsNo.append(noResponseStrenthHistFig)

    allRegionCoactivationNormalizedPlots.append(
        createPlot(siteData.coactivationNorm.similarity, siteData.coactivationNorm.y * 100, "Normalized coactivation probability in %", "coactivation_normalized" + os.sep + fileDescription, True, ticktextCoactivation))
    allRegionZScoresPlots.append(
        createBoxPlot(uniqueSimilarities, siteData.zScoresNorm.values, "Mean zscores", "zscores", "zscores" + os.sep + fileDescription))
    allRegionFiringRatesPlots.append(
        createBoxPlot(uniqueSimilarities, siteData.firingRatesNorm.values, "Normalized firing rates", "firing_rates", "firing_rates" + os.sep + fileDescription))
    allRegionCohensDPlots.append(
        createBoxPlot(uniqueSimilarities, siteData.cohensD.values, "Mean cohens d", "cohensd", "cohensd" + os.sep + fileDescription))
    allRegionResponseStrengthPlots.append(
        createBoxPlot(uniqueSimilarities, siteData.responseStrength.values, "Mean response strength (median - meanBaseline) / maxMedian", "responseStrength", "responseStrength" + os.sep + fileDescription))
    allRegionSpearmanPlots.append(
        createBoxPlot(np.arange(0.0, 1.0 + corStepSize, corStepSize), siteData.spearmanCorSteps, "Spearman correlation dependent on semantic similarity", "spearmanCorSteps", "spearmanCorSteps" + os.sep + fileDescription, 'all')) 
    allRegionPearsonPlots.append(
        createBoxPlot(np.arange(0.0, 1.0 + corStepSize, corStepSize), siteData.pearsonCorSteps, "Pearson correlation dependent on semantic similarity", "spearmanCorSteps", "spearmanCorSteps" + os.sep + fileDescription, 'all')) 

    counts, bins = np.histogram(siteData.numResponsesHist, bins=range(args.max_responses_unit + 1))
    numResponsesFig = px.bar(x=bins[:-1], y=counts, labels={'x':'Number of units', 'y':'Number of responses'})
    allRegionNumResponsesPlots.append(numResponsesFig)
    saveImg(numResponsesFig, "num_responses" + os.sep + paradigm + '_' + site)


spearmanPlot.update_layout(title="Spearman correlation for responding units",)
saveImg(spearmanPlot, paradigm + "_" + args.metric + "_spearmanPlot")

spearmanPPlot.update_layout(title="Spearman p-value for responding units",)
saveImg(spearmanPPlot, paradigm + "_" + args.metric + "_spearmanPPlot")

pearsonPlot.update_layout(title="Pearson correlation for responding units",)
saveImg(pearsonPlot, paradigm + "_" + args.metric + "_pearsonPlot")

pearsonPPlot.update_layout(title="Pearson p-value for responding units",)
saveImg(pearsonPPlot, paradigm + "_" + args.metric + "_pearsonPPlot")

print("\nTime creating figures: " + str(time.time() - figurePrepareTime) + " s\n")

coactivationDiv, coactivationFigId, coactivationTableId = createRegionsDiv("Coactivation")
copresentationDiv, copresentationFigId, copresentationTableId = createRegionsDiv("Copresentation")
coactivationNormalizedDiv, coactivationNormalizedFigId, coactivationNormalizedTableId = createRegionsDiv("Coactivation - Normalized")
zscoresDiv, zscoresFigId, zscoresTableId = createRegionsDiv("Mean zscores dependent on semantic similarity to best response")
firingRatesDiv, firingRatesFigId, firingRatesTableId = createRegionsDiv("Normalized firing rates dependent on semantic similarity to best response")
firingRatesBarsDiv, firingRatesBarsFigId, firingRatesBarsTableId = createRegionsDiv("Error bars for normalized firing rates dependent on semantic similarity to best response")
cohensDDiv, cohensDFigId, cohensDTableId = createRegionsDiv("Mean cohens d dependent on semantic similarity to best response")
responseStrengthDiv, responseStrengthFigId, responseStrengthTableId = createRegionsDiv("Mean response strength dependent on semantic similarity to best response")
spearmanCorStepsDiv, spearmanCorStepsFigId, spearmanCorStepsTableId = createRegionsDiv("Spearman correlation dependent on semantic similarity to best response")
pearsonCorStepsDiv, pearsonCorStepsFigId, pearsonCorStepsTableId = createRegionsDiv("Pearson correlation dependent on semantic similarity to best response")
firingRatesScatterDiv, firingRatesScatterFigId, firingRatesScatterTableId = createRegionsDiv("Firing rates of concepts dependent on semantic similarity to best response")
numRespDiv, numRespFigId, numRespTableId = createRegionsDiv("Number of units with respective response counts")
responseStrengthHistDiv, responseStrengthHistFigId, responseStrengthHistTableId = createRegionsDiv("Response strength histogram for responsive units")
responseStrengthHistDivNo, responseStrengthHistFigIdNo, responseStrengthHistTableIdNo = createRegionsDiv("Response strength histogram for non responsive units")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Semantic Tuning'),
    html.H3('Spearman correlation'),
    dcc.Graph(id='spearman-plot', figure=spearmanPlot),
    dcc.Graph(id='spearman-p-plot', figure=spearmanPPlot),
    spearmanCorStepsDiv,
    html.H3('Pearson correlation'),
    dcc.Graph(id='pearson-plot', figure=pearsonPlot),
    dcc.Graph(id='pearson-p-plot', figure=pearsonPPlot),
    pearsonCorStepsDiv,
    #coactivationDiv, 
    #copresentationDiv, 
    coactivationNormalizedDiv, 
    firingRatesDiv, 
    responseStrengthHistDiv, 
    responseStrengthHistDivNo, 
    #firingRatesScatterDiv,
    #firingRatesBarsDiv,
    responseStrengthDiv, 
    zscoresDiv, 
    cohensDDiv,
    numRespDiv, 
    
])

print("\n--- Ready! ---\n\n")


def getActivePlot(data, activeCell) : 
    if(activeCell == None) :
        return data[0]
    else : 
        return data[activeCell['row']]

@app.callback(
    Output(component_id=spearmanCorStepsFigId, component_property='figure'), 
    Input(spearmanCorStepsTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSpearmanPlots, active_cell)

@app.callback(
    Output(component_id=pearsonCorStepsFigId, component_property='figure'), 
    Input(pearsonCorStepsTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionPearsonPlots, active_cell)

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
    Output(component_id=firingRatesScatterFigId, component_property='figure'), 
    Input(firingRatesScatterTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionFiringRateScatterPlots, active_cell)

@app.callback(
    Output(component_id=responseStrengthHistFigId, component_property='figure'), 
    Input(responseStrengthHistTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionRespStrengthHistPlots, active_cell)

@app.callback(
    Output(component_id=responseStrengthHistFigIdNo, component_property='figure'), 
    Input(responseStrengthHistTableIdNo, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionRespStrengthHistPlotsNo, active_cell)


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

    
if __name__ == '__main__':
    app.run_server(debug=False) # why ?