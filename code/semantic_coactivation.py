#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/14 17:10:12
@Author  :   Katharina Karkowski 
"""

import os
import math
from unittest.mock import NonCallableMagicMock
import numpy as np
import pandas as pd
import time
import argparse
import itertools
import statistics 

from typing import List
from dataclasses import dataclass, field
from scipy import stats
import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# utilility modules
from plot_helper import *
from data_manip import DataHandler
#from data_manip import get_THINGS_indices
from data_manip import get_mean_firing_rate_normalized

parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str, #"90_1_aos" / None ; 90_3_aos, channel 68 cluster 1
                    #"88_1_aos", "88_3_aos", .. 89_3
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
parser.add_argument('--step', type=float, default=0.1,
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
                    default='../data/aos_after_manual_clustering/') 
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation/') 

args=parser.parse_args()

@dataclass 
class SimilaritiesArray:
    values : List = field(default_factory=lambda: [[] for i in range(nSimilarities)])

    def addValue(self, index, value) : 
        self.values[index].append(value)

@dataclass
class NormArray:
    y : List = field(default_factory=lambda: np.zeros((nSimilarities-1)))
    normalizer : List = field(default_factory=lambda: np.zeros((nSimilarities-1)))
    similarity : List = field(default_factory=lambda: [])

    def normalize(self) : 
        self.similarity = uniqueSimilarities
        for i in range(len(self.normalizer)) :
            if self.normalizer[i] > 0 : 
                self.y[i] = self.y[i] / self.normalizer[i]
        
    def addValue(self, index, value) : 
        self.y[index] += value
        self.normalizer[index] += 1


@dataclass
class Region:
    sitename: str
    coactivationNorm : NormArray = field(default_factory=lambda: NormArray())
    numResponsesPerConcept : List = field (default_factory=lambda: np.zeros(len(data.df_metadata['uniqueID'])))
    zScoresNorm : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    cohensD : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    responseStrength : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    firingRatesNorm : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    numResponsesHist : List = field (default_factory=lambda: [])
    similaritiesArray : List = field (default_factory=lambda: [])
    firingRatesScatterSimilarities : List = field (default_factory=lambda: [])
    firingRatesScatter : List = field (default_factory=lambda: [])
    responseStrengthHistResp : List = field (default_factory=lambda: [])
    responseStrengthHistNoResp : List = field (default_factory=lambda: [])
    steepestSlopeLog : List = field (default_factory=lambda: [])

    spearmanCor : List = field (default_factory=lambda: [])
    spearmanP : List = field (default_factory=lambda: [])
    spearmanCorSteps : List = field(default_factory=lambda: [[] for i in range(numCorSteps)])
    pearsonCor : List = field (default_factory=lambda: [])
    pearsonP : List = field (default_factory=lambda: [])
    pearsonCorSteps : List = field(default_factory=lambda: [[] for i in range(numCorSteps)])

    logisticFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(fitLogisticFunc, fitLogisticFuncParams, ["x0", "K", "A", "C"], p0=[0.5, 1, 0, 1], bounds=[[0, -1000, 0, 0], [1, 1000, 1, 1]]))
    stepFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(fitStep, fitStepParams, ["x0", "a", "b"], p0=[0.5, 0, 1], bounds=[[0, 0, 0], [1, 1, 1]]))
    gaussFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(halfGauss, halfGaussParams, ["x0", "a", "b", "sigma"], p0=[0.5, 1, 0, 1], bounds=[[0, 0, 0, 0], [1, 100, 1, 100]]))
    rDiffLog : List = field (default_factory=lambda: [])
    rDiffGauss : List = field (default_factory=lambda: [])
    rDiffLogGauss : List = field (default_factory=lambda: [])


def createAndSave(func, filename) : 
    fig = func 
    saveImg(fig, filename)
    return fig

def saveImg(fig, filename) : 

    file = args.path2images + filename + ".png"

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fig.write_image(file)


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

startPrepareDataTime = time.time()
onlyTwoSessions = False  # for testing purposes (first session has no responses)
paradigm = args.path2data.split(os.sep)[-1].split('/')[-2].split('_')[0]
includeSelfSimilarity = False
nTHINGS = len(data.df_metadata.uniqueID)

uniqueSimilarities = np.arange(0.0, 1.0 + (1.0 % args.step), args.step)
nSimilarities = len(uniqueSimilarities)
similarityMatrixToIndex = (data.similarity_matrix.to_numpy().round(decimals=4) / args.step).astype(int)
corStepSize = 0.1
numCorSteps = math.ceil(1.0 / corStepSize) + 1
numRespUnitStimuli = 0
numNoRespUnitStimuli = 0
logisticFitStepSize = 0.1
numLogisticFit = math.ceil(1.0 / logisticFitStepSize) + 1

startBaseline = -500
startTimeAvgFiringRate = 100 #100 #should fit response interval, otherwise spikes of best response can be outside of this interval and normalization fails
stopTimeAvgFiringRate = 800 #800 # 800 for rodrigo
minRatioActiveTrials = 0.5
minFiringRateConsider = 0.6
firingRateFactor = (1000 / (stopTimeAvgFiringRate - startTimeAvgFiringRate))
firingRateFactorBaselines = (1000 / (0 - startBaseline))

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

regions[allRegionsName] = Region(allRegionsName)
alphaBestResponse = []

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

    countBestResponseIsResponse = 0
    countBestResponseIsNoResponse = 0

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
        firingRates, consider = get_mean_firing_rate_normalized(trials, stimuliIndices, startTimeAvgFiringRate, stopTimeAvgFiringRate, minRatioActiveTrials, minFiringRateConsider)
        responseStimuliIndices = np.where((pvals < args.alpha) & (consider > 0))[0]
        responses = [thingsIndices[i] for i in responseStimuliIndices]
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
            #bestResponse = responseStimuliIndices[np.argmax(firingRates[responseStimuliIndices])] # best Response = highest z? highest response strength?
            #firingRates /= firingRates[bestResponse]
            indexBest = thingsIndices[bestResponse]
            alphaBestResponse.append(pvals[bestResponse])

            if indexBest not in responses : 
                countBestResponseIsNoResponse += 1
                #print("WARNING: best response is not a response! Subj " + str(subjectNum) + ", sess " + str(sessionNum) 
                #    + " (" + str(sessionParadigm) + "), chan " + str(channel) + ", clus " + str(cluster))
            else : 
                countBestResponseIsResponse += 1

            for i in range(numStimuli) : # responseIndices
                #if i == bestResponse and not includeSelfSimilarity : 
                #    continue
                index = thingsIndices[i]
                similarity = data.similarity_matrix[index][indexBest]
                similarityIndex = similarityMatrixToIndex[index, indexBest]
                #regions[allRegionsName].zScoresNorm.addValue(similarityIndex, zscores[i])
                #regions[site].zScoresNorm.addValue(similarityIndex, zscores[i])
                regions[allRegionsName].firingRatesNorm.addValue(similarityIndex, firingRates[i])
                regions[site].firingRatesNorm.addValue(similarityIndex, firingRates[i])

                regions[allRegionsName].firingRatesScatterSimilarities.append(similarity)
                regions[site].firingRatesScatterSimilarities.append(similarity)
                regions[allRegionsName].firingRatesScatter.append(firingRates[i])
                regions[site].firingRatesScatter.append(firingRates[i])

                regions[allRegionsName].similaritiesArray.append(similarity)
                regions[site].similaritiesArray.append(similarity) # 89 3 aos, channel 74 cluster 2, 1913, 1690

                #88 3 aos, channel 18, cluster 2
                if index in responses : 
                    regions[allRegionsName].responseStrengthHistResp.append(firingRates[i])
                    regions[site].responseStrengthHistResp.append(firingRates[i])
                    numRespUnitStimuli += 1
                else : 
                    regions[allRegionsName].responseStrengthHistNoResp.append(firingRates[i])
                    regions[site].responseStrengthHistNoResp.append(firingRates[i])
                    numNoRespUnitStimuli += 1

                if not i == bestResponse : ##and index in responses: ###
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

                zscore = (mean1 - meanAll) / stddevAll # meanBaseline ?, firingRates[stimNum]
                regions[allRegionsName].zScoresNorm.addValue(similarityIndex, zscore)
                regions[site].zScoresNorm.addValue(similarityIndex, zscore)

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
            
            ## fit step function
            if len(valuesCor) >= 3 : 
                rSquaredLog = regions[site].logisticFit.addFit(similaritiesCor, valuesCor)
                rSquaredStep = regions[site].stepFit.addFit(similaritiesCor, valuesCor)
                rSquaredGauss = regions[site].gaussFit.addFit(similaritiesCor, valuesCor)
                if rSquaredLog >= 0 and rSquaredStep >= 0 : 
                    regions[site].rDiffLog.append(rSquaredLog - rSquaredStep)
                if rSquaredGauss >= 0 and rSquaredStep >= 0 : 
                    regions[site].rDiffGauss.append(rSquaredGauss - rSquaredStep)
                if rSquaredLog >= 0 and rSquaredGauss >= 0 : 
                    regions[site].rDiffGauss.append(rSquaredGauss - rSquaredLog)
                if rSquaredLog >= 0 :
                    stepSlope = 0.05
                    logFit = regions[site].logisticFit
                    for i in range(len(logFit.params[0])) :
                        params = [logFit.params[0][i], logFit.params[1][i], logFit.params[2][i], logFit.params[3][i]]
                        yFit = regions[site].logisticFit.funcParams(np.arange(0,1,0.001), params)
                        steepestSlope = np.max(abs(yFit[:-1] - yFit[1:]))
                        regions[site].steepestSlopeLog.append(steepestSlope / stepSlope)
                        regions[allRegionsName].steepestSlopeLog.append(steepestSlope / stepSlope)
                
    
    print("Best response is response in " + str(countBestResponseIsResponse) + " cases and no response in " + str(countBestResponseIsNoResponse) + " cases.")
        #+ "Subj " + str(subjectNum) + ", sess " + str(sessionNum) 
        #+ " (" + str(sessionParadigm) + "), chan " + str(channel) + ", clus " + str(cluster))

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
allRegionCohensDPlots = []
allRegionResponseStrengthPlots = []
allRegionNumResponsesPlots = []
allRegionSpearmanPlots = []
allRegionSpearmanPPlots = []
allRegionPearsonPlots = []
allRegionPearsonPPlots = []
allRegionRespStrengthHistPlots = []
allRegionRespStrengthHistPlotsNo = []
allRegionLogisticFitBoxX0 = []
allRegionLogisticFitBoxK = []
allRegionLogisticFitBoxRSquared = []
allRegionLogisticFitPlots = []
allRegionLogisticFitAlignedPlots = []
allRegionGaussFitPlots = []
allRegionGaussFitAlignedPlots = []
allRegionRDiffPlots = []
allRegionSlopeLogPlots = []
#allRegionRDiffGaussPlots = []

figurePrepareTime = time.time()
spearmanPPlot = go.Figure()
spearmanPlot = go.Figure()
pearsonPPlot = go.Figure()
pearsonPlot = go.Figure()

for site in allSiteNames : 
    if site == allRegionsName : 
        continue
    regions[allRegionsName].logisticFit.append(regions[site].logisticFit)
    regions[allRegionsName].stepFit.append(regions[site].stepFit)
    regions[allRegionsName].gaussFit.append(regions[site].gaussFit)
    regions[allRegionsName].rDiffLog.extend(regions[site].rDiffLog)
    regions[allRegionsName].rDiffGauss.extend(regions[site].rDiffGauss)
    regions[allRegionsName].rDiffLogGauss.extend(regions[site].rDiffLogGauss)


for site in allSiteNames : 

    siteData = regions[site]
    print("Create figures for " + site)
    
    spearmanPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanCor))
    spearmanPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanP))
    pearsonPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonCor))
    pearsonPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonP))

    coactivationBeforeNormalization = siteData.coactivationNorm.y.copy()
    siteData.coactivationNorm.normalize()

    ticktextCoactivation = np.asarray([str(round(siteData.coactivationNorm.similarity[i], 2))
        #+ ": " + str(siteData.coactivationNorm.y[i] * 100) 
        #+ ("%.2f" % coactivationBeforeNormalization[i])
        + " (" + str(round(siteData.coactivationNorm.y[i] * 100, 5)) 
        + " = " + str(coactivationBeforeNormalization[i])
        + "/" + str(siteData.coactivationNorm.normalizer[i]) + ")" for i in range(len(coactivationBeforeNormalization))])

    fileDescription = paradigm + '_' + args.metric + '_' + site 

    totalNumResponseStrengthHist = max(1.0, len(siteData.responseStrengthHistResp) + len(siteData.responseStrengthHistNoResp)) #numRespUnitStimuli

    if len(regions[site].logisticFit.yFit[0]) > 0 :

        pTmp = regions[site].logisticFit.params
        
        for i in range(len(pTmp[0])) :
            logisticFitFigSingle = go.Figure(
                go.Scatter(
                    x=regions[site].logisticFit.xFit,
                    y=fitLogisticFunc(regions[site].logisticFit.xFit, pTmp[0][i], pTmp[1][i], pTmp[2][i], pTmp[3][i]),
                )
            )

            logisticFitFigSingle.update_layout(
                title_text="Logistic fit. R: " + str(round(regions[site].logisticFit.rSquared[i],2)) + ", X0: " + str(round(pTmp[0][i],2)) + ", K: " + str(round(pTmp[1][i],2)) + ", a: " + str(round(pTmp[2][i],2)) + ", c: " + str(round(pTmp[3][i],2)),
                xaxis_title='Semantic similarity',
                yaxis_title='Firing rate',
                showlegend=False 
            )
            saveImg(logisticFitFigSingle, "fit" + os.sep + "logistic_fit_single" + os.sep + fileDescription + "_" + str(i))

        #for i in range(len(regions[site].logisticFitK)) : 
        #    logisticFitFig.add_trace(go.Scatter(
        #        x=xLogisticFit,
        #        y=fitLogisticFunc(xLogisticFit, regions[site].logisticFitX0[i], regions[site].logisticFitK[i], regions[site].logisticFitA[i], regions[site].logisticFitC[i]),
        #    ))


    allRegionGaussFitPlots.append(createAndSave(
        createFitPlot(regions[site].gaussFit, "Gauss"), 
        "fit" + os.sep + "gaussian_fit" + os.sep + fileDescription))
    allRegionGaussFitAlignedPlots.append(createAndSave(
        createFitPlotAligned(regions[site].gaussFit, "Gauss"), 
        "fit" + os.sep + "gaussian_fit_aligned" + os.sep + fileDescription))
    allRegionLogisticFitPlots.append(createAndSave(
        createFitPlot(regions[site].logisticFit, "Logistic"), 
        "fit" + os.sep + "logistic_fit" + os.sep + fileDescription))
    allRegionLogisticFitAlignedPlots.append(createAndSave(
        createFitPlotAligned(regions[site].logisticFit, "Logistic"), 
        "fit" + os.sep + "logistic_fit_aligned" + os.sep + fileDescription))
    allRegionLogisticFitBoxX0.append(createAndSave(
        createBoxPlot([regions[site].logisticFit.params[0], regions[site].stepFit.params[0]], ["Log", "Step"], "Fit X0"), 
        "fit" + os.sep + "box_x0" + os.sep + fileDescription))
    allRegionLogisticFitBoxK.append(createAndSave(
        createBoxPlot( [regions[site].logisticFit.params[1]], ["K"], "Logistic fit K"), 
        "fit" + os.sep + "logistic_fit_box_k" + os.sep + fileDescription))
    allRegionLogisticFitBoxRSquared.append(createAndSave(
        createBoxPlot([regions[site].logisticFit.rSquared, regions[site].gaussFit.rSquared, regions[site].stepFit.rSquared], ["Log", "Gauss", "Step"], "Fit R Squared"), 
        "fit" + os.sep + "box_r_squared" + os.sep + fileDescription))
    allRegionRDiffPlots.append(createAndSave(
        createBoxPlot([regions[site].rDiffLog, regions[site].rDiffGauss, regions[site].rDiffLogGauss], ["r(Log) - r(Step)", "r(Gauss) - r(Step)", "r(Log) - r(Gauss)"], "Diff of R squared of gaussian fit and R squared of step fit"), 
        "fit" + os.sep + "box_r_diff" + os.sep + fileDescription))
    #allRegionRDiffGaussPlots.append(createAndSave(
    #    createBoxPlot([regions[site].rDiffLog], ["r(Log) - r(Step)"], "Diff of R squared of logistic fit and R squared of step fit"), 
    #    "fit" + os.sep + "box_r_diff_log" + os.sep + fileDescription))
    allRegionRespStrengthHistPlots.append(createAndSave(
        createHist(siteData.responseStrengthHistResp, np.arange(0,1,0.01), 100.0 / float(totalNumResponseStrengthHist), 'Firing rate', 'Stimuli in %'),
        "response_strength_hist" + os.sep + fileDescription)) 
    allRegionRespStrengthHistPlotsNo.append(createAndSave(
        createHist(siteData.responseStrengthHistNoResp, np.arange(0,1,0.01), 100.0 / float(totalNumResponseStrengthHist), 'Firing rate', 'Stimuli in %'),
        "response_strength_hist_no" + os.sep + fileDescription))
    allRegionNumResponsesPlots.append(createAndSave(
        createHist(siteData.numResponsesHist, range(args.max_responses_unit + 1), 1, 'Number of units', 'Number of responses'),
        "num_responses" + os.sep + fileDescription))
    allRegionCoactivationNormalizedPlots.append(createAndSave(
        createPlot(siteData.coactivationNorm.similarity[:-1], siteData.coactivationNorm.y * 100, "Normalized coactivation probability in %", "coactivation normalized", True, ticktextCoactivation), 
        "coactivation_normalized" + os.sep + fileDescription))
    allRegionZScoresPlots.append(createAndSave(
        createStepBoxPlot(uniqueSimilarities, siteData.zScoresNorm.values, "Mean zscores", "zscores", args.alpha_box), 
        "zscores" + os.sep + fileDescription))
    allRegionFiringRatesPlots.append(createAndSave(
        createStepBoxPlot(uniqueSimilarities, siteData.firingRatesNorm.values, "Normalized firing rates", "firing_rates", args.alpha_box), #False, True), 
        "firing_rates" + os.sep + fileDescription))
    allRegionCohensDPlots.append(createAndSave(
        createStepBoxPlot(uniqueSimilarities, siteData.cohensD.values, "Mean cohens d", "cohensd", args.alpha_box), 
        "cohensd" + os.sep + fileDescription))
    allRegionResponseStrengthPlots.append(createAndSave(
        createStepBoxPlot(uniqueSimilarities, siteData.responseStrength.values, "Mean response strength (median - meanBaseline) / maxMedian", "responseStrength", args.alpha_box), 
        "responseStrength" + os.sep + fileDescription))
    allRegionSpearmanPlots.append(createAndSave(
        createStepBoxPlot(np.arange(0.0, 1.0 + corStepSize, corStepSize), siteData.spearmanCorSteps, "Spearman correlation dependent on semantic similarity", "spearmanCorSteps", args.alpha_box, 'all'), 
        "spearmanCorSteps" + os.sep + fileDescription)) 
    allRegionPearsonPlots.append(createAndSave(
        createStepBoxPlot(np.arange(0.0, 1.0 + corStepSize, corStepSize), siteData.pearsonCorSteps, "Pearson correlation dependent on semantic similarity", "spearmanCorSteps", args.alpha_box, 'all'), 
        "spearmanCorSteps" + os.sep + fileDescription)) 
    allRegionSlopeLogPlots.append(createAndSave(
        createBoxPlot([regions[site].steepestSlopeLog], ["Slope"], "Steepest slope of logistic fit"), 
        "fit" + os.sep + "slope_log" + os.sep + fileDescription))


spearmanPlot.update_layout(title="Spearman correlation for responding units",)
saveImg(spearmanPlot, paradigm + "_" + args.metric + "_spearmanPlot")

spearmanPPlot.update_layout(title="Spearman p-value for responding units",)
saveImg(spearmanPPlot, paradigm + "_" + args.metric + "_spearmanPPlot")

pearsonPlot.update_layout(title="Pearson correlation for responding units",)
saveImg(pearsonPlot, paradigm + "_" + args.metric + "_pearsonPlot")

pearsonPPlot.update_layout(title="Pearson p-value for responding units",)
saveImg(pearsonPPlot, paradigm + "_" + args.metric + "_pearsonPPlot")

print("\nTime creating figures: " + str(time.time() - figurePrepareTime) + " s\n")

coactivationDiv, coactivationFigId, coactivationTableId = createRegionsDiv("Coactivation", allSiteNames)
copresentationDiv, copresentationFigId, copresentationTableId = createRegionsDiv("Copresentation", allSiteNames)
coactivationNormalizedDiv, coactivationNormalizedFigId, coactivationNormalizedTableId = createRegionsDiv("Coactivation - Normalized", allSiteNames)
zscoresDiv, zscoresFigId, zscoresTableId = createRegionsDiv("Mean zscores dependent on semantic similarity to best response", allSiteNames)
firingRatesDiv, firingRatesFigId, firingRatesTableId = createRegionsDiv("Normalized firing rates dependent on semantic similarity to best response", allSiteNames)
cohensDDiv, cohensDFigId, cohensDTableId = createRegionsDiv("Mean cohens d dependent on semantic similarity to best response", allSiteNames)
responseStrengthDiv, responseStrengthFigId, responseStrengthTableId = createRegionsDiv("Mean response strength dependent on semantic similarity to best response", allSiteNames)
spearmanCorStepsDiv, spearmanCorStepsFigId, spearmanCorStepsTableId = createRegionsDiv("Spearman correlation dependent on semantic similarity to best response", allSiteNames)
pearsonCorStepsDiv, pearsonCorStepsFigId, pearsonCorStepsTableId = createRegionsDiv("Pearson correlation dependent on semantic similarity to best response", allSiteNames)
numRespDiv, numRespFigId, numRespTableId = createRegionsDiv("Number of units with respective response counts", allSiteNames)
responseStrengthHistDiv, responseStrengthHistFigId, responseStrengthHistTableId = createRegionsDiv("Response strength histogram for responsive stimuli", allSiteNames)
responseStrengthHistDivNo, responseStrengthHistFigIdNo, responseStrengthHistTableIdNo = createRegionsDiv("Response strength histogram for non responsive stimuli", allSiteNames)
logisticFitDiv, logisticFitFigId, logisticFitTableId = createRegionsDiv("Logistic fit for all responsive units", allSiteNames)
logisticFitAlignedDiv, logisticFitAlignedFigId, logisticFitAlignedTableId = createRegionsDiv("Logistic fit for all responsive units - aligned", allSiteNames)
gaussianFitDiv, gaussianFitFigId, gaussianFitTableId = createRegionsDiv("Gaussian fit for all responsive units", allSiteNames)
gaussianFitAlignedDiv, gaussianFitAlignedFigId, gaussianFitAlignedTableId = createRegionsDiv("Gaussian fit for all responsive units - aligned", allSiteNames)
logisticFitX0Div, logisticFitX0FigId, logisticFitX0TableId = createRegionsDiv("Logistic fit for all responsive units: X0", allSiteNames)
logisticFitKDiv, logisticFitKFigId, logisticFitKTableId = createRegionsDiv("Logistic fit for all responsive units: K", allSiteNames)
logisticFitRSquaredDiv, logisticFitRSquaredFigId, logisticFitRSquaredTableId = createRegionsDiv("Logistic fit for all responsive units: R squared", allSiteNames)
logisticFitRDiffDiv, logisticFitRDiffFigId, logisticFitRDiffTableId = createRegionsDiv("Diff of R squared between logistic and step fit", allSiteNames)
#gaussianFitRDiffDiv, gaussianFitRDiffFigId, gaussianFitRDiffTableId = createRegionsDiv("Diff of R squared between gaussian and step fit", allSiteNames)
slopeLogDiv, slopeLogFigId, slopeLogTableId = createRegionsDiv("Deepest slope of log fit of firing rate to similarity", allSiteNames)

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
    slopeLogDiv,
    logisticFitDiv,
    logisticFitAlignedDiv,
    gaussianFitDiv,
    gaussianFitAlignedDiv,
    logisticFitX0Div,
    logisticFitKDiv,
    logisticFitRSquaredDiv,
    logisticFitRDiffDiv,
    #gaussianFitRDiffDiv,
    zscoresDiv, 
    cohensDDiv,
    responseStrengthDiv, 
    numRespDiv, 
])


#print("pvals best responses: " + str(alphaBestResponse))
print("pvals median: " + str(statistics.median(alphaBestResponse)))
print("pvals mean: " + str(statistics.mean(alphaBestResponse)))
print("pvals < 0.001: " + str(len(np.where(np.asarray(alphaBestResponse) < 0.001)[0])))
print("pvals > 0.001: " + str(len(np.where(np.asarray(alphaBestResponse) > 0.001)[0])))
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
    Output(component_id=logisticFitFigId, component_property='figure'), 
    Input(logisticFitTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionLogisticFitPlots, active_cell)

@app.callback(
    Output(component_id=logisticFitAlignedFigId, component_property='figure'), 
    Input(logisticFitAlignedTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionLogisticFitAlignedPlots, active_cell)

@app.callback(
    Output(component_id=gaussianFitFigId, component_property='figure'), 
    Input(gaussianFitTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionGaussFitPlots, active_cell)

@app.callback(
    Output(component_id=gaussianFitAlignedFigId, component_property='figure'), 
    Input(gaussianFitAlignedTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionGaussFitAlignedPlots, active_cell)

@app.callback(
    Output(component_id=logisticFitX0FigId, component_property='figure'), 
    Input(logisticFitX0TableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionLogisticFitBoxX0, active_cell)

@app.callback(
    Output(component_id=logisticFitKFigId, component_property='figure'), 
    Input(logisticFitKTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionLogisticFitBoxK, active_cell)

@app.callback(
    Output(component_id=logisticFitRSquaredFigId, component_property='figure'), 
    Input(logisticFitRSquaredTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionLogisticFitBoxRSquared, active_cell)

@app.callback(
    Output(component_id=logisticFitRDiffFigId, component_property='figure'), 
    Input(logisticFitRDiffTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionRDiffPlots, active_cell)

#@app.callback(
#    Output(component_id=gaussianFitRDiffFigId, component_property='figure'), 
#    Input(gaussianFitRDiffTableId, 'active_cell')
#)
#def update_output_div(active_cell):
#    return getActivePlot(allRegionRDiffGaussPlots, active_cell)

@app.callback(
    Output(component_id=slopeLogFigId, component_property='figure'), 
    Input(slopeLogTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSlopeLogPlots, active_cell)

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