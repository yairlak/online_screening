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
import shutil

from typing import List
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# utilility modules
from utils import *
from plot_helper import *
from data_manip import DataHandler

parser = argparse.ArgumentParser()

# 88_1_aos (great TC), 88_3, 89_1 (.8 lower), 89_2 (.7 and .9 lower), 89_3 (maybe .8), 89_5 (bad!), 
# 90_1_aos (.7 lower), 90_2_aos, 90_3_aos, 90_4 (0.7 lower), 90_5 (0.7 and 0.8 lower), 94_1, 95_1, 96_1, 97_1, 97_2 (0.8 lower), 98_1
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
parser.add_argument('--response_metric', default='firing_rates', # zscores, or pvalues or firing_rates
                    help='Metric to rate responses') # best firing_rates = best zscore ?!

# FLAGS
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--only_SU', default=False, 
                    help='If True, only single units are considered')
parser.add_argument('--only_responses', default=False, 
                    help='If True, only stimuli ecliciting responses are considered')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Alpha for responses') 
parser.add_argument('--alpha_box', type=float, default=0.001,
                    help='Alpha for box plot significance')
parser.add_argument('--alpha_fit', type=float, default=0.1,
                    help='Alpha for logistic fit')
parser.add_argument('--thresh_gof', type=float, default=-0.5,
                    help='Goodness of fit threshold for logistic fit')

# PLOT
parser.add_argument('--step', type=float, default=0.1,
                    help='Plotting detail')
parser.add_argument('--max_stdev_outliers', type=float, default=5,
                    help='Limit for excluding outliers')   
parser.add_argument('--max_responses_unit', type=float, default=20,
                    help='Limit for counting responses per unit for histogram')      
parser.add_argument('--plot_regions', default='collapse_hemispheres',
                    help='"full"->all regions, "hemispheres"->split into hemispheres, "collapse_hemispheres"->regions of both hemispheres are collapsed')      
parser.add_argument('--step_span', type=float, default=0.05,
                    help='Plotting detail for span')
parser.add_argument('--step_correlation_split', type=int, default=10,
                    help='How many datapoints for each step in the split correlation plot')

# PATHS
parser.add_argument('--path2metadata',
                    default='../data/THINGS/things_concepts.tsv',
                    help='TSV file containing semantic categories, etc.')
parser.add_argument('--path2categories',
                    default='../data/THINGS/category_mat_manual.tsv')
parser.add_argument('--path2wordembeddings',
                    default='../data/THINGS/sensevec_augmented_with_wordvec.csv')
parser.add_argument('--path2worndetids',
                    default='../data/THINGS/wordnet_id.csv')
parser.add_argument('--path2semanticdata',
                    default='../data/semantic_data/')
parser.add_argument('--path2data', 
                    default='../data/aos_after_manual_clustering/') #aos_after_manual_clustering aos_one_session aos_selected_sessions
parser.add_argument('--path2images', 
                    default='../figures/semantic_coactivation') 

args=parser.parse_args()

@dataclass 
class SimilaritiesArray:
    values : List = field(default_factory=lambda: [])
    similarities : List = field(default_factory=lambda: [])

    def addValue(self, similarity, value) : 
        self.similarities.append(similarity)
        self.values.append(value)

@dataclass
class NormArray:
    y : List = field(default_factory=lambda: np.zeros((nSimilarities-1)))
    normalizer : List = field(default_factory=lambda: np.zeros((nSimilarities-1)))
    similarity : List = field(default_factory=lambda: [])

    def normalize(self) : 
        self.similarity = uniqueSimilarities
        for i in range(len(self.normalizer)) :
            if self.normalizer[i] > 0 : 
                self.y[i] = self.y[i] / float(self.normalizer[i])
        
    def addValue(self, index, value) : 
        self.y[index] += value
        self.normalizer[index] += 1

@dataclass
class MedianSplit:
    median: float = 0.0
    xLower: List = field (default_factory=lambda: [])
    xHigher: List = field (default_factory=lambda: [])
    yLower: List = field (default_factory=lambda: [])
    yHigher: List = field (default_factory=lambda: [])
    
    def createMedianSplitPlot(self, title, yLabel) :
        if len(self.xLower) == 0 or len(self.xHigher) == 0: 
            return createStdErrorMeanPlot([0], [[0]], title, yLabel)
        
        x = [statistics.mean(self.xLower), statistics.mean(self.xHigher)]
        #y = [statistics.mean(self.yLower), statistics.mean(self.yHigher)]
        #error=[sem(self.yLower), sem(self.yHigher)]


        fig = createStdErrorMeanPlot(x, [self.yLower, self.yHigher], title + ". Median: " + str(round(self.median, 3)), yLabel, False)
        fig.update_layout(
            width=600,
            height=600,
        )

        return fig

    def getMedianSplit(x, y) :
        x = np.array(x)
        y = np.array(y)
        median = statistics.median(x)
        lowerIndex = np.where(x <= median)[0]
        higherIndex = np.where(x > median)[0]
        xLower = x[lowerIndex]
        xHigher = x[higherIndex]
        yLower = y[lowerIndex]
        yHigher = y[higherIndex]
        return MedianSplit(median, xLower, xHigher, yLower, yHigher)
    
@dataclass
class Region:
    sitename: str
    coactivationNorm : NormArray = field(default_factory=lambda: NormArray())
    coactivationProb : NormArray = field(default_factory=lambda: NormArray())
    numResponsesPerConcept : List = field (default_factory=lambda: np.zeros(len(data.df_metadata['uniqueID'])))
    zScoresNorm : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    firingRatesNorm : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    similaritiesConcepts : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    similaritiesConceptsDiscrete : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    similaritiesResiduals : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    #oneDimData : defaultdict = field(default_factory=lambda: defaultdict())
    #combinationsData : defaultdict = field(default_factory=lambda: defaultdict(lambda: [])) #defaultdict(lambda: "Not Present") 
    pvalues : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    numResponsesHist : List = field (default_factory=lambda: [])
    similaritiesArray : List = field (default_factory=lambda: [])
    responseStrengthHistResp : List = field (default_factory=lambda: [])
    responseStrengthHistNoResp : List = field (default_factory=lambda: [])
    responseStrengthHistRespSelf : List = field (default_factory=lambda: [])
    responseStrengthHistNoRespSelf : List = field (default_factory=lambda: [])
    maxDistResp : List = field (default_factory=lambda: [])

    spearmanCor : List = field (default_factory=lambda: [])
    spearmanP : List = field (default_factory=lambda: [])
    spearmanCorSteps : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    spearmanCorSplit : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    spearmanCorMSplit : MedianSplit = field(default_factory=lambda: MedianSplit()) 
    pearsonCor : List = field (default_factory=lambda: [])
    pearsonP : List = field (default_factory=lambda: [])
    pearsonCorSteps : SimilaritiesArray = field(default_factory=lambda: SimilaritiesArray())
    copresentationSpan = np.zeros(int(math.ceil(1/args.step_span)))

    logisticFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(logFunc, logFuncParams, ["x0", "K", "A", "C"], p0=[0.5, 1, 0, 1], bounds=[[0, -1000, 0, 0], [1, 1000, 1, 1]]))
        #Fitter.getFitter(logFunc, logFuncParams, ["x0", "K", "A", "C"], p0=[0.5, 1, 0, 1], bounds=[[0, -1000, -10, -10], [1, 1000, 10, 10]]))
    logisticFitGood : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(logFunc, logFuncParams, ["x0", "K", "A", "C"], p0=[0.5, 1, 0, 1], bounds=[[0, -1000, -10, -10], [1, 1000, 10, 10]]))
    stepFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(step, stepParams, ["x0", "a", "b"], p0=[0.5, 0, 1], bounds=[[0, 0, 0], [1, 1, 1]]))
    gaussFit : Fitter = field(default_factory=lambda: 
        Fitter.getFitter(halfGauss, halfGaussParams, ["x0", "a", "b", "sigma"], p0=[0.5, 1, 0, 1], bounds=[[0, 0, 0, 0], [1, 100, 1, 100]]))
    rDiffLog : List = field (default_factory=lambda: [])
    rDiffGauss : List = field (default_factory=lambda: [])
    rDiffLogGauss : List = field (default_factory=lambda: [])


def createAndSave(func, filename) : 
    fig = func #updateFigure(func) 
    saveImg(fig, filename)
    return fig

def getImgpath() : 
    if args.only_SU : 
        unitPath = "SU"
    else : 
        unitPath = "MU_SU"
    return args.path2images + "_" + args.response_metric + os.sep + args.plot_regions + "_" + unitPath

def saveImg(fig, filename) : 
    pathWordEmbeddings = "" # args.path2wordembeddings.split(".")[-2].split("/")[-1] + "_"

    file = getImgpath() + os.sep + filename 

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        #fig.write_image(file + ".svg")
        fig.write_image(file + ".png")

def getCategory(indexThings) : 
    categoryResponse = np.where(data.df_categories.loc[indexThings] == 1)[0]
    if isinstance(categoryResponse, (list, tuple, np.ndarray)) : 
        if len(categoryResponse) == 0 : 
            return -1
        categoryResponse = categoryResponse[0]

    return categoryResponse


#############
# LOAD DATA #
#############
print("\n--- START ---")
startLoadData = time.time()

data = DataHandler(args) # class for handling neural and feature data
#data.load_categories()
#data.get_category_distances()
data.load_metadata() # -> data.df_metadata
data.load_categories() # -> data.df_categories
data.load_word_embeddings() # -> data.df_word_embeddings
data.load_wordnet_ids()
categorySimilarities, categorySimilaritiesBinary = data.get_category_similarities()
#data.load_word_embeddings_tsne() # -> data.df_word_embeddings_tsne
data.load_similarity_matrix() # -> data.similarity_matrix
data.load_neural_data() # -> data.neural_data

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
numExludeConceptsResiduals = 4

uniqueSimilarities = np.arange(0.0, 1.0 + (1.0 % args.step), args.step)
nSimilarities = len(uniqueSimilarities)
numConcepts = len(categorySimilarities)
similarityMatrixToIndex = (data.similarity_matrix.to_numpy().round(decimals=4) / args.step).astype(int)
corStepSize = 0.1
numCorSteps = math.ceil(1.0 / corStepSize) + 1
numRespUnitStimuli = 0
numNoRespUnitStimuli = 0
logisticFitStepSize = 0.1
numLogisticFit = math.ceil(1.0 / logisticFitStepSize) + 1
#copresentationSpanX = np.arange(0.0, 1.0 + (1.0 % stepSpan), stepSpan)

startBaseline = -500
#startTimeAvgFiringRate = 100 #100 #should fit response interval, otherwise spikes of best response can be outside of this interval and normalization fails
#stopTimeAvgFiringRate = 800 #800 # 800 for rodrigo
minRatioActiveTrials = 0.5
minFiringRateConsider = 1
#firingRateFactor = (1000 / (stopTimeAvgFiringRate - startTimeAvgFiringRate))
#firingRateFactorBaselines = (1000 / (0 - startBaseline))

allRegionsName = 'All'
allSiteNames = [allRegionsName]
regions = {}

regions[allRegionsName] = Region(allRegionsName)
#regions[allRegionsName].data[""]
regions[allRegionsName].coactivationProb.normalizer = np.zeros((nTHINGS))
regions[allRegionsName].coactivationProb.y = np.zeros((nTHINGS,nTHINGS))
alphaBestResponse = []
sitesToConsider = ["LA", "RA", "LEC", "REC", "LAH", "RAH", "LMH", "RMH", "LPHC", "RPHC", "LPIC", "RPIC"]
#sitesToExclude = ["LFa", "LTSA", "LTSP", "Fa", "TSA", "TSP", "LPL", "LTP", "LTB", "RMC", "RAI", "RAC", "RAT", "RFO", "RFa", "RFb", "RFc", "RT", "RFI", "RFM", "RIN", "LFI", "LFM", "LIN"]
#sitesToExclude = ["LFa", "LTSA", "LTSP", "Fa", "TSA", "TSP", "LTP", "LTB", "RMC", "RAI", "RAC", "RAT", "RFO", "RFa", "RFb", "RFc"] 

unitCounter = 0
unitCounterLeft = 0
unitCounterRight = 0
responsiveUnitCounter = 0
responsiveUnitCounterLeft = 0
responsiveUnitCounterRight = 0
sessionCounter = 0
unitsPerSite = []
unitsPerSiteResponsive = []
numUnitsTotal = {}
numUnitsResponsive = {}

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
    #objectNames = data.neural_data[session]['objectnames']
    stimuliIndices = data.neural_data[session]['objectindices_session']
    thingsIndices = data.get_THINGS_indices(data.neural_data[session]['stimlookup'])
    numStimuli = len(stimlookup)

    # do it before to make it faster
    allSitesSession = []
    numUnits = 0

    for unit in units:
        site = data.neural_data[session]['units'][unit]['site']

        if not site in sitesToConsider : 
            continue

        site = getSite(site, args.plot_regions)

        if site not in allSitesSession : 
            allSitesSession.append(site)

        if site not in allSiteNames : 
            allSiteNames.append(site)
            regions[site] = Region(site)
            regions[site].coactivationProb.normalizer = np.zeros((nTHINGS))
            regions[site].coactivationProb.y = np.zeros((nTHINGS,nTHINGS))
        
        if not (not data.neural_data[session]['units'][unit]['kind'] == 'SU' and args.only_SU): 
            numUnits = numUnits + 1
            
            if site in numUnitsTotal : 
                numUnitsTotal[site] = numUnitsTotal[site] + 1 
            else : 
                numUnitsTotal[site] = 1


    for site in allSitesSession :  
        if not site in sitesToConsider : 
            continue
        for i1, i2 in itertools.product(thingsIndices, thingsIndices) : 
            if i2 > i1 or (i1 == i2 and not includeSelfSimilarity) :
                continue
            regions[allRegionsName].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += numUnits # copresentation
            regions[site].coactivationNorm.normalizer[similarityMatrixToIndex[i1, i2]] += numUnits
            distStep = int(1 - data.similarity_matrix[i1][i2] / args.step_span)
            regions[allRegionsName].copresentationSpan[distStep] += numUnits
            regions[site].copresentationSpan[distStep] += numUnits

    countBestResponseIsResponse = 0
    countBestResponseIsNoResponse = 0
    

    for unit in units:
        unitData = data.neural_data[session]['units'][unit]
        if (not unitData['kind'] == 'SU' and args.only_SU) or not unitData['site'] in sitesToConsider : 
            continue
        pvals = unitData['p_vals']
        site = getSite(unitData['site'], args.plot_regions)
        trials = unitData['trial']
        channel = unitData['channel_num']
        cluster = unitData['class_num']
        zscores = unitData['zscores'] 
        firingRates = unitData['firing_rates']
        responseStimuliIndices = unitData['responses']
        responseIndicesFit = np.where((pvals < args.alpha_fit) & (unitData['consider']  == 1))[0] 
        #responseStimuliIndicesInverted = np.where((pvals >= args.alpha) | (consider == 0))[0]
        responses = [thingsIndices[i] for i in responseStimuliIndices]
        zscoresSortedIndices = np.argsort(np.asarray(zscores))[::-1]
        bestZscoresIndices = zscoresSortedIndices[:30] 

        similaritiesCor = []
        valuesCor = []
        similaritiesCorSteps = [[] for i in range(numCorSteps)]
        valuesCorSteps = [[] for i in range(numCorSteps)]
        #zscores = zscores / max(zscores)
        
        #if subjectNum == 103 and sessionNum == 1 and channel == 70 and cluster == 1 : # TODO!!!!!!!!!!!!! Here, there are many responses to car parts which drives the very high end of the coactivation plot
        #    continue

        unitCounter += 1
        isLeftSite = True if unitData['site'][0] == 'L' else False
        isRightSite = True if unitData['site'][0] == 'R' else False

        if isLeftSite : 
            unitCounterLeft += 1
        if isRightSite : 
            unitCounterRight += 1
        if not isLeftSite and not isRightSite : 
            print("No side: " + unitData['site'])

        #if unitData['kind'] == 'SU' : 
        siteIndex = np.where(np.asarray(allSiteNames) == site)[0][0]
        if siteIndex > len(unitsPerSite)-1 : 
            unitsPerSite.append(1)
            unitsPerSiteResponsive.append(1)
        else : 
            unitsPerSite[siteIndex] += 1
            if len(responses) > 0 :  
                unitsPerSiteResponsive[siteIndex] += 1


        if args.response_metric == "zscores" : 
            metric = zscores
        else : 
            if args.response_metric == "pvalues" : 
                metric = pvals
            else : 
                metric = firingRates
        
        if not len(firingRates) == numStimuli or not len(zscores) == numStimuli : 
            print("ERROR: array length does not fit for zscores or firing rates!")

        for i1 in responses :  
            regions[allRegionsName].numResponsesPerConcept[i1] += 1
            regions[site].numResponsesPerConcept[i1] += 1
            
            regions[allRegionsName].coactivationProb.normalizer[i1] +=1
            regions[site].coactivationProb.normalizer[i1] +=1
            
            for i2 in responses : 
                if i2 > i1 or (i1 == i2 and not includeSelfSimilarity) :
                    continue
                regions[allRegionsName].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1
                regions[site].coactivationNorm.y[similarityMatrixToIndex[i1, i2]] += 1

                regions[allRegionsName].coactivationProb.y[i1,i2] +=1
                regions[site].coactivationProb.y[i1,i2] +=1

                #if similarityMatrixToIndex[i1, i2] >= similarityMatrixToIndex.max() - 2 : 
                #    print("pat: " + str(subjectNum) + ", session: " + str(sessionNum) + ", channel: " + str(channel) + ", cluster: " + str(cluster) + ", index: " + str(similarityMatrixToIndex[i1, i2]))

        if len(responses) > 0 :
            responsiveUnitCounter += 1 
            if isLeftSite :
                responsiveUnitCounterLeft += 1 
            if isRightSite : 
                responsiveUnitCounterRight += 1 
            regions[allRegionsName].numResponsesHist.append(len(responses))
            regions[site].numResponsesHist.append(len(responses))

            if site in numUnitsResponsive : 
                numUnitsResponsive[site] += 1 
            else : 
                numUnitsResponsive[site] = 1

            # zscores
            #metric_only_consider = metric
            if args.response_metric == "pvalues" : 
                #metric_only_consider[responseStimuliIndicesInverted] = 100
                bestResponse = np.argmin(metric) # best Response = highest z? highest response strength?
            else : 
                #metric_only_consider[responseStimuliIndicesInverted] = 0
                bestResponse = np.argmax(metric) # best Response = highest z? highest response strength?

            indexBest = thingsIndices[bestResponse]
            alphaBestResponse.append(pvals[bestResponse])

            if indexBest not in responses : 
                countBestResponseIsNoResponse += 1
            else : 
                countBestResponseIsResponse += 1

            for i in range(numStimuli) : # responseIndices
                #if i == bestResponse and not includeSelfSimilarity : 
                #    continue
                index = thingsIndices[i]

                indexHighestFiring = np.argmax(firingRates)
                #88 3 aos, channel 18, cluster 2
                if index in responses : ## Rodrigos histograms
                    regions[allRegionsName].responseStrengthHistRespSelf.append(firingRates[i])
                    regions[site].responseStrengthHistRespSelf.append(firingRates[i])
                    if not i == indexHighestFiring : 
                        regions[allRegionsName].responseStrengthHistResp.append(firingRates[i])
                        regions[site].responseStrengthHistResp.append(firingRates[i])
                    numRespUnitStimuli += 1
                else : 
                    regions[allRegionsName].responseStrengthHistNoRespSelf.append(firingRates[i])
                    regions[site].responseStrengthHistNoRespSelf.append(firingRates[i])
                    if not i == indexHighestFiring : 
                        regions[allRegionsName].responseStrengthHistNoResp.append(firingRates[i])
                        regions[site].responseStrengthHistNoResp.append(firingRates[i])
                    numNoRespUnitStimuli += 1

                if index == indexBest and not includeSelfSimilarity : 
                    continue

                if not index in responses and args.only_responses: 
                    continue

                similarity = data.similarity_matrix[index][indexBest]
                #similarityIndex = similarityMatrixToIndex[index, indexBest]

                if not (index == indexBest and args.response_metric == "firing_rates" ): 
                    regions[allRegionsName].firingRatesNorm.addValue(similarity, firingRates[i])
                    regions[site].firingRatesNorm.addValue(similarity, firingRates[i])
                if not (index == indexBest and args.response_metric == "pvalues" ): 
                    regions[allRegionsName].pvalues.addValue(similarity, pvals[i])
                    regions[site].pvalues.addValue(similarity, pvals[i])
                if not (index == indexBest and args.response_metric == "zscores"): 
                    regions[allRegionsName].zScoresNorm.addValue(similarity, zscores[i])
                    regions[site].zScoresNorm.addValue(similarity, zscores[i])
                    #regions[allRegionsName].combinationsData["zscores"].append(zscores[i])
                    #regions[site].combinationsData["zscores"].append(zscores[i])

                categoryResponse = getCategory(index)
                categoryBest = getCategory(indexBest)
                if not index==indexBest and categoryResponse >= 0 and categoryBest >= 0 : 
                    similarityConcepts = categorySimilarities[categoryResponse][categoryBest]
                    regions[allRegionsName].similaritiesConcepts.addValue(similarityConcepts, metric[i])
                    regions[site].similaritiesConcepts.addValue(similarityConcepts, metric[i])

                    similarityConceptsDiscrete = categorySimilaritiesBinary[categoryResponse][categoryBest]
                    regions[allRegionsName].similaritiesConceptsDiscrete.addValue(similarityConceptsDiscrete, metric[i])
                    regions[site].similaritiesConceptsDiscrete.addValue(similarityConceptsDiscrete, metric[i])

                    if similarityConceptsDiscrete < numConcepts - numExludeConceptsResiduals : 
                        regions[allRegionsName].similaritiesResiduals.addValue(similarity, metric[i])
                        regions[site].similaritiesResiduals.addValue(similarity, metric[i])

                        #regions[allRegionsName].combinationsData["similarities_concepts"].append(similarityConcepts)
                        #regions[site].combinationsData["similarities_concepts"].append(similarityConcepts) 
                        #regions[allRegionsName].combinationsData["zscores_concepts"].append(zscores[i])
                        #regions[site].combinationsData["zscores_concepts"].append(zscores[i]) 

                #regions[allRegionsName].similaritiesArray.append(similarity)
                #regions[site].similaritiesArray.append(similarity) # 89 3 aos, channel 74 cluster 2, 1913, 1690
                #regions[allRegionsName].combinationsData["similarities"].append(similarity)
                #regions[site].combinationsData["similarities"].append(similarity) 
                #regions[allRegionsName].combinationsData["site"].append(site)
                #regions[site].combinationsData["site"].append(site) 
                
                

                if not i == bestResponse : #and i in responseIndicesFit : #responseIndicesFit, bestZscoresIndices
                    corStep = int(similarity / corStepSize)
                    similaritiesCor.append(similarity)
                    valuesCor.append(metric[i])

                    #for j in range(0, corStep) : 
                    for j in range(corStep, len(similaritiesCorSteps)-1) : 
                    #j = corStep
                        similaritiesCorSteps[j].append(similarity)
                        valuesCorSteps[j].append(metric[i])
                

        if len(responses) > 1 :
            maxDist = 0
            for i in responses : 
                for j in responses : 
                    dist = 1 - data.similarity_matrix[i][j]
                    if dist > maxDist : 
                        maxDist = dist
            
            regions[allRegionsName].maxDistResp.append(maxDist)
            regions[site].maxDistResp.append(maxDist)

        if len(valuesCor) >= 5 : 
            regions[site].spearmanCorMSplit = MedianSplit.getMedianSplit(similaritiesCor, valuesCor)
            regions[allRegionsName].spearmanCorMSplit = MedianSplit.getMedianSplit(similaritiesCor, valuesCor)
            
            spearman = stats.spearmanr(similaritiesCor, valuesCor)
            if not math.isnan(spearman.correlation) : 
                regions[site].spearmanCor.append(spearman.correlation)
                regions[site].spearmanP.append(spearman.pvalue)
                regions[allRegionsName].spearmanCor.append(spearman.correlation)
                regions[allRegionsName].spearmanP.append(spearman.pvalue)
            else : 
                print("WARNING!! spearman is nan")

            pearson = stats.pearsonr(similaritiesCor, valuesCor)
            regions[site].pearsonCor.append(pearson[0])
            regions[site].pearsonP.append(pearson[1])
            regions[allRegionsName].pearsonCor.append(pearson[0])
            regions[allRegionsName].pearsonP.append(pearson[1])

            for i in range(numCorSteps) : 
                similarityCorStep = i * corStepSize
                if len(valuesCorSteps[i]) > 0 : 
                    spearman = stats.spearmanr(similaritiesCorSteps[i], valuesCorSteps[i]) 
                    if not math.isnan(spearman.correlation) : 
                        regions[site].spearmanCorSteps.addValue(similarityCorStep, spearman.correlation)
                        regions[allRegionsName].spearmanCorSteps.addValue(similarityCorStep, spearman.correlation)
                    
                if len(valuesCorSteps[i]) >= 2 : 
                    pearson = stats.pearsonr(similaritiesCorSteps[i], valuesCorSteps[i]) 
                    if not math.isnan(pearson[0]) and not math.isnan(pearson[1]) : 
                        regions[site].pearsonCorSteps.addValue(similarityCorStep, pearson[0])
                        regions[allRegionsName].pearsonCorSteps.addValue(similarityCorStep, pearson[0])

            similaritiesCorSortedIndices = np.argsort(similaritiesCor)
            similaritiesCorSorted = [similaritiesCor[index] for index in similaritiesCorSortedIndices] 
            valuesCorSorted = [valuesCor[index] for index in similaritiesCorSortedIndices] 

            for j in range(len(valuesCorSorted)-1, 0, -args.step_correlation_split) : 
                i = max(j-args.step_correlation_split+1,0) #min(i+args.step_correlation_split, len(valuesCor)-1)
                if j < i + 2 :
                    break
                spearmanX = statistics.median(similaritiesCorSorted[i:j])
                spearmanSplit = stats.spearmanr(similaritiesCorSorted[i:j], valuesCorSorted[i:j])
                if not math.isnan(spearmanSplit.correlation) :
                    regions[site].spearmanCorSplit.addValue(spearmanX, spearmanSplit.correlation)
                    regions[allRegionsName].spearmanCorSplit.addValue(spearmanX, spearmanSplit.correlation)
            
            ## fit step function
            if len(valuesCor) >= 3 : 
                plotDetails = session + "_ch" + str(channel) + "_cluster" + str(cluster)
                rSquaredLog = regions[site].logisticFit.addFit(similaritiesCor, valuesCor, plotDetails, spearman.correlation) ### TODO: all values, not only responses?
                rSquaredLog = regions[allRegionsName].logisticFit.addFit(similaritiesCor, valuesCor, plotDetails, spearman.correlation, site) ### TODO: all values, not only responses?
                rSquaredStep = regions[site].stepFit.addFit(similaritiesCor, valuesCor, plotDetails)
                rSquaredStep = regions[allRegionsName].stepFit.addFit(similaritiesCor, valuesCor, plotDetails, site)
                rSquaredGauss = regions[site].gaussFit.addFit(similaritiesCor, valuesCor, plotDetails)
                rSquaredGauss = regions[allRegionsName].gaussFit.addFit(similaritiesCor, valuesCor, plotDetails, site)
                if rSquaredLog >= 0 and rSquaredStep >= 0 : 
                    regions[site].rDiffLog.append(rSquaredLog - rSquaredStep)
                    regions[allRegionsName].rDiffLog.append(rSquaredLog - rSquaredStep)
                if rSquaredGauss >= 0 and rSquaredStep >= 0 : 
                    regions[site].rDiffGauss.append(rSquaredGauss - rSquaredStep)
                    regions[allRegionsName].rDiffGauss.append(rSquaredGauss - rSquaredStep)
                if rSquaredLog >= 0 and rSquaredGauss >= 0 : 
                    regions[site].rDiffGauss.append(rSquaredGauss - rSquaredLog)
                    regions[allRegionsName].rDiffGauss.append(rSquaredGauss - rSquaredLog)
        else: 
            if len(responses) > 1 :
                print("WARNING! NOT ENOUGH DATAPOINTS FOR SPEARMAN CORRELATION")
                
    
    print("Best response is response in " + str(countBestResponseIsResponse) + " cases and no response in " + str(countBestResponseIsNoResponse) + " cases.")
        #+ "Subj " + str(subjectNum) + ", sess " + str(sessionNum) 
        #+ " (" + str(sessionParadigm) + "), chan " + str(channel) + ", clus " + str(cluster))

    print("Prepared data of session " + session + ". Time: " + str(time.time() - startPrepareSessionData) + " s" )

    sessionCounter += 1
    if onlyTwoSessions and sessionCounter >= 2 : 
        break

print("\nTime preparing data: " + str(time.time() - startPrepareDataTime) + " s")
print("\nNum sessions: " + str(sessionCounter) )
print("\nNum units: " + str(unitCounter))
print("Num units left: " + str(unitCounterLeft))
print("Num units right: " + str(unitCounterRight))
print("\nNum responsive units: " + str(responsiveUnitCounter))
print("Num responsive units left: " + str(responsiveUnitCounterLeft))
print("Num responsive units right: " + str(responsiveUnitCounterRight))


allRegionCoactivationPlots = []
allRegionCoactivationProbPlots = []
allRegionCopresentationPlots = []
allRegionCoactivationNormalizedPlots = []
allRegionZScoresPlots = []
allRegionPValuesPlots = []
allRegionFiringRatesPlots = []
allRegionNumResponsesPlots = []
allRegionMaxDistPlots = []
allRegionSpearmanPlots = []
allRegionSpearmanSlopePlots = []
allRegionSpearmanSplitPlots = []
allRegionSpearmanMSplitPlots = []
allRegionSpearmanPPlots = []
allRegionPearsonPlots = []
allRegionPearsonPPlots = []
allRegionRespStrengthHistPlots = []
allRegionRespStrengthHistPlotsNo = []
allRegionRespStrengthHistSelfPlots = []
allRegionRespStrengthHistSelfPlotsNo = []
allRegionLogisticFitBoxX0 = []
allRegionLogisticFitBoxK = []
allRegionLogisticFitBoxRSquared = []
allRegionLogisticFitPlots = []
allRegionLogisticFitAlignedPlots = []
allRegionGaussFitPlots = []
allRegionGaussFitAlignedPlots = []
allRegionRDiffPlots = []
allRegionSlopePlots = []
#allRegionRDiffGaussPlots = []

figurePrepareTime = time.time()
spearmanPPlot = go.Figure()
spearmanPlot = createStdErrorMeanPlot([regions[site].sitename for site in allSiteNames], [regions[site].spearmanCor for site in allSiteNames], "Spearman correlation for responding units", "")
pearsonPPlot = go.Figure()
pearsonPlot = go.Figure()

#for site in allSiteNames : 

#    if site == allRegionsName : 
#        continue
#    regions[allRegionsName].logisticFit.append(regions[site].logisticFit)
#    regions[allRegionsName].stepFit.append(regions[site].stepFit)
#    regions[allRegionsName].gaussFit.append(regions[site].gaussFit)
#    regions[allRegionsName].rDiffLog.extend(regions[site].rDiffLog)
#    regions[allRegionsName].rDiffGauss.extend(regions[site].rDiffGauss)
#    regions[allRegionsName].rDiffLogGauss.extend(regions[site].rDiffLogGauss)

colorDiscreteMap={'A':'purple',
                    'EC':'blue',
                    'PHC':'green',
                    'H':'red',
                    'PIC':'orange'}
if args.plot_regions == "hemispheres" :  
    colorDiscreteMap={'R':'blue',
                        'L':'yellow',}


numUnit_df = pd.DataFrame()
numUnit_df["sites"] = allSiteNames[1:]
numUnit_df["numUnits"] = unitsPerSite[1:]
numUnit_df["numUnitsResponsive"] = unitsPerSiteResponsive[1:]

numUnitPlot = px.bar(numUnit_df, x="sites", y="numUnits")
saveImg(numUnitPlot, "num_units")
 
numUnitPlot = px.bar(numUnit_df, x="sites", y="numUnitsResponsive")
saveImg(numUnitPlot, "num_units_responsive")

pieLabelsFull = ['A', 'H', 'EC', 'PHC', 'PIC']
pieLabels = [label for label in pieLabelsFull if label in list(numUnitsTotal.keys())] #np.sort(np.array(list(numUnitsTotal.keys())))
colorDiscreteMapAll = { key:val for key, val in colorDiscreteMap.items() if key in pieLabels} 
pieValues = np.array([numUnitsTotal[p] for p in pieLabels])
pieTotalPlot = go.Figure(
    go.Pie(values=pieValues, 
        labels=pieLabels, sort=False,
        direction ='clockwise',
        #textfont_size=20,
        marker_colors=[colorDiscreteMapAll[p] for p in pieLabels], 
        title=str(sum(pieValues)) + " units in total"))
pieTotalPlot.update_layout(font=dict(size=18))
pieTotalPlot.update_traces(textinfo='label+value')
if args.only_SU : 
    saveImg(pieTotalPlot, "num_units_pie_SU")
else: 
    saveImg(pieTotalPlot, "num_units_pie_SU_MU")


pieLabels = [label for label in pieLabelsFull if label in list(numUnitsResponsive.keys())]
colorDiscreteMapResponsive = { key:val for key, val in colorDiscreteMap.items() if key in pieLabels} 
pieValues = np.array([numUnitsResponsive[p] for p in pieLabels])
pieResponsivePlot = go.Figure(
    go.Pie(values=pieValues, labels=pieLabels, sort=False, 
        direction ='clockwise',
        marker_colors=[colorDiscreteMapResponsive[p] for p in pieLabels], 
        title=str(sum(pieValues)) + " responsive units in total"))
pieResponsivePlot.update_traces(textinfo='label+value')
pieResponsivePlot.update_layout(font=dict(size=18))
if args.only_SU : 
    saveImg(pieResponsivePlot, "num_units_pie_responsive_SU")
else: 
    saveImg(pieResponsivePlot, "num_units_pie_responsive_SU_MU")

if not args.dont_plot :
    try: 
        shutil.rmtree(getImgpath() + os.sep + "fit" + os.sep + "logistic_fit_single")
    except : 
        print("Warning. Single fit plots can not be removed")
#os.rmdir(getImgpath() + os.sep + "fit" + os.sep + "logistic_fit_single")
        
#sloped_df = pd.DataFrame({'slopes' : regions[allRegionsName].logisticFitGood.steepestSlopes, 'sites': regions[allRegionsName].logisticFitGood.sites})
#regions[allRegionsName].logisticFit.calculateSteepestSlopes()
#regions[allRegionsName].logisticFitGood = regions[allRegionsName].logisticFit.getGood(regions[allRegionsName].logisticFit.rSquared, 0.1)
for site in allSiteNames : 
    
    regions[site].logisticFit.calculateSteepestSlopes()
    regions[site].gaussFit.calculateSteepestSlopes()
    #goodFit = np.where(logFit.gof < args.thresh_gof)[0]

    #regions[site].logisticFitGood = logFit.getGood(gof, -args.thresh_gof)
    regions[site].logisticFitGood = regions[site].logisticFit.getGood(regions[site].logisticFit.rSquared, 0.1)

slopesStart = -2
slopesStop = 7
slopesStep = 0.2
#uniqueSites = np.unique(np.array(regions[allRegionsName].logisticFit.sites))

createAndSave(createGroupedHist(
    regions[allRegionsName].logisticFit.steepestSlopes, regions[allRegionsName].logisticFit.sites, slopesStart, 3, 0.5, "Steepest slope of logistic fit data per neuron", xLabel="Slopes"), 
    "fit" + os.sep + "slopes_grouped" + os.sep + "all")
createAndSave(createGroupedHist(
    regions[allRegionsName].logisticFitGood.steepestSlopes, regions[allRegionsName].logisticFitGood.sites, slopesStart, 3, 0.5, "Steepest slope of logistic fit per neuron", xLabel="Slopes"), 
    "fit" + os.sep + "slopes_grouped" + os.sep + "all_good")
createAndSave(createGroupedHist(
    regions[allRegionsName].logisticFit.params[0], regions[allRegionsName].logisticFit.sites, 0.0, 1.01, 0.1, "x0"), 
    "fit" + os.sep + "x0_grouped" + os.sep + "all")

for site in allSiteNames : 

    siteData = regions[site]
    print("Create figures for " + site)
    
    #spearmanPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanCor))
    spearmanPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.spearmanP))
    pearsonPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonCor))
    pearsonPPlot.add_trace(createCorrelationPlot(siteData.sitename, siteData.pearsonP))

    coactivationBeforeNormalization = siteData.coactivationNorm.y.copy()
    siteData.coactivationNorm.normalize()

    siteData.coactivationProb.y = np.asarray(siteData.coactivationProb.y)
    coactivationProbBeforeNormalization = siteData.coactivationProb.y.copy()
    siteData.coactivationProb.normalize()
    
    semProbPositiveIndices = np.where(siteData.coactivationProb.normalizer > 0)[0]
    coactivationProbTriu = np.transpose(siteData.coactivationProb.y)[semProbPositiveIndices,:][:,semProbPositiveIndices]
    semanticSimTriu = np.asarray(data.similarity_matrix)[semProbPositiveIndices,:][:,semProbPositiveIndices]

    triu_indices = np.triu_indices(len(semProbPositiveIndices), k = 1)
    coactivationProbTriu = coactivationProbTriu[triu_indices]
    semanticSimTriu = semanticSimTriu[triu_indices]
    semanticProbabilitySimilarity = SimilaritiesArray(values=coactivationProbTriu, similarities=semanticSimTriu)

    ticktextCoactivation = np.asarray([str(round(siteData.coactivationNorm.similarity[i], 2)) for i in range(len(coactivationBeforeNormalization))])
        #+ ": " + str(siteData.coactivationNorm.y[i] * 100) 
        #+ ("%.2f" % coactivationBeforeNormalization[i])
        #+ " (" + str(round(siteData.coactivationNorm.y[i] * 100, 5)) 
        #+ " = " + str(coactivationBeforeNormalization[i])
        #+ "/" + str(siteData.coactivationNorm.normalizer[i]) + ")" for i in range(len(coactivationBeforeNormalization))])

    fileDescription = paradigm + '_' + args.metric + '_' + site 

    totalNumResponseStrengthHist = max(1.0, len(siteData.responseStrengthHistResp) + len(siteData.responseStrengthHistNoResp)) #numRespUnitStimuli
    totalNumResponseStrengthHistSelf = max(1.0, len(siteData.responseStrengthHistRespSelf) + len(siteData.responseStrengthHistNoRespSelf)) #numRespUnitStimuli

    logFit = regions[site].logisticFit

    if len(logFit.yFit[0]) > 0 and not site == 'All':

        pTmp = logFit.params
        
        for i in range(len(pTmp[0])) :
            logisticFitFigSingle = go.Figure(
                go.Scatter(
                    x=logFit.xFit,
                    y=np.asarray(logFit.yFit).T[i],#logFunc(logFit.xFit, pTmp[0][i], pTmp[1][i], pTmp[2][i], pTmp[3][i]),
                    marker = {'color' : 'blue'}
                )
            )
            #logisticFitFigSingle.add_trace(
            #    go.Scatter(
            #        x=siteData.gaussFit.xFit,
            #        y=np.asarray(siteData.gaussFit.yFit).T[i],
            #        marker = {'color' : 'red'}
            #    )
            #)

            logisticFitFigSingle.add_trace(
                go.Scatter(
                    x=logFit.x[i],
                    y=logFit.y[i],
                    mode='markers',
                    marker_color='blue'
                )
            )
            if args.response_metric == "firing_rates" : 
                logisticFitFigSingle.update(layout_yaxis_range = [-0.05,1.05])

            rStr = str(round(logFit.rSquared[i],2))
            gof = str(round(logFit.gof[i],2))

            logisticFitFigSingle.update_layout(
                title_text="Logistic fit. R: " + rStr + ", gof: " + gof + ", X0: " + str(round(pTmp[0][i],2)) + ", K: " + str(round(pTmp[1][i],2)) + ", a: " + str(round(pTmp[2][i],2)) + ", c: " + str(round(pTmp[3][i],2)),
                xaxis_title='Semantic similarity',
                yaxis_title='Firing rate',
                showlegend=False 
            )
            saveImg(logisticFitFigSingle, "fit" + os.sep + "logistic_fit_single" + os.sep + "session" + os.sep + logFit.plotDetails[i] + "_" + site)
            #saveImg(logisticFitFigSingle, "fit" + os.sep + "logistic_fit_single" + os.sep + "sorted_gof" + os.sep + gof + "_" + logFit.plotDetails[i] + "_" + site)
            saveImg(logisticFitFigSingle, "fit" + os.sep + "logistic_fit_single" + os.sep + "sorted_r" + os.sep + rStr + "_" + logFit.plotDetails[i] + "_" + site)

        #for i in range(len(regions[site].logisticFitK)) : 
        #    logisticFitFig.add_trace(go.Scatter(
        #        x=xLogisticFit,
        #        y=fitLogisticFunc(xLogisticFit, regions[site].logisticFitX0[i], regions[site].logisticFitK[i], regions[site].logisticFitA[i], regions[site].logisticFitC[i]),
        #    ))

    if not site == "All" : 
        for i in range(len(regions[site].logisticFit.yNoFit)) : 
            if len(logFit.yNoFit[i]) == 0: 
                continue
            logisticFitFigSingle = go.Figure(go.Scatter(x=logFit.xNoFit[i], y=logFit.yNoFit[i],mode='markers'))
            saveImg(logisticFitFigSingle, "fit" + os.sep + "logistic_fit_single" + os.sep + "session" + os.sep + logFit.plotDetailsNoFit[i] + "_" + site + "_nofit")

    #regions[site].gaussFit.calculateSteepestSlopes()
    #regions[site].logisticFit.calculateSteepestSlopes()
    #goodFit = np.where(logFit.gof < args.thresh_gof)[0]

    #regions[site].logisticFitGood = logFit.getGood(gof, -args.thresh_gof)
    #regions[site].logisticFitGood = logFit.getGood(logFit.rSquared, 0.1)

    spearman_slope_df = pd.DataFrame(data={'spearman': siteData.logisticFit.spearman, 'slope': siteData.logisticFit.steepestSlopes})    
    allRegionSpearmanSlopePlots.append(createAndSave(px.scatter(spearman_slope_df, x='spearman', y='slope'), 
        "fit" + os.sep + "logistic_fit_spearman" + os.sep + fileDescription))

    allRegionGaussFitPlots.append(createAndSave(
        createFitPlot(regions[site].gaussFit, "Gauss"), 
        "fit" + os.sep + "gaussian_fit" + os.sep + fileDescription))
    allRegionGaussFitAlignedPlots.append(createAndSave(
        createFitPlotAligned(regions[site].gaussFit, "Gauss"), 
        "fit" + os.sep + "gaussian_fit_aligned" + os.sep + fileDescription))
    allRegionLogisticFitPlots.append(createAndSave(
        createFitPlot(regions[site].logisticFit, "Logistic"), 
        "fit" + os.sep + "logistic_fit" + os.sep + fileDescription))
    createAndSave(
        createFitPlot(regions[site].logisticFitGood, "Logistic"), 
        "fit" + os.sep + "logistic_fit_good" + os.sep + fileDescription)
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
    #allRegionCoactivationProbPlots.append(createAndSave(
    #    createStepBoxPlot(regions[site].logisticFit.rSquared, "rsquared logistic fit", "log_fit_rsquared", args.alpha_box), 
    #    "log_fit_rsquared" + os.sep + fileDescription))
    #allRegionCoactivationProbPlots.append(createAndSave(
    #    createStepBoxPlot(regions[site].logisticFit.gof, "gof logistic fit", "log_fit_gof", args.alpha_box), 
    #    "log_fit_gof" + os.sep + fileDescription))
    createAndSave(createHist(regions[site].logisticFit.rSquared, np.concatenate(([-np.inf], np.arange(-1.0,1.02,0.01), [np.inf])), 1.0, 'r squared', 'Num Units'),
        "fit" + os.sep + "logistic_fit_rsquared" + os.sep + fileDescription)
    createAndSave(createHist(regions[site].logisticFit.gof, np.concatenate(([-np.inf],np.arange(-1.5,1.1,0.01), [np.inf])), 1.0, 'gof', 'Num Units'),
        "fit" + os.sep + "logistic_fit_gof" + os.sep + fileDescription)
    #createAndSave(createHist(siteData.logisticFit.gof, [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], labelX='Alpha value', labelY='Counts'), 
    #    "fit" + os.sep + "gof" + os.sep + fileDescription)
    allRegionRDiffPlots.append(createAndSave(
        createBoxPlot([regions[site].rDiffLog, regions[site].rDiffGauss, regions[site].rDiffLogGauss], ["r(Log) - r(Step)", "r(Gauss) - r(Step)", "r(Log) - r(Gauss)"], "Diff of R squared of gaussian fit and R squared of step fit"), 
        "fit" + os.sep + "box_r_diff" + os.sep + fileDescription))
    #allRegionRDiffGaussPlots.append(createAndSave(
    #    createBoxPlot([regions[site].rDiffLog], ["r(Log) - r(Step)"], "Diff of R squared of logistic fit and R squared of step fit"), 
    #    "fit" + os.sep + "box_r_diff_log" + os.sep + fileDescription))

    createAndSave(createHistCombined(
        np.arange(0,1.02,0.01), siteData.responseStrengthHistNoResp, siteData.responseStrengthHistResp, 100.0 / float(totalNumResponseStrengthHist), 'Non-Responses', 'Responses'), 
        "response_strength_hist_combined" + os.sep + fileDescription)
    createAndSave(createHistCombined(
        np.arange(0,1.02,0.01), siteData.responseStrengthHistNoRespSelf, siteData.responseStrengthHistRespSelf, 100.0 / float(totalNumResponseStrengthHistSelf), 'Non-Responses', 'Responses'), 
        "response_strength_hist_combined_self" + os.sep + fileDescription)
    allRegionRespStrengthHistPlots.append(createAndSave(createHist(
        siteData.responseStrengthHistResp, np.arange(0,1.02,0.01), 100.0 / float(totalNumResponseStrengthHist), 'Normalized firing rate', 'Stimuli in %', "blue"),
        "response_strength_hist" + os.sep + fileDescription)) 
    allRegionRespStrengthHistPlotsNo.append(createAndSave(createHist(
        siteData.responseStrengthHistNoResp, np.arange(0,1.02,0.01), 100.0 / float(totalNumResponseStrengthHist), 'Normalized firing rate', 'Stimuli in %', "red"),
        "response_strength_hist_no" + os.sep + fileDescription))
    allRegionRespStrengthHistSelfPlots.append(createAndSave(createHist(
        siteData.responseStrengthHistRespSelf, np.arange(0,1.02,0.01), 100.0 / float(totalNumResponseStrengthHistSelf), 'Normalized firing rate', 'Stimuli in %', "blue"),
        "response_strength_hist_self" + os.sep + fileDescription)) 
    allRegionRespStrengthHistSelfPlotsNo.append(createAndSave(createHist(
        siteData.responseStrengthHistNoRespSelf, np.arange(0,1.02,0.01), 100.0 / float(totalNumResponseStrengthHistSelf), 'Normalized firing rate', 'Stimuli in %', "red"),
        "response_strength_hist_no_self" + os.sep + fileDescription))
    allRegionNumResponsesPlots.append(createAndSave(
        createHist(siteData.numResponsesHist, range(args.max_responses_unit + 1), 1, 'Number of responses', 'Number of units', "blue"),
        "num_responses" + os.sep + fileDescription))

    #xSpan = np.arange(0.0, 0.99, args.span_step)
    counts, bins = np.histogram(siteData.maxDistResp, bins = np.arange(0.0, 1.001, args.step_span))
    #fig = px.bar(x=bins[:-1], y=counts / siteData.copresentationSpan, labels={'x':'Span of responses', 'y':'Number of units'})
    allRegionMaxDistPlots.append(createAndSave(
        px.bar(x=bins[:-1], y=counts/siteData.copresentationSpan, labels={'x':'Span of responses', 'y':'Number of units'}), 
        "span_responses" + os.sep + fileDescription))
    allRegionCoactivationNormalizedPlots.append(createAndSave(
        createPlot(siteData.coactivationNorm.similarity[:-1], siteData.coactivationNorm.y * 100, "Normalized coactivation probability in %", "coactivation normalized", True, ticktextCoactivation), 
        "coactivation_normalized" + os.sep + fileDescription))
    allRegionCoactivationProbPlots.append(createAndSave(
        createStepBoxPlot(semanticProbabilitySimilarity, "Coactivation probability in %", "coactivation_probability", args.alpha_box), 
        "coactivation_probability" + os.sep + fileDescription))
    
    #regions[allRegionsName].combinationsData["similarities_concepts"]
    allRegionZScoresPlots.append(createAndSave(
        createStepBoxPlot(siteData.zScoresNorm, "Mean zscores", "zscores", args.alpha_box), 
        "zscores" + os.sep + fileDescription))
    
    #regions[allRegionsName].combinationsData = pd.DataFrame(regions[allRegionsName].combinationsData)
    #siteCombinationsData = regions[allRegionsName].combinationsData
    #if site != allRegionsName :  
        #siteCombinationsDataIndices = np.where(np.asarray(regions[allRegionsName].combinationsData["site"]) == site)[0]
        #siteCombinationsData = siteCombinationsData.loc[siteCombinationsData['site'] == site] #regions[allRegionsName].combinationsData[siteCombinationsDataIndices]
    
    #allRegionZScoresPlots.append(createAndSave(
    #    createStepBoxPlot(SimilaritiesArray(values=siteCombinationsData["zscores"], similarities=siteCombinationsData["similarities"]), "Mean zscores", "zscores", args.alpha_box, addLog=True, addPartialGaussian=False), 
    #    "zscores_combined" + os.sep + fileDescription))
    
    createAndSave(createStepBoxPlot(siteData.similaritiesResiduals, "Mean zscores for residuals", "zscores", args.alpha_box), "similarities_residuals" + os.sep + fileDescription)
    createAndSave(createStepBoxPlot(siteData.similaritiesConcepts, "Mean zscores for concept distances", "zscores", args.alpha_box), "similarities_concepts" + os.sep + fileDescription)
    createAndSave(createStepBoxPlot(siteData.similaritiesConceptsDiscrete, "Mean zscores for concept distances discrete", "zscores", args.alpha_box, stepBox=1.0, max=27.0), 
                  "similarities_concepts_discrete" + os.sep + fileDescription)

    #createAndSave(createBoxPlot(siteData.zScoresConceptsBinary.values, siteData.zScoresConceptsBinary.similarities, "Mean zscores for discrete concept distances"), "zscores_concepts_binary" + os.sep + fileDescription)
    
    #allRegionZScoresPlots.append(createAndSave(
    #    createStepBoxPlot(siteData.zScoresConcepts, "Mean zscores", "zscores", args.alpha_box), 
    #    "zscores_concepts" + os.sep + fileDescription))
    
    #zScoresConceptsBinary
    #allRegionZScoresPlots.append(createAndSave(
    #    createStepBoxPlot(SimilaritiesArray(values=siteCombinationsData["zscores_concepts"], similarities=siteCombinationsData["similarities_concepts"]), "Mean zscores", "zscores", args.alpha_box, addLog=True, addPartialGaussian=False), 
    #    "zscores_concepts_combined" + os.sep + fileDescription))
    
    allRegionPValuesPlots.append(createAndSave(
        createStepBoxPlot(siteData.pvalues, "Mean pvalues", "pvalues", args.alpha_box), 
        "pvalues" + os.sep + fileDescription))
    allRegionFiringRatesPlots.append(createAndSave(
        createStepBoxPlot(siteData.firingRatesNorm, "Normalized firing rates", "Normalized firing rates", args.alpha_box), 
        "firing_rates" + os.sep + fileDescription))
    allRegionSpearmanPlots.append(createAndSave(
        createStepBoxPlot(siteData.spearmanCorSteps, "Spearman correlation dependent on semantic similarity", "spearmanCorSteps", args.alpha_box), 
        "spearmanCorSteps" + os.sep + fileDescription)) 
    allRegionSpearmanMSplitPlots.append(createAndSave(
        siteData.spearmanCorMSplit.createMedianSplitPlot("Spearman correlation median split", "spearmanCor"), 
        "spearmanCorMSplit" + os.sep + fileDescription)) 
    allRegionPearsonPlots.append(createAndSave(
        createStepBoxPlot(siteData.pearsonCorSteps, "Pearson correlation dependent on semantic similarity", "pearsonCorSteps", args.alpha_box), 
        "pearsonCorSteps" + os.sep + fileDescription)) 
    allRegionSpearmanSplitPlots.append(createAndSave(
        createStepBoxPlot(siteData.spearmanCorSplit, "Spearman correlation dependent on semantic similarity - stepsize: " + str(args.step_correlation_split), "spearmanSplit", args.alpha_box, addLog=False, addPartialGaussian=False), 
        "spearmanSplit" + os.sep + fileDescription)) 
    #allRegionSlopePlots.append(createAndSave(
    #    createBoxPlot([regions[site].logisticFit.steepestSlopes], [""], "Steepest slope of fitted data per neuron"), 
    #    "fit" + os.sep + "slopes" + os.sep + fileDescription))
    #createAndSave(
    #    createBoxPlot([regions[site].logisticFitGood.steepestSlopes], [""], "Steepest slope of fitted data per neuron"), 
    #    "fit" + os.sep + "slopes_good" + os.sep + fileDescription)
    allRegionSlopePlots.append(createAndSave(
        createHist([regions[site].logisticFit.steepestSlopes], np.arange(slopesStart,slopesStop,slopesStep), factorY=1.0, labelX="Steepest slopes", labelY="Num"),
        "fit" + os.sep + "slopes" + os.sep + fileDescription))
    createAndSave(
        createHist([regions[site].logisticFitGood.steepestSlopes], np.arange(slopesStart,slopesStop,slopesStep), factorY=1.0, labelX="Steepest slopes", labelY="Num"),
        "fit" + os.sep + "slopes_good" + os.sep + fileDescription)
    createAndSave(
        createHist([regions[site].logisticFit.params[0]], np.append(np.append(np.arange(0.0,1.0,0.1), np.inf), np.inf), factorY=1.0, labelX="x0", labelY="Num"),
        "fit" + os.sep + "x0_steps" + os.sep + fileDescription)
    createAndSave(
        createHist([regions[site].logisticFitGood.params[0]], np.append(np.append(np.arange(0.0,1.0,0.1), np.inf), np.inf), factorY=1.0, labelX="x0", labelY="Num"),
        "fit" + os.sep + "x0_steps_good" + os.sep + fileDescription)
    #allRegionSlopePlots.append(createAndSave(
    #    createBoxPlot([regions[site].logisticFit.steepestSlopes, regions[site].gaussFit.steepestSlopes], ["Log", "Gauss"], "Steepest slope of fitted data"), 
    #    "fit" + os.sep + "slopes" + os.sep + fileDescription))


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
pValuesDiv, pValuesFigId, pValuesTableId = createRegionsDiv("Mean pValues dependent on semantic similarity to best response", allSiteNames)
firingRatesDiv, firingRatesFigId, firingRatesTableId = createRegionsDiv("Normalized firing rates dependent on semantic similarity to best response", allSiteNames)
spearmanCorMSplitDiv, spearmanCorMSplitFigId, spearmanCorMSplitTableId = createRegionsDiv("Spearman correlation median split", allSiteNames)
spearmanCorStepsDiv, spearmanCorStepsFigId, spearmanCorStepsTableId = createRegionsDiv("Spearman correlation dependent on semantic similarity to best response", allSiteNames)
spearmanCorSplitDiv, spearmanCorSplitFigId, spearmanCorSplitTableId = createRegionsDiv("Spearman correlation dependent on semantic similarity to best response; split data", allSiteNames)
pearsonCorStepsDiv, pearsonCorStepsFigId, pearsonCorStepsTableId = createRegionsDiv("Pearson correlation dependent on semantic similarity to best response", allSiteNames)
spearmanSlopesDiv, spearmanSlopesFigId, spearmanSlopesTableId = createRegionsDiv("Spearman slopes scatter plot", allSiteNames)
numRespDiv, numRespFigId, numRespTableId = createRegionsDiv("Number of units with respective response counts", allSiteNames)
maxDistDiv, maxDistFigId, maxDistTableId = createRegionsDiv("Max span of responsive field of a neuron", allSiteNames)
responseStrengthHistDiv, responseStrengthHistFigId, responseStrengthHistTableId = createRegionsDiv("Response strength histogram for responsive stimuli", allSiteNames)
responseStrengthHistDivNo, responseStrengthHistFigIdNo, responseStrengthHistTableIdNo = createRegionsDiv("Response strength histogram for non responsive stimuli", allSiteNames)
responseStrengthHistSelfDiv, responseStrengthHistSelfFigId, responseStrengthHistSelfTableId = createRegionsDiv("Response strength histogram for responsive stimuli - including response used for normalization", allSiteNames)
responseStrengthHistSelfDivNo, responseStrengthHistSelfFigIdNo, responseStrengthHistSelfTableIdNo = createRegionsDiv("Response strength histogram for non responsive stimuli - including response used for normalization", allSiteNames)
logisticFitDiv, logisticFitFigId, logisticFitTableId = createRegionsDiv("Logistic fit for all responsive units", allSiteNames)
logisticFitAlignedDiv, logisticFitAlignedFigId, logisticFitAlignedTableId = createRegionsDiv("Logistic fit for all responsive units - aligned", allSiteNames)
gaussianFitDiv, gaussianFitFigId, gaussianFitTableId = createRegionsDiv("Gaussian fit for all responsive units", allSiteNames)
gaussianFitAlignedDiv, gaussianFitAlignedFigId, gaussianFitAlignedTableId = createRegionsDiv("Gaussian fit for all responsive units - aligned", allSiteNames)
logisticFitX0Div, logisticFitX0FigId, logisticFitX0TableId = createRegionsDiv("Logistic fit for all responsive units: X0", allSiteNames)
logisticFitKDiv, logisticFitKFigId, logisticFitKTableId = createRegionsDiv("Logistic fit for all responsive units: K", allSiteNames)
logisticFitRSquaredDiv, logisticFitRSquaredFigId, logisticFitRSquaredTableId = createRegionsDiv("Logistic fit for all responsive units: R squared", allSiteNames)
logisticFitRDiffDiv, logisticFitRDiffFigId, logisticFitRDiffTableId = createRegionsDiv("Diff of R squared between logistic and step fit", allSiteNames)
#gaussianFitRDiffDiv, gaussianFitRDiffFigId, gaussianFitRDiffTableId = createRegionsDiv("Diff of R squared between gaussian and step fit", allSiteNames)
slopeDiv, slopeFigId, slopeTableId = createRegionsDiv("Steepest slope of log fit of firing rate to similarity", allSiteNames)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Semantic Tuning'),
    html.H3('Spearman correlation'),
    dcc.Graph(id='spearman-plot', figure=spearmanPlot),
    dcc.Graph(id='spearman-p-plot', figure=spearmanPPlot),
    spearmanCorStepsDiv,
    spearmanCorSplitDiv,
    spearmanCorMSplitDiv,
    spearmanSlopesDiv,
    html.H3('Pearson correlation'),
    dcc.Graph(id='pearson-plot', figure=pearsonPlot),
    dcc.Graph(id='pearson-p-plot', figure=pearsonPPlot),
    pearsonCorStepsDiv,
    #coactivationDiv, 
    #copresentationDiv, 
    coactivationNormalizedDiv, 
    firingRatesDiv, 
    zscoresDiv,
    pValuesDiv,
    responseStrengthHistDiv, 
    responseStrengthHistDivNo, 
    responseStrengthHistSelfDiv, 
    responseStrengthHistSelfDivNo, 
    slopeDiv,
    logisticFitDiv,
    logisticFitRSquaredDiv,
    logisticFitRDiffDiv,
    logisticFitKDiv,
    logisticFitX0Div,
    #gaussianFitRDiffDiv,
    maxDistDiv, 
    numRespDiv, 
    logisticFitAlignedDiv,
    gaussianFitDiv,
    gaussianFitAlignedDiv,
])

#print("pvals best responses: " + str(alphaBestResponse))
print("pvals median: " + str(statistics.median(alphaBestResponse)))
print("pvals mean: " + str(statistics.mean(alphaBestResponse)))
print("pvals < 0.01: " + str(len(np.where(np.asarray(alphaBestResponse) < 0.01)[0])))
print("pvals > 0.01: " + str(len(np.where(np.asarray(alphaBestResponse) > 0.01)[0])))
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
    Output(component_id=spearmanCorSplitFigId, component_property='figure'), 
    Input(spearmanCorSplitTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSpearmanSplitPlots, active_cell)

@app.callback(
    Output(component_id=spearmanCorMSplitFigId, component_property='figure'), 
    Input(spearmanCorMSplitTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSpearmanMSplitPlots, active_cell)

@app.callback(
    Output(component_id=spearmanSlopesFigId, component_property='figure'), 
    Input(spearmanSlopesTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSpearmanSlopePlots, active_cell)

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
    Output(component_id=responseStrengthHistSelfFigId, component_property='figure'), 
    Input(responseStrengthHistSelfTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionRespStrengthHistSelfPlots, active_cell)

@app.callback(
    Output(component_id=responseStrengthHistSelfFigIdNo, component_property='figure'), 
    Input(responseStrengthHistSelfTableIdNo, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionRespStrengthHistSelfPlotsNo, active_cell)

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
    Output(component_id=slopeFigId, component_property='figure'), 
    Input(slopeTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionSlopePlots, active_cell)

@app.callback(
    Output(component_id=zscoresFigId, component_property='figure'), 
    Input(zscoresTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionZScoresPlots, active_cell)

@app.callback(
    Output(component_id=pValuesFigId, component_property='figure'), 
    Input(pValuesTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionPValuesPlots, active_cell)

@app.callback(
    Output(component_id=firingRatesFigId, component_property='figure'), 
    Input(firingRatesTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionFiringRatesPlots, active_cell)

@app.callback(
    Output(component_id=numRespFigId, component_property='figure'), 
    Input(numRespTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionNumResponsesPlots, active_cell)

@app.callback(
    Output(component_id=maxDistFigId, component_property='figure'), 
    Input(maxDistTableId, 'active_cell')
)
def update_output_div(active_cell):
    return getActivePlot(allRegionMaxDistPlots, active_cell)

    
if __name__ == '__main__':
    app.run_server(debug=False) # why ?