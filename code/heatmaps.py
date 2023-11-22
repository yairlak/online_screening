#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/13 13:04:41
@Author  :   Katharina Karkowski 
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import operator

import time
from typing import List
from dataclasses import field
from dataclasses import dataclass
import scipy
import scipy.io
import scipy.interpolate 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from skimage.measure import label
import matplotlib.pyplot as plt
from PIL import Image

import dash
#import dash_table
#import dash_core_components as dcc
#import dash_html_components as html
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# utilility modules
from plot_helper import plotRaster
from plot_helper import RasterInput
from data_manip import DataHandler
from data_manip import get_mean_firing_rate_normalized

parser = argparse.ArgumentParser()

# SESSION/UNIT
parser.add_argument('--session', default=None, type=str,
                    help="If None, all sessions in folder are processed. \
                        Otherwise, format should be '{subject}_{session}, \
                            e.g., '90_1'.")

# ANALYSIS
parser.add_argument('--data_type', default="zstatistics", type=str, # zscores or firing_rates or zsctatistics
                    help="Determines underlying datatype for heatmaps. \
                        Currently, zscores or firing_rate are implemented.")
parser.add_argument('--min_t', default=100, type=int,
                    help="Relevant for calculating mean firing_rate. \
                        Min time offset of spike after stimulus onset.")
parser.add_argument('--max_t', default=800, type=int,
                    help="Relevant for calculating mean firing_rate. \
                        Max time offset of spike after stimulus onset.")

# FLAGS
parser.add_argument('--show_all', default=True,
                    help='If True, all heatmaps are shown on dashboard and saved')
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha for stats')
parser.add_argument('--alpha_region', type=float, default=0.01,
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
                    default='../data/aos_after_manual_clustering/') # also work with nos? aos_after_manual_clustering, aos_selected_sessions, aos_one_session
parser.add_argument('--path2images', 
                    default='../figures/heatmaps/') 

args=parser.parse_args()

@dataclass
class Tuner:
    subjectsession: str
    channel: int
    cluster: int
    unitType: str
    name: str
    paradigm: str
    stimuli: List[int] 
    stimuliX: List[float]
    stimuliY: List[float]
    stimuliNames: List[str]
    zscores : List[float]
    firingRates: List[float]
    pvalues : List[float] = field(default_factory=lambda: [])
    responses: List[RasterInput] = field(default_factory=lambda: [])
    allRasters: List[RasterInput] = field(default_factory=lambda: [])
    responseIndices : List[int] = field(default_factory=lambda: [])
    site : str = ""
    zstatistics : List[float] = field(default_factory=lambda: []) 




def save_img(filename) : 

    file = args.path2images + os.sep + args.data_type + os.sep + filename

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file + ".png", bbox_inches="tight")
        plt.clf()

    plt.close()


def createRasterPlot(rasterToPlot, figureWidth, figureHeight) : 
    
    ## Raster plots
    numCols = 5
    numRowsRaster = int(len(rasterToPlot) / numCols) + 1

    specs=[]
    for rowNum in range(numRowsRaster) : 
        specsCols = []
        for colNum in range(numCols) : 
            specsCols.append({})
        specs.append(specsCols)

    responsesSorted = sorted(rasterToPlot, key=lambda x: x.pval)

    subplot_titles=[]
    for response in responsesSorted : 
        pval = response.pval
        if pval < 0.0001 : 
            pval = np.format_float_scientific(pval, precision=3)
        else : 
            pval = round(pval, 7)
        subplot_titles.append(response.stimulusName + ', pval: ' + str(pval))

    rasterGrid = make_subplots(rows=numRowsRaster, cols=numCols, specs=specs, subplot_titles=subplot_titles)

    for title in rasterGrid['layout']['annotations']:
        title['font'] = dict(size=10)

    rowNum = 0
    colNum = 0
    for response in responsesSorted : 
        rasterFigure = plotRaster(response, linewidth=1.5)
        for line in rasterFigure.data : 
            rasterGrid.add_trace(line, row=rowNum+1, col=colNum+1)

        colNum += 1
        if colNum == numCols : 
            rowNum += 1
            colNum = 0

    rasterGrid.update_layout(go.Layout(
        showlegend=False, 
        autosize=False,
        height = int(figureHeight / 4) * numRowsRaster + 100,
        width = int(figureWidth), 
    ))

    for ax in rasterGrid['layout']:
        if ax[:5]=='xaxis':
            rasterGrid['layout'][ax]['range']=[-500,1500]
            rasterGrid['layout'][ax]['tickmode']='array'
            rasterGrid['layout'][ax]['tickvals']=[0, 1000]
            rasterGrid['layout'][ax]['tickfont']=dict(size=8)
        if ax[:5]=='yaxis':
            rasterGrid['layout'][ax]['visible']=False
            rasterGrid['layout'][ax]['showticklabels']=False
            #rasterGrid['layout'][ax]['range']=[0, len(response.spikes)]  # can only be set for all together (?)
            
    return rasterGrid

def rescaleX(x) :
    xPadding = np.asarray([xOut / args.padding_factor for xOut in x])
    return args.interpolation_factor * (xPadding - xMinThings) / (xMaxThings - xMinThings) - 0.5

def rescaleY(y) :
    yPadding = np.asarray([yOut / args.padding_factor for yOut in y])
    return args.interpolation_factor * (yPadding - yMinThings) / (yMaxThings - yMinThings) - 0.5

def getInterpolatedMap(x, y, z, pvalues=False) : 
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
    if pvalues : 
        zWithBorder = np.log(zWithBorder)

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


    if pvalues : 
        rbf = scipy.interpolate.Rbf(xWithBorder, yWithBorder, zWithBorder, function='linear')
        zi = rbf(xi, yi)
        zi[zi > np.log(args.alpha_region)] = 0.0 # allow 10*alpha to connect

        #zi = np.log(zi)
        #zi[zi < 0.0] = 0.0
        #rbf.di[rbf.di > args.alpha] = 0.0
    
    else : 
        # Interpolate missing data
        rbf = scipy.interpolate.Rbf(xWithBorder, yWithBorder, zWithBorder, function='linear')
        #rbf = scipy.interpolate.Rbf(xWithBorder, yWithBorder, zWithBorder)
        zi = rbf(xi, yi)

    return px.imshow(zi,aspect=0.8,color_continuous_scale='RdBu_r',origin='lower'), xi, yi #, zmax=1

#def createHeatMapZScores(tuner, figureHeight, savePath=outputPath, addName=False) : 
#    createHeatMap(tuner, tuner.zscores, figureHeight, savePath, addName)

#def createHeatMapFiringRate(tuner, figureHeight, savePath=outputPath, addName=False) : 
#    createHeatMap(tuner, tuner.firingRate, figureHeight, savePath, addName)


def createHeatMap(tuner, figureHeight, savePath="", addName=False) : 

    outputPath = args.path2images + os.sep + args.data_type + os.sep + savePath
    
    if args.data_type == "firing_rates" :
        targetValue = tuner.firingRates
    elif args.data_type == "zstatistics": 
        targetValue = tuner.zstatistics
    else : 
        targetValue = tuner.zscores

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    ## Heatmap
    heatmap = getInterpolatedMap(np.array(tuner.stimuliX), np.array(tuner.stimuliY), np.array(targetValue))[0]
    heatmapOnlyResponses, xi, yi = getInterpolatedMap(np.array(tuner.stimuliX), np.array(tuner.stimuliY), np.array(tuner.pvalues), pvalues=True)

    regionsHeatmap = heatmapOnlyResponses.data[0].z.copy()
    regionsHeatmap[regionsHeatmap != 0.0] = 1.0
    regionsLabels = label(regionsHeatmap, connectivity=2)

    
    #remove areas without response (including consider)
    regionsLabels[regionsLabels > 0] += 1000
    newLabelCount = 1
    for responseIndex in tuner.responseIndices :
        xResponse = tuner.stimuliX[responseIndex]
        yResponse = tuner.stimuliY[responseIndex]
        #xHeatmap = heatmapOnlyResponses.data[0].x.copy()
        xResponseHeatmap = (np.abs(xi[0] - xResponse)).argmin()
        yResponseHeatmap = (np.abs(yi[:,0] - yResponse)).argmin()
        responseLabel = regionsLabels[yResponseHeatmap][xResponseHeatmap]
        heatmapOnlyResponses.data[0].z[yResponseHeatmap][xResponseHeatmap] = 1
        
        regionFound = False
        if responseLabel == 0 : 
            for xstep in [-1, 1] : 
                for ystep in [-1, 1] : 
                    responseLabel = regionsLabels[yResponseHeatmap + xstep][xResponseHeatmap + ystep]
                    if responseLabel > 0 : 
                        regionFound = True
                        print("found neighboring region. xstep: " + str(xstep) + ", ystep: " + str(ystep))
                        break
                if regionFound : 
                    break

        if responseLabel > 1000 : 
            labelIndices = np.where(regionsLabels == responseLabel)
            regionsLabels[labelIndices] = newLabelCount
            newLabelCount += 1
        else: 
            print("label is " + str(responseLabel))
        
        #print("test")
        
    noResponseIndices = np.where(regionsLabels > 1000)
    regionsLabels[noResponseIndices] = 0
    heatmapOnlyResponses.data[0].z[noResponseIndices] = 0
    #heatmapOnlyResponses.data[0].z = regionsLabels

    numRegions = np.amax(regionsLabels)

    #heatmapPvalues.data[0].z[heatmapPvalues.data[0].z > args.alpha] = 0.0

    ## Create reduced heatmap
    #targetValuesOnlyResponses = targetValue.copy()
    #noResponseIndices = list(set(range(len(targetValuesOnlyResponses))) - set(tuner.responseIndices))
    #targetValuesOnlyResponses[noResponseIndices] = 0.0
    #heatmapOnlyResponses = heatmapPvalues #getInterpolatedMap(np.array(tuner.stimuliX), np.array(tuner.stimuliY), np.array(targetValuesOnlyResponses))
    
    targetValues = np.copy(targetValue)
    targetValues -= min(targetValues)
    targetValues /= max(targetValues)

    ## Text / labels 
    for stimulusNum in range(len(tuner.stimuliNames)) :
        opacityStim = targetValues[stimulusNum]
        heatmap.add_trace(
            go.Scatter(
                mode='text',
                x=rescaleX([tuner.stimuliX[stimulusNum]]), y=rescaleY([tuner.stimuliY[stimulusNum]]),
                #text=[tuner.stimuliNames[stimulusNum]],
                hovertext=[tuner.stimuliNames[stimulusNum] + ", z: " + str(round(targetValues[stimulusNum], 2))],
                opacity=opacityStim,
                textfont=dict(
                    size=12,
                    color="black"
                ),
                name='zscore'
            )
        )

    
    for stimulusNum in tuner.responseIndices :
        text = tuner.stimuliNames[stimulusNum]
        heatmapOnlyResponses.add_trace(
            go.Scatter(
                mode='text',
                text=text,
                x=rescaleX([tuner.stimuliX[stimulusNum]]), y=rescaleY([tuner.stimuliY[stimulusNum]]),
                #hovertext=[tuner.stimuliNames[stimulusNum] + ", z: " + str(round(targetValues[stimulusNum], 2))],
                textfont=dict(size=12,color="black"),
                #name='zscore'
            )
        )

    figureWidth = figureHeight*3/2
    titleText = "session: " + tuner.subjectsession + ", channel: " + str(tuner.channel) + ", cluster: " + str(tuner.cluster) + ", " + tuner.unitType + ", " + tuner.site
    graphLayout = go.Layout(
        title_text=titleText, 
        xaxis=dict(ticks='', showticklabels=False),
        yaxis=dict(ticks='', showticklabels=False),
        showlegend=False, 
        autosize=False,
        height=figureHeight,
        width=figureWidth
    )

    heatmap.update_layout(graphLayout)
    heatmapOnlyResponses.update_layout(graphLayout)
    heatmapOnlyResponses.update_layout(go.Layout(title_text = titleText + ", numRegions: " + str(numRegions)))


    """ rasterToPlot = tuner.responses # tuner.responses

    ## Raster plots
    numCols = 5
    numRowsRaster = int(len(rasterToPlot) / numCols) + 1

    specs=[]
    for rowNum in range(numRowsRaster) : 
        specsCols = []
        for colNum in range(numCols) : 
            specsCols.append({})
        specs.append(specsCols)

    responsesSorted = sorted(rasterToPlot, key=lambda x: x.pval)

    subplot_titles=[]
    for response in responsesSorted : 
        pval = response.pval
        if pval < 0.0001 : 
            pval = np.format_float_scientific(pval, precision=3)
        else : 
            pval = round(pval, 7)
        subplot_titles.append(response.stimulusName + ', pval: ' + str(pval))

    rasterGrid = make_subplots(rows=numRowsRaster, cols=numCols, specs=specs, subplot_titles=subplot_titles)

    for title in rasterGrid['layout']['annotations']:
        title['font'] = dict(size=10)

    rowNum = 0
    colNum = 0
    for response in responsesSorted : 
        rasterFigure = plotRaster(response, linewidth=1.5)
        for line in rasterFigure.data : 
            rasterGrid.add_trace(line, row=rowNum+1, col=colNum+1)

        colNum += 1
        if colNum == numCols : 
            rowNum += 1
            colNum = 0

    rasterGrid.update_layout(go.Layout(
        showlegend=False, 
        autosize=False,
        height = int(figureHeight / 4) * numRowsRaster + 100,
        width = int(figureWidth), 
    ))

    for ax in rasterGrid['layout']:
        if ax[:5]=='xaxis':
            rasterGrid['layout'][ax]['range']=[-500,1500]
            rasterGrid['layout'][ax]['tickmode']='array'
            rasterGrid['layout'][ax]['tickvals']=[0, 1000]
            rasterGrid['layout'][ax]['tickfont']=dict(size=8)
        if ax[:5]=='yaxis':
            rasterGrid['layout'][ax]['visible']=False
            rasterGrid['layout'][ax]['showticklabels']=False
            #rasterGrid['layout'][ax]['range']=[0, len(response.spikes)]  # can only be set for all together (?)
             """
    
    allRasters = createRasterPlot(tuner.allRasters, figureWidth, figureHeight)
    rasterGrid = createRasterPlot(tuner.responses, figureWidth, figureHeight)

    if not args.dont_plot : 
        filename = tuner.subjectsession + "_ch" + str(tuner.channel) + "_cl" + str(tuner.cluster) 
        if addName : 
            filename = outputPath + os.sep + filename + "_" + tuner.name
        else : 
            filename = outputPath + os.sep + str(numRegions) + "regions" + os.sep + filename
        heatmapFilename = filename + "_heatmap.png"
        heatmapOnlyResponsesFilename = filename + "_heatmap_responses.png"
        rasterFilename = filename + "_rasterplots.png"
        rastersAllFilename = filename + "_rasterplots_all.png"
        completeFilename = filename 
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        heatmapOnlyResponses.write_image(heatmapOnlyResponsesFilename)
        heatmap.write_image(heatmapFilename)
        heatmap.write_image(filename + "_heatmap.svg")
        rasterGrid.write_image(rasterFilename)
        rasterGrid.write_image(filename + "_rasterplots.svg")
        allRasters.write_image(rastersAllFilename)

        pltHeatmap = Image.open(heatmapFilename)
        pltRaster = Image.open(rasterFilename)

        totalWidth = max([pltHeatmap.size[0], pltRaster.size[0]])
        totalHeight = pltHeatmap.size[1] + pltRaster.size[1]
        completeImage = Image.new('RGB', (totalWidth, totalHeight))

        completeImage.paste(pltHeatmap, (0,0))
        completeImage.paste(pltRaster, (0,pltHeatmap.size[1]))
        completeImage.save(completeFilename + ".png")
        #completeImage.save(completeFilename + ".svg")

        os.remove(heatmapFilename)
        os.remove(rasterFilename)


    tunerDivGrid = html.Div([
        html.Div(children=[dcc.Graph(id='heatmap-' + tuner.name, figure=heatmap)], style={'margin-bottom': 0}),
        html.Div(children=[dcc.Graph(id='rasterGrid-' + tuner.name, figure=rasterGrid)], style={'margin-top': 0})
    ])

    return tunerDivGrid, numRegions

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

print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")


tuners = [ # clusters might not fit (manual clustering took place)
    #Tuner("088e03aos1", 17, 1, "Pacifier", "aos", [], [], [], [], []),
    Tuner("103_1", 70, 1, [], "Car parts", "aos", [], [], [], [], [], []),
    Tuner("88_1", 77, 1, [], "Engine", "aos", [], [], [], [], [], []),
    Tuner("88_1", 75, 2, [], "Lizard", "aos", [], [], [], [], [], []), 
    Tuner("88_1", 87, 2, [], "Zucchini", "aos", [], [], [], [], [], []), 
    Tuner("88_3", 73, 2, [], "Terrarium", "aos",  [], [], [], [], [], []),
    Tuner("88_3", 92, 1, [], "Photograph", "aos",  [], [], [], [], [], []),
    Tuner("89_1", 84, 1, [], "Ambulance", "aos",  [], [], [], [], [], []),
    #Tuner("89_2", 77, 2, [], "Machine Gun", "aos",  [], [], [], [], []),
    Tuner("89_3", 62, 1, [], "Machine Gun", "aos",  [], [], [], [], [], []),
    Tuner("90_1", 49, 1, [], "Waffle1", "aos",  [], [], [], [], [], []),
    Tuner("90_1", 49, 2, [], "Waffle2", "aos",  [], [], [], [], [], []),
    Tuner("90_1", 49, 3, [], "Waffle3", "aos",  [], [], [], [], [], []),
    Tuner("90_1", 60, 2, [], "Ferry1", "aos",  [], [], [], [], [], []),
    Tuner("90_1", 60, 3, [], "Ferry2", "aos",  [], [], [], [], [], []),
    Tuner("90_2", 65, 3, [], "Hamburger1", "aos",  [], [], [], [], [], []),
    Tuner("90_2", 65, 4, [], "Hamburger2", "aos",  [], [], [], [], [], []),
    Tuner("90_2", 68, 3, [], "Pancake", "aos",  [], [], [], [], [], []),
    Tuner("90_3", 49, 4, [], "Lipstick", "aos",  [], [], [], [], [], []),
    #Tuner("90_3", 52, 1, [], "Onion1", "aos",  [], [], [], [], []),
    #Tuner("90_3", 52, 2, [], "Onion2", "aos",  [], [], [], [], []),
    #Tuner("90_3", 52, 3, [], "Onion2", "aos",  [], [], [], [], []),
    Tuner("90_4", 52, 1, [], "Potato", "aos",  [], [], [], [], [], []),
    Tuner("90_5", 52, 2, [], "Coin", "aos",  [], [], [], [], [], []),
    Tuner("90_5", 56, 1, [], "Hamburger1", "aos",  [], [], [], [], [], []),
    Tuner("90_5", 56, 3, [], "Hamburger2", "aos",  [], [], [], [], [], []),
    Tuner("90_5", 67, 1, [], "Donkey - Petfood - Carrot", "aos",  [], [], [], [], [], []),
    Tuner("101_4", 28, 1, [], "Rocket", "aos",  [], [], [], [], [], []),
    Tuner("102_1", 51, 1, [], "Food", "aos",  [], [], [], [], [], []),
    Tuner("103_1", 57, 1, [], "Car1", "aos",  [], [], [], [], [], []),
    Tuner("103_1", 58, 2, [], "Car2", "aos",  [], [], [], [], [], []),
    Tuner("100_1", 107, 1, [], "Animals and clothing", "aos",  [], [], [], [], [], []),
    Tuner("103_3", 64, 1, [], "Sea animals and dolly", "aos",  [], [], [], [], [], []),
]

figureHeight = 600
figureHeightBig = 750

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
outputPath =  args.path2images + os.sep + args.data_type

allUnits = []

for session in sessions:
    if not hasattr(args, 'unit'):
        units = list(set(data.neural_data[session]['units'].keys()))
    else:
        units = [args.unit]

    subjectNum = int(session.split("_")[0])
    sessionNum = int(session.split("_")[1])
    sessionParadigm = session.split("_")[2]

    stimuliIndices = data.neural_data[session]['objectindices_session']
    stimuliNames = data.neural_data[session]['stimlookup']
    stimuliNamesTrials = data.neural_data[session]['objectnames']

    stimuliNums = []
    stimuliX = []
    stimuliY = []

    for stim in stimuliNames : 
        stimInThings = np.where(data.df_metadata.uniqueID == stim)[0][0]
        stimuliNums.append(stimInThings)
        stimuliX.append(xThings[stimInThings])
        stimuliY.append(yThings[stimInThings])
    
    for unit in units:
        site = data.neural_data[session]['units'][unit]['site']
        pvals = data.neural_data[session]['units'][unit]['p_vals']
        zscores = data.neural_data[session]['units'][unit]['zscores']
        zstatistics = data.neural_data[session]['units'][unit]['zstatistics']
        channelName = data.neural_data[session]['units'][unit]['channel_name']
        channel = data.neural_data[session]['units'][unit]['channel_num']
        cluster = data.neural_data[session]['units'][unit]['class_num']
        trials = data.neural_data[session]['units'][unit]['trial']
        kind = data.neural_data[session]['units'][unit]['kind']
        firingRates = get_mean_firing_rate_normalized(trials, stimuliIndices, args.min_t, args.max_t)[0]
        name = "pat " + str(subjectNum) + ", session " + str(sessionNum) + ", " + channelName + ", channel " + str(channel) + ", cluster " + str(cluster) + ", " + kind
        
        responses = []
        allRasters = []
        responseIndices = data.neural_data[session]['units'][unit]['responses']
        for responseIndex in responseIndices : 
            stimulusName = stimuliNames[responseIndex]
            trialIndices = np.where(np.asarray(stimuliNamesTrials) == stimulusName)[0]
            stimulusTrials = trials[trialIndices]
            responses.append(RasterInput(stimulusName, pvals[responseIndex], trials[trialIndices]))

        for i in range(len(pvals)) : 
            stimulusName = stimuliNames[i]
            trialIndices = np.where(np.asarray(stimuliNamesTrials) == stimulusName)[0]
            stimulusTrials = trials[trialIndices]
            allRasters.append(RasterInput(stimulusName, pvals[i], trials[trialIndices]))

        allRasters = sorted(allRasters, key=lambda x: x.pval)
        allRasters = allRasters[:30]
        #allRasters.sort(key=operator.attrgetter('pvalues'))

        for tuner in tuners : 
            if tuner.subjectsession + "_" + tuner.paradigm == session and tuner.channel == channel and tuner.cluster == cluster : 
                tuner.zscores = zscores
                tuner.firingRates = firingRates
                tuner.stimuli = stimuliNums
                tuner.stimuliNames = stimuliNames
                tuner.stimuliX = stimuliX
                tuner.stimuliY = stimuliY
                tuner.responses = responses
                tuner.unitType = kind
                tuner.responseIndices = responseIndices
                tuner.allRasters = allRasters
                tuner.pvalues = pvals
                tuner.site = site
                tuner.zstatistics = zstatistics

        #if all(pval >= args.alpha for pval in pvals) : 
            #print("Skipping " + subjectSession + ", cell " + str(cellNum))
            #continue

        if len(responseIndices) > 0 : 
            allUnits.append(Tuner(session, channel, cluster, kind, name, "aos", 
                stimuliNums, stimuliX, stimuliY, stimuliNames, zscores, firingRates, pvals, responses, allRasters, responseIndices, site, zstatistics))

    print("Prepared session " + session)

print("Time preparing data: " + str(time.time() - startPrepareData) + " s\n")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

table_options_heatmap = [{'tuners-heatmap-column': tuners[i].name, 'id': i} for i in range(len(tuners))]


startHeatmap = getInterpolatedMap(np.array(tuners[0].stimuliX), np.array(tuners[0].stimuliY), np.array(tuners[0].zscores))

startTimeTunerPlots = time.time()
tunerHeatmaps = []
for tuner in tuners : 
    if len(tuner.pvalues) == 0 : 
        print("WARNING! Tuner " + tuner.name + " not found ")
        continue
    tunerHeatmaps.append(
        createHeatMap(tuner, figureHeightBig, "interesting", True)[0])

    print("Created heatmap for " + tuner.name)    

print("Time preparing tuner plots: " + str(time.time() - startTimeTunerPlots) + " s\n")

numRegions = []
allHeatmaps = []
if args.show_all : 
    for unit in allUnits : 
        heatMapData = createHeatMap(unit, figureHeight)
        heatMapFigure = heatMapData[0]
        numRegions.append(heatMapData[1])
        allHeatmaps.append(
            html.Div([
                html.H3(children='Activation heatmap ' + unit.name),
                html.Div(children=[heatMapFigure]),
                #html.Div([
                #    dcc.Graph(id='heatmap-' + cell.name, figure=heatMapFigure)
                #], className="nine columns"),
            ], className="row"),
        ) 
        print("Created heatmap for " + unit.name)

    print("Done loading all heatmaps!")


counts, bins = np.histogram(numRegions, bins=np.append(np.arange(0,15,1), np.inf))
#counts = np.append(counts,0)
numRegionsPlot = sns.barplot(x=bins[:-1], y=counts)
save_img("numRegions")


app.layout = html.Div(children=[
    html.H1(children='Tuners'),
    html.H2(children='Activation heatmap'),

    html.Div([
        
        html.Div(id='heatmapDiv', className = "nine columns"),

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
                    'height': figureHeightBig - 200,
                    'overflowY': 'scroll'
                }
            ), 
        ], className="two columns"),
    ], className="row"),

    html.Div(children = allHeatmaps),
])

print("\n--- Ready! ---\n\n")


@app.callback(
    Output(component_id='heatmapDiv', component_property='children'), #src
    Input('tuners-heatmap-table', 'active_cell')
)
def update_output_div(active_cell):
 
    if(active_cell == None) :
        tunerIndex = 0
    else : 
        tunerIndex = active_cell['row']

    return tunerHeatmaps[tunerIndex]


if __name__ == '__main__':
    app.run_server(debug=False) # why ?
