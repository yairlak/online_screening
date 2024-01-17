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
from statannotations.Annotator import Annotator
#from statannot import add_stat_annotation
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
from utils import *
from plot_helper import *
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
parser.add_argument('--data_type', default="zscores", type=str, # zscores or firingRates or zsctatistics
                    help="Determines underlying datatype for heatmaps. \
                        Currently, zscores or firingRates are implemented.")
parser.add_argument('--min_t', default=0, type=int, #100
                    help="Relevant for calculating mean firing_rate. \
                        Min time offset of spike after stimulus onset.")
parser.add_argument('--max_t', default=1000, type=int, # 800
                    help="Relevant for calculating mean firing_rate. \
                        Max time offset of spike after stimulus onset.")
parser.add_argument('--min_num_trials', default=5, 
                    help='Min number of trials for a stimulus to be used for creating heatmap')

# FLAGS
parser.add_argument('--save_data', default=False,
                    help='If True, all heatmaps are saved')
parser.add_argument('--load_data', default=True,
                    help='If True, all heatmaps are loaded')
parser.add_argument('--dont_plot', action='store_true', default=False, 
                    help='If True, plotting to figures folder is supressed')
parser.add_argument('--only_SU', default=True, 
                    help='If True, only single units are considered')
parser.add_argument('--plot_all_rasters', default=False, 
                    help='If True, extended rasters are plotted')
parser.add_argument('--num_rasters', default=150, #currently not in use
                    help='Numbers of rasters to be plotted if plot_all_rasters is True')
parser.add_argument('--load_cat2object', default=False, 
                    help='If True, cat2object is loaded')

# STATS
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha for stats')
parser.add_argument('--alpha_region', type=float, default=0.01,
                    help='alpha for stats')

# PLOT
parser.add_argument('--interpolation_factor', type=float, default=1000,
                    help='heatmap interpolation grid size')
parser.add_argument('--padding_factor', type=float, default=1.1,
                    help='padding around datapoints')
parser.add_argument('--plot_regions', default='hemispheres',
                    help='"full"->all regions, "hemispheres"->split into hemispheres, "collapse_hemispheres"->regions of both hemispheres are collapsed')  

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
parser.add_argument('--path2heatmapdata', 
                    default='../data/heatmaps') 
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
    numTrials : List[int] = field(default_factory=lambda: [])

def getImgPath() : 
    if args.only_SU : 
        unitPath = "SU"
    else : 
        unitPath = "MU_SU"
    return args.path2images + os.sep + args.data_type + "_" + args.plot_regions + "_" + unitPath + "_" + str(args.interpolation_factor)

def getHeatmapPath() : 

    return args.path2heatmapdata + os.sep +  'heatmaps_' + args.data_type + "_" + str(args.interpolation_factor) 

def saveImgFont(filename, plotlyFig = None, snsFig = None): 
    adjustFontSize(snsFig)
    plt.xticks(rotation='horizontal')
    save_img(filename, plotlyFig)

def save_img(filename, plotlyFig = None) : 

    file = getImgPath() + os.sep + filename

    if not args.dont_plot : 
        os.makedirs(os.path.dirname(file), exist_ok=True)

        if plotlyFig is None: 
            plt.savefig(file + ".png", bbox_inches="tight")
            plt.clf()
        else : 
            plotlyFig.write_image(file + ".png")

    plt.close()

def createRasterPlot(rasterToPlot, figureWidth, figureHeight, savePath="") : 
    
    ## Raster plots
    numCols = 3
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
        title['font'] = dict(size=32)

    rowNum = 0
    colNum = 0
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    
    for response in responsesSorted : 
        rasterFigure = plotRaster(response, linewidth=5.0)
        if len(savePath) > 0  : 
            newRaster = go.Figure(rasterFigure)
            newRaster.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            newRaster.update_layout(title=None, xaxis={'showticklabels': False})
            newRaster.update_layout(plot_bgcolor='white')
            #fig.update_layout(xaxis_title=None)
            newRaster.write_image(savePath + os.sep + response.stimulusName + ".png")
            #save_img("raster" + os.sep + savePath + os.sep + response.stimulusName)
        for line in rasterFigure.data : 
            rasterGrid.add_trace(line, row=rowNum+1, col=colNum+1)

        colNum += 1
        if colNum == numCols : 
            rowNum += 1
            colNum = 0

    rasterGrid.update_layout(go.Layout(
        showlegend=False, 
        autosize=False,
        height = int(0.8 * figureWidth / numCols * numRowsRaster), #figureWidth, # int(figureHeight / 4) * numRowsRaster + 100,
        width = int(figureWidth), 
        paper_bgcolor='lightgrey',
        plot_bgcolor='white',
    ))

    for ax in rasterGrid['layout']:
        if ax[:5]=='xaxis':
            rasterGrid['layout'][ax]['range']=[-500,1500]
            #rasterGrid['layout'][ax]['color']='lightskyblue'#(0.5,0.5,0.5,1.0)
            rasterGrid['layout'][ax]['tickmode']='array'
            rasterGrid['layout'][ax]['tickvals']=[0, 1000]
            rasterGrid['layout'][ax]['tickfont']=dict(size=20)
            #rasterGrid['layout'][ax]['plot_bgcolor']='white'
        if ax[:5]=='yaxis':
            rasterGrid['layout'][ax]['visible']=False
            rasterGrid['layout'][ax]['showticklabels']=False
            #rasterGrid['layout'][ax]['plot_bgcolor']='white'
            #rasterGrid['layout'][ax]['range']=[0, len(response.spikes)]  # can only be set for all together (?)
            
    return rasterGrid

def addStimNames(heatmap, x, y, names): 
    heatmap.add_trace(go.Scatter(mode='text', x=rescaleX(x), y=rescaleY(y), text=names, textfont=dict(size=12,color="black"),))
    heatmap.update_layout(go.Layout(showlegend=False))
    return heatmap

def rescaleX(x) :
    xPadding = np.asarray([xOut / args.padding_factor for xOut in x])
    return args.interpolation_factor * (xPadding - xMinThings) / (xMaxThings - xMinThings) - 0.5

def rescaleY(y) :
    yPadding = np.asarray([yOut / args.padding_factor for yOut in y])
    return args.interpolation_factor * (yPadding - yMinThings) / (yMaxThings - yMinThings) - 0.5

def heatmapFromZ(z) : 
    #fig = plt.imshow(z, cmap='RdBu_r', interpolation='nearest')
    #fig = sns.heatmap(z, cmap='RdBu')
    
    # Plot the heatmap
    #fig = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    fig = px.imshow(z,aspect=1.0,color_continuous_scale='RdBu_r',origin='lower')
    font = dict(weight='bold', size=32)
    plt.rc('font', **font)
    return fig

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
        #zWithBorder *= -1.0

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
        zi = np.exp(zi)
        zi[zi > args.alpha_region] = 0.0 
        #zi[zi < np.log(args.alpha_region)] = 0.0 
    else : 
        # Interpolate missing data
        rbf = scipy.interpolate.Rbf(xWithBorder, yWithBorder, zWithBorder, function='linear')
        zi = rbf(xi, yi)

    return heatmapFromZ(zi), xi, yi#, zmax=1

def createHeatMap(tuner, figureHeight, savePath="", addName=False) : 

    outputPath = getImgPath() + os.sep + savePath
    #if args.only_8_trials : 
    relevant = np.where(tuner.numTrials >= args.min_num_trials)[0]
    #else : 
    #    relevant = np.arange(len(tuner.pvals))
    
    if args.data_type == "firing_rates" :
        targetValue = tuner.firingRates
    elif args.data_type == "zstatistics": 
        targetValue = tuner.zstatistics
    else : 
        targetValue = tuner.zscores

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    ## Heatmap
    heatmap = getInterpolatedMap(np.array(tuner.stimuliX)[relevant], np.array(tuner.stimuliY)[relevant], np.array(targetValue)[relevant])[0]
    heatmapOnlyResponses, xi, yi = getInterpolatedMap(np.array(tuner.stimuliX)[relevant], np.array(tuner.stimuliY)[relevant], np.array(tuner.pvalues)[relevant], pvalues=True)
    #heatmapMasked = getInterpolatedMap(np.array(tuner.stimuliX), np.array(tuner.stimuliY), np.array(targetValue))[0]

    regionsHeatmap = heatmapOnlyResponses.data[0].z.copy()
    regionsHeatmap[regionsHeatmap != 0.0] = 1.0
    regionsLabels = label(regionsHeatmap, connectivity=2)
    heatmapCopy = heatmap.data[0].z.copy()

    #remove areas without response (including consider)
    numRegionsStart = np.amax(regionsLabels)
    regionsLabels[regionsLabels > 0] += 1000
    newLabelCount = 1
    numStimPerRegion = np.zeros((numRegionsStart+1,), dtype=int)
    numNoRegionFound = 0

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
            if numStimPerRegion[newLabelCount] != 0 : 
                print("ERROR!!")
            numStimPerRegion[newLabelCount] = 1
            newLabelCount += 1
        elif responseLabel == 0 : 
            print("WARNING! Response without region")
            numNoRegionFound += 1
            numStimPerRegion = np.append(numStimPerRegion, [0])

            regionsLabels[xResponseHeatmap][yResponseHeatmap] = newLabelCount
            if numStimPerRegion[newLabelCount] != 0 : 
                print("ERROR!!")
            numStimPerRegion[newLabelCount] = 1
            newLabelCount += 1
            #numStimPerRegion[0] += 1
        else : 
            numStimPerRegion[responseLabel] += 1
            #print("label is " + str(responseLabel))
        
        #print("test")
        
    
    noResponseIndices = np.where(regionsLabels > 1000)
    numNoResponse = len(np.unique(regionsLabels[noResponseIndices]))
    regionsLabels[noResponseIndices] = 0
    heatmapOnlyResponses.data[0].z[noResponseIndices] = 0

    #regionSize = np.zeros(len(numStimPerRegion))
    numTotalFields = args.interpolation_factor * args.interpolation_factor
    sizeFields = []
    for reg in range(1,len(numStimPerRegion)) : 
        if numStimPerRegion[reg] < 2 : 
            continue
        numFieldsReg = float(len(np.where(regionsLabels == reg)[0]))
        sizeFields.append(numFieldsReg / numTotalFields)

    mask = heatmapOnlyResponses.data[0].z
    mask[np.where(regionsLabels == 0)] = 0.5
    mask[np.where(regionsLabels > 0)] = 1

    numRegions = len(np.where(numStimPerRegion > 1)[0]) # np.amax(regionsLabels) = np.where(numStimPerRegion > 0) # criterion: at least 2 stim in blob
    targetValues = np.copy(targetValue)
    targetValues -= min(targetValues)
    targetValues /= max(targetValues)
    
    for stimulusNum in tuner.responseIndices :
        textTrace = go.Scatter(
                mode='text',
                text=tuner.stimuliNames[stimulusNum],
                x=rescaleX([tuner.stimuliX[stimulusNum]]), y=rescaleY([tuner.stimuliY[stimulusNum]]),
                #hovertext=[tuner.stimuliNames[stimulusNum] + ", z: " + str(round(targetValues[stimulusNum], 2))],
                textfont=dict(size=40,color="black"),
                #name='zscore'
            )
        heatmapOnlyResponses.add_trace(textTrace)
        heatmap.add_trace(textTrace)

    figureWidth = figureHeight#*3/2
    titleText = "session: " + tuner.subjectsession + ", channel: " + str(tuner.channel) + ", cluster: " + str(tuner.cluster) + ", " + tuner.unitType + ", " + tuner.site
    graphLayout = go.Layout(
        title=dict(
            text=titleText,
            font=dict(size=42,)),
        #title_text=titleText, 
        xaxis=dict(ticks='', showticklabels=False),
        yaxis=dict(ticks='', showticklabels=False),
        showlegend=False, 
        autosize=True,
        height=figureHeight,
        width=figureWidth
    )

    heatmapOnlyResponses.update_layout(graphLayout)
    heatmapOnlyResponses.update_layout(go.Layout(title_text = titleText + ", numRegions: " + str(numRegions)))
    heatmap.update_layout(graphLayout)
    heatmap.update_layout(go.Layout(width=figureWidth+140))

    filenameRaster = tuner.subjectsession + "_ch" + str(tuner.channel) + "_cl" + str(tuner.cluster) 
    if addName : 
        filename = outputPath + os.sep + filenameRaster + "_" + tuner.name
    else : 
        filename = outputPath + os.sep + str(numRegions) + "regions" + os.sep + filenameRaster
    filenameRaster = filename + os.sep #+ "raster"
    
    figureWidthRaster = figureWidth #figureHeightRaster*3/2
    figureHeightRaster = figureWidthRaster * 2/3
    if args.plot_all_rasters : 
        allRasters = createRasterPlot(tuner.allRasters, figureWidthRaster, figureHeightRaster, filenameRaster)
    rasterGrid = createRasterPlot(tuner.responses, figureWidthRaster, figureHeightRaster, filenameRaster)

    if not args.dont_plot : 
        heatmapFilename = filename + "_heatmap.png"
        heatmapMaskedFilename = filename + "_heatmap_masked.png"
        heatmapOnlyResponsesFilename = filename + "_heatmap_responses.png"
        rasterFilename = filename + "_rasterplots.png"
        rastersAllFilename = filename + "_rasterplots_all.png"
        completeFilename = filename 
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        heatmap.update_coloraxes(colorbar_tickfont_size=50)
        heatmap.update_coloraxes(colorbar_thickness=100)
        #heatmap.update_layout(width=2000)
        heatmapOnlyResponses.write_image(heatmapOnlyResponsesFilename)
        heatmap.write_image(heatmapFilename)
        heatmap.write_image(filename + "_heatmap.svg")
        #go.Image(z=images[1], opacity=0.1)
        #heatmap.add_trace(go.Image(z=mask, opacity=0.1))
        #px.imshow(mask, alpha=alphas, **imshow_kwargs)
        heatmapTmp = heatmap.data[0].z.copy()
        heatmap.data[0].z = mask * heatmap.data[0].z 
        heatmap.update_coloraxes(showscale=False)
        heatmap.update_layout(go.Layout(width=figureWidth, title_text=""))
        heatmap.update_layout(go.Layout(margin=go.layout.Margin(l=30, r=30, b=30, t=30)))
        heatmap.write_image(heatmapMaskedFilename)
        #heatmapTmp.data[0].z = heatmapTmp
        #heatmap.update_coloraxes(showscale=False)

        rasterGrid.write_image(rasterFilename)
        rasterGrid.write_image(filename + "_rasterplots.svg")
        if args.plot_all_rasters : 
            allRasters.write_image(rastersAllFilename)

        pltHeatmap = Image.open(heatmapFilename)
        pltRaster = Image.open(rasterFilename)
        pltMask = Image.open(heatmapMaskedFilename)

        totalWidth = max([pltHeatmap.size[0], pltRaster.size[0]])
        totalHeight = pltHeatmap.size[1] + pltRaster.size[1]
        completeImage = Image.new('RGB', (totalWidth, totalHeight))

        completeImage.paste(pltHeatmap, (0,0))
        completeImage.paste(pltRaster, (0,pltHeatmap.size[1]))
        completeImage.save(completeFilename + ".png")
        #completeImage.save(completeFilename + ".svg")
        
        ##pltMask.putalpha(127)  # Half alpha; alpha argument must be an int
        pltHeatmap.paste(pltMask, (0, 0))
        #pltHeatmap.imshow(pltMask, alpha=0.5)
        pltHeatmap.save(heatmapMaskedFilename)
        #ax.set_axis_off()

        #os.remove(heatmapFilename)
        #os.remove(rasterFilename)


    tunerDivGrid = html.Div([
        html.Div(children=[dcc.Graph(id='heatmap-' + tuner.name, figure=heatmap)], style={'margin-bottom': 0}),
        html.Div(children=[dcc.Graph(id='rasterGrid-' + tuner.name, figure=rasterGrid)], style={'margin-top': 0})
    ])

    return tunerDivGrid, numRegions, numStimPerRegion[1:], np.asarray(sizeFields), numNoRegionFound, numNoResponse, heatmapCopy

#############
# LOAD DATA #
#############
startLoadData = time.time()

data = DataHandler(args) # class for handling neural and feature data
data.load_metadata() # -> data.df_metadata
#data.load_neural_data(min_active_trials=4) # -> data.neural_data
#data.load_word_embeddings() # -> data.df_word_embeddings
data.load_word_embeddings_tsne() # -> data.df_word_embeddings_tsne


print("\nTime loading data: " + str(time.time() - startLoadData) + " s\n")

sitesToConsider = ["LA", "RA", "LEC", "REC", "LAH", "RAH", "LMH", "RMH", "LPHC", "RPHC", "LPIC", "RPIC"]
tuners = [ # clusters might not fit (manual clustering took place)
    #Tuner("088e03aos1", 17, 1, "Pacifier", "aos", [], [], [], [], []),
    Tuner("103_2", 50, 2, [], "Sceleton", "aos", [], [], [], [], [], []),
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
    Tuner("104_1", 16, 2, [], "Wood", "aos",  [], [], [], [], [], []),
    Tuner("104_1", 15, 1, [], "Charcoal, Eyedropper", "aos",  [], [], [], [], [], []),
    Tuner("104_1", 89, 2, [], "Backpack", "aos",  [], [], [], [], [], []),
]

figureHeight = 1200
figureHeightBig = 1500

nConcepts = data.df_word_embeddings_tsne.shape[0]
xThings = data.df_word_embeddings_tsne[:][1]
yThings = data.df_word_embeddings_tsne[:][0]

xMinThings = xThings.min()
xMaxThings = xThings.max()
yMinThings = yThings.min()
yMaxThings = yThings.max()

#xThingsRescaled = rescaleX(xThings)
#yThingsRescaled = rescaleY(yThings)

startPrepareData = time.time()
inputPath = args.path2data #+ tuner.paradigm + "_after_manual_clustering/"
outputPath =  args.path2images + os.sep + args.data_type

allUnits = []

print("Time preparing data: " + str(time.time() - startPrepareData) + " s\n")

startTimeFullAnalysis = time.time()
#numRegions = []
#allHeatmaps = []
#allSitesComplete = []
#allSites = []
#allSitesFields = []
#numStimPerRegion = np.array([])#np.zeros((100,),dtype=int)
#sizeFieldsRegion = np.array([])#np.zeros((100,),dtype=int)
#numStimPerRegionMatrixAllSites = []
#sizeFieldsMatrixAllSites = []
#heatmaps = []
heatmapsComplete = []
numStimPerRegionUnitComplete = []
sizeFieldsComplete = []
numRegionsComplete = []


if args.load_data : 
    heatmap_df = pd.read_pickle(args.path2heatmapdata + os.sep + "heatmaps_" + args.data_type + "_" + str(args.interpolation_factor) + ".pk1")
else : 
    
    data.load_neural_data() # -> data.neural_data
    sessions = list(data.neural_data.keys())

    numNoRegionFound = 0
    numNoResponseFound = 0

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
            firingRates = data.neural_data[session]['units'][unit]['firing_rates']
            name = "pat " + str(subjectNum) + ", session " + str(sessionNum) + ", " + channelName + ", channel " + str(channel) + ", cluster " + str(cluster) + ", " + kind
            
            if site not in sitesToConsider : 
                continue

            responses = []
            allRasters = []
            responseIndices = data.neural_data[session]['units'][unit]['responses']
            numTrials = []

            for i in range(len(pvals)) : 
                stimulusName = stimuliNames[i]
                trialIndices = np.where(np.asarray(stimuliNamesTrials) == stimulusName)[0]
                stimulusTrials = trials[trialIndices]
                allRasters.append(RasterInput(stimulusName, pvals[i], stimulusTrials))
                numTrials.append(len(stimulusTrials))

                if i in responseIndices : 
                    responses.append(RasterInput(stimulusName, pvals[i], stimulusTrials))

            if args.plot_all_rasters : 
                allRasters = sorted(allRasters, key=lambda x: x.pval)
                #allRasters = allRasters[:args.num_rasters]
            else : 
                allRasters = []
            #allRasters.sort(key=operator.attrgetter('pvalues'))
            numTrials = np.asarray(numTrials)

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
                    tuner.allRasters = allRasters
                    tuner.responseIndices = responseIndices
                    tuner.pvalues = pvals
                    tuner.site = site
                    tuner.zstatistics = zstatistics
                    tuner.numTrials = numTrials

            #if all(pval >= args.alpha for pval in pvals) : 
                #print("Skipping " + subjectSession + ", cell " + str(cellNum))
                #continue

            if len(responseIndices) > 0 : 
                allUnits.append(Tuner(session, channel, cluster, kind, name, "aos", 
                    stimuliNums, stimuliX, stimuliY, stimuliNames, zscores, firingRates, pvals, responses, allRasters, responseIndices, site, zstatistics, numTrials))

        print("Prepared session " + session)

        
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


    for unit in allUnits : 
        heatMapData = createHeatMap(unit, figureHeight)
        heatMapFigure = heatMapData[0]
        numStimPerRegionUnit = heatMapData[2]
        sizeFields = heatMapData[3]
        numNoRegionFound += heatMapData[4]
        numNoResponseFound += heatMapData[5]
        heatmap = heatMapData[6]

        heatmapsComplete.append(heatmap)
        sizeFieldsComplete.append(sizeFields)
        numStimPerRegionUnitComplete.append(numStimPerRegionUnit)
        numRegionsComplete.append(heatMapData[1])
        print("Created heatmap for " + unit.name + ", num stimuli per field: " + str(numStimPerRegionUnit))

    print("For " + str(numNoRegionFound) + " responses there was no region found!")
    print("For " + str(numNoResponseFound) + " regions there was no response found!")


    heatmap_df = pd.DataFrame()
    heatmap_df["heatmaps"] = heatmapsComplete
    heatmap_df["sizeFields"] = sizeFieldsComplete
    heatmap_df["numRegions"] = numRegionsComplete
    heatmap_df["numStimPerRegion"] = numStimPerRegionUnitComplete
    heatmap_df["sites"] = np.array([unit.site for unit in allUnits])
    heatmap_df["names"] = np.array([unit.name for unit in allUnits])
    heatmap_df["zscores"] = [unit.zscores for unit in allUnits]
    heatmap_df["stimuli"] = [unit.stimuli for unit in allUnits]
    heatmap_df["stimuliNames"] = [unit.stimuliNames for unit in allUnits]
    heatmap_df["stimuliX"] = [unit.stimuliX for unit in allUnits]
    heatmap_df["stimuliY"] = [unit.stimuliY for unit in allUnits]
    heatmap_df["unitType"] = [unit.unitType for unit in allUnits]
    heatmap_df["allRasters"] = [unit.allRasters for unit in allUnits]
    heatmap_df["responseIndices"] = [unit.responseIndices for unit in allUnits]
    heatmap_df["pvalues"] = [unit.pvalues for unit in allUnits]
    heatmap_df["numTrials"] = [unit.numTrials for unit in allUnits]
    heatmap_df["firing_rates"] = [unit.firingRates for unit in allUnits]
    heatmap_df["pvals"] = [unit.pvalues for unit in allUnits]


print("Done loading all heatmaps!")

if args.save_data : 
    os.makedirs(os.path.dirname(getHeatmapPath()), exist_ok=True)
    heatmap_df.to_pickle(getHeatmapPath() + ".pk1")
    

#unitIndices = np.range(len(allUnits))
if args.only_SU : 
    #unitIndices = np.where(heatmap_df["unitType"] == 'SU')[0]
    heatmap_df = heatmap_df.loc[heatmap_df['unitType'] == 'SU']
    heatmap_df.index = range(len(heatmap_df))

heatmaps = heatmap_df["heatmaps"]
allSites = [getSite(site, args.plot_regions) for site in heatmap_df["sites"]]
heatmap_df["sites"] = allSites
numStimPerRegionMatrixAllSites = heatmap_df["numStimPerRegion"]
sizeFieldsMatrixAllSites = heatmap_df["sizeFields"]
numRegions = heatmap_df["numRegions"]
numStimPerRegion = np.array([])
sizeFieldsRegion = np.array([])
allSitesFields = np.array([])

for index, unit in heatmap_df.iterrows() : 
    sizeFields = unit["sizeFields"]
    numStimPerRegionUnit = unit["numStimPerRegion"]
    site = unit["sites"]

    numStimPerRegion = np.concatenate((numStimPerRegion, np.asarray(numStimPerRegionUnit)), axis=None)
    sizeFieldsRegion = np.concatenate((sizeFieldsRegion, np.asarray(sizeFields)), axis=None)
    allSitesFields = np.concatenate((allSitesFields, np.repeat(site, len(sizeFields))), axis=None)

counts, bins = np.histogram(numStimPerRegion, bins=np.arange(1,max(numStimPerRegion)+1, dtype=int))
numStimPlot = sns.barplot(x=bins[:-1], y=counts, color='blue')
saveImgFont("numStimuliPerField", snsFig=numStimPlot)

binsSizeFields = np.round(np.append(np.arange(0.0, 0.05, 0.005), 1.0),3)
binsSizeFields = np.append(np.array([0] + [0.001 * 2**i for i in range(10)]),1.0)
counts, bins = np.histogram(sizeFieldsRegion[np.where(sizeFieldsRegion > 0)[0]], bins=binsSizeFields)
sizeFieldsPlot = sns.barplot(x=bins[:-1], y=counts, color='blue')
saveImgFont("sizeFields", snsFig=sizeFieldsPlot)

counts, bins = np.histogram(numRegions, bins=np.append(np.arange(1,10), np.inf))
numFieldsPlot = sns.barplot(x=bins[:-1], y=counts, color='blue')
saveImgFont("numFields", snsFig=numFieldsPlot)

allSites = np.array(allSites)
uniqueSites = np.unique(allSites)
numUnitsPerSite = pd.Series(allSites).value_counts()
numRegions = np.array(numRegions)

#rightHemisphere = np.asarray([i for i in range(len(allSites)) if allSites[i][0] == "R"])
#counts, bins = np.histogram(numRegions[rightHemisphere], bins=np.append(np.asarray(range(11)), 10000))
#sns.barplot(x=bins[:-1], y=counts)
#save_img("numRegions_right")

#leftHemisphere = np.asarray([i for i in range(len(allSites)) if allSites[i][0] == "L"])
#counts, bins = np.histogram(numRegions[leftHemisphere], bins=np.append(np.asarray(range(11)), 10000))
#sns.barplot(x=bins[:-1], y=counts)
#save_img("numRegions_left")

binsRegions = np.append(np.arange(1,5),np.inf)
binsStim = np.append(np.arange(1,9),np.inf)
histMatrixRegions = []
histMatrixStim = []
histMatrixSize = []
sizeFieldsCountsSorted = []
numAtLeastOneField = len(np.where(numRegions > 0))
heatmapsSites = []

combinedPath = getImgPath() + os.sep + "combined" + os.sep 
combinedPathAverage = combinedPath + "average" + os.sep
combinedResponsesPath = combinedPath + "responses" + os.sep
combinedAllPath = combinedPath + "all_stimuli" + os.sep
os.makedirs(os.path.dirname(combinedPathAverage), exist_ok=True)
os.makedirs(os.path.dirname(combinedResponsesPath), exist_ok=True)
os.makedirs(os.path.dirname(combinedAllPath), exist_ok=True)

for site in uniqueSites : 
    siteIndices = np.where(allSites == site)[0]
    heatmap_site_df = heatmap_df.loc[heatmap_df['sites'] == site]

    bestX = []
    bestY = []
    bestZ = []
    bestNames = []
    for index, unit in heatmap_site_df.iterrows():
        bestIndex = unit[args.data_type].argmax()
        bestValue = unit[args.data_type][bestIndex]
        bestX.append(unit["stimuliX"][bestIndex])
        bestY.append(unit["stimuliY"][bestIndex])
        bestZ.append(bestValue)
        bestNames.append(unit["stimuliNames"][bestIndex])

    newBestX = []
    newBestY = []
    newBestZ = []
    newBestNames = []
    for x in np.unique(bestX) : 
        indicesBest = np.where(bestX == x)[0]
        newBestX.append(x)
        newBestY.append(bestY[indicesBest[0]])
        newBestZ.append(np.average(np.array(bestZ)[indicesBest]))
        newBestNames.append(bestNames[indicesBest[0]])

    allX = np.array([])
    allY = np.array([])
    allZ = np.array([])
    allNames = np.array([])
    for index, unit in heatmap_site_df.iterrows() :
        allX = np.concatenate((allX, np.asarray(unit["stimuliX"])), axis=None)
        allY = np.concatenate((allY, np.asarray(unit["stimuliY"])), axis=None)
        allZ = np.concatenate((allZ, np.asarray(unit[args.data_type])), axis=None)
        allNames = np.concatenate((allNames, np.asarray(unit["stimuliNames"])), axis=None)

    newAllX = []
    newAllY = []
    newAllZ = []
    newAllNames = []
    for x in np.unique(allX) :
        indices = np.where(allX == x)[0]
        newAllX.append(x)
        newAllY.append(allY[indices[0]])
        newAllZ.append(np.average(allZ[indices]))
        newAllNames.append(allNames[indices[0]])

    heatmapResponsesSite = getInterpolatedMap(np.array(newBestX), np.array(newBestY), np.array(newBestZ))[0]
    heatmapResponsesSite = addStimNames(heatmapResponsesSite, newBestX, newBestY, newBestNames)
    heatmapResponsesSite.write_image(combinedResponsesPath + site + ".png")

    heatmapResponsesSite = getInterpolatedMap(np.array(newAllX), np.array(newAllY), np.array(newAllZ))[0] # reuse to save space
    heatmapResponsesSite = addStimNames(heatmapResponsesSite, newBestX, newBestY, newBestNames)
    heatmapResponsesSite.write_image(combinedAllPath + site + ".png")

    heatmapsSite = np.matrix(heatmaps[siteIndices[0]])
    for unit in range(1,len(siteIndices)) : 
        heatmapsSite += np.matrix(heatmaps[siteIndices[unit]])
    heatmapsSite = heatmapsSite / len(siteIndices)
    heatmapsSites.append(heatmapsSite) 
    heatmapIm = heatmapFromZ(heatmapsSite)
    heatmapIm = addStimNames(heatmapIm, bestX, bestY, bestNames)
    heatmapIm.write_image(combinedPathAverage + site + ".png")
    
    #adjustFontSize()
    numRegionsSite = numRegions[siteIndices] 
    counts, bins = np.histogram(numRegionsSite, bins=binsRegions)
    adjustFontSize()
    numRegionsPlot = sns.barplot(x=bins[:-1], y=counts)
    #plt.xticks(rotation=90, ha='right')
    sumCounts = max(1.0, float(sum(counts)))
    histMatrixRegions.append(counts.astype(np.float32) / sumCounts)
    saveImgFont("numFields_" + site)

    numStimSiteCounts = np.array([])
    sizeFieldsSiteCounts = np.array([])
    for siteIndex in siteIndices : 
        numStimSiteCounts = np.concatenate((numStimSiteCounts, np.asarray(numStimPerRegionMatrixAllSites[siteIndex])), axis=None)
        sizeFieldsSiteCounts = np.concatenate((sizeFieldsSiteCounts, np.asarray(sizeFieldsMatrixAllSites[siteIndex])), axis=None)
    sizeFieldsCountsSorted.append(sizeFieldsSiteCounts)
    counts, bins = np.histogram(numStimSiteCounts, bins=binsStim)
    numStimPlot = sns.barplot(x=bins[:-1], y=counts)
    #plt.xticks(rotation=90, ha='right')
    sumCounts = max(1.0, float(sum(counts)))
    histMatrixStim.append(counts/sumCounts)
    saveImgFont("numStimuli_" + site, snsFig=numStimPlot)
    
    counts, bins = np.histogram(sizeFieldsSiteCounts[np.where(sizeFieldsSiteCounts > 0)[0]], bins=binsSizeFields)
    sizeFieldsSitePlot = sns.barplot(x=bins[:-1], y=counts)
    #adjustFontSize()
    #plt.xticks(rotation=90, ha='right')
    sumCounts = max(1.0, float(sum(counts)))
    histMatrixSize.append(counts/sumCounts)
    saveImgFont("sizeFields_" + site, snsFig=sizeFieldsSitePlot)



siteFields_df = pd.DataFrame(data=sizeFieldsRegion, columns=["size"])
siteFields_df["sites"] = allSitesFields
siteFieldsPlot = sns.boxplot(data=siteFields_df, x="sites", y="size")
anova = stats.f_oneway(*histMatrixSize)

uniqueSitesFields = np.unique(allSitesFields)
if len(uniqueSitesFields) > 1 :
    annotationSites = [(uniqueSitesFields[site1], uniqueSitesFields[site2]) for site1 in range(len(uniqueSitesFields)) for site2 in range(len(uniqueSitesFields)) if site1 < site2]
    annotator = Annotator(siteFieldsPlot, annotationSites, data=siteFields_df, x="sites", y="size")
    annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    annotator.apply_and_annotate() 
saveImgFont("sizeFieldsBox", snsFig=siteFieldsPlot)


sizeFieldsPlot = createStdErrorMeanPlot(uniqueSites, sizeFieldsCountsSorted, "Average size of semantic fields, p: " + str(anova.pvalue), yLabel="", xLabel="Size fields")
saveImgFont("sizeFieldsBars", sizeFieldsPlot)

numRegions_df = pd.DataFrame(np.asarray(histMatrixRegions).transpose(), columns = uniqueSites)
numRegions_df["index"] = binsRegions[:-1] #np.append(binsRegions[1:], [len(binsRegions)])
numRegions_df.plot(kind="bar", x="index", figsize=(10,6), ylabel="Fraction of number of fields per unit", xlabel="")
#numRegionsPlot.set_xticklabels(np.append(bins, [bins[-1]+1]))
#plt.xticks(rotation=90, ha='right')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
saveImgFont("numFields_grouped")

numStim_df = pd.DataFrame(np.asarray(histMatrixStim).transpose(), columns = uniqueSites)
#binsStim[-1] = len(binsStim)-1
numStim_df["index"] = binsStim[:-1] #np.append(binsStim[1:], [len(binsStim)])
numStim_df.plot(kind="bar", x="index", figsize=(10,6), ylabel="Fraction of number of stimuli per field", xlabel="")
#plt.xticks(rotation=90, ha='right')
#numRegionsPlot.set_xticklabels(np.append(binsStim, [bins[-1]+1]))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
saveImgFont("numStimuli_grouped")

numStim_df = pd.DataFrame(np.asarray(histMatrixSize).transpose(), columns = uniqueSites)
#binsStim[-1] = len(binsStim)-1
numStim_df["index"] = np.round(binsSizeFields[:-1],3) #np.append(binsStim[1:], [len(binsStim)])
numStim_df.plot(kind="bar", x="index", figsize=(10,6), ylabel="Fraction of field sizes")
#plt.xticks(rotation=90, ha='right')
#numRegionsPlot.set_xticklabels(np.append(binsStim, [bins[-1]+1]))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
saveImgFont("sizeFields_grouped")

print("Time accumulated analysis: " + str(time.time() - startTimeFullAnalysis) + " s\n")
