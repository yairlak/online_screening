#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/24 13:41:48
@Author  :   Katharina Karkowski 
"""

import os

import numpy as np
import statistics 
from fit_data import *
from scipy import stats
from scipy.signal import savgol_filter

from typing import List
from dataclasses import field
from dataclasses import dataclass

from dash import dash_table
from dash import dcc
from dash import html

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pcolors
from matplotlib import colors as mcolors



@dataclass
class RasterInput: 
    stimulusName: str
    pval: float
    spikes: List[List] = field (default_factory=lambda: [[]])
    minX : int = field (default_factory=lambda: -500)
    maxX : int = field (default_factory=lambda: 1500)


def plotRaster(rasterInput, linewidth=1) : 

    fig=go.Figure()

    numTrials = len(rasterInput.spikes)
    for trial in range(numTrials) : 
        spikesTrial = rasterInput.spikes[trial][0]
        spikes = spikesTrial[np.where(spikesTrial <= rasterInput.maxX)[0]]
        spikes = spikesTrial[np.where(spikesTrial > rasterInput.minX)[0]]

        for spikeTime in spikes : 
            fig.add_trace(
                go.Scatter(
                    x=[spikeTime, spikeTime], 
                    y=[trial, trial + 1],
                    mode='lines',
                    line_color='black', 
                    line_width=linewidth
            ))
    
    pval = rasterInput.pval
    if pval < 0.0001 : 
        pval = np.format_float_scientific(pval, precision=3)
    else : 
        pval = round(pval, 7)

    fig.update_layout(
        title=rasterInput.stimulusName + ', pval: ' + str(pval),
        title_font=dict(size=10),
        showlegend=False,
        yaxis_visible=False, 
        yaxis_showticklabels=False, 
        yaxis_range=[0, numTrials],

        xaxis_range=[rasterInput.minX,rasterInput.maxX],
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0, 1000],
            tickfont=dict(size=8),
        )
    )

    return fig


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

def createRegionsDiv(name, allSiteNames) : 
    figureId = name + "-overview"
    tableId = name + "-regions-table"
    columnId = name + "-regions-column"
    columnData = [{columnId: site, 'id': site} for site in allSiteNames]

    return createTableDiv(
        name, figureId, tableId, "Regions", columnId, columnData), figureId, tableId

def createHist(x, inputBins, factorY, labelX, labelY) : 
    counts, bins = np.histogram(x, bins=inputBins)
    return px.bar(x=bins[:-1], y=counts.astype(float)*float(factorY), labels={'x':labelX, 'y':labelY})

def createCorrelationPlot(sitename, correlation) : 
    return go.Box(
        y=correlation, 
        name=sitename + " (" + str(len(correlation)) + ")",
        boxpoints='all',
        #boxmean='sd'
    )

def createBoxPlot(values, xNames, title, boxpoints='all') :
    fig = go.Figure()
    for i in range(len(xNames)) : 
        #stddev = statistics.stdev(values[i])
        fig.add_trace(go.Box(
            y=values[i],
            name=xNames[i],
            #boxpoints=boxpoints,
            #boxmean='sd'
        ))
    fig.update_layout(
        title_text=title,
        showlegend=False
    )
    return fig

def createStepBoxPlot(similaritiesArray, title, yLabel="", alpha=0.001, boxpoints=False, addLog=False, addPartialGaussian=True, stepBox=0.1) :   
    
    x = np.asarray(similaritiesArray.similarities)
    values = np.asarray(similaritiesArray.values)

    yFit=[]
    xFit=[]
    
    lowerfences = []
    upperfences = []
    means = []
    q1 = []
    q3 = []

    fig = go.Figure()

    for step in np.arange(-0.01,1.001,stepBox) : # due to rounding errors
        indices = np.where((x >= step) & (x < step+stepBox))[0]
        if len(indices) == 0 : 
            continue
        y = values[indices]
        if len(y) >= 2 :
            mean = statistics.mean(y)
            stddev = statistics.stdev(y)
            means.append(mean)
            lowerfences.append(mean-stddev)
            upperfences.append(mean+stddev)
            q1.append(np.percentile(y, 25))
            q1.append(np.percentile(y, 75))
    
        fig.add_trace(go.Box(
            x0=step,
            y=values[indices],
            name="{:.2f} ({})".format(step, y),
            boxpoints=boxpoints,
            #boxmean='sd'
        ))

    fig.update_traces(q1=q1, median=means,
        q3=q3, lowerfence=lowerfences,
        upperfence=upperfences )

    """ for i in range(len(values)) : 
        if(len(values[i]) >= 1) : 
            fig.add_trace(go.Box(
                #xaxis='x',
                x0=x[i],
                y=values[i],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
                #boxpoints='all',
            ))
            xFit = np.concatenate((xFit, np.repeat(x[i], len(values[i]))))
            yFit = np.concatenate((yFit, values[i]))
        #if(len(values[i]) >= 10) : 
        #    xFit.append(x[i])
        #    yFit.append(statistics.median(values[i]))
        else : 
            fig.add_trace(go.Box(
                xaxis='x',
                x0=x[i],
                y=[0.0],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
            )) """

        #if i < len(values)-1 : 
        #    t_value, p_value = stats.ttest_ind(values[i], values[i+1]) 

        #    if p_value <= alpha : 
        #        print(title + ': p_value=%.8f' % p_value,
        #            'for value=%i ' % i)

    xFit = x
    yFit = values

    if addLog and len(yFit) > 0: 
        xFitted, yFitted = fitLog(xFit, yFit)#, x[1]-x[0])
        #fig.add_trace(
        #    go.Scatter(
        #        x=xFitted,
        #        #xaxis='x',
        #        y=yFitted,
        #        mode='lines',
        #    ))
        addPlot(fig, xFitted, yFitted, 'lines', 'Logistic fit')
        
    if addPartialGaussian and len(yFit) > 0: 
        xFitted, yFitted = fitPartialGaussian(xFit, yFit)#, x[1]-x[0])
        addPlot(fig, xFitted, yFitted, 'lines', 'Gaussian fit')

    fig.update_layout(
        title_text=title,
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False,
    )

    fig.update_xaxes(
        tickmode = 'array',
        tickvals = np.arange(0,1.05,0.1)
    )

    return fig

def createPlot(x, y, yLabel, title, plotHalfGaussian, ticktext=[], plotStep=0.01) :

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
        xGauss, yGauss = fitGauss(xWithoutOutliers, yWithoutOutliers, plotStep)

    try : 
        yFitted = savgol_filter(yWithoutOutliers, 15, 3) # window size 51, polynomial order 3
    except Exception : 
        #print("WARNING: Error applying filter")
        yFitted = yWithoutOutliers

    xLog, yLog = fitLog(xWithoutOutliers, yWithoutOutliers, plotStep)

    fig = go.Figure()

    addPlot(fig, xWithoutOutliers, yWithoutOutliers, 'markers', 'Data')
    #addPlot(fig, xWithoutOutliers, smooth(yWithoutOutliers, 5), 'lines', 'Smoothed 5 point avg')
    #addPlot(fig, xGauss, yGauss, 'lines', 'Gaussian fit')
    #addPlot(fig, xLog, yLog, 'lines', 'Logistic fit')
    #addPlot(fig, xWithoutOutliers, yFitted, 'lines', 'Savgol filter')
    
    if plotHalfGaussian : 
        xPartialGauss, yPartialGauss = fitPartialGaussian(xWithoutOutliers, yWithoutOutliers, plotStep)
        addPlot(fig, xPartialGauss, yPartialGauss, 'lines', 'Half gaussian fit')

    fig.update_layout(
        title_text=title,
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

    return fig

def addMeanStddevPlots(fig, x, meanFit, stddevFit) :
    color="blue"
    lower = meanFit - stddevFit
    upper = meanFit + stddevFit
    colorBetweenLines(fig, x, meanFit, lower, color)
    colorBetweenLines(fig, x, meanFit, upper, color)
    addPlotColor(fig, x, meanFit, "lines", "Mean fit", color)
    #addPlotColor(fig, x, lower, "lines", "Mean - stddev", [color, 0.1],)
    #addPlotColor(fig, x, upper, "lines", "Mean + stddev", [color, 0.1])

def createFitPlotAligned(regionFit, name) : 
    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :
        xAligned, meanFit, medianFit, paramsMedianFit, stddevFit = regionFit.getMeanStddevAligned()
        addMeanStddevPlots(fig, xAligned, meanFit, stddevFit)
        addPlot(fig, xAligned, medianFit, "lines", "Median fit")
        addPlotColor(fig, xAligned, paramsMedianFit, "lines", "Median params fit", "red")
        #addPlot(fig, xAligned, meanFit, "lines", "Mean fit")
        #addPlot(fig, xAligned, meanFit - stddevFit, "lines", "Mean - stddev")
        #addPlot(fig, xAligned, meanFit + stddevFit, "lines", "Mean + stddev")

    
    fig.update_layout(
        title_text=name + " fit",
        xaxis_title='Semantic similarity',
        yaxis_title='Normalized firing rate',
    )

    return fig

def createFitPlot(regionFit, name) :

    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :

        meanFit, medianFit, stddevFit, meanParams, medianParams, paramsMedian = regionFit.getMeanMedianStddevFit()
        #addPlot(fig, regionFit.xFit, medianFit, "lines", "Median fit")
        addMeanStddevPlots(fig, regionFit.xFit, meanFit, stddevFit)
        #addPlotColor(fig, regionFit.xFit, meanParams, "lines", "Mean params", "green")
        addPlotColor(fig, regionFit.xFit, medianParams, "lines", "Median params", "red")
        ##addPlot(fig, regionFit.xFit, meanFit, "lines", "Mean fit")
        ##addPlot(fig, regionFit.xFit, meanFit - stddevFit, "lines", "Mean - stddev")
        ##addPlot(fig, regionFit.xFit, meanFit + stddevFit, "lines", "Mean + stddev")

        medianSlope = statistics.median(regionFit.steepestSlopes)
        x0 = paramsMedian[0]
        y0 = regionFit.funcParams([x0], paramsMedian)[0]
        y1 = min(medianParams) - 0.05
        y2 = max(medianParams) + 0.05
        y = [y1, y0, y2]
        x = (y - y0) / medianSlope + x0

        #x1 = max(x0-y0/medianSlope, 0) #max(max(x0-y0/medianSlope, min(medianParams)),0)
        #x2 = min((1-y0) / medianSlope + x0, 1) #min(min((1-y0) / medianSlope + x0, min(medianParams)),1)
        #x= [x1, x0, x2]
        #y = medianSlope * (x-x0) + y0

        addPlotColor(fig, x, y, "lines", "Median steepest slope: " + str(round(medianSlope,2)), "green")
        
        #for i in range(len(regions[site].logisticFitK)) : 
        #    logisticFitFig.add_trace(go.Scatter(
        #        x=xLogisticFit,
        #        y=fitLogisticFunc(xLogisticFit, regions[site].logisticFitX0[i], regions[site].logisticFitK[i], regions[site].logisticFitA[i], regions[site].logisticFitC[i]),
        #    ))
            
    fig.update_layout(
        title_text=name + " fit",
        xaxis_title='Semantic similarity',
        yaxis_title='Normalized firing rate',
    )
    return fig

def addOpacity(color, opacity) : 
    listColor = list(mcolors.to_rgba(color))
    listColor[3] = opacity
    return pcolors.convert_to_RGB_255(tuple(listColor))

def rgb_to_rgba(rgb_value, alpha):
    colorList = list(mcolors.to_rgba(rgb_value))
    return "rgba("+str(colorList[0])+","+str(colorList[1])+","+str(colorList[2])+","+str(alpha)+")" 

def colorBetweenLines(fig, x, y1, y2, color) :
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y1, y2[::-1]]),
        fill='toself',
        fillcolor=rgb_to_rgba(color, 0.2),
        line_color=rgb_to_rgba(color, 0.1),
        showlegend=False,
    ))

def addPlot(fig, x, y, mode, name) : 
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name
        ))

def addPlotColor(fig, x, y, mode, name, color) : 
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name, 
            marker=dict(color=color)
        ))
