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
    )

def createBoxPlot(values, xNames, title, boxpoints='all') :
    fig = go.Figure()
    for i in range(len(xNames)) : 
        fig.add_trace(go.Box(
            y=values[i],
            name=xNames[i],
            boxpoints=boxpoints,
        ))
    fig.update_layout(
        title_text=title,
        showlegend=False
    )
    return fig

def createStepBoxPlot(x, values, title, yLabel="", alpha=0.001, boxpoints=False, addFit=True) :   
    
    yFit=[]
    xFit=[]
    
    fig = go.Figure()
    for i in range(len(values)) : 
        if(len(values[i]) >= 1) : 
            fig.add_trace(go.Box(
                #xaxis='x',
                x0=x[i],
                y=values[i],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
                #boxpoints='all',
            ))
            xFit.append(x[i])
            yFit.append(statistics.median(values[i]))
        else : 
            fig.add_trace(go.Box(
                xaxis='x',
                x0=x[i],
                y=[0.0],
                name="{:.2f} ({})".format(x[i], len(values[i])),
                boxpoints=boxpoints,
            ))

        if i < len(values)-1 : 
            t_value, p_value = stats.ttest_ind(values[i], values[i+1]) 

            if p_value <= alpha : 
                print(title + ': p_value=%.8f' % p_value,
                    'for value=%i ' % i)

    if addFit and len(yFit) > 0: 
        xFitted, yFitted = fitLog(xFit, yFit, x[1]-x[0])
        fig.add_trace(
            go.Scatter(
                x=xFitted,
                #xaxis='x',
                y=yFitted,
                mode='lines',
            ))
        #addPlot(fig, x, yFitted, 'lines', 'Logistic fit')

    fig.update_layout(
        title_text=title,
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False,
    )
    #fig.update_xaxes(
        #range=[0,1]
        #tickvals=x
    #)

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
        print("WARNING: Error applying filter")
        yFitted = yWithoutOutliers

    xLog, yLog = fitLog(xWithoutOutliers, yWithoutOutliers, plotStep)

    fig = go.Figure()

    addPlot(fig, xWithoutOutliers, yWithoutOutliers, 'markers', 'Data')
    #addPlot(fig, xWithoutOutliers, smooth(yWithoutOutliers, 5), 'lines', 'Smoothed 5 point avg')
    #addPlot(fig, xGauss, yGauss, 'lines', 'Gaussian fit')
    addPlot(fig, xLog, yLog, 'lines', 'Logistic fit')
    addPlot(fig, xWithoutOutliers, yFitted, 'lines', 'Savgol filter')
    
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

def createFitPlotAligned(regionFit, name) : 
    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :
        xAligned, meanFit, medianFit, stddevFit = regionFit.getMeanStddevAligned()
        addPlot(fig, xAligned, meanFit, "lines", "Mean fit")
        addPlot(fig, xAligned, medianFit, "lines", "Median fit")
        addPlot(fig, xAligned, meanFit - stddevFit, "lines", "Mean - stddev")
        addPlot(fig, xAligned, meanFit + stddevFit, "lines", "Mean + stddev")

    
    fig.update_layout(
        title_text=name + " fit",
        xaxis_title='Semantic similarity',
        yaxis_title='Firing rate',
    )

    return fig

def createFitPlot(regionFit, name) :

    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :

        meanFit, medianFit, stddevFit, meanParams, medianParams = regionFit.getMeanMedianStddevFit()
        addPlot(fig, regionFit.xFit, meanFit, "lines", "Mean fit")
        #addPlot(fig, regionFit.xFit, medianFit, "lines", "Median fit")
        addPlot(fig, regionFit.xFit, meanParams, "lines", "Mean params")
        addPlot(fig, regionFit.xFit, medianParams, "lines", "Median params")
        addPlot(fig, regionFit.xFit, meanFit - stddevFit, "lines", "Mean - stddev")
        addPlot(fig, regionFit.xFit, meanFit + stddevFit, "lines", "Mean + stddev")

        
        #for i in range(len(regions[site].logisticFitK)) : 
        #    logisticFitFig.add_trace(go.Scatter(
        #        x=xLogisticFit,
        #        y=fitLogisticFunc(xLogisticFit, regions[site].logisticFitX0[i], regions[site].logisticFitK[i], regions[site].logisticFitA[i], regions[site].logisticFitC[i]),
        #    ))
            
    fig.update_layout(
        title_text=name + " fit",
        xaxis_title='Semantic similarity',
        yaxis_title='Firing rate',
    )
    return fig

def addPlot(fig, x, y, mode, name) : 
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name
        ))
