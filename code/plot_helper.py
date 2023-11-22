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

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pcolors
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt



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
        if len(rasterInput.spikes) == 0 or len(rasterInput.spikes[trial]) == 0 :  
            fig.add_trace(
                go.Scatter(
                    x=[0, 0], 
                    y=[trial, trial + 1],
                    mode='lines',
                    opacity=0.0,
                    line_color='blue', 
                    line_width=linewidth
            ))
            continue
        spikesTrial = rasterInput.spikes[trial][0]
        if type(spikesTrial) == np.float64 : 
            spikesTrial = np.asarray([spikesTrial])
        spikes = spikesTrial[(np.where((spikesTrial <= rasterInput.maxX) & (spikesTrial > rasterInput.minX)))[0]]
        #spikes = spikesTrial[np.where(spikesTrial > rasterInput.minX)[0]]

        if len(spikes) == 0: 
            fig.add_trace(
                go.Scatter(
                    x=[0, 0], 
                    y=[trial, trial + 1],
                    mode='lines',
                    opacity=0.0,
                    line_color='blue', 
                    line_width=linewidth
            ))
        else : 
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

def create2DhemispherePlt(valuesSites, sitenames) : 
    left = []
    right = []
    sitename_r_squared = []
    for site in sitenames:
        if site.startswith("L") :
            site_general_name = site[1:]
            if "R" + site_general_name in sitenames : 
                left.append(statistics.mean(valuesSites["L" + site_general_name]))
                right.append(statistics.mean(valuesSites["R" + site_general_name]))
                sitename_r_squared.append(site_general_name)

    fig, ax = plt.subplots()
    ax.scatter(left, right)
    ax.set_aspect('equal')
    for i, txt in enumerate(sitename_r_squared):
        offset = 0.001
        ax.annotate(txt + " (" + str(len(valuesSites["L" + txt])) + "/" + str(len(valuesSites["R" + txt])) + ")", (left[i], right[i]+ offset))

    low = min(min(left), min(right)) - 0.005
    high = max(max(left), max(right)) + 0.03
    #corr = stats.spearmanr(left, right)

    #plt.title("corr: " + str(corr.statistic) + ", pvalue: " + str(corr.pvalue))
    plt.plot([low,high], [low,high], color = 'b')
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel("left hemisphere")
    plt.ylabel("right hemisphere")


def createHistPlt(x, inputBins, factorY=1.0, labelX="", labelY="", color="blue", pvalues=False) : 

    if len(inputBins) == 0 :
        print("WARNING: empty bins for histogram!")
        return
    bins = np.asarray(inputBins)
    if not pvalues : 
        bins = np.append(np.asarray(bins), np.inf)

    counts, bins = np.histogram(x, bins=bins)
    if pvalues : 
        xPlot=bins[1:] # because it is "less than"
        yPlot=counts.astype(float)*float(factorY)
    else : 
        xPlot = bins[:-1].astype(int)
        yPlot=counts.astype(float)*float(factorY)
        
    plot = sns.barplot(x=xPlot, y=yPlot, color=color) # .astype(int) TODO
    plot.set(xlabel=labelX, ylabel=labelY)

def createHist(x, inputBins, factorY, labelX, labelY, color="blue") : 
    counts, bins = np.histogram(x, bins=inputBins)
    fig = px.bar(x=bins[:-1], y=counts.astype(float)*float(factorY), labels={'x':labelX, 'y':labelY})
    fig.update_traces(marker_color=color)
    return fig

def createHistCompleteXTick(x, inputBins, factorY, labelX, labelY="count", color="blue") :
    hist = createHist(x, inputBins, factorY, labelX, labelY, color)
    return hist.update_xaxes(dtick=1)

def createCorrelationPlot(sitename, correlation) : 
    return go.Box(
        y=correlation, 
        name=sitename + " (" + str(len(correlation)) + ")",
        boxpoints='all',
        #boxmean='sd'
    )

def createStdErrorMeanNamesPlot(values, xNames, title, boxpoints='all') :
    fig = go.Figure()

    for i in range(len(values)) : 
        if len(values[i]) == 0 : 
            values[i] = [0]

    #sems = [sem(values[i]) for i in range(len(values))] 
    #y = [statistics.mean(values[i]) for i in range(len(values))] 

    #fig.add_trace(
    #    go.Scatter(
    #        name=xNames,
    #        y=y, 
    #        error_y=dict(type='data', array=sems, visible=True)
   #     ))
    for i in range(len(xNames)) : 
        #stddev = statistics.stdev(values[i])
        fig.add_trace(go.Scatter(
            y=[statistics.mean(values[i])],
            name=xNames[i],
            error_y=dict(type='data', array=[sem(values[i])], visible=True)
            #boxpoints=boxpoints,
            #boxmean='sd'
            #marker_color = 'blue'
        ))

    fig.update_layout(
        title_text=title, 
        showlegend=False
    )
    return fig

def createBoxPlot(values, xNames, title, boxpoints='all') :
    fig = go.Figure()

    for i in range(len(xNames)) : 
        #stddev = statistics.stdev(values[i])
        fig.add_trace(go.Box(
            y=values[i],
            name=xNames[i],
            #boxpoints=boxpoints,
            #boxmean='sd'
            marker_color = 'blue'
        ))

    fig.update_layout(
        title_text=title, 
        showlegend=False
    )
    return fig

def createStdErrorMeanPlt(x, data, title, yLabel, ylim = []) :
    for i in range(len(data)) : 
        if len(data[i]) == 0 : 
            data[i] = [0]

    sems = [sem(data[i]) for i in range(len(data))] 
    y = [statistics.mean(data[i]) for i in range(len(data))] 
    plt.errorbar(x, y, sems, fmt='o', capsize=7)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    plt.ylabel(yLabel)
    plt.title(title)

    if len(ylim) > 0 : 
        plt.ylim(ylim)


def createStdErrorMeanPlot(x, data, title, yLabel, minZero=True) : 
    fig = go.Figure()

    for i in range(len(data)) : 
        if len(data[i]) == 0 : 
            data[i] = [0]

    sems = [sem(data[i]) for i in range(len(data))] 
    y = [statistics.mean(data[i]) for i in range(len(data))] 

    fig.add_trace(
        go.Scatter(
            x=x, 
            y=y, 
            mode='markers',
            error_y=dict(type='data', array=sems, visible=True)
        ))
    
    fig.update_layout(
        title_text=title, ###
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False,
    )

    if minZero: 
        fig.add_hline(y=0.0, line_color="rgba(0,0,0,0)") # hack: set transparent line at y = 0 to have lower limit of y-axis

    return fig

def createStepBoxPlot(similaritiesArray, title, yLabel="", alpha=0.001, boxpoints='all', addLog=True, addPartialGaussian=True, stepBox=0.1) :   
    
    x = np.asarray(similaritiesArray.similarities)
    values = np.asarray(similaritiesArray.values)

    yFit=[]
    xFit=[]
    
    lowerfences = []
    upperfences = []
    means = []
    sems = []
    q1 = []
    q3 = []

    fig = go.Figure()

    xPlot = np.arange(-0.0001,1.0001,stepBox)
    for step in xPlot : # due to rounding errors
        indices = np.where((x >= step) & (x < step+stepBox))[0]
        if len(indices) == 0 : 
            continue
        y = values[indices]
        if len(y) >= 2 :
            mean = statistics.mean(y)
            stddev = statistics.stdev(y)
            means.append(mean)
            sems.append(sem(y))
            lowerfences.append(mean-stddev)
            upperfences.append(mean+stddev)
            q1.append(np.percentile(y, 25)) #(mean-error)
            q3.append(np.percentile(y, 75))
    
            
            #go.Box(
            #x0=step,
            #y=values[indices],
            #name="{:.2f} ({})".format(step, y),
            #boxpoints=boxpoints,
            #marker_color = 'blue'
            ##boxmean='sd'
        #))

    
    fig.add_trace(
        go.Scatter(
            x=xPlot,
            y=means,
            error_y=dict(type='data', array=sems, visible=True)
        ))

    #fig.update_traces(q1=q1, median=means,
    #    q3=q3, lowerfence=lowerfences,
    #    upperfence=upperfences )
    #fig.update_traces(q1=q1, q3=q3, median=means)


        #if i < len(values)-1 : 
        #    t_value, p_value = stats.ttest_ind(values[i], values[i+1]) 

        #    if p_value <= alpha : 
        #        print(title + ': p_value=%.8f' % p_value,
        #            'for value=%i ' % i)

    xFit = x
    yFit = values

    if addLog and len(yFit) > 0: 
        xFitted, yFitted, rSquared = fitLog(xFit, yFit)#, x[1]-x[0])
        #fig.add_trace(
        #    go.Scatter(
        #        x=xFitted,
        #        #xaxis='x',
        #        y=yFitted,
        #        mode='lines',
        #    ))
        try : 
            rSquared = round(rSquared, 4)
        except : 
            rSquared = 0.0
        title = title + "." + "<br>" + "Rsquared for logistic fit: " + str(rSquared)
        addPlot(fig, xFitted, yFitted, 'lines', 'Logistic fit')
        
    if addPartialGaussian and len(yFit) > 0: 
        xFitted, yFitted, rSquared = fitPartialGaussian(xFit, yFit)#, x[1]-x[0])
        try : 
            rSquared = round(rSquared, 4)
        except : 
            rSquared = 0.0
        title = title + "." + "<br>" + "R squared for Gaussian fit: " + str(rSquared) 
        addPlot(fig, xFitted, yFitted, 'lines', 'Gaussian fit')

        
    spearman = stats.spearmanr(x, values) 
    #not math.isnan(spearman.correlation) : 
    try : 
        spearmanString = str(round(spearman.correlation, 3))
    except : 
        spearmanString = "could not be computed."
    title = title + "." + "<br>" + "Spearman rho: " + spearmanString

    fig.update_layout(
        width=600,
        height=500,
        title_text=title, ###
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False,
    )

    fig.update_xaxes(
        tickmode = 'array',
        tickvals = np.arange(0,1.05,0.1)
    )
    
    fig.update_yaxes(automargin=True)


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
        xGauss, yGauss, rSquaredGauss = fitGauss(xWithoutOutliers, yWithoutOutliers, plotStep)

    try : 
        yFitted = savgol_filter(yWithoutOutliers, 15, 3) # window size 51, polynomial order 3
    except Exception : 
        #print("WARNING: Error applying filter")
        yFitted = yWithoutOutliers

    xLog, yLog, rSquared = fitLog(xWithoutOutliers, yWithoutOutliers, plotStep)

    fig = go.Figure()

    addPlot(fig, xWithoutOutliers, yWithoutOutliers, 'markers', 'Data')
    #addPlot(fig, xWithoutOutliers, smooth(yWithoutOutliers, 5), 'lines', 'Smoothed 5 point avg')
    #addPlot(fig, xGauss, yGauss, 'lines', 'Gaussian fit')
    #addPlot(fig, xLog, yLog, 'lines', 'Logistic fit')
    #addPlot(fig, xWithoutOutliers, yFitted, 'lines', 'Savgol filter')
    
    if plotHalfGaussian : 
        xPartialGauss, yPartialGauss, rSquared = fitPartialGaussian(xWithoutOutliers, yWithoutOutliers, plotStep)
        addPlot(fig, xPartialGauss, yPartialGauss, 'lines', 'Half gaussian fit. R squared = ' + str(rSquared))

    fig.update_layout(
        title_text=title,
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel, 
        showlegend=False
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

def addMeanStddevPlots(fig, x, meanFit, error, title) :
    color="blue"
    lower = meanFit - error
    upper = meanFit + error
    colorBetweenLines(fig, x, meanFit, lower, color)
    colorBetweenLines(fig, x, meanFit, upper, color)
    addPlotColor(fig, x, meanFit, "lines", title, color)
    #addPlotColor(fig, x, lower, "lines", "Mean - stddev", [color, 0.1],)
    #addPlotColor(fig, x, upper, "lines", "Mean + stddev", [color, 0.1])

def createFitPlotAligned(regionFit, name) : 
    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :
        xAligned, meanFit, medianFit, paramsMedianFit, stddevFit = regionFit.getMeanStddevAligned()
        addMeanStddevPlots(fig, xAligned, meanFit, stddevFit, "Mean over fits for all neurons")
        addPlot(fig, xAligned, medianFit, "lines", "Median over fits for all neurons")
        addPlotColor(fig, xAligned, paramsMedianFit, "lines", "Median of fitted parameters", "red")
        #addPlot(fig, xAligned, meanFit, "lines", "Mean fit")
        #addPlot(fig, xAligned, meanFit - stddevFit, "lines", "Mean - stddev")
        #addPlot(fig, xAligned, meanFit + stddevFit, "lines", "Mean + stddev")

    
    fig.update_layout(
        title_text=name + " fit",
        xaxis_title='Semantic similarity',
        yaxis_title='Normalized firing rate',
    )

    return fig

def getLineWithSlope(slope, params, paramsY, regionFit) :
    
    x0 = params[0]
    y0 = regionFit.funcParams([x0], params)[0]
    y1 = min(paramsY) - 0.05#, y0 - slope * x0)
    y2 = max(paramsY) + 0.05#, (1-y0)/slope - x0)
    y = [y1, y0, y2]
    x = (y - y0) / slope + x0
    return x, y

def createFitPlot(regionFit, name) :

    fig = go.Figure()
    if len(regionFit.yFit[0]) > 0 :

        meanFit, medianFit, stddevFit, meanParams, medianParams, paramsMedian = regionFit.getMeanMedianStddevFit()

        medianSlope = statistics.median(regionFit.steepestSlopes)
        q1 = np.percentile(regionFit.steepestSlopes, 25)
        q3 = np.percentile(regionFit.steepestSlopes, 75)
        semSlope = sem(regionFit.steepestSlopes)
        #x0 = paramsMedian[0]
        #y0 = regionFit.funcParams([x0], paramsMedian)[0]
        #y1 = min(medianParams) - 0.05
        #y2 = max(medianParams) + 0.05
        #y = [y1, y0, y2]
        #x = (y - y0) / medianSlope + x0

        xMedianSlope, yMedianSlope = getLineWithSlope(medianSlope, paramsMedian, medianParams, regionFit)
        xSlopeLower, ySlopeLower = getLineWithSlope(q1, paramsMedian, medianParams, regionFit)
        xSlopeUpper, ySlopeUpper = getLineWithSlope(q3, paramsMedian, medianParams, regionFit)

        #x1 = max(x0-y0/medianSlope, 0) #max(max(x0-y0/medianSlope, min(medianParams)),0)
        #x2 = min((1-y0) / medianSlope + x0, 1) #min(min((1-y0) / medianSlope + x0, min(medianParams)),1)
        #x= [x1, x0, x2]
        #y = medianSlope * (x-x0) + y0
        
        #addPlot(fig, regionFit.xFit, medianFit, "lines", "Median fit")
        addMeanStddevPlots(fig, regionFit.xFit, meanFit, stddevFit, "Mean over fits for all neurons")
        #addPlotColor(fig, regionFit.xFit, meanParams, "lines", "Mean params", "green")
        addPlotColor(fig, regionFit.xFit, medianParams, "lines", "Median of fitted parameters", "red")
        ##addPlot(fig, regionFit.xFit, meanFit, "lines", "Mean fit")
        ##addPlot(fig, regionFit.xFit, meanFit - stddevFit, "lines", "Mean - stddev")
        ##addPlot(fig, regionFit.xFit, meanFit + stddevFit, "lines", "Mean + stddev")

        addPlotColor(fig, xMedianSlope, yMedianSlope, "lines", "Median steepest slope: " + str(round(medianSlope,2)), "indigo")
        #colorBetweenLines(fig, xSlopeUpper, yMedianSlope, ySlopeUpper, "indigo")
        #colorBetweenLines(fig, xSlopeLower, yMedianSlope, ySlopeLower, "indigo")
        
        #for i in range(len(regions[site].logisticFitK)) : 
        #    logisticFitFig.add_trace(go.Scatter(
        #        x=xLogisticFit,
        #        y=fitLogisticFunc(xLogisticFit, regions[site].logisticFitX0[i], regions[site].logisticFitK[i], regions[site].logisticFitA[i], regions[site].logisticFitC[i]),
        #    ))
            
    fig.update_layout(
        title_text=name + " fit for all responsive units",
        xaxis_title='Semantic similarity',
        yaxis_title='Normalized firing rate',
        xaxis = dict(tickvals = np.arange(0, 1, 0.1),)
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
            name=name, 
            #hoverinfo='text'
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
