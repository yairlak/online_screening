#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/24 13:41:48
@Author  :   Katharina Karkowski 
"""

import os

import numpy as np
import pandas as pd
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


def adjustFontSize(snsFig=None) :
    labelsize=20
    
    font = dict(weight='normal', size=labelsize)
    plt.rc('font', **font)
    #if not snsFig == None : 
    #    snsFig.set_yticklabels(snsFig.get_yticks(), size=labelsize)
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

def save_plt(file) : 
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()


def plotRaster(rasterInput, linewidth=1) : 

    upperBound = 1000
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
        

    fig.add_trace(
        go.Scatter(x=[upperBound, upperBound], 
            y=[0, numTrials],
            mode='lines',
            #rasterGrid['layout'][ax]['color']='lightskyblue'#(0.5,0.5,0.5,1.0)
            line_color='lightskyblue', 
            line_dash='dash',
            #line = dict(color='royalblue', width=4, dash='dash')
            line_width=1.2*linewidth))
    
    fig.add_trace(
        go.Scatter(x=[0, 0], 
            y=[0, numTrials],
            mode='lines',
            line_color='lightskyblue', 
            line_dash='dash',
            line_width=1.2*linewidth))

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
        #paper_bgcolor='lightgrey',
        #plot_bgcolor='white',

        xaxis_range=[rasterInput.minX,rasterInput.maxX],
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0, upperBound],
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

    if len(sitename_r_squared) == 0: 
        print("WARNING! 2D hemisphere plots could not be created. This happens when the hemispheres are collapsed.")
        return

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

def createHistPlt(x, inputBins, factorY=1.0, labelX="", labelY="", color="blue", pvalues=False, title="") : 

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
    plot.set(xlabel=labelX, ylabel=labelY, title=title)

def createHist(x, inputBins, factorY=1.0, labelX="", labelY="", color="blue", title="") : 
    counts, bins = np.histogram(x, bins=inputBins)
    fig = px.bar(x=bins[:-1], y=counts.astype(float)*float(factorY), labels={'x':labelX, 'y':labelY})
    fig.update_traces(marker_color=color)#, font=dict(size=24))
    if len(title) > 0 : 
        fig.update_layout(title=title)
    return fig

def createHistCompleteXTick(x, inputBins, factorY, labelX, labelY="count", color="blue") :
    hist = createHist(x, inputBins, factorY, labelX, labelY, color)
    return hist.update_xaxes(dtick=1)

def createHistCombined(x, y1, y2, factorY, yLabel1, yLabel2) : 
    counts1, bins1 = np.histogram(y1, bins=x)
    counts2, bins2 = np.histogram(y2, bins=x)
    counts1 = counts1.astype(float)*float(factorY)
    counts2 = counts2.astype(float)*float(factorY)

    xLine = [-0.015]
    yLine1 = [0, 0]
    yLine2 = [0, 0]

    for i in range(len(counts1)) : 
        xLine.append(bins1[i])
        xLine.append(bins1[i])
        yLine1.append(counts1[i]) 
        yLine1.append(counts1[i])
        yLine2.append(counts2[i]) 
        yLine2.append(counts2[i])
    
    yLine1.append(0)
    yLine1.append(0)
    yLine1.append(0)
    yLine2.append(0)
    yLine2.append(0)
    yLine2.append(0)
    xLine.append(max(xLine)+0.01)
    xLine.append(max(xLine))
    xLine.append(max(xLine)+0.01)
    #xLine.append(1.1)
    xticks = np.arange(stop=1.1, step=0.2).round(1)

    fig = go.Figure(data=[
        go.Scatter(name=yLabel1, x=xLine, y=yLine1, yaxis='y', mode='lines', offsetgroup=1, line_color='red'),
        go.Scatter(name=yLabel2, x=xLine, y=yLine2, yaxis='y2', mode='lines', offsetgroup=2, line_color='blue'),
        #go.Scatter(x=[0,1], y=[-0.2,-0.2], line_color='white')
        #go.Bar(name=yLabel1, x=bins1[:-1], y=counts1.astype(float)*float(factorY), yaxis='y', offsetgroup=1),
        #go.Bar(name=yLabel2, x=bins1[:-1], y=counts2.astype(float)*float(factorY), yaxis='y2', offsetgroup=2)
    ],layout={
        'yaxis': dict(title=yLabel1, color='red', showline=True, mirror=True, linecolor='grey', showgrid=True, gridcolor='lightgray'),
        'yaxis2': dict(title=yLabel2, color='blue', showline=True, mirror=True, linecolor='grey', showgrid=False, overlaying='y', side='right'),
        'xaxis': dict(showgrid=False, showline=True, mirror=True, linecolor='grey', tickvals=xticks, ticktext=xticks),
        'xaxis_range': [-0.015,1.02],
        'autosize':False,
        'width':1000,
        'height':600,
        #'yaxis_range': [-0.2,max(yLine1)+max(yLine1)*0.02],
        #'yaxis2_range': [-0.2,max(yLine2)+max(yLine2)*0.02],
        'barmode': 'group',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'legend': dict(y=0.8, xref="container", yref="container")})
    
    #fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')#,
        #plot_bgcolor='rgba(0,0,0,0)', 
    

    return updateFigure(fig)

def createGroupedHistNew(values, sites, bins, title="", xLabel="") : 

    newBins = np.append(np.round(np.asarray(bins), 2), np.inf)
    uniqueSites = np.unique(sites)
    hist_df = pd.DataFrame(columns=uniqueSites)
    hist_df['bins'] = newBins[:-1]
    hist_df.set_index('bins', inplace=True)

    for site in uniqueSites : 
        counts, bins = np.histogram(np.asarray(values)[np.where(np.asarray(sites)==site)[0]], bins=newBins)
        hist_df[site] = counts

    hist_df.plot(kind="bar", stacked=False)
    plt.ylabel("Count")
    plt.xlabel(xLabel)
    plt.title(title)

def createAllHistsSites(data_dict, path, filename, title="", xLabel="") : 
    adjustFontSize()
    #df_copy = data_dict.copy()
    #data_dict["all"] = data_dict.sum(axis=1)

    hist_array = []
    hist_array_all = np.array([])
    maxTotal = 0
    maxSizeTotal = max([len(data_dict[site]) for site in data_dict.keys()])

    for site in data_dict.keys() : 
        hist_array.append(data_dict[site])
        site_data_filled = np.hstack([data_dict[site], np.zeros(maxSizeTotal- len(data_dict[site]))])
        hist_array_all = np.concatenate((hist_array_all, site_data_filled), axis=None)
        maxTotal = max(maxTotal, max(data_dict[site])+1)
        createHistPlt(data_dict[site], range(max(data_dict[site])+1), title=title, labelX=xLabel, labelY="Count")
        save_plt(path + os.sep + site + "_" + filename)


    createHistPlt(hist_array_all, range(int(max(hist_array_all)+1)), title=title, labelX=xLabel, labelY="Count")
    save_plt(path + os.sep + "all_" + filename)
    createGroupedHistNew(hist_array, np.array(list(data_dict.keys())), range(maxTotal+1), title=title, xLabel=xLabel)
    save_plt(path + os.sep + "all_" + filename + "_grouped")

def createGroupedHist(values, sites, binsStart, binsStop, binsStep=1, title="", xLabel="") : 
    values = np.asarray(values)
    sites = np.array(sites)

    fig = go.Figure()
    for site in np.unique(sites) : 
        fig.add_trace(
            createHist(values[np.where(sites==site)[0]], np.arange(binsStart, binsStop+binsStep, binsStep))
            ##go.Histogram(
            ##x=values[np.where(sites==site)[0]],
            #histnorm='percent',
            ##name=site, # name used in legend and hover labels
            ##xbins=dict( start=binsStart, end=binsStop, size=binsStep),
            #marker_color='#EB89B5',
            #opacity=0.75
            ##)
        )
        #fig.add_trace(createHist(values[np.where(sites==site)[0]], np.append(np.arange(binsStart, binsStop, binsStep), np.inf)))

    fig.update_layout(
        #font=dict(size=20), 
        title_text=title,
        xaxis_title_text=xLabel, # xaxis label
        yaxis_title_text='Count', # yaxis label
    #    bargap=0.2, # gap between bars of adjacent location coordinates
    #    bargroupgap=0.1 # gap between bars of the same location coordinates
        #xaxis = dict(
        #    tickvals = np.arange(binsStart, binsStop+binsStep, binsStep),
        #    ticktext = np.arange(binsStart, binsStop+binsStep, binsStep)
            #tickmode = 'linear',
        ##    tick0 = binsStart,
            #dtick = binsStep
        #)
    )

    return updateFigure(fig)


def updateFigure(fig) : 
    fig.update_layout(
        font=dict(size=20), 
        #zeroline = False,
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)', 
        )
    return fig


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

def createStdErrorMeanPlt(x, data, title, yLabel, ylim = [], sort=False, horizontal=False) :
    for i in range(len(data)) : 
        if len(data[i]) == 0 : 
            data[i] = [0]

    y = [statistics.mean(data[i]) for i in range(len(data))] 
    sems = [sem(data[i]) for i in range(len(data))] 

    if sort : 
        sortIndices = np.argsort(y)#[::-1] #np.asarray(y).sort(order="descending")
        y = np.asarray(y)[sortIndices]
        sems = np.asarray(sems)[sortIndices]
        x = np.asarray(x)[sortIndices]

    if horizontal : 
        plt.errorbar(y, x, xerr=sems, fmt='o', capsize=7)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    else : 
        plt.errorbar(x, y, sems, fmt='o', capsize=7)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    plt.ylabel(yLabel)
    plt.title(title)

    if len(ylim) > 0 : 
        plt.ylim(ylim)

def createStdErrorMeanPltCompare(x, data1, data2, title, label1="", label2="", yLabel="", ylim = [], sort=False, horizontal=False) :
    for i in range(len(data1)) : 
        if len(data1[i]) == 0 : 
            data1[i] = [0]

    for i in range(len(data2)) : 
        if len(data2[i]) == 0 : 
            data2[i] = [0]

    y1 = [statistics.mean(data1[i]) for i in range(len(data1))] 
    sems1 = [sem(data1[i]) for i in range(len(data1))] 
    
    y2 = [statistics.mean(data2[i]) for i in range(len(data2))] 
    sems2 = [sem(data2[i]) for i in range(len(data2))] 

    if len(y1) != len(y2) : 
        print("ERROR! Data size does not match for stderrormean plot to compare!")
        return

    if sort : 
        sortIndices = np.argsort(y1)#[::-1] #np.asarray(y).sort(order="descending")
        y1 = np.asarray(y1)[sortIndices]
        sems1 = np.asarray(sems1)[sortIndices]
        x1 = np.asarray(x1)[sortIndices]

        y2 = np.asarray(y2)[sortIndices]
        sems2 = np.asarray(sems2)[sortIndices]

    if horizontal : 
        plt.errorbar(y1, x, xerr=sems1, fmt='o', capsize=7, label=label1)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
        plt.errorbar(y2, x, xerr=sems2, fmt='o', capsize=7, mfc='green', mec='green', label=label2)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    else : 
        plt.errorbar(x, y1, sems1, fmt='o', capsize=7, label=label1)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
        plt.errorbar(x, y2, sems2, fmt='o', capsize=7, mfc='green', mec='green', ecolor='green', label=label2)#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend()

    if len(ylim) > 0 : 
        plt.ylim(ylim)

def createStdErrorMeanPlot(x, data, title, yLabel="", minZero=True, xLabel="Semantic similarity") : 
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
        xaxis_title=xLabel,
        yaxis_title=yLabel,
        showlegend=False,
    )

    if minZero: 
        fig.add_hline(y=0.0, line_color="rgba(0,0,0,0)") # hack: set transparent line at y = 0 to have lower limit of y-axis

    return fig

def createStepBoxPlot(similaritiesArray, title, yLabel="", alpha=0.001, addLog=True, addPartialGaussian=False, stepBox=0.1, max=1.0) :   
    
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

    xPlot = np.arange(-0.0001, max+0.0001, stepBox)
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
        else : 
            means.append(0)
            sems.append(0)
            lowerfences.append(0)
            upperfences.append(0)
            q1.append(0)
            q3.append(0)
            
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
        #title_text=title, ###
        xaxis_title='Semantic similarity',
        yaxis_title=yLabel,
        showlegend=False,
    )

    fig.update_xaxes(
        tickmode = 'array',
        #tickvals = np.arange(0,1.05,0.1)
        tickvals = np.arange(0,max+0.05,stepBox)
    )
    
    fig.update_yaxes(automargin=True)

    return updateFigure(fig)

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
        #sortedSlopes = sorted(regionFit.steepestSlopes)
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
