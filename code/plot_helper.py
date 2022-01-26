#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/01/24 13:41:48
@Author  :   Katharina Karkowski 
"""

import numpy as np

from typing import List
from dataclasses import field
from dataclasses import dataclass

import plotly.graph_objects as go

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

