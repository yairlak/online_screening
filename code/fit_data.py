import math
from re import I
import numpy as np
import scipy
import statistics
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import List

@dataclass
class Fitter :
    stepSize : float = 0.1
    numSteps : int = 0 # is initialized
    func = lambda : fitLogisticFunc
    funcParams = lambda : fitLogisticFuncParams
    p0 : List = field (default_factory=lambda: [])
    bounds : List = field (default_factory=lambda: [[]])
    xFit : List = field (default_factory=lambda: [])
    yFit : List = field (default_factory=lambda: [[]])

    paramsNames : List = field (default_factory=lambda: [])
    params : List = field(default_factory=lambda: [[]])
    rSquared : List = field (default_factory=lambda: [])

    def getFitter(func, funcParams, paramsNames, p0, bounds, stepSize=0.02) : 
        newFitter = Fitter()
        newFitter.stepSize = stepSize
        newFitter.p0 = p0
        newFitter.bounds = bounds
        newFitter.func = func
        newFitter.funcParams = funcParams
        newFitter.paramsNames = paramsNames
        newFitter.numSteps = math.ceil(1.0 / stepSize) + 1
        newFitter.xFit = np.arange(0, stepSize*newFitter.numSteps, stepSize)
        newFitter.yFit = [[] for i in range(newFitter.numSteps)]
        newFitter.params = [[] for i in range(len(paramsNames))]

        return newFitter

    def calculateRSquared(self, y, yFitted) : 
        ssRes = np.sum((y - yFitted)**2)
        ssTot = np.sum((y - statistics.mean(y))**2)
        return 1 - ssRes/ssTot

    def addFit(self, xToFit, yToFit) :
        try : 
            popt, pcov = curve_fit(self.func, xToFit, yToFit, p0=self.p0, bounds=self.bounds)
        except Exception as e : 
            print("WARNING: No logistic curve fitting found: " + str(e))
            return -1
        
        rSquared = self.calculateRSquared(yToFit, self.funcParams(xToFit, popt))
        self.rSquared.append(rSquared)

        for i in range(len(popt)):
            self.params[i].append(popt[i])
        
        #xFitTmp = self.xFit - popt[0]
        #popt[0] = 0

        yFit = self.funcParams(self.xFit, popt)
        for i in range(len(yFit)) :
            self.yFit[i].append(yFit[i])

        #if yLogisticFit[0] >= yLogisticFit[-1] : 
        #    print("--- BAD fitting of logistic function. K: " + str(popt[1]))
        #else : 
        #    print("--- GOOD fitting of logistic function. K: " + str(popt[1]))
        
        return rSquared
    
    def getMeanMedianStddevFit(self) :

        if len(self.yFit) == 0 : 
            return np.zeros(self.numSteps), np.zeros(self.numSteps), np.zeros(self.numSteps)
        
        #paramsMean, paramsMedian = np.zeros(len(self.params))
        #for i in range(len(self.params)) : 
        #    if len(self.params[i]) > 0 : 

        paramsMean = np.array([statistics.mean(self.params[i]) for i in range(len(self.params))])
        paramsMedian = np.array([statistics.median(self.params[i]) for i in range(len(self.params))])
        meanParams = self.funcParams(self.xFit, paramsMean)
        medianParams = self.funcParams(self.xFit, paramsMedian)

        meanFit = np.array([statistics.mean(self.yFit[i]) for i in range(self.numSteps)])
        medianFit = np.array([statistics.median(self.yFit[i]) for i in range(self.numSteps)])
        
        if len(self.yFit[0]) == 1 : 
            return meanFit, medianFit, np.zeros(self.numSteps), meanParams, medianParams

        stddevFit = np.array([statistics.stdev(self.yFit[i]) for i in range(self.numSteps)])

        return meanFit, medianFit, stddevFit, meanParams, medianParams

    def append(self, input) :
        for i in range(len(input.params)) :
            self.params[i].extend(input.params[i])

        for i in range(len(input.yFit)) :
            self.yFit[i].extend(input.yFit[i])

        self.rSquared.extend(input.rSquared)



def mirrorInputAtMax(x, y) : 
    maxIndex = np.amax(np.where(y > 0)) + 1

    xOut = x[:maxIndex]
    xOut = np.append(xOut, xOut[-1]-xOut[-2] + xOut[-1])
    for i in range(1,maxIndex) : 
        xOut = np.append(xOut, x[maxIndex-i]-x[maxIndex-i-1] + xOut[-1])

    yPart = y[:maxIndex]
    yOut = np.concatenate((yPart, yPart[::-1]))

    return xOut, yOut, maxIndex


def smooth(y, numPoints):
    if len(y) == 0 : 
        return y 
    else : 
        return np.convolve(y, np.ones(numPoints)/numPoints, mode='same')


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

    return yGaussPart 

def fitGauss(x, y, xPlot=np.arange(0.0, 1-0.01, 0.01)) : 
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    try : 
        popt,pcov = curve_fit(Gauss, x, y, p0=[mean, max(y), 0, sigma])
    except Exception: 
        print("WARNING: Error fitting gauss")
        return [], []

    yGauss = Gauss(xPlot, *popt)

    return x, yGauss

def Gauss(x, x0, a, b, sigma):
    return b + a * np.exp(-(x - x0)**2 / (2 * sigma**2))


#def partialGauss(x, x0, a, sigma) : 
#    xMirrored = mirrorInputAtMax(x, y)
#    yMirrored = Gauss(xMirrored, x0, a, sigma)
#    y = yMirrored[:maxIndex]

def halfGaussParams(x, params) :
    return halfGauss(x, params[0], params[1], params[2], params[3])

def halfGauss(x, x0, a, b, sigma) : 
    xDoubled = np.concatenate((x, x + max(x) + (x[-1]-x[-2]) - x[0]))
    yGauss = Gauss(xDoubled, max(x), a, b, sigma)
    yGauss = yGauss[:int(len(xDoubled) / 2)]

    #xMirrored = np.concatenate((x, np.flip(x)))
    #yGauss = Gauss(xMirrored, 1, a, b, sigma)
    #yGauss = yGauss[:int(len(xMirrored) / 2)]
    return yGauss

def fitLogisticFuncParams(x, params) : 
    return fitLogisticFunc(x, params[0], params[1], params[2], params[3])

def fitLogisticFunc(x, x0, k, a, c) :
    return a + (c - a) * scipy.special.expit((x-x0)*(-k))

def fitStepParams(x, params) :
    return fitStep(x, params[0], params[1], params[2])

def fitStep(x, x0, a, b) :
    y = np.zeros(len(x))
    y[np.where(x <= x0)[0]] = a
    y[np.where(x > x0)[0]] = b
    return fitLogisticFunc(x, x0, 1000, a, b)  #a * (np.heaviside(x-x0, 0) + b) #a * (np.sign(x-x0) + b)

