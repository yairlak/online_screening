from cmath import isnan
import math
#from re import I
import statsmodels.api as sm
import numpy as np
import scipy
import statistics
from scipy.stats import sem # chisquare, kstest, sem
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import List
from utils import get_indexed_array

@dataclass
class Fitter :
    stepSize : float = 0.01
    numSteps : int = 0 # is initialized
    stepSlope : float = 0.01
    func = lambda : logFunc
    funcParams = lambda : logFuncParams
    p0 : List = field (default_factory=lambda: [])
    bounds : List = field (default_factory=lambda: [[]])
    xFit : List = field (default_factory=lambda: [])
    yFit : List = field (default_factory=lambda: [[]])
    x : List = field (default_factory=lambda: [[]])
    y : List = field (default_factory=lambda: [[]])
    xNoFit : List = field (default_factory=lambda: [[]])
    yNoFit : List = field (default_factory=lambda: [[]])

    paramsNames : List = field (default_factory=lambda: [])
    params : List = field(default_factory=lambda: [[]])
    rSquared : List = field (default_factory=lambda: [])
    gof : List = field (default_factory=lambda: [])
    gof2 : List = field (default_factory=lambda: [])
    steepestSlopes : List = field (default_factory=lambda: [])
    spearman : List = field (default_factory=lambda: [])

    plotDetails : List = field (default_factory=lambda: [])
    plotDetailsNoFit : List = field (default_factory=lambda: [])

    sites : List = field (default_factory=lambda: [])

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
    
    def getGood(self, gof, thresh) : 
        newFitter = Fitter()
        goodFit = np.where(np.asarray(gof) >= thresh)[0] # np.where(np.absolute(np.asarray(gof)) >= thresh)[0]
        
        newFitter.stepSize = self.stepSize
        newFitter.numSteps = self.numSteps
        newFitter.stepSlope = self.stepSlope
        newFitter.func = self.func
        newFitter.funcParams = self.funcParams
        newFitter.p0 = self.p0
        newFitter.bounds = self.bounds
        newFitter.paramsNames = self.paramsNames

        newFitter.xFit = self.xFit
        newFitter.yFit = np.asarray(self.yFit)[:,np.asarray(goodFit)]
        newFitter.x = get_indexed_array(self.x, goodFit) #np.asarray(self.x)[goodFit]
        newFitter.y = get_indexed_array(self.y, goodFit)
        #newFitter.xNoFit = self.xNoFit[goodFit]
        #newFitter.yNoFit = self.yNoFit[goodFit]

        newFitter.params = np.asarray(self.params)[:,np.asarray(goodFit)]
        newFitter.rSquared = np.asarray(self.rSquared)[goodFit]
        newFitter.gof = np.asarray(self.gof)[goodFit]
        newFitter.steepestSlopes = np.asarray(self.steepestSlopes)[goodFit]
        newFitter.spearman = np.asarray(self.spearman)[goodFit]

        newFitter.plotDetails = np.asarray(self.plotDetails)[goodFit]
        #newFitter.plotDetailsNoFit = self.plotDetailsNoFit[goodFit]
        newFitter.sites = np.asarray(self.sites)[goodFit]
        return newFitter


    #def calculateRSquared(self, y, yFitted) : 
    #    ssRes = np.sum((y - yFitted)**2)
    #    ssTot = np.sum((y - statistics.mean(y))**2)
    #    return 1 - ssRes/ssTot

    def addFit(self, xToFit, yToFit, plotDetails="", spearman=0.0, site=None) :
        yToFitNormalized = yToFit - min(yToFit)
        yToFitNormalized = yToFitNormalized / max(yToFitNormalized)
        #yToFit = yToFitNormalized

        try : 
            fitted_data = curve_fit(self.func, xToFit, yToFit, p0=self.p0, bounds=self.bounds, full_output=True)
            popt = fitted_data[0]
            infodict = fitted_data[2]
            #1 / max(yToFit) # in case of only 0 values don't do curve fitting
            #popt, pcov, infodict
        except Exception as e : 
            print("WARNING: No logistic curve fitting found: " + str(e))
            if len(self.xNoFit[0]) == 0 and len(self.xNoFit) == 1 : 
                self.xNoFit[0] = xToFit
                self.yNoFit[0] = yToFit
                self.plotDetailsNoFit.append(plotDetails)
            else : 
                self.xNoFit.append(xToFit)
                self.yNoFit.append(yToFit)
                self.plotDetailsNoFit.append(plotDetails)
            return -1
        
        yLogisticFit = self.funcParams(xToFit, popt)
        rSquared = calculateRSquared(yToFit, yLogisticFit)
        #chisquared = chisquare(f_obs=yToFit, f_exp=yLogisticFit)
        #ks = kstest(yToFit, yLogisticFit) # ks.pvalue
        self.rSquared.append(rSquared)
        self.spearman.append(spearman)
        #self.chiSquared.append(chisquared.pvalue)
        #self.gof2.append(infodict.fvec) # infodict.fvec gives residuals

        # Calculate McFadden's R-squared
        #logit_model = sm.Logit(yToFit, xToFit)
        #yhat = logit_model.predict(self.xFit[0]) 
        #result = logit_model.fit()

        #null_deviance = result.llnull
        #model_deviance = result.llf
        McFadden_R2 = 1 # 1 - (model_deviance / null_deviance)
        self.gof.append(McFadden_R2) 


        for i in range(len(popt)):
            self.params[i].append(popt[i])
        
        #xFitTmp = self.xFit - popt[0]
        #popt[0] = 0
        if len(self.x[0]) == 0 and len(self.x) == 1 : 
            self.x[0] = xToFit
            self.y[0] = yToFit
        else: 
            self.x.append(xToFit)
            self.y.append(yToFit)

        self.plotDetails.append(plotDetails)

        yFit = self.funcParams(self.xFit, popt)
        for i in range(len(yFit)) :
            self.yFit[i].append(yFit[i])

        #if yLogisticFit[0] >= yLogisticFit[-1] : 
        #    print("--- BAD fitting of logistic function. K: " + str(popt[1]))
        #else : 
        #    print("--- GOOD fitting of logistic function. K: " + str(popt[1]))
            
        if not site is None :
            self.sites.append(site) 
        else : 
            self.sites.append("") 
        
        return rSquared

    def getMeanStddevAligned(self) : 
        meanX0 = statistics.mean(self.params[0])
        xAligned = np.arange(-meanX0, -meanX0+self.stepSize*self.numSteps, self.stepSize)
        
        yAligned = [[] for i in range(self.numSteps)]

        for i in range(len(self.params[0])) :
            paramsAligned = [self.params[j][i] for j in range(len(self.params))]
            paramsAligned[0] = 0
            yAlignedSingle = self.funcParams(xAligned, paramsAligned)
            for i in range(len(yAligned)) :
                yAligned[i].append(yAlignedSingle[i])

        
        meanFit = np.array([statistics.mean(yAligned[i]) for i in range(self.numSteps)])
        medianFit = np.array([statistics.median(yAligned[i]) for i in range(self.numSteps)])

        paramsMedian = np.array([statistics.median(self.params[i]) for i in range(len(self.params))])
        paramsMedian[0] = 0
        paramsMedianFit = self.funcParams(xAligned, paramsMedian)
        
        if len(yAligned[0]) == 1 : 
            stddevFit = np.zeros(self.numSteps)
        else: 
            stddevFit = np.array([sem(yAligned[i]) for i in range(self.numSteps)]) #statistics.stdev

        return xAligned, meanFit, medianFit, paramsMedianFit, stddevFit
    
    def getMeanMedianStddevFit(self) :

        if len(self.yFit) == 0 : 
            return np.zeros(self.numSteps), np.zeros(self.numSteps), np.zeros(self.numSteps), np.zeros(self.numSteps), np.zeros(self.numSteps), np.zeros(self.numSteps)
        
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
            return meanFit, medianFit, np.zeros(self.numSteps), meanParams, medianParams, paramsMedian

        stddevFit = np.array([sem(self.yFit[i]) for i in range(self.numSteps)])

        return meanFit, medianFit, stddevFit, meanParams, medianParams, paramsMedian


    def calculateSteepestSlopes(self) :
        self.steepestSlopes = []
    
        for i in range(len(self.params[0])) :
            params = []
            for j in range(len(self.params)) :
                params.append(self.params[j][i])
            yFit = self.funcParams(np.arange(0,1,self.stepSlope), params)
            steepestSlopeUp = np.max(yFit[1:] - yFit[:-1])
            steepestSlopeDown = np.min(yFit[1:] - yFit[:-1])
            steepestSlope = steepestSlopeUp if abs(steepestSlopeUp) > abs(steepestSlopeDown) else steepestSlopeDown
            self.steepestSlopes.append(steepestSlope / self.stepSlope)
            #self.steepestSlopes.append(steepestSlope / self.stepSlope)

        return 

    def append(self, input) :
        for i in range(len(input.params)) :
            self.params[i].extend(input.params[i])

        for i in range(len(input.yFit)) :
            self.yFit[i].extend(input.yFit[i])

        self.rSquared.extend(input.rSquared)


def calculateRSquared(y, yFitted) : 
        ssRes = np.sum((y - yFitted)**2)
        ssTot = np.sum((y - statistics.mean(y))**2)
        return 1 - ssRes/ssTot

def sortByX(x, y) :
    sortedIndices = np.argsort(x)
    return np.sort(x), y[sortedIndices]

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

def fitPartialGaussian(x, y, plotStep=0.01) : 
    if len(y) <= 2 or not np.any(np.asarray(y) > 0): 
        return x, y, 1

    x, y = sortByX(x, y)

    maxIndex = np.amax(np.where(np.asarray(y) > 0)) + 1

    xGauss = x[:maxIndex]
    xGauss = np.append(xGauss, xGauss[-1]-xGauss[-2] + xGauss[-1])
    for i in range(1,maxIndex) : 
        xGauss = np.append(xGauss, x[maxIndex-i]-x[maxIndex-i-1] + xGauss[-1])

    yPart = y[:maxIndex]
    yGaussInput = np.concatenate((yPart, yPart[::-1]))

    return fitGauss(xGauss, yGaussInput, plotStep)

def fitLog(x, y, plotStep=0.01) : 
    try : 
        popt, pcov = curve_fit(logFunc, x, y, p0=[0.5, 1, 0, 1], bounds=[[0, -1000, 0, 0], [1, 1000, 1, 1]])  
        xPlot = np.arange(min(x), max(x) + plotStep, plotStep) #
        yLog = logFuncParams(xPlot, popt)
        rSquared = calculateRSquared(y, logFuncParams(x, popt))
        return xPlot, yLog, rSquared

    except Exception as e : 
        print("Error fitting logistic function: " + str(e))
        return [], [], []

def fitGauss(x, y, plotStep = 0.01, minX = 0.0, maxX = 1.0) : 
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    if isnan(sigma) :
        sigma = 1.0

    try : 
        popt,pcov = curve_fit(Gauss, x, y, p0=[0.5, max(y), 0, 5], bounds=[[0, 0, 0, 0.1], [10, 5000, 10, 5000]]) #p0=[mean, max(y), 0, sigma]
    except Exception as e: 
        print("WARNING: Error fitting gauss" + str(e))
        return [], [], []

    xPlot = np.arange(minX, maxX, plotStep)
    yGauss = Gauss(xPlot, *popt)
    rSquared = calculateRSquared(y, Gauss(x, *popt))

    return xPlot, yGauss, rSquared

def Gauss(x, x0, a, b, sigma):
    return b + a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def halfGaussParams(x, params) :
    return halfGauss(x, params[0], params[1], params[2], params[3])

def halfGauss(x, x0, a, b, sigma) : 
    if len(x) == 1 : 
        return Gauss(x, max(x), a, b, sigma)
    #x, y = sortByX(x, y)
    xDoubled = np.concatenate((x, x + max(x) + (x[-1]-x[-2]) - x[0]))
    yGauss = Gauss(xDoubled, max(x), a, b, sigma)
    yGauss = yGauss[:int(len(xDoubled) / 2)]
    return yGauss

def logFuncParams(x, params) : 
    return logFunc(x, params[0], params[1], params[2], params[3])

def logFunc(x, x0, k, a, c) :
    return scipy.special.expit((x-x0)*(k)) # why not -k? because expit also introduces *-1
    #return a + (c - a) * scipy.special.expit((x-x0)*(-k))

def stepParams(x, params) :
    return step(x, params[0], params[1], params[2])

def step(x, x0, a, b) :
    y = np.zeros(len(x))
    y[np.where(x <= x0)[0]] = a
    y[np.where(x > x0)[0]] = b
    return logFunc(x, x0, 10000, a, b)  #a * (np.heaviside(x-x0, 0) + b) #a * (np.sign(x-x0) + b)

    #return 1 * scipy.special.expit((x-x0)*(-k))



