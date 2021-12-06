#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:44:34 2021

@author: yoshidanao
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# AOTF shape parameters
aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

# Calibration coefficients
cfaotf  = [1.34082e-7, 0.1497089, 305.0604]                   # Frequency of AOTF [cm-1 from kHz]
cfpixel = [1.75128E-08, 5.55953E-04, 2.24734E+01]             # Blaze free-spectral-range (FSR) [cm-1 from pixel]
ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency
norder  = 4                                                   # Number of +/- orders to be considered in order-addition
npix    = 30                                                  # Number of +/- pixels to be analyzed
forder  = 3                                                   # Polynomial order used to fit the baseline


def F_blaze_new(m, p, aotf, tp):
    
    line = np.polyval(cfaotf, aotf)
    line += aotfts*tp*line
    
    blazew = np.polyval(blazep, line-3700.0)
    blazew += blazew*np.polyval(ncoeff,tp)
    order = round(line/blazew)
    
    ipix = range(320)
    xdat = np.polyval(cfpixel,ipix)*m
    sh0 = xdat*np.polyval(ncoeff,tp)
    xdat += xdat*np.polyval(ncoeff, tp)

    blaze0 = m*blazew
    dx = xdat - blaze0
    
    blaze = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2
    
    return blaze
    
def AOTF_CWN_new(aotf, tp):
    #aotf = 26008 # test, order 190, frequency
    freq = np.polyval(cfaotf,aotf) # ExoMars/aotf.pyでは + dfa がされている。必要？ => 必要ないんじゃないか、っていう話になった w/ Ian
    freq += aotfts*tp*freq
    
    return freq
    
