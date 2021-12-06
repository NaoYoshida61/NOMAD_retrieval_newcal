#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:54:01 2021

@author: yoshidanao
nomad_psg.pyを元に自分でcalibrationに必要な部分だけ抜き出してくる
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

# Define auxliary functions
def sinc(dx,width,lobe,asym,offset,gauss):
	sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	sinc += offset/(2.0*norder + 1.0)
	sigma = 50.0
	sinc += gauss*np.exp(-0.5*(dx/sigma)**2.0)
	return sinc
#End sinc-function

#aotf = 26008 # test, order 190, frequency
#aotf = 25864 # test, order 189, frequency
#aotf = 20052 # tesr, order 149, frequency
#aotf =  25430 # tesr, order 186, frequency
#aotf = 25719 # order 188
#aotf = 26153  # order 191
#aotf = 26297 # order 192
aotf = 19907 # order 148
line  = np.polyval(cfaotf,aotf) # 多分 AOTF CWN (TBC), lineの設定の仕方がわからん (聞かないと)

#fdf5f = h5py.File(wdir) (TBD)
#temp = np.array(list(hdf5f['/Channel/InterpolatedTemperature'])) 新しくファイルダウンロードしないと存在しない！
temp = 1.0

"""
# Define initial AOTF parameters and the line
aotfw = np.polyval(aotfwc,line)
aotfs = np.polyval(aotfsc,line)
aotfa = np.polyval(aotfac,line)
aotfo = np.polyval(aotfoc,line)
lorder = round(line/(np.polyval(cfpixel,160.0)))
xdat  = np.polyval(cfpixel,range(320))*lorder
lpix = int(np.interp(line, xdat, range(320)))
"""

# Define initial AOTF parameters and the line
aotfw = np.polyval(aotfwc,line)
aotfs = np.polyval(aotfsc,line)
aotfa = np.polyval(aotfac,line)
aotfo = np.polyval(aotfoc,line)
aotfg = np.polyval(aotfgc,line)
blazew =  np.polyval(blazep,line-3700.0)
lorder = round(line/blazew)
xdat   = np.polyval(cfpixel,range(320))*lorder
lpix   = int(np.interp(line, xdat, range(320)))

dx = np.arange(-300,300,0.1)
plt.plot(dx,sinc(dx,aotfw,aotfs,aotfa,aotfo,aotfg))
plt.title('AOTF function')
plt.xlabel('Wavenumber [cm^-1]')
plt.ylabel('Intensity')
plt.show() # plot aotf function

aotf_func = sinc(dx,aotfw,aotfs,aotfa,aotfo,aotfg)

data = np.array([dx, aotf_func])
data = data.T

save_dir = '/Users/yoshidanao/Python/NOMAD/calibration'
np.savetxt(save_dir+'/new_goddard_148_order+-4_211012.dat', data, fmt=["%8.3f","%12.8f"], header='dwn, AOTF func')


freq = np.polyval(cfaotf,aotf) # ExoMars/aotf.pyでは + dfa がされている。必要？
freq += aotfts*temp*freq 


#------------------------------------
# Define Blaze function
blazew = np.polyval(blazep, line-3700.0)
blazew += blazew*np.polyval(ncoeff,temp)
order = round(line/blazew)

ipix = range(320)
xdat = np.polyval(cfpixel,ipix)*order
sh0 = xdat*np.polyval(ncoeff,temp)
xdat += xdat*np.polyval(ncoeff, temp)

blaze0 = order*blazew
dx = xdat - blaze0

blaze = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2

plt.plot(xdat, blaze)
plt.title('Blaze function')
plt.xlabel('Wavenumber [cm^-1]')
plt.ylabel('Intensity')
plt.show()