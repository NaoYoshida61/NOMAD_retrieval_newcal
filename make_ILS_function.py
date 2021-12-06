#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:09:49 2021

@author: yoshidanao

yotubemaking ILS function (based on Geronimo's parameter')
"""
import numpy as np
import statistics
import math


def get_ils_21(order,pixels,pix_mid=160,rp=17000,wn_ref=3700,
               A1=1.,A2=0.27,shift1=0.):
    '''
    Update July 2021 mail Geronimo 20210706
    Pixels is the array containing the pixels.
    Provides coeffs as center1, width1, amplitude1, center2, width2, amplitude2
    rp is the resolving power of NOMAD-SO.
    wn_ref is the reference wn for the coefficients for the shift
    of the second Gaussian.

    Returns coeffs: (6params) x (npixels)
    '''
    
    s_to_fwhm = 2.35482
    
    temp = -1.0 #ということは毎回違うファイルを作成するべき、ということか？？
    
    pix0 = temp*-0.8276
    wvn = (22.4701 + (pixels+pix0)*5.480e-4 +(pixels+pix0)**2*3.32e-8)*order
    
    #wn_mid=pixtown(order,pix_mid)
    
    print(wvn)
    
    wn_mid = statistics.median(wvn)
    print(wn_mid)
    width=wn_mid/s_to_fwhm/rp
    params_21=np.array([-3.06665339e-06,1.71638815e-03,1.31671485e-03])
    shift2=-np.polyval(params_21*wn_mid/wn_ref,pixels)
    coeffs=np.zeros((pixels.size,6))
    coeffs[:,0]=shift1
    coeffs[:,1]=width
    coeffs[:,2]=A1
    coeffs[:,3]=shift2
    coeffs[:,4]=width
    coeffs[:,5]=A2
    return coeffs


pixels = np.arange(0,320,1)
#pixels = 200
coeffs = get_ils_21(189,pixels)
#print(coeffs)

np.savetxt('/Users/yoshidanao/Python/Instrument/Order_189.dat', coeffs, fmt="%.8f")

