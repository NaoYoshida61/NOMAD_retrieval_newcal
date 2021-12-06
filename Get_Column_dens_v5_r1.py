#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:42:16 2021

@author: yoshidanao
"""

# Name; Get_Column_dens
# Purpose; This is the mainroutine for calculating the column density.
#                It includes fitting of ILS function or Doppler profile.

# ILS function is arranged for order 149 and 190.
# Dopper-broadened profile is applied for order 148.

# output; ####_Norm, YErrorNorm is applied.

# last change; version 5_r1 is made. 
# It includes the backgound fit using Asymmetric Least Square.


# The newest recipe needs to modify AOTF function, Blaze shape, AOTF CWN by myself.

import glob
import h5py
import numpy as np
from numpy import exp, linspace, random
from scipy.optimize import curve_fit
from scipy.integrate import quad
import scipy.integrate as integrate
import math
from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/yoshidanao/Python/NOMAD/20210105_revision')
import Line_strength

from decimal import Decimal, ROUND_HALF_UP

import csv
from scipy import sparse

order, ideal_wn, p, lam = '190', '4291.35', 0.05, 10000. # for Order 190
#order, ideal_wn, p, lam = '149', '3364.48', 0.03, 1000. # for Order 149
#order, ideal_wn, p, lam = '189', '4268.83', 0.05, 10000.# for Order 189
#order, ideal_wn = '191', '4314.04' # for Order 191
#order, ideal_wn = '188', '4246.16' # for Order 188
#order, ideal_wn = '148', '3342.036' # for Order 148

"""
Inputfileも変更するように注意！！
"""

alt_min, alt_max = 60, 110

#p, lam = 0.05, 10000. # for CO
#p, lam = 0.03, 1000. # for 149, 148

#-------------------------------------------------------------------------
# Input correction of Equivalent width due to AOTF transfer function change

#file_AOTF = '/Users/yoshidanao/Python/NOMAD/20210308_orderaddition/EquivalentW_correct_'+order+'.npz'
#aotf_E = np.load(file_AOTF)


def profile_saving(altitude, fit0, fit1, coefficient):#, lat, lon):
    array = [str(altitude), str(fit0), str(fit1), str(coefficient[0])]
    """
    array[0] = str(altitude)
    array[1] = str(fit0)
    array[2] = str(fit1)
    array[3] = str(coefficient[0])
    """
    #array = str(array)
    print(array)
    
    #with open('/Volumes/NekoPen/nomad/edited_results/fitting_ILS'+str(wave_cen)+'.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(array)
    
 
     
    #np.savetxt('/Volumes/NekoPen/nomad/for_idl/'+obs_name+'_a758_Shohei_AOTF.txt', array, delimiter=',', newline='\n', header='altitude, density, uncertainty, Latitude, Longitude', comments='#')
    
    #np.savez('/Volumes/NekoPen/nomad/edited_results/fitting_'++'.npz', altitude=altitude, density=density, uncertainty=error)


def baseline_als_forPool(args):
    y,lam,p=args
    return baseline_als(y, lam, p)

#Ian's parameters:
#    lam=250
#    p=0.95

def baseline_als(y, lam, p, niter=10):
    '''
    # als: asymetric least squares
    #lam: lambda: smoothness
    #p: asymmetry
    #There are two parameters: p for asymmetry and lam for smoothness.
    #Both have to be tuned to the data at hand. We found that generally
    #0.001 < p < 0.1 is a good choice (for a signal with positive peaks)
    #and 10^2 <lam< 10^9 , but exceptions may occur. In any case one
    #should vary lam on a grid that is approximately linear for log lam
    Good values:
    lam_als=1e3 #smoothness
    p_als=0.2 #asymmetry

    smoothness needs to vary if the number of points changes (like with zerofill)
    '''
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = (1-p) * (y > z) + (p) * (y < z)   #ligne changée par rapport a version originale
    return z


#-------------------------------------------------------------------------
# Input the NOMAD SO file
# individual order
wdir = "/Volumes/KingPen/nomad/hdf5_level_1p0a"
            
list_file = glob.glob(wdir+'/2019/07/*/*'+order+'.h5', recursive=True) # order numberは任意
#list_file = '/Volumes/NekoPen/nomad/hdf5_level_1p0a/2018/09/30/20180930_113957_1p0a_SO_A_I_190.h5'
#print(list_file)

count_01 = 0; count_02 = 0

for i_data in range(len(list_file)):
    input_file = list_file[i_data]
    #input_file = '/Volumes/KingPen/nomad/hdf5_level_1p0a/2019/06/24/20190624_055238_1p0a_SO_A_I_149.h5'
    #input_file = '/Volumes/KingPen/nomad/hdf5_level_1p0a/2018/05/02/20180502_130154_1p0a_SO_A_I_190.h5'
    


    print('reading file name', input_file)
    
    # すでにcolumn densityの結果がある場合は計算しないようにする
    
    find_result = glob.glob("/Volumes/KingPen/nomad/retrieval_results/column_dens/"+input_file[78:81]+"_211125/"+input_file[50:81]+"*_bin1_error.npz")
    #find_result = glob.glob("/Volumes/NekoPen/nomad/retrieval_results/column_dens/"+input_file[50:81]+"*.npz")
    print(find_result)
    if len(find_result) == 6:
        print("Already column density is calculated")
        continue
      
    fig = plt.figure(figsize=(11.69, 8.27))

    plt.subplots_adjust(wspace=1.0, hspace=0.4)
    """
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    """
    
    ax1 = plt.subplot2grid((3,6), (0,0),  colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((3,6), (1,3),  colspan=3)
    ax3 = plt.subplot2grid((3,6), (2,0),  colspan=2)
    ax4 = plt.subplot2grid((3,6), (2,2),  colspan=2)
    ax5 = plt.subplot2grid((3,6), (2,4),  colspan=2)
    ax6 = plt.subplot2grid((3,6), (0,3),  colspan=3) 
    

    
    h5file = h5py.File(input_file, "r")
    
    filename = input_file[50:81]
    DD = input_file[56:58]
    MM = input_file[54:56]
    YY = input_file[50:54]
    order_num = int(input_file[78:81])
    observation_type = input_file[74:75]
    
    #print(observation_type)
    
    if observation_type == 'G' or observation_type == 'L':
        continue

    #print(type(int(order_num)))

    print(filename,DD,MM,YY,order_num)

    bin_num = h5file["Channel/IndBin"][()]
    bin1 = np.where(bin_num[:] == 1.)[0].tolist()
    bin2 = np.where(bin_num[:] == 2.)[0].tolist()
    bin3 = np.where(bin_num[:] == 3.)[0].tolist()
    #bin1 = bin3

    #obs_alt = h5file["Geometry/Point0/TangentAlt"][()]
    obs_alt = h5file["Geometry/Point0/TangentAltAreoid"][()]
    obs_alt_bin1 = obs_alt[bin1,0]
    obs_alt_bin2 = obs_alt[bin2,0]

    # search the altitude range
    # とりあえず高度40-110 kmで行う
    a_1d = np.where((obs_alt[bin1,0] >= alt_min) & (obs_alt[bin1,0] <= alt_max))[0].tolist()

    wavenum = h5file["Science/X"][()]
    transmit = h5file["Science/Y"][()]
    SNR = h5file["Science/SNR"][()]
    transmit_error = h5file["Science/YErrorNorm"][()]
    
    # Geometry
    Latitude = h5file["Geometry/Point0/Lat"][()]
    Longitude = h5file["Geometry/Point0/Lon"][()]
    LST = h5file["Geometry/Point0/LST"][()] # local solar time
    LsubS = h5file["Geometry/LSubS"][()]

    # AOTF CWN
    AOTF_cwn = h5file["Channel/AOTFCentralWavenb"][()]
    #print('AOTF_cwn', AOTF_cwn[0])
    #AOTF_cwn[0] = 4289.6468
    #print('AOTF cwn modified', AOTF_cwn[0])
    
    # AOTF? interpolated temperature # need to consider the shift, blaze shape
    Int_temp = h5file["Channel/InterpolatedTemperature"][()]
    aotf = h5file["Channel/AOTFFrequency"][()] 
    #line = 4289.384689893248
    
    #----------------------------------------------------------------------------------------------
    # AOTF CWNを元に、AOTF function, solar irradianceで重み付けられた関数を計算する
    """
    from nomad_sim2 import NOMAD_sim2
    sim = NOMAD_sim2(order=int(order), adj_orders=4,
          dirAtmosphere='/Volumes/NekoPen/nomad/Auxiliary_files/Atmosphere/',
          dirLP='/Volumes/NekoPen/nomad/Auxiliary_files/Spectroscopy',
          atmo_filename='/Volumes/NekoPen/nomad/Auxiliary_files/Atmosphere/apriori_1_1_1_GEMZ_wz_mixed/gem-mars-a585_AllSeasons_AllHemispheres_AllTime_mean_atmo.dat',
          solar_file='/Volumes/NekoPen/nomad/Auxiliary_files/Solar/Solar_irradiance_ACESOLSPEC_2015.dat',
          ils_type='dg', dirInstrument='/Users/yoshidanao/Python/Instrument', ilsFile='dg_ils_co.dat', pixel_shift=0, nu0_shift=AOTF_cwn[0])
    sim.forward_model(nu0_shift=AOTF_cwn[0])
    
    b0_I0 = sim.order_so_ir
    
    total_ir = np.zeros(320)
    
    for iord in range(9):
        total_ir += b0_I0[:,iord]
        
    #pix = np.arange(320)
    #plt.plot(pix, b0_I0[:,4]/total_ir[:])
    #plt.show()
        
    coef = np.zeros(320)
    coef[:] = b0_I0[:,4] / total_ir[:]
    """
    
    # plot transmittance in fig.ax1
    #print(a_1d)
    #b_1da = [a_1d[20], a_1d[-35], a_1d[-5]]
    for c1 in a_1d:
        wave = wavenum[bin1[c1],:]
        trans = transmit[bin1[c1],:]
        f = round(obs_alt_bin1[c1])
        
        #wave = wavenum[bin1[b_1da[c1]],:]
        #trans = transmit[bin1[b_1da[c1]],:]
        #f = round(obs_alt_bin1[b_1da[c1]])
        
        ax1.plot(wave, trans, label=f)
        ax1.set_title(input_file[50:])
        ax1.set_xlabel("wave number [$cm^{{-}1}$]")
        ax1.set_ylabel("transmittance")
        #ax1.legend(title = 'Altitude [km]')


    # target of absorption line
    # line center, ±width, del_wn
    if order_num == 148:
        abs_line = [[3343.1, 1.0, 0.5], [3344.8, 1.0, 0.5], [3346.36, 1.0, 0.5]]
        coef_pix = [[206],[226],[245]]
        

    if order_num == 149:
        abs_line = [[3365.02, 1.0, 0.5], [3366.52, 1.0, 0.5], [3358.85, 1.0, 0.5],\
                    [3355.65, 1.0, 0.5], [3357.20, 1.0, 0.5], [3360.28, 1.0, 0.5]]#, [3360.0, 1.0, 0.5]]
        coef_pix= [[196],[214],[122],[85],[104],[141]]#, [148]]

    if order_num == 190:
        abs_line = [[4291.50, 1.2, 0.5], [4288.29, 1.0, 0.5], [4285.01, 1.0, 0.5]]
        coef_pix = [[203],[173],[142]]
        #abs_line = [[4288.29, 1.0, 0.5], [4285.01, 1.0, 0.5]]
        #coef_pix = [[173],[142]]
        
    if order_num == 189:
        abs_line = [[4263.84, 1.0, 0.5], [4267.54, 1.0, 0.5]]
        coef_pix = [[155], [190]] #[[156], [198]] コメントアウトはまちがい。nomad_simからpixelの位置を持って来る必要がある
        
    if order_num == 191:
        abs_line = [[4306., 0.6, 0.5]] #隣の吸収の強い方に持っていかれる。
        coef_pix = [[132]]
    
    if order_num == 188:
        abs_line = [[4244., 1.0, 0.5]]
        coef_pix = [[183]]
        
    color = ['#4169E1' ,'#FF8C00',  '#32CD32', 'blueviolet', 'tomato', 'dimgray', 'k']

    #--------------------------------------------------------------------------------------
    # Fit the ILS function or Dopplar profile to get equivalent width
    # For individual absorption line



    for i_l in range(len(abs_line[:])):
        print("searching %i" % i_l)
        
        wave_cen = str(abs_line[i_l][0])
        l = ['Altiutde, fitting1, fitting2, coefficient']
        #with open('/Volumes/NekoPen/nomad/edited_results/fitting_ILS'+str(wave_cen)+'.csv', 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(l)
        
        E_width_alt = []
        Alt_prof = []
        
        E_width_alt_dn = []
        E_width_alt_up = []

        E_width_alt_cor = []
        E_width_alt_err_cor = []
        
        E_width_err = []
        
        Lat_prof = []
        Lon_prof = []
        LST_prof = []
        Ls_prof = []
        
        #N_std = np.zeros(a_1d)
        #E_std = np.zeros(a_1d)
        
        for i_bin in range(1):
            #print(i_bin)
            if i_bin == 0:
                bin00 = bin1
            if i_bin == 1:
                bin00 = bin2

            for i_alt in a_1d:
                index_off = np.where((wavenum[bin00[i_alt],:] >= (abs_line[i_l][0] - abs_line[i_l][1])) & (wavenum[bin00[i_alt],:] <= (abs_line[i_l][0] + abs_line[i_l][1])))[0].tolist()
                
                if len(index_off) == 0:
                    continue

                y1_idx = transmit[bin00[i_alt], index_off]
                x1_idx = wavenum[bin00[i_alt], index_off]

                idx_min = np.argmin(y1_idx)

                #idx_4_fitting = np.where((x1_idx >= (x1_idx[idx_min] - abs_line[i_l][2])) & (x1_idx <= (x1_idx[idx_min] + abs_line[i_l][2])))
                
                # for Order 190, 波長のシフトがおきているので、indexの抜き出す領域もずらす。
                index_off2 = np.where((wavenum[bin00[i_alt],:] >= (x1_idx[idx_min] - abs_line[i_l][1])) & (wavenum[bin00[i_alt],:] <= (x1_idx[idx_min] + abs_line[i_l][1])))[0].tolist() # for integration

                y1_idx2 = transmit[bin00[i_alt], index_off2]
                x1_idx2 = wavenum[bin00[i_alt], index_off2]
                
                idx_4_fitting_left = np.where(x1_idx2 < (x1_idx[idx_min] - abs_line[i_l][2]))[0].tolist()
                idx_4_fitting_right = np.where(x1_idx2 > (x1_idx[idx_min] + abs_line[i_l][2]))[0].tolist()

                #x1_4_fitting = x1_idx[idx_4_fitting]
                #y1_4_fitting = y1_idx[idx_4_fitting]
                
                #xx_0 = x1_idx[:idx_4_fitting[0][0]-1]
                #xx_1 = x1_idx[idx_4_fitting[0][-1]+1:]
                #xx = np.append(xx_0, xx_1[::]) 

                #yy = np.append(y1_idx[:idx_4_fitting[0][0]-1], y1_idx[idx_4_fitting[0][-1]+1:])
                
                xx_0 = x1_idx2[idx_4_fitting_left]
                xx_1 = x1_idx2[idx_4_fitting_right]
                
                xx = np.append(xx_0, xx_1[::]) # shift分ずらしたあとの、backgroundを引き算するための部分
                yy = np.append(y1_idx2[idx_4_fitting_left], y1_idx2[idx_4_fitting_right])
                
                # transmittanceのsigma
                yy_err = transmit_error[bin00[i_alt], index_off2]

                #-------------------------------------------------------
                # substitute the background level #ここ一応で本当にいいのか？
                # normalize the transmittance!!

                if order_num == 148 or order_num == 149:
                    a0, b0, c0 = np.polyfit(xx,yy,2, cov = False)
                    f_lin = a0 * xx**2 + b0 * xx + c0 # revised to 1 order to 2 order

                    #f_lin_rm = []
                    #f_lin_rm = a0 * x1_4_fitting + b0

                    f_lin_fit = []
                    f_lin_fit = a0 * x1_idx2**2 + b0*x1_idx2 + c0

                    res_y1 = y1_idx2 / f_lin_fit
                    use_y1 = -1 * (res_y1 - 1)

                    #ax6.plot(x1_idx2, use_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, f_lin_fit, linestyle='dashed')
                    #ax6.plot(xx, f_lin)
                    #ax6.set_title = ('subtraction of background')
                    #ax6.set_xlim(3355, 3367)
                    #ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    #use_yy_err = yy_err / f_lin_fit
                    
                    new_fit = baseline_als(transmit[bin00[i_alt],:], lam, p, niter=10)
                    new_tr = transmit[bin00[i_alt],index_off2] / new_fit[index_off2]
                    new_tr1 = -1 * (new_tr - 1)
                    new_tr_err = transmit_error[bin00[i_alt],index_off2] / new_fit[index_off2]
                    
                    ax6.plot(x1_idx2, new_tr, label=round(obs_alt[bin00[i_alt],0]))
                    ax6.set_title = ('subtraction of background')
                    ax6.set_xlim(3355, 3367)
                    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    

                if order_num == 190:
                    a0, b0, c0 = np.polyfit(xx,yy,2, cov = False)
                    f_lin = a0 * xx**2 + b0 * xx + c0

                    #f_lin_rm = []
                    #f_lin_rm = a0 * x1_4_fitting**2 + b0*x1_4_fitting  + c0

                    f_lin_fit = []
                    f_lin_fit = a0 * x1_idx2**2 + b0*x1_idx2 + c0

                    res_y1 = y1_idx2 / f_lin_fit
                    use_y1 = -1 * (res_y1 - 1)
                    
                    #ax6.plot(x1_idx2, use_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, res_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, f_lin_fit, linestyle='dashed')
                    #ax6.plot(xx, f_lin)
                    #ax6.set_title = ('original')
                    #ax6.set_xlim(4283, 4300)
                    #ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    #use_yy_err = yy_err / f_lin_fit                  
                    
                    new_fit = baseline_als(transmit[bin00[i_alt],:], lam, p, niter=10)
                    new_tr = transmit[bin00[i_alt],index_off2] / new_fit[index_off2]
                    new_tr1 = -1 * (new_tr - 1)
                    new_tr_err = transmit_error[bin00[i_alt],index_off2] / new_fit[index_off2]
                    
                    ax6.plot(x1_idx2, new_tr, label=round(obs_alt[bin00[i_alt],0]))
                    ax6.set_title = ('subtraction of background')
                    ax6.set_xlim(4283, 4300)
                    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                if order_num == 189:
                    a0, b0, c0 = np.polyfit(xx,yy,2, cov = False)
                    f_lin = a0 * xx**2 + b0 * xx + c0 # revised to 1 order to 2 order

                    #f_lin_rm = []
                    #f_lin_rm = a0 * x1_4_fitting + b0

                    f_lin_fit = []
                    f_lin_fit = a0 * x1_idx2**2 + b0*x1_idx2 + c0

                    #res_y1 = y1_idx2 / f_lin_fit
                    #use_y1 = -1 * (res_y1 - 1)

                    #ax6.plot(x1_idx2, use_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, res_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, f_lin_fit, linestyle='dashed')
                    #ax6.plot(xx, f_lin)
                    #ax6.set_title = ('subtraction of background')
                    #ax6.set_xlim(4247, 4281)
                    #ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    #use_yy_err = yy_err / f_lin_fit
                    
                    new_fit = baseline_als(transmit[bin00[i_alt],:], lam, p, niter=10)
                    new_tr = transmit[bin00[i_alt],index_off2] / new_fit[index_off2]
                    new_tr1 = -1 * (new_tr - 1)
                    new_tr_err = transmit_error[bin00[i_alt],index_off2] / new_fit[index_off2]
                    
                    ax6.plot(x1_idx2, new_tr, label=round(obs_alt[bin00[i_alt],0]))
                    ax6.set_title = ('subtraction of background')
                    ax6.set_xlim(4247, 4281)
                    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    
                if order_num == 191:
                    a0, b0, c0 = np.polyfit(xx,yy,2, cov = False)
                    f_lin = a0 * xx**2 + b0 * xx + c0 # revised to 1 order to 2 order

                    #f_lin_rm = []
                    #f_lin_rm = a0 * x1_4_fitting + b0

                    f_lin_fit = []
                    f_lin_fit = a0 * x1_idx2**2 + b0*x1_idx2 + c0

                    res_y1 = y1_idx2 / f_lin_fit
                    use_y1 = -1 * (res_y1 - 1)

                    #ax6.plot(x1_idx2, use_y1, label=round(obs_alt[bin00[i_alt],0]))
                    ax6.plot(x1_idx2, res_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, f_lin_fit, linestyle='dashed')
                    #ax6.plot(xx, f_lin)
                    ax6.set_title = ('subtraction of background')
                    ax6.set_xlim(4300, 4310)
                    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    use_yy_err = yy_err / f_lin_fit
                    
                if order_num == 188:
                    a0_1, b0, c0 = np.polyfit(xx,yy,2, cov = False)
                    f_lin = a0_1 * xx**2 + b0 * xx + c0 # revised to 1 order to 2 order

                    #f_lin_rm = []
                    #f_lin_rm = a0 * x1_4_fitting + b0

                    f_lin_fit = []
                    f_lin_fit = a0_1 * x1_idx2**2 + b0*x1_idx2 + c0

                    res_y1 = y1_idx2 / f_lin_fit
                    use_y1 = -1 * (res_y1 - 1)

                    #ax6.plot(x1_idx2, use_y1, label=round(obs_alt[bin00[i_alt],0]))
                    ax6.plot(x1_idx2, res_y1, label=round(obs_alt[bin00[i_alt],0]))
                    #ax6.plot(x1_idx2, f_lin_fit, linestyle='dashed')
                    #ax6.plot(xx, f_lin)
                    ax6.set_title = ('Original')
                    ax6.set_xlim(4242, 4246)
                    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    use_yy_err = yy_err / f_lin_fit


                idx_cent = np.where(new_tr1 == max(new_tr1))
                pix_num_idx = np.where(wavenum[bin00[i_alt],:] == x1_idx2[idx_cent]) 
                pix_num = list(pix_num_idx[0])
                
                if len(pix_num_idx[0].tolist()) == 0:
                    continue
                
                #----------------------------------------------------
                # Fittingをする前に、いま見ている高度におけるAOTF CWN, Blaze shapeを計算する必要がある
                Int_temp_use = Int_temp[bin00[i_alt]]
                #print('AOTF_cwn[0]', AOTF_cwn[0])
                
                #----------------------------------------------------------------------------------------------
                # AOTF CWNを元に、AOTF function, solar irradianceで重み付けられた関数を計算する
                from nomad_sim3 import NOMAD_sim3
                sim = NOMAD_sim3(order=int(order), adj_orders=4,
                                 dirAtmosphere='/Volumes/KingPen/nomad/Auxiliary_files/Atmosphere/',
                                 dirLP='/Volumes/KingPen/nomad/Auxiliary_files/Spectroscopy',
                                 atmo_filename='/Volumes/KingPen/nomad/Auxiliary_files/Atmosphere/apriori_1_1_1_GEMZ_wz_mixed/gem-mars-a585_AllSeasons_AllHemispheres_AllTime_mean_atmo.dat',
                                 solar_file='/Volumes/KingPen/nomad/Auxiliary_files/Solar/Solar_irradiance_ACESOLSPEC_2015.dat',
                                 ils_type='dg', dirInstrument='/Users/yoshidanao/Python/Instrument', ilsFile='dg_ils_co.dat', pixel_shift=0, 
                                 nu0_shift=AOTF_cwn[0], aotf=aotf[0], temp=Int_temp_use)
                sim.forward_model(temp=Int_temp_use, aotf=aotf[0], nu0_shift=AOTF_cwn[0])
    
                b0_I0 = sim.order_so_ir
    
                total_ir = np.zeros(320)
    
                for iord in range(9):
                    total_ir += b0_I0[:,iord]
        
                #pix = np.arange(320)
                #plt.plot(pix, b0_I0[:,4]/total_ir[:])
                #plt.show()
        
                coef = np.zeros(320)
                coef[:] = b0_I0[:,4] / total_ir[:]
                

                #----------------------------------------------------
                # Fitting

                def ILS_149(wn_x, x0, con_t):
                    return con_t*coef_v*(a3 * exp(-0.5 * ((wn_x-x0+a1)/a2)**2) + a6 * exp(-0.5 * ((wn_x-x0-a4)/a5)**2))

                def ILS_190(wn_x, x0, con_t):
                    return con_t*coef_v*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_x-a2-x0)**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_x-a5-x0)**2 / (a6**2)))

                def gaussian(s, amp, cen, wid):
                    return amp * exp(-(s-cen)**2 / (2*wid**2))
                
                def ILS_190_new(wn_x,x0,con_t):
                    #return con_t*coef_v*(a3*exp(-0.5*((wn_x+a1-x0)/a2)**2) + a6*exp(-0.5*((wn_x+a4-x0)/a5)**2))
                    return con_t*coef_v*(a3*exp(-0.5*((wn_x+a1-x0)/a2)**2) + a6*exp(-0.5*((wn_x-a4-x0)/a5)**2))


                if order_num == 149 or order_num == 148:
                    print(order_num)
                    print("Fitting with ILS finction")
                    
                    """ old
                    a1 = 0.0548001 + pix_num[0]*(-0.000121804)
                    a2 = 0.0627182 + pix_num[0]*(7.5972e-5)
                    a3 = 0.920421 + pix_num[0]*(-0.000192975) 
                    a4 = -0.0414220 + pix_num[0]*(-0.000395962)
                    a5 = -0.0133858 + pix_num[0]*(0.000676179)
                    a6 = 0.683544 + pix_num[0]*(-0.00169758)
                    """
                    
                    file = '/Users/yoshidanao/Python/Instrument/Order_'+order+'.dat'
                    data = np.loadtxt(file)

                    a1 = data[pix_num[0],0] # もしかしたらpix-1が必要かも知れん
                    a2 = data[pix_num[0],1]
                    a3 = data[pix_num[0],2]
                    a4 = data[pix_num[0],3]
                    a5 = data[pix_num[0],4]
                    a6 = data[pix_num[0],5]

                    #init_vals = [abs_line[i_l][0],0.05]
                    init_vals = [x1_idx[idx_min], 1.0]
                    
                    coef_v = coef[coef_pix[i_l]] # pixel#として、pixel shiftする前の各lineの位置を知りたい。
                    
                    try:
                        best_vals, covar = curve_fit(ILS_149, x1_idx2, new_tr1, p0=init_vals, sigma=new_tr_err, absolute_sigma=True, maxfev=10000)
                    except RuntimeError:
                        print("ILS fitting isn't converged.")
                        #input()
                        continue
                    
                    profile_saving(obs_alt[bin00[i_alt],0], best_vals[0], best_vals[1], coef_v)
                    
                    stderr = np.sqrt(np.diag(covar))
                    
                    def res_fit_ILS149(wn_xx):
                        return best_vals[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                    
                    def ILS_area(wn_x):
                        return a3 * exp(-0.5 * ((wn_x-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5 * ((wn_x-best_vals[0]-a4)/a5)**2)
                    
                    Area = list(integrate.quad(ILS_area, x1_idx2[0], x1_idx2[-1]))
                    E_width = best_vals[1]*Area[0]
                    #print('covar', covar, best_vals)
                    #print(stderr)
                        
                    ## エラーの計算をする
                    # New error estimation!!
                    rel_ori_area = Area[1] / Area[0]
                    rel_amp_area = stderr[1] / best_vals[1]
                    error_W = E_width * (rel_ori_area + rel_amp_area)
                    
                    """
                    def err_ILS149(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a3*(wn_xx-best_vals[0]+a1)/(a2**2) * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + \
                                          a6*(wn_xx-best_vals[0]-a4)/(a5**2) * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        f2_c = stderr[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        
                        #print(f1_x0, f2_c)
                    
                        return np.sqrt(f1_x0**2 + f2_c**2)
                    

                    def res_fit_ILS149(wn_xx):
                        return best_vals[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                    
                    E_width = list(integrate.quad(res_fit_ILS149, x1_idx2[0], x1_idx2[-1]))
                    
                    def err_up_ILS149(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a3*(wn_xx-best_vals[0]+a1)/(a2**2) * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + \
                                          a6*(wn_xx-best_vals[0]-a4)/(a5**2) * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        f2_c = stderr[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2)) + derr
                    
                    
                    def err_dn_ILS149(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a3*(wn_xx-best_vals[0]+a1)/(a2**2) * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + \
                                          a6*(wn_xx-best_vals[0]-a4)/(a5**2) * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        f2_c = stderr[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a3 * exp(-0.5*((wn_xx-best_vals[0]+a1)/a2)**2) + a6 * exp(-0.5*((wn_xx-best_vals[0]-a4)/a5)**2)) - derr
                    
                    E_width_up = list(integrate.quad(err_up_ILS149, x1_idx2[0], x1_idx2[-1]))
                    E_width_dn = list(integrate.quad(err_dn_ILS149, x1_idx2[0], x1_idx2[-1])) 
                    """
                    
                    """
                    RMS = use_y1 - res_fit_ILS149(x1_idx2)
                    RMS_idx = np.where(RMS > 0.005)[0].tolist()
                    if len(RMS_idx) > 0:
                        continue
                    """
                    
                    ax2.plot(x1_idx2, (new_tr1-res_fit_ILS149(x1_idx2)*coef_v), label=round(obs_alt[bin00[i_alt],0]))
                    ax2.axvline(x=abs_line[i_l][0])
                    ax2.axvline(x=(abs_line[i_l][0]+0.5), linestyle='dashed')
                    ax2.axvline(x=(abs_line[i_l][0]-0.5), linestyle='dashed')
                    ax2.set_title("Fitting RMS")
                    ax2.set_xlim(3355, 3367)
                    #ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
                    
                    
                if order_num == 190 or order_num == 189:# or order_num == 191:
                    #print(order_num)
                    #print("Fitting with ILS function")

                    """ old
                    a1 = 1.0753 + pix_num[0]*(1.430e-3)
                    a2 = 0.
                    a3 = 0.1121 + pix_num[0]*(-2.086e-5)
                    a4 = -0.1036 + pix_num[0]*(1.554e-3)
                    a5 = -0.1746 + pix_num[0]*(-4.127e-4)
                    a6 = -3.814e-2 + pix_num[0]*(7.546e-4)
                    """
                    
                    file = '/Users/yoshidanao/Python/Instrument/Order_'+str(order_num)+'.dat'
                    data = np.loadtxt(file)

                    a1 = data[pix_num[0],0] # もしかしたらpix-1が必要かも知れん
                    a2 = data[pix_num[0],1]
                    a3 = data[pix_num[0],2]
                    a4 = data[pix_num[0],3]
                    a5 = data[pix_num[0],4]
                    a6 = data[pix_num[0],5]

                    init_vals = [abs_line[i_l][0], 1.0]
                    
                    #print(pix_num)
                    coef_v = coef[coef_pix[i_l]] # pixel#として、pixel shiftする前の各lineの位置を知りたい。
                    
                    try:
                        #best_vals, covar = curve_fit(ILS_190, x1_idx2, new_tr1, p0=init_vals, sigma=new_tr_err, absolute_sigma = True, maxfev=10000)
                        best_vals, covar = curve_fit(ILS_190_new, x1_idx2, new_tr1, p0=init_vals, sigma=new_tr_err, absolute_sigma = True, maxfev=10000)
                    except RuntimeError:
                        print("ILS fitting isn't converged.")
                        #input()
                        continue
                    
                    #print(best_vals, coef_v)
                        
                    stderr = np.sqrt(np.diag(covar))
                    #print('covar', covar, best_vals)
                    #print(stderr)
                        
                    ## エラーの計算をする
                    """
                    def err_ILS190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                    
                        return np.sqrt(f1_x0**2 + f2_c**2)
                    """
                    
                    #def res_fit_ILS190(wn_xx):
                    #    return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2)))
                    
                    def res_fit_ILS190_new(wn_x):
                        #return best_vals[1]*(a3*exp(-0.5*((wn_x+a1-best_vals[0])/a2)**2) + a6*exp(-0.5*((wn_x+a4-best_vals[0])/a5)**2))
                        return best_vals[1]*(a3*exp(-0.5*((wn_x+a1-best_vals[0])/a2)**2) + a6*exp(-0.5*((wn_x-a4-best_vals[0])/a5)**2))

                    def ILS_area(wn_x):
                        #return best_vals[1]*(a3*exp(-0.5*((wn_x+a1-best_vals[0])/a2)**2) + a6*exp(-0.5*((wn_x+a4-best_vals[0])/a5)**2))
                        return a3*exp(-0.5*((wn_x+a1-best_vals[0])/a2)**2) + a6*exp(-0.5*((wn_x-a4-best_vals[0])/a5)**2)
                    
                    Area = list(integrate.quad(ILS_area, x1_idx2[0], x1_idx2[-1]))
                    
                    #E_width = list(integrate.quad(res_fit_ILS190, x1_idx2[0], x1_idx2[-1]))
                    #E_width = list(integrate.quad(res_fit_ILS190_new, x1_idx2[0], x1_idx2[-1]))
                    E_width = best_vals[1]*Area[0]
                    
                    """
                    def err_up_ILS190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2))) + derr
                    
                    
                    def err_dn_ILS190(wn_xx):
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2))) - derr
                    
                    def derr_190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return derr
                    
                    
                    E_width_up = list(integrate.quad(err_up_ILS190, x1_idx2[0], x1_idx2[-1]))
                    E_width_dn = list(integrate.quad(err_dn_ILS190, x1_idx2[0], x1_idx2[-1])) 
                    
                    
                    derr = derr_190(x1_idx2)
                    """
                    
                    #print('Area = ', obs_alt[bin00[i_alt],0], E_width[0], best_vals[1]*Area[0], stderr[0]*best_vals[1], E_width[0] - E_width_dn[0])
                    
                    # New error estimation!!
                    rel_ori_area = Area[1] / Area[0]
                    rel_amp_area = stderr[1] / best_vals[1]
                    error_W = E_width * (rel_ori_area + rel_amp_area)
                    
                    #print('Error W', error_W)
                    
                    """
                    diff = (res_fit_ILS190(x1_idx2))*coef_v - use_y1
                    #plt.plot(x1_idx2, transmit_error[bin00[i_alt],index_off2], label='YerrorNorm')
                    #plt.plot(x1_idx2, derr, label='sigma')
                    plt.plot(x1_idx2, diff, label='difference')
                    plt.plot(x1_idx2, (res_fit_ILS190(x1_idx2))*coef_v, label='Fitting')
                    plt.plot(x1_idx2, (res_fit_ILS190(x1_idx2)+derr)*coef_v, label='+error')
                    plt.plot(x1_idx2, (res_fit_ILS190(x1_idx2)-derr)*coef_v, label='-error')
                    plt.plot(x1_idx2, use_y1, label='real')
                    plt.plot(x1_idx2, res_fit_ILS190(x1_idx2), label='no coeff.')
                    plt.plot(x1_idx2, (res_fit_ILS190(x1_idx2)+derr), label='no coeff.+derr')
                    print(obs_alt[bin00[i_alt],0])
                    print((E_width_up[0] - E_width[0]))
                    print('derr', sum(derr))
                    #print('YerrorNorm', sum(transmit_error[bin00[i_alt],index_off2]))
                    #plt.plot(x1_idx2, use_y1, linestyle=':')
                    #plt.plot(x1_idx2, (res_fit_ILS190(x1_idx2))*coef_v, linestyle='--')
                    plt.legend()
                    plt.show()
                    #print, 'derr', derr
                    """
                    
                    
                    ax2.plot(x1_idx2, new_tr1-(res_fit_ILS190_new(x1_idx2))*coef_v)
                    #ax2.axvline(x=abs_line[i_l][0])
                    #ax2.axvline(x=(abs_line[i_l][0]+0.5), linestyle='dashed')
                    #ax2.axvline(x=(abs_line[i_l][0]-0.5), linestyle='dashed')
                    ax2.set_title("Fitting RMS")
                    if order == '189':
                        ax2.set_xlim(4247, 4281)
                    if order == '190':
                        ax2.set_xlim(4283, 4300)
                    #ax2.legend()
                
                if order_num == 188 or order_num == 191:
                    print(order_num)
                    
                    ## 基本はorder 190と同じ。ただしfittingする時に隣のラインをconstantで埋める。 
                    a1 = 1.0753 + pix_num[0]*(1.430e-3)
                    a2 = 0.
                    a3 = 0.1121 + pix_num[0]*(-2.086e-5)
                    a4 = -0.1036 + pix_num[0]*(1.554e-3)
                    a5 = -0.1746 + pix_num[0]*(-4.127e-4)
                    a6 = -3.814e-2 + pix_num[0]*(7.546e-4)

                    init_vals = [abs_line[i_l][0], 1.0]
                    
                    #print(pix_num)
                    coef_v = coef[coef_pix[i_l]] # pixel#として、pixel shiftする前の各lineの位置を知りたい。
                    
                    # x1_idx2のうち、基準をとってconstantを入れたい。
                    # 2つの吸収線の間にあるtransmittanceはその左右に比べてmaxになるはずだから、それを探す。
                    
                    idx_leftside = np.where(x1_idx2 < x1_idx[idx_min])[0].tolist()
                    idx_rightside = np.where(x1_idx2 > x1_idx[idx_min])[0].tolist()
                    
                    idx_min_left = np.argmax(res_y1[idx_leftside])
                    idx_min_right = np.argmax(res_y1[idx_rightside])
                    
                    print('left', x1_idx2[idx_leftside][idx_min_left])
                    print('right', x1_idx2[idx_rightside][idx_min_right])
                    
                    rest_left = x1_idx2[:idx_min_left]
                    rest_right = x1_idx2[idx_rightside[idx_min_right]:]
                    
                    #rest_left[:] = res_y1[idx_leftside][idx_min_left]
                    #rest_right[:] = res_y1[idx_rightside][idx_min_right]
                    
                    # use_y1の形を変更する
                    res_y1[:idx_min_left] = res_y1[idx_leftside][idx_min_left]
                    res_y1[idx_rightside[idx_min_right]:] = res_y1[idx_rightside][idx_min_right]
                    
                    ax6.plot(x1_idx2, res_y1, label=round(obs_alt[bin00[i_alt],0]), linestyle='--', color='k')
                    
                    use_y1 = -1 * (res_y1 - 1)
                    
                    try:
                        best_vals, covar = curve_fit(ILS_190, x1_idx2, use_y1, p0=init_vals, sigma=use_yy_err, absolute_sigma = True, maxfev=10000)
                    except RuntimeError:
                        print("ILS fitting isn't converged.")
                        #input()
                        continue
                    
                    print(best_vals, coef_v)
                        
                    stderr = np.sqrt(np.diag(covar))
                    #print('covar', covar, best_vals)
                    #print(stderr)
                        
                    ## エラーの計算をする
                    def err_ILS190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                    
                        return np.sqrt(f1_x0**2 + f2_c**2)
                    
                    def res_fit_ILS190(wn_xx):
                        return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2)))
                    def res_fit_ILS190_new(wn_x):
                        return best_vals[1]*(a2*exp(-0.5*((wn_x+a0-best_vals[0])/a1)**2) + a5*exp(-0.5*((wn_x-a3-best_vals[0])/a4)**2))

                    #E_width = list(integrate.quad(res_fit_ILS190, x1_idx2[0], x1_idx2[-1]))
                    E_width = list(integrate.quad(res_fit_ILS190_new, x1_idx2[0], x1_idx2[-1]))
                    
                    """
                    def err_up_ILS190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2))) + derr
                    
                    
                    def err_dn_ILS190(wn_xx):
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return best_vals[1]*(a1 * (1/(math.sqrt(2*math.pi)*a3)) * exp(-0.5* (wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4 * (1/(math.sqrt(2*math.pi)*a6)) * exp(-0.5* (wn_xx-a5-best_vals[0])**2 / (a6**2))) - derr
                    
                    def derr_190(wn_xx):
                        
                        f1_x0 = stderr[0]*best_vals[1]*(a1*(wn_xx-a2-best_vals[0])*(1/(a3**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + a4*(wn_xx-a5-best_vals[0])*(1/(a6**3 * math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        f2_c = stderr[1]*(a1*(1/(a3*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a2-best_vals[0])**2 / (a3**2)) + \
                                         a4*(1/(a6*math.sqrt(2*math.pi))) * exp(-0.5*(wn_xx-a5-best_vals[0])**2 / (a6**2)))
                        
                        derr = np.sqrt(f1_x0**2 + f2_c**2)
                        
                        return derr

                    
                    E_width_up = list(integrate.quad(err_up_ILS190, x1_idx2[0], x1_idx2[-1]))
                    E_width_dn = list(integrate.quad(err_dn_ILS190, x1_idx2[0], x1_idx2[-1])) 
                    
                    
                    derr = derr_190(x1_idx2)
                    """
                    
                    ax2.plot(x1_idx2, res_fit_ILS190(x1_idx2)*coef_v)
                    #ax2.axvline(x=abs_line[i_l][0])
                    #ax2.axvline(x=(abs_line[i_l][0]+0.5), linestyle='dashed')
                    #ax2.axvline(x=(abs_line[i_l][0]-0.5), linestyle='dashed')
                    ax2.set_title("Fitting RMS")
                    ax2.set_xlim(4300,4310)#(4242, 4246)
                    #ax2.legend()
                
                """
                if order_num == 148:
                    print(order_num)
                    print("Fitting with theoretical Doppler profile")
                    #print(obs_alt[bin[i_alt],0])
                    init_vals = [0.005, abs_line[i_l][0], 0.8]
                    
                    try:
                        best_vals, covar = curve_fit(gaussian, x1_idx, use_y1, p0 = init_vals, maxfev = 10000)
                    except RuntimeError:
                        print("Gaussian fitting isn't converged.")
                        #input()
                        continue
                    
                    stderr = np.sqrt(np.diag(covar))

                    def res_fit_148(s):
                        return best_vals[0] * exp(-(s-best_vals[1])**2 / (2*best_vals[2]**2))

                    E_width = list(integrate.quad(res_fit_148, x1_idx[0], x1_idx[-1]))
                """
                    
                E_width_alt.append(E_width)
                Alt_prof.append(obs_alt[bin00[i_alt],0])
                
                #E_width_alt_up.append(E_width_up[0] - E_width[0])
                #E_width_alt_dn.append(E_width[0] - E_width_dn[0])
                
                E_width_err.append(error_W)

                """
                # Equivalent widthをAOTF transfer functionの変化を考慮して補正する
                # 見ている吸収線・波長・AOTF central wnの情報が必要
                fit_result = aotf_E['E_correct']
                print(fit_result[:, i_l, 0])
                if i_alt < len(bin00)-1:
                    print(obs_alt[bin00[i_alt],0], obs_alt[bin00[i_alt+1],0])
                    alt_fit = np.where((fit_result[:, i_l, 0] >= obs_alt[bin00[i_alt],0]) & (fit_result[:,i_l,0] < obs_alt[bin00[i_alt+1],0]))[0].tolist()
                    print(alt_fit)
                    print('focusing wavenumber', fit_result[alt_fit, i_l, 1]) 

                    if len(alt_fit) == 0:
                        alt_fit = np.where(fit_result[:,i_l,0] <= obs_alt[bin00[i_alt],0])[0].tolist()
                        alt_fit = np.flipud(alt_fit)

                    E_AOTF_ideal = fit_result[alt_fit[0],i_l,2]*float(ideal_wn)**2 + fit_result[alt_fit[0],i_l,3]*float(ideal_wn) + fit_result[alt_fit[0],i_l,4]
                    E_shift = fit_result[alt_fit[0],i_l,2]*AOTF_cwn[0]**2 + fit_result[alt_fit[0],i_l,3]*AOTF_cwn[0] + fit_result[alt_fit[0],i_l,4]
                
                    Err_AOTF_ideal = fit_result[alt_fit[0], i_l, 5]
                    Err_shift = fit_result[alt_fit[0], i_l, 5]

                    E_correct = E_width[0] * (E_AOTF_ideal/E_shift)
                
                    # 相対誤差を計算
                    E_derr = ((E_width_up[0] - E_width_dn[0])/2)
                    relerr = np.sqrt((E_derr/E_width[0])**2 + (Err_AOTF_ideal/E_AOTF_ideal)**2 + (Err_shift/E_shift)**2)

                    err_Ecor = E_correct*relerr # これが修正されたエラーの大きさ
                
                    print('E change', E_width[0], E_correct)
                    E_width_alt_cor.append(E_correct)    
                    E_width_alt_err_cor.append(err_Ecor)
                    
                else:
                    E_width_alt_cor.append(np.nan)    
                    E_width_alt_err_cor.append(np.nan)
                
                #if i_l == 0:
                #    E_std[i_alt] = E_correct
                
                #plt_x = E_correct/E_std
                """

                
                #ax3.errorbar(E_width_alt, Alt_prof, xerr = [E_width_alt_dn, E_width_alt_up], yerr = None, alpha=1.0, color=color[i_l], label = str(abs_line[i_l][0]))
                ax3.errorbar(E_width_alt, Alt_prof, xerr = E_width_err, yerr = None, alpha=1.0, color=color[i_l], label = str(abs_line[i_l][0]))
                #ax3.plot(E_width_alt_cor, Alt_prof)
                ax3.set_title("Curve of growth")
                ax3.set_xlabel("Equivalent width")
                ax3.set_ylabel("Altitude [km]")
                ax3.set_xscale("log")
                ax3.grid(True)
                #ax3.legend()
                
                Lat_prof.append(Latitude[bin00[i_alt],0])
                Lon_prof.append(Longitude[bin00[i_alt],0])
                LST_prof.append(LST[bin00[i_alt],0])
                Ls_prof.append(LsubS[bin00[i_alt],0])
                
                """
                plt.plot(Alt_prof, E_width_alt, marker='.', linestyle='None', label='wn ~4285')
                plt.yscale('log')
                plt.xlabel("Altitude [km]", fontsize=14)
                plt.ylabel("Equivalent width", fontsize=14)
                plt.legend()
                plt.show()
                """

            #------------------------------------------------------------------------------
            # Get line strength via model temperature by GEM-Mars
            # Use the subroutine calc_S_w_GEM
            # Output; line strength, maximum sigma, Temperature

            # If text file of line strength already calculated, just open its file.

            # search the file
            print(i_l)
            wave_cen = str(abs_line[i_l][0])
            wave_cen = wave_cen[0:4]

            wdir_4_gem = '/Users/yoshidanao/Python/NOMAD/line_strength'
            Gem_list = glob.glob(wdir_4_gem+'/'+str(order_num)+'/'+filename[0:27]+'**'+wave_cen+'**_a652.txt', recursive=True)
            print('Already have file list', Gem_list)

            if len(Gem_list) == 1:
                S_table = np.loadtxt(Gem_list[0], delimiter=',')
                Alt_gem = S_table[:,0]
                Temp_gem = S_table[:,1]
                line_S = S_table[:,2]
                xsec = S_table[:,3]

            if len(Gem_list) == 0:
                #continue # inserted in 
                make_list = Line_strength.calc_S_w_GEM(filename, abs_line, order_num, i_l)
                Gem_list = glob.glob(wdir_4_gem+'/'+str(order_num)+'/'+filename[0:27]+'**'+wave_cen+'**_a652.txt', recursive=True)
                if len(Gem_list) == 1:
                    S_table = np.loadtxt(Gem_list[0], delimiter=',')
                    Alt_gem = S_table[:,0]
                    Temp_gem = S_table[:,1]
                    line_S = S_table[:,2]
                    xsec = S_table[:,3]
                if len(Gem_list) == 0:

                    continue


                #print(Alt_gem)

            #-------------------------------------------------------------------
            # Alt_gemから観測に近い高度範囲だけを抜き出す
            # GEM_MARSの計算結果、高度が逆方向から入ってるから注意

            idx_alt = np.where((Alt_gem >= alt_min) & (Alt_gem <= alt_max))[0].tolist()

            Alt_gem_rev = np.flipud(Alt_gem[idx_alt])
            line_S_rev = np.flipud(line_S[idx_alt])
            xsec_rev = np.flipud(xsec[idx_alt])

            print("modeling=", Alt_gem_rev, len(Alt_gem_rev))
            print("observation=", Alt_prof, len(Alt_prof))
            
            if len(Alt_prof) == 0:
                continue

            #---------------------------------------------------------
            # check the total number of index (from GEM and NOMAD)
            # 変なところで止まるやつまだあるぞ
            if len(Alt_gem_rev) > len(Alt_prof):
                print(int(Alt_gem_rev[0]), int(Alt_prof[0]))
                print(int(Alt_gem_rev[-1]), int(Alt_prof[-1]))
                #四捨五入に変更
                if (Decimal(str(Alt_prof[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)- \
                Decimal(str(Alt_gem_rev[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) >=1:
                    Alt_gem_rev = Alt_gem_rev[1:]
                    line_S_rev = line_S_rev[1:]
                    xsec_rev = xsec_rev[1:]
                    #print('modified', Alt_prof)
                #if (Decimal(str(Alt_prof[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)- \
                #Decimal(str(Alt_gem_rev[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) <1:
                else:
                    Alt_gem_rev = Alt_gem_rev[1:]
                    line_S_rev = line_S_rev[1:]
                    xsec_rev = xsec_rev[1:]
                    count_01 = count_01 + 1

            if len(Alt_gem_rev) < len(Alt_prof):
                if (Decimal(str(Alt_gem_rev[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)- \
                Decimal(str(Alt_prof[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) >=1:
                    Alt_prof = Alt_prof[1:]
                    E_width_alt = E_width_alt[1:]
                    
                    #E_width_alt_dn = E_width_alt_dn[1:]
                    #E_width_alt_up = E_width_alt_up[1:]
                    E_width_err = E_width_err[1:]
                    
                    #E_width_alt_cor = E_width_alt_cor[1:]
                    #E_width_alt_err_cor = E_width_alt_err_cor[1:]
                    print('modified', Alt_prof)
                else:
                #if (Decimal(str(Alt_gem_rev[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)- \
                #Decimal(str(Alt_prof[0])).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) <1:
                    Alt_prof = Alt_prof[1:]
                    E_width_alt = E_width_alt[1:]
                    
                    E_width_err = E_width_err[1:]
                    
                    #E_width_alt_dn = E_width_alt_dn[1:]
                    #E_width_alt_up = E_width_alt_up[1:]
                    
                    #E_width_alt_cor = E_width_alt_cor[1:]
                    #E_width_alt_err_cor = E_width_alt_err_cor[1:]
                    count_02 = count_02 + 1

            # errorに対しても同じ作業が必要!!

            #---------------------------------------------------------
            # Get the column density

            #N_dens = []
            if len(Alt_prof) != len(Alt_gem_rev):
                print("We have a problem with the total number of GEM Mars array and NOMAD array")
                f=open('/Users/yoshidanao/Python/NOMAD/noequal_array_GEM.dat', 'a')
                f.write(filename+'\n')
                f.close()
                continue

            N_dens = E_width_alt / line_S_rev
            
            #N_dens_err_dn = E_width_alt_dn / line_S_rev
            #N_dens_err_up = E_width_alt_up / line_S_rev
            
            N_dens_err = E_width_err / line_S_rev
            
            #N_dens_cor = E_width_alt_cor / line_S_rev
            #N_dens_cor_err = E_width_alt_err_cor / line_S_rev
            
            #if i_l == 0:
            #    N_std = N_dens_cor
            
            #print('N dens', N_dens)
            #print(N_dens_err_dn, N_dens_err_up)
            
            #print('Percentage', N_dens_err_dn/N_dens * 100)
            
            #print('correction', N_dens_cor / N_dens)
            
            #ax4.errorbar(N_dens, Alt_prof, xerr = [N_dens_err_dn, N_dens_err_up], yerr=None, label=str(wave_cen), alpha=1.0)
            ax4.errorbar(N_dens, Alt_prof, xerr = N_dens_err, yerr=None, label=str(wave_cen), alpha=1.0)
            #ax4.plot(N_dens_cor, Alt_prof, label=str(wave_cen))
            ax4.set_ylabel("Altitude [km]")
            ax4.set_xlabel("Slant column density [$cm^{{-}2}$]")
            ax4.set_xscale('log')
            ax4.grid(True)
            ax4.legend()
            
            """
            plt.plot(N_dens, Alt_prof, marker='.', label='4285')
            plt.ylabel("Altitude [km]", fontsize=14)
            plt.xlabel("Column density [$cm^{{-}2}$]", fontsize=14)
            plt.xscale('log')
            plt.grid()
            plt.legend()
            plt.show()
            """

            #---------------------------------------------------------
            # Check the slant opacity

            tau_alt = []

            for i_t in range(len(N_dens)):
                tau = xsec_rev[i_t]*N_dens[i_t]
                tau_alt.append(tau)

            ax5.plot(tau_alt, Alt_prof, marker='.', label=str(wave_cen))
            ax5.set_xlabel("tau")
            ax5.set_ylabel("Altitude [km]")
            ax5.legend()
            ax5.grid(True)
            """
            plt.plot(tau_alt, Alt_prof, marker='.', label='4285')
            plt.xlabel("tau")
            plt.ylabel("Altitude [km]")
            plt.legend()
            plt.show()
            """

            #---------------------------------------------------------
            # save the result
            # for individual absorption line

            mk_ar_order = np.zeros(len(N_dens), dtype=np.int64)
            mk_ar_wn = np.zeros(len(N_dens), dtype=np.int64)

            mk_ar_order[:] = order_num
            mk_ar_wn[:] = wave_cen

            #print(mk_ar_order)
            #print(N_dens)

            results = [[mk_ar_order], [mk_ar_wn], [N_dens], [Alt_prof], [tau_alt]]
            #print(results)

            #np.savez('/Volumes/NekoPen/nomad/retrieval_results/column_dens/'+str(order_num)+'_check/'+filename+"_"+str(wave_cen), mk_ar_order, mk_ar_wn, E_width_alt_cor, N_dens_cor, N_dens_err_dn, N_dens_err_up, Alt_prof, tau_alt, Ls_prof, Lat_prof, Lon_prof, LST_prof)
            #np.savez('/Volumes/KingPen/nomad/retrieval_results/column_dens/'+str(order_num)+'_newcal/'+filename+"_"+str(wave_cen)+"_newfit_newils_areoid_bin1", mk_ar_order, mk_ar_wn, E_width_alt, N_dens, N_dens_err_dn, N_dens_err_up, Alt_prof, tau_alt, Ls_prof, Lat_prof, Lon_prof, LST_prof)
            np.savez('/Volumes/KingPen/nomad/retrieval_results/column_dens/'+str(order_num)+'_211125/'+filename+"_"+str(wave_cen)+"_newfit_newils_areoid_bin1_error", mk_ar_order, mk_ar_wn, E_width_alt, N_dens, N_dens_err, N_dens_err, Alt_prof, tau_alt, Ls_prof, Lat_prof, Lon_prof, LST_prof)
            #np.savez('/Users/yoshidanao/Python/'+filename+"_"+str(wave_cen)+"_aotf", mk_ar_order, mk_ar_wn, E_width_alt, N_dens, N_dens_err_dn, N_dens_err_up, Alt_prof, tau_alt, Ls_prof, Lat_prof, Lon_prof, LST_prof)

            print("Done!!")
    fig.savefig("/Volumes/KingPen/nomad/retrieval_results/figures/"+str(order_num)+"_211125/"+filename+"_"+str(wave_cen)+"_newfit_newils_areoird_bin1.png", dpi=200)
    #plt.show()
            #test = np.load('/Volumes/NekoPen/nomad/retrieval_results/column_dens/'+str(order_num)+'/'+filename+str(wave_cen)+'.npz')
            #print(test)
                       
print(count_01, count_02)