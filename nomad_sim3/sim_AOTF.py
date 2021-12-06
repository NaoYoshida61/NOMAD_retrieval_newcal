# name; sim_AOTF.py
# purpose; SO_simulation.pyを元に、Solar irradiance, instrumentに関連する部分のみ計算するように変更（つまり、大気状態の計算は特にしない）

# input; AOTF CWN, order#


from __future__ import print_function

import os
import sys
from matplotlib import pyplot as plt


# LOG_LEVEL: 0->None, 1->warning, 2->info, 3->debug
LOG_LEVEL = 2


def get_solar_hr(nu_hr, solar_file=None, **kwargs):
    '''  '''
    import numpy as np
    from scipy import interpolate

    if solar_file == None:
        import nomadtools
        solar_file = os.path.join(nomadtools.rcParams['paths.dirSolar'], 'Solar_irradiance_ACESOLSPEC_2015.dat')

    if LOG_LEVEL >= 2:
        print('Reading in solar file %s'%solar_file)

    nu_solar = []
    I0_solar = []
    nu_min = nu_hr[0] - 1.
    nu_max = nu_hr[-1] + 1.
    with open(solar_file) as f:
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        for line in f:

            nu, I0 = [float(val) for val in line.split()]
            if nu < nu_min:
                continue
            if nu > nu_max:
                break
            nu_solar.append(nu)
            I0_solar.append(I0)
  
    f_solar = interpolate.interp1d(nu_solar, I0_solar)
    I0_solar_hr = f_solar(nu_hr)

    return I0_solar_hr

class NOMAD_sim3(object):
    def __init__(self, order=121, adj_orders=2, **kwargs):
        #molecule='H2O', str_min=1.0e-25, iso_num=None, vmr=None,
        #apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime',
        #TangentAlt=None, spec_res=None, pixel_shift=0.0
        

        import numpy as np
        #from scipy import interpolate

        from nomad_sim3.NOMAD_instrument import freq_mp

        #
        self.order = order
        self.adj_orders = adj_orders

        #nu0_shift = nu0_shift # inserted by nao

        nu_hr_min = freq_mp(order-adj_orders, 0) - 5.
        nu_hr_max = freq_mp(order+adj_orders, 320.) + 5.
        #print('101行目', nu_hr_min, nu_hr_max)
        dnu = 0.001
        Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
        #print('Nbnu_hr', Nbnu_hr)
        nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
        dnu = nu_hr[1]-nu_hr[0]
        #print(nu_hr, dnu)

        self.Nbnu_hr = Nbnu_hr
        self.nu_hr = nu_hr
        self.dnu = dnu

        #
        if LOG_LEVEL >= 2:
          print('hih resolution range %.1f to %.1f (with %d points)' % (nu_hr_min, nu_hr_max, Nbnu_hr))
        I0_solar_hr = get_solar_hr(nu_hr, **kwargs)
        self.I0_hr = I0_solar_hr

        #
        self.init_instrument(**kwargs)
        
        
    def init_instrument(self, aotf, temp, pixel_shift=0.0, ils_type='sg', spec_res=None, dirInstrument='', ilsFile=None, conv_method="Wsplit", **kwargs):

        import numpy as np
        from nomad_sim3.NOMAD_instrument import freq_mp, F_blaze
        from nomad_sim3.instrument_func_newcal import AOTF_CWN_new, F_blaze_new

        order = self.order
        adj_orders = self.adj_orders
        #temp = self.temp #これでいいのか？
        #line = self.line

        #
        pixels = np.arange(320)
        NbP = len(pixels)
        self.pixels = pixels
        self.NbP = NbP
        self.nu_p = freq_mp(order, pixels, p0=pixel_shift)


        if ils_type == 'sg':
          #
          if spec_res is None:
              nu_m = np.mean(self.nu_p)
              R = 4.59995e4 + nu_m*(-1.73554e1 + nu_m*2.15817e-3)
              spec_res = nu_m/R
              if LOG_LEVEL >= 2:
                  print("Using ILS FWHM = %.3f cm-1 (R=%.1f)" % (spec_res,R))
          else:
              if LOG_LEVEL >= 2:
                  print("Using ILS FWHM = %.3f cm-1" % spec_res)
                  sconv = spec_res/2.355
              
        elif ils_type == 'dg':

            if dirInstrument == '':
                import nomadtools
                dirInstrument = nomadtools.rcParams['paths.dirInstrument']

            if ilsFile == None:
                raise Exception('ilsFile not given')

            filename = os.path.join(dirInstrument, ilsFile)
            a1, a2, a3, a4, a5, a6 = np.loadtxt(filename, comments='#', unpack=True)
            if LOG_LEVEL >= 2:
                print("Using ILS from file: %s" % ilsFile)

            else:
                raise Exception('ils_type not recognized')

        #
        self.conv_method = conv_method
        if LOG_LEVEL >= 2:
            print("Computing convolution matrix (conv_method='%s')"%conv_method)

        if conv_method == "Wfull":
            #
            W_conv = np.zeros((self.NbP,self.Nbnu_hr))
            for iord in range(order-adj_orders, order+adj_orders+1):
                nu_pm = freq_mp(iord, pixels, p0=pixel_shift)
                if LOG_LEVEL >= 2:
                    print('order %d: %.1f to %.1f' % (iord, nu_pm[0], nu_pm[-1]))
                W_blaze = F_blaze(iord, pixels)
                for ip in pixels:
                    inu1 = np.searchsorted(self.nu_hr, nu_pm[ip] - 5.*sconv)
                    inu2 = np.searchsorted(self.nu_hr, nu_pm[ip] + 5.*sconv)
                    if ils_type == 'sg':
                        ilsF = 1.0/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(self.nu_hr[inu1:inu2]-nu_pm[ip])**2/(2.*sconv**2))
                    elif ils_type == 'dg':
                        ilsF = a1[ip]/(np.sqrt(2.*np.pi)*a3[ip])*np.exp(-0.5*((self.nu_hr[inu1:inu2]-a2[ip]-nu_pm[ip])/a3[ip])**2) \
                          + a4[ip]/(np.sqrt(2.*np.pi)*a6[ip])*np.exp(-0.5*((self.nu_hr[inu1:inu2]-a5[ip]-nu_pm[ip])/a6[ip])**2)
                    else:
                        raise Exception('ils_type not recognized')
                    W_conv[ip,inu1:inu2] += (W_blaze[ip]*self.dnu)*ilsF
            self.W_conv = W_conv

        elif conv_method == "Wsplit":
            #
            if ils_type == 'sg':
                Nbnu_w = int(np.ceil(2*5.*sconv/self.dnu))
            elif ils_type == 'dg':
                sconv = 0.1
                Nbnu_w = int(np.ceil(2*5.*sconv/self.dnu))
                print('Nbnu_w', Nbnu_w)
            NbTotalOrders = 2*adj_orders + 1
            W2_conv = np.zeros((NbTotalOrders, self.NbP, Nbnu_w))
            W2_conv_inu1 = np.zeros((NbTotalOrders, self.NbP), dtype=int)

            for i in range(NbTotalOrders):
                iord = order - adj_orders + i
                nu_pm = freq_mp(iord, pixels, p0=pixel_shift)
                if LOG_LEVEL >= 2:
                    print('W2_conv, order %d: %.1f to %.1f' % (iord, nu_pm[0], nu_pm[-1]))
                W_blaze = F_blaze(iord, pixels)
                
                #-------------------------------------------------
                # New recipe: Blaze shape change with AOTF temp
                
                W_blaze = F_blaze_new(iord, pixels, aotf, temp)
                
                #print(W_blaze)
                #plt.plot(pixels, W_blaze)
                #plt.show()
                for ip in pixels:
                    inu1 = np.searchsorted(self.nu_hr, nu_pm[ip] - 5.*sconv)
                    inu2 = inu1 + Nbnu_w
                    #print(self.nu_hr[inu1:inu2])
                    if ils_type == 'sg':
                        ilsF = 1.0/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(self.nu_hr[inu1:inu2]-nu_pm[ip])**2/(2.*sconv**2))
                    elif ils_type == 'dg':
                        ilsF = a1[ip]/(np.sqrt(2.*np.pi)*a3[ip])*np.exp(-0.5*((self.nu_hr[inu1:inu2]-a2[ip]-nu_pm[ip])/a3[ip])**2) \
                          + a4[ip]/(np.sqrt(2.*np.pi)*a6[ip])*np.exp(-0.5*((self.nu_hr[inu1:inu2]-a5[ip]-nu_pm[ip])/a6[ip])**2)
                    #print(self.nu_hr[inu1:inu2])
                    #print(nu_pm[ip])
                    #print(len(ilsF))
                    #plt.plot(self.nu_hr[inu1:inu2], ilsF)
                    #plt.show()
                    else:
                        raise Exception('ils_type not recognized')
                    W2_conv[i,ip,:] = (W_blaze[ip]*self.dnu)*ilsF
                    #print(len(W2_conv[i,ip,:]))
                    #print('blaze', W_blaze[ip])
                    #print('かけざん', W_blaze[ip]*self.dnu)
                    W2_conv_inu1[i,ip] = inu1

            if LOG_LEVEL >= 2:
                print("Nbnu_w = ", Nbnu_w)
                #print("W2_conv", len(W2_conv))
                #print("ILS F", len(ilsF))
                self.Nbnu_w = Nbnu_w
                self.NbTotalOrders = NbTotalOrders
                self.W2_conv = W2_conv
                self.W2_conv_inu1 = W2_conv_inu1

        else:
            raise Exception("conv_method '%s' not supported"%self.conv_method)
        
    def forward_model(self, temp, aotf, nu0_shift):

        import numpy as np

        if LOG_LEVEL >= 2:
            print("Forward model")

        self.forward_model_instrument(temp, aotf, nu0_shift)
        
    def forward_model_instrument(self, temp, aotf, nu0_shift):

        import numpy as np
        from nomad_sim3.NOMAD_instrument import F_aotf_goddard18b
        from scipy import interpolate
        
        from nomad_sim3.instrument_func_newcal import AOTF_CWN_new
    
        if self.conv_method == "Wfull":
            # 
            W_aotf = F_aotf_goddard18b(self.order, self.nu_hr)
            I0_hr = W_aotf * self.I0_hr       # nhr
            I0_p = np.matmul(self.W_conv, I0_hr)  # np x 1
            I_hr = I0_hr[None,:] * self.Trans_hr  # nz x nhr
            I_p = np.matmul(self.W_conv, I_hr.T).T  # nz x np
            self.Trans_p = I_p / I0_p[None,:]     # nz x np

        elif self.conv_method == "Wsplit":
            #
            """
            print('original AOTF function')
            print(nu0_shift)
            W_aotf = F_aotf_goddard18b(self.order, self.nu_hr, offset=0., nu0 = nu0_shift) # nomad_instrumentから
            
            plt.plot(self.nu_hr, W_aotf)
            plt.show()
            """

            #-------------------------------------------------------------------------
            # Input new AOTF function (new Goddard) # shift考慮されてる？
            """
            Goddard_n_aotf = "/Users/yoshidanao/Documents/Plan_of_study/Homopause_NOMAD/AOTF_test/AOTF_Goddard2021Apri_Order190.dat"
            print('We are reading new Goddard AOTF function')
            AOTF_data = np.loadtxt(Goddard_n_aotf, skiprows=1)

            dwn_aotf = AOTF_data[:,0]
            dwn_aotf = dwn_aotf + nu0_shift
            func_aotf = AOTF_data[:,1]
            func_aotf = np.flipud(func_aotf)

            # interpolate the AOTF function with d = 0.001
            f_interp = interpolate.interp1d(dwn_aotf, func_aotf, kind='linear')

            #plt.plot(dwn_aotf, func_aotf)
            #plt.plot(self.nu_hr, f_interp(self.nu_hr), ':')
            #plt.show()

            W_aotf = f_interp(self.nu_hr)
            """

            #---------------------------------------------------------------
            # Input Shohei's AOTF function
            """
            Shohei_n_aotf = "/Users/yoshidanao/Documents/Plan_of_study/Homopause_NOMAD/AOTF_test/Shohei/AOTF_Function_Order151_test.dat"

            print("We are reading Shohei's AOTF function")
            print(nu0_shift)

            AOTF_data = np.loadtxt(Shohei_n_aotf)

            dwn_aotf = AOTF_data[:,0]
            dwn_aotf = dwn_aotf + nu0_shift
            func_aotf = AOTF_data[:,1]
            func_aotf = np.flipud(func_aotf)

            # interpolate the AOTF function with d = 0.001
            f_interp = interpolate.interp1d(dwn_aotf, func_aotf, kind='linear')

            #plt.plot(dwn_aotf, func_aotf)
            #plt.plot(self.nu_hr, f_interp(self.nu_hr), ':')
            #plt.show()

            W_aotf = f_interp(self.nu_hr)
            """
            #---------------------------------------------------------------
            # Input PSG AOTF function
            # orderごとに違う値を入れるので注意
            """
            Goddard_n_aotf = "/Users/yoshidanao/Python/NOMAD/calibration/new_goddard_"+str(self.order)+"_order+-4.dat"
            
            print("We are reading Goddard's new AOTF function order+-4")
            print(Goddard_n_aotf[-15:])
            print(nu0_shift)

            AOTF_data = np.loadtxt(Goddard_n_aotf)

            dwn_aotf = AOTF_data[:,0]
            dwn_aotf = dwn_aotf + nu0_shift
            func_aotf = AOTF_data[:,1]
            #func_aotf = np.flipud(func_aotf)

            # interpolate the AOTF function with d = 0.001
            f_interp = interpolate.interp1d(dwn_aotf, func_aotf, kind='linear')

            #fig = plt.figure()
            #plt.plot(dwn_aotf, func_aotf)
            #plt.plot(self.nu_hr, f_interp(self.nu_hr))
            #plt.show()

            W_aotf = f_interp(self.nu_hr)
            """
            # New recipe
            # 2021 Oct. 13
            Goddard_n_aotf = "/Users/yoshidanao/Python/NOMAD/calibration/new_goddard_"+str(self.order)+"_order+-4_211012.dat"
            
            print("We are reading Goddard's new AOTF function order+-4 with new recipe")
            print(Goddard_n_aotf[-25:])
            print(nu0_shift)

            AOTF_data = np.loadtxt(Goddard_n_aotf)
            
            # AOTF CWNを計算する必要がある
            
            nu0_shift = AOTF_CWN_new(aotf, temp)
            print("nu0_shift", nu0_shift, aotf, temp)
            
            dwn_aotf = AOTF_data[:,0]
            dwn_aotf = dwn_aotf + nu0_shift # nu0_shiftがAOTF CWNに相当
            
            func_aotf = AOTF_data[:,1]
            #func_aotf = np.flipud(func_aotf)

            # interpolate the AOTF function with d = 0.001
            f_interp = interpolate.interp1d(dwn_aotf, func_aotf, kind='linear')

            #fig = plt.figure()
            #plt.plot(dwn_aotf, func_aotf, label='aotf')
            #plt.plot(self.nu_hr, f_interp(self.nu_hr))
            #plt.show()
            #print(f_interp)
            W_aotf = f_interp(self.nu_hr)
            

            #
            # go-back to original program
                
            I0_hr = W_aotf * self.I0_hr       # nhr
            print('solar irradiance matrix', I0_hr.shape)

            I0_p = np.zeros(self.NbP)
      
            I0_p_order = np.zeros((self.NbP, self.NbTotalOrders))


            # AOTF central wavenumberの変動を見るため
            aotf_conv = np.zeros((self.NbP, self.NbTotalOrders))
            #total = np.zeros((self.NbP))
    
            for iord in range(self.NbTotalOrders):
                for ip in range(self.NbP):
                    inu1 = self.W2_conv_inu1[iord,ip]
                    inu2 = inu1 + self.Nbnu_w
                    I0_p[ip] += np.sum(I0_hr[inu1:inu2]*self.W2_conv[iord,ip,:])

                    I0_p_order[ip,iord] += np.sum(I0_hr[inu1:inu2]*self.W2_conv[iord,ip,:]) #残す

                    aotf_conv[ip, iord] += np.sum(W_aotf[inu1:inu2]*self.W2_conv[iord,ip,:])
                    
                    #total[ip] += I0_p_order[ip,iord]
                    
            #pix_num = np.arange(320)
            #plt.plot(pix_num, I0_p_order[:,4]/total[:])
            #plt.show()

            self.order_so_ir = I0_p_order
      
        else:
            raise Exception("conv_method '%s' not supported"%self.conv_method)