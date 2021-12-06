"""

This module contain function related to the NOMAD SO instrument

"""

import numpy as np
from matplotlib import pyplot as plt

def freq_mp(m, p, p0=0., F0=22.473422, F1=5.559526e-4, F2=1.751279e-8):
  """ maps order and pixel to wavenumber

  Parameters
  ----------
  m : int
    difraction order
  p : int (or array)
    pixel(s)
  p0 : float
    first pixel (or pixel offset

  Returns
  -------
  float
    wavenumber(s)

  """
  f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
  return f


def pixel_mf(m, f, p0=0., F0=22.473422, F1=5.559526e-4, F2=1.751279e-8):
  """ maps order and wavenumber to pixel

  Parameters
  ----------
  m : int
    difraction order
  f : float (or array)
    wavenumber(s)
  p0 : float
    first pixel (or pixel offset

  Returns
  -------
  float
    pixels(s)

  """
  p = -p0 + (-F1 + np.sqrt(F1**2 - 4.*F2*(F0-f/m)))/(2*F2)
  return p


def p0_blaze(m, b0=160.25, b1=0.23, b2=0.0):
  """ maps order to blaze central pixel

  Parameters
  ----------
  m : int
    difraction order


  Returns
  -------
  float
    blaze central pixel

  """
  p0 = b0 + m*(b1 + m*b2)
  return p0


#def wp_blaze(m, wp0=730.02044, wp1=-4.76191969, wp2=0.0100485578):
def wp_blaze(m, wp0=811.133822, wp1=-5.29102188, wp2=0.0111650642):
  """ maps order blaze width

  Parameters
  ----------
  m : int
    difraction order

  Returns
  -------
  float
    blaze width

  """
  wp = wp0 + m*(wp1 + m*wp2)
  return wp


def F_blaze(m, p, p0=None, wp=None):
  """ maps order and pixel to blaze shape

  Parameters
  ----------
  m : int
    difraction order
  p : float (or array)
    pixels(s)
  p0 : float (optional)
    blaze central pixel
  wp : float (optional)
    blaze width

  Returns
  -------
  float
    blaze shape

  """
  if p0 == None:
    p0 = p0_blaze(m)
  if wp == None:
    wp = wp_blaze(m)
  F = np.sinc((p-p0)/wp)**2
  return F


A_aotf = {
  110:14332, 111:14479, 112:14627, 113:14774, 114:14921, 115:15069, 116:15216, 117:15363, 118:15510, 119:15657,
  120:15804, 121:15951, 122:16098, 123:16245, 124:16392, 125:16539, 126:16686, 127:16832, 128:16979, 129:17126,
  130:17273, 131:17419, 132:17566, 133:17712, 134:17859, 135:18005, 136:18152, 137:18298, 138:18445, 139:18591,
  140:18737, 141:18883, 142:19030, 143:19176, 144:19322, 145:19468, 146:19614, 147:19761, 148:19907, 149:20052,
  150:20198, 151:20344, 152:20490, 153:20636, 154:20782, 155:20927, 156:21074, 157:21219, 158:21365, 159:21510,
  160:21656, 161:21802, 162:21947, 163:22093, 164:22238, 165:22384, 166:22529, 167:22674, 168:22820, 169:22965,
  170:23110, 171:23255, 172:23401, 173:23546, 174:23691, 175:23836, 176:23981, 177:24126, 178:24271, 179:24416,
  180:24561, 181:24706, 182:24851, 183:24996, 184:25140, 185:25285, 186:25430, 187:25575, 188:25719, 189:25864,
  190:26008, 191:26153, 192:26297, 193:26442, 194:26586, 195:26731, 196:26875, 197:27019, 198:27163, 199:27308,
  200:27452, 201:27596, 202:27740, 203:27884, 204:28029, 205:28173, 206:28317, 207:28461, 208:28605, 209:28749,
  210:28893,
}
"""
AOTF driving frequencies as a dict 
"""



def nu0_aotf(A, G0=313.91768, G1=0.1494441, G2=1.340818E-7): 
  """ maps AOTF driving frequency to AOTF central wavenumber

  Parameters
  ----------
  A : float
    AOTF driving frequency

  Returns
  -------
  float
    AOTF central wavenumber

  """
  nu0 = G0 + A*(G1 + A*G2)
  return nu0


def w0_aotf(nu0, a0=2.18543e1, a1=5.82007e-4, a2=-2.18387e-7):
  """ maps AOTF center wavenumber to AOTF width

  Parameters
  ----------
  nu0 : float
    AOTF central wavenumber

  Returns
  -------
  float
    AOTF width

  """
  w0 = a0 + nu0*(a1 + a2*nu0)
  return w0


def slf_aotf(nu0, a0=4.24031, a1=-2.24849e-3, a2=4.25071e-7):
  """ maps AOTF center wavenumber to AOTF side lobe factor

  Parameters
  ----------
  nu0 : float
    AOTF central wavenumber

  Returns
  -------
  float
    AOTF side lobe factor

  """
  slf = a0 + nu0*(a1 + a2*nu0)
  return slf


def offset_aotf(nu0, a0=-4.99704e-1, a1=2.80952e-4, a2=-3.51707e-8):
  """ maps AOTF center wavenumber to AOTF offset

  Parameters
  ----------
  nu0 : float
    AOTF central wavenumber

  Returns
  -------
  float
    AOTF offset

  """
  offset = np.max([a0 + nu0*(a1 + a2*nu0),0.])
  return offset


def F_aotf_goddard18b(m, nu, A=None, nu0=None, w0=None, slf=None, offset=None):
  """ maps order and wavenumber to AOTF shape

  Parameters
  ----------
  m : int
    difraction order
  nu : float (or array)
    wavenumbers

  Returns
  -------
  float
    AOTF shape

  """
  if A is None:
    A = A_aotf[m]
  if nu0 is None:
    nu0 = nu0_aotf(A)
  if w0 is None:
    w0 = w0_aotf(nu0)
  if slf is None:
    slf = slf_aotf(nu0)
  if offset is None:
    offset = offset_aotf(nu0)
  #print(w0, slf, offset)

  F = np.sinc((nu-nu0)/w0)**2
  
  #plt.plot(nu, F, '-')
  
  F[np.abs(nu-nu0)>=w0] *= slf
    
  #plt.plot(nu, F, ':')

  F = offset + (1.-offset)*F
    
  #plt.plot(nu, F, '--')

  #F += offset
  #F /= np.max(F)
    
  #plt.show()
  #print('AOTF function', F.shape)

  return F

def F_aotf_3sinc(m, nu, A=None, nu0=None, w0=None, I1=0.19, nu1=35.0, w1=21.0, I2=0.33, nu2=-35.0, w2=20.):

  if A is None:
    A = A_aotf[m]
  if nu0 is None:
    nu0 = nu0_aotf(A)
  if w0 is None:
    w0 = w0_aotf(nu0)

  F = np.sinc((nu-nu0)/w0)**2 + I1*np.sinc((nu-nu0-nu1)/w1)**2 + I2*np.sinc((nu-nu0-nu2)/w2)**2

  return F



