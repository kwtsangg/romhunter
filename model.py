#!/usr/bin/env python
__file__       = "model.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Jul-27"

Description=""" To store the function for evaluation.
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import mpmath

#===============================================================================
#  Main
#===============================================================================

    #===============================#
    # Amplification factor for point mass lens 
    #===============================#

"""
  arxiv :: astro-ph/0305055.pdf
"""
def AmpFactor_PointMassLens(freqList, M_Lz, y):
  mpmath.pretty = True
  xm   = (y + np.sqrt(y*y + 4.))/2.
  phim = (xm - y)*(xm - y)/2. - np.log(xm)
  Msun_time = 4.92567e-6

  vecList = []
  for ifreq in freqList:
    w = 8.*np.pi*M_Lz*ifreq*Msun_time
    if abs(w) > 246.:
      vecList.append( AmpFactor_PointMassLens_geometrical(ifreq, M_Lz, y) )
    else:
      vecList.append( np.exp(np.pi*w/4. + 1j*w/2.*(np.log(w/2.)-2.*phim)) * complex(mpmath.gamma(1.-1j*w/2.) * mpmath.hyp1f1(1j*w/2., 1, 1j*w*y*y/2., maxterms=100000)) )

  return vecList

def AmpFactor_PointMassLens_geometrical(f, M_Lz, y):
  y2 = y*y
  abs_mu_p = np.abs(0.5 + (y2+2.)/(2.*y*np.sqrt(y2+4.)))
  abs_mu_m = np.abs(0.5 - (y2+2.)/(2.*y*np.sqrt(y2+4.)))
  dt_d     = 4.*M_Lz*( y*np.sqrt(y2+4.) + np.log( (np.sqrt(y2+4.)+y)/(np.sqrt(y2+4.)-y) ) )
  return np.sqrt(abs_mu_p) - 1j * np.sqrt(abs_mu_m) * np.exp(1j*2.*np.pi*f*dt_d)


