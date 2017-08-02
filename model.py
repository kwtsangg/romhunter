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

import gwhunter.utils.uniConst as uniConst

#===============================================================================
#  Main
#===============================================================================

    #===============================#
    # Amplification factor for point mass lens 
    #===============================#
  
def AmpFactor_PointMassLens(freqList, M_Lz, y):
  mpmath.pretty = True
  xm   = (y + np.sqrt(y*y + 4.))/2.
  phim = (xm - y)*(xm - y)/2. - np.log(xm)

  vecList = []
  for ifreq in freqList:
    w  = 8.*np.pi*M_Lz*ifreq*uniConst.Msun_time
#    vecList.append( np.exp(np.pi*w/4. + 1j*w/2.*(np.log(w/2.)-2.*phim)) * mpmath.gamma(1.-1j*w/2.) * mpmath.hyp1f1(1j*w/2., 1, 1j*w*y*y/2., maxterms=100000) )
    vecList.append( complex(mpmath.gamma(1.-1j*w/2.) * mpmath.hyp1f1(1j*w/2., 1, 1j*w*y*y/2., maxterms=100000)) )
  return vecList
