#!/usr/bin/env python
__file__       = "trainingset.pyx"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Jan-30"

Description=""" To get the frequency vector at each point in phase space for different waveform models.
"""

#===============================================================================
#  Module
#===============================================================================
import sys
from gwhunter.utils.general import printf
from gwhunter.waveform.lalwaveform import IMRPhenomPv2FD
import gwhunter.utils.vector as vec

#===============================================================================
#  Main
#===============================================================================

"""
  Input:
    freqList     : a list to store frequency nodes being evaluated
    paramsMatrix : a matrix containing parameters needed for that model
    modelName    : item in [ "IMRPhenomPv2FD" ]
    modelTag     : item in [ "hp", "hc", "hphp", "hphc", "hchc", "hchc", "hpPlushcSquared" ]
  Output:
    vecMatrix    : a list of vec.vector1D (defined in vectorUtils.py)
"""
def evaluateModel(freqList, paramsMatrix, modelName, modelTag):
  vecMatrix = []
  if modelName == "IMRPhenomPv2FD":
    for iparams in paramsMatrix:
      iVecList  = IMRPhenomPv2FD(freqList, iparams[0], iparams[1], iparams[2], iparams[3], iparams[4], iparams[5], iparams[6])
      vecMatrix.append(evaluateModelTag(iVecList, modelTag))
  else:
    printf("In evaluateModel, the model (%s) is not supported. Exiting ...", __file__, "error") 
    sys.exit()
  return vecMatrix

"""
  Input:
    iVecList  : The list with component [0] to be hplus vector list and component [1] to be hcross vector list
    modelTag  : item in [ "hp", "hc", "hphp", "hphc", "hchc", "hchc", "hpPlushcSquared" ]
  Output:
    a vec.vector1D object (defined in vectorUtils.py) with tag considered.
"""
def evaluateModelTag(iVecList, modelTag):
  # hp
  if modelTag == "hp":
    iVechp = vec.vector1D(iVecList[0])
    return iVechp
  # hc
  elif modelTag == "hc":
    iVechc = vec.vector1D(iVecList[1])
    return iVechc
  # hp * conj(hp)
  elif modelTag == "hphp":
    iVechp = vec.vector1D(iVecList[0])
    return iVechp*(iVechp.conj())
  # hp * conj(hc)
  elif modelTag == "hphc":
    iVechp = vec.vector1D(iVecList[0])
    iVechc = vec.vector1D(iVecList[1])
    return iVechp*(iVechc.conj())
  # hc * conj(hp)
  elif modelTag == "hchp":
    iVechp = vec.vector1D(iVecList[0])
    iVechc = vec.vector1D(iVecList[1])
    return iVechc*(iVechp.conj())
  # hc * conj(hc)
  elif modelTag == "hchc":
    iVechc = vec.vector1D(iVecList[1])
    return iVechc*(iVechc.conj())
  # (hp+hc) * conj(hp+hc)
  elif modelTag == "hpPLUShcSquared":
    iVechp = vec.vector1D(iVecList[0])
    iVechc = vec.vector1D(iVecList[1])
    iVechpPLUShc = iVechp + iVechc
    return iVechpPLUShc*(iVechpPLUShc.conj())
  else:
    printf("In evaluateModelTag, the modelTag (%s) is not supported. Exiting...", __file__, "error")
    sys.exit()

