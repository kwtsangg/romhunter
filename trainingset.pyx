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
from gwhunter.utils.generalPythonFunc import printf
from gwhunter.waveform.waveform import IMRPhenomPv2FD
import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================

"""
  Input:
    freqList          : a list to store frequency nodes being evaluated
    TSParams_FilePath : a file containing parameters needed for that model
    modelName         : item in [ "IMRPhenomPv2FD" ]
    modelTag          : item in [ "hp", "hc", "hphp", "hphc", "hchc", "hchc", "hpPlushcSquared" ]
  Output:
    TSMatrix          : a list of vec.vector1D (defined in vectorUtils.py)
"""
def evaluateModel(freqList, TSParams_FilePath, modelName, modelTag):
  TSMatrix = []
  if modelName == "IMRPhenomPv2FD":
    TSParams_File = open(TSParams_FilePath, "r")
    for iTSParams in TSParams_File:
      iTSParams = iTSParams.split()
      for j in xrange(len(iTSParams)):
        iTSParams[j] = float(iTSParams[j])
      iTSList  = IMRPhenomPv2FD(freqList, iTSParams[0], iTSParams[1], iTSParams[2], iTSParams[3], iTSParams[4], iTSParams[5], iTSParams[6])
      TSMatrix.append(evaluateModelTag(iTSList, modelTag))
    TSParams_File.close()
  else:
    printf("In evaluateModel, the model (%s) is not supported. Exiting ...", __file__, "error") 
    sys.exit()
  return TSMatrix

"""
  Input:
    iTSList  : The trainingset vector list with component [0] to be hp and component [1] to be hc
    modelTag : item in [ "hp", "hc", "hphp", "hchc", "hphc", "hpPlushcSquared" ]
  Output:
    a vec.vector1D object (defined in vectorUtils.py) with tag considered.
"""
def evaluateModelTag(iTSList, modelTag):
  # hp
  if modelTag == "hp":
    iTSVechp = vec.vector1D(iTSList[0])
    return iTSVechp.unitVector()
  # hc
  elif modelTag == "hc":
    iTSVechc = vec.vector1D(iTSList[1])
    return iTSVechc.unitVector()
  # hp * conj(hp)
  elif modelTag == "hphp":
    iTSVechp = vec.vector1D(iTSList[0])
    return (iTSVechp*(iTSVechp.conj())).unitVector()
  # hp * conj(hc)
  elif modelTag == "hphc":
    iTSVechp = vec.vector1D(iTSList[0])
    iTSVechc = vec.vector1D(iTSList[1])
    return (iTSVechp*(iTSVechc.conj())).unitVector()
  # hc * conj(hp)
  elif modelTag == "hchp":
    iTSVechp = vec.vector1D(iTSList[0])
    iTSVechc = vec.vector1D(iTSList[1])
    return (iTSVechc*(iTSVechp.conj())).unitVector()
  # hc * conj(hc)
  elif modelTag == "hchc":
    iTSVechc = vec.vector1D(iTSList[1])
    return (iTSVechc*(iTSVechc.conj())).unitVector()
  # (hp+hc) * conj(hp+hc)
  elif modelTag == "hpPLUShcSquared":
    iTSVechp = vec.vector1D(iTSList[0])
    iTSVechc = vec.vector1D(iTSList[1])
    iTSVechpPLUShc = iTSVechp + iTSVechc
    return (iTSVechpPLUShc*(iTSVechpPLUShc.conj())).unitVector()
  else:
    printf("In evaluateModelTag, the modelTag (%s) is not supported. Exiting...", __file__, "error")
    sys.exit()

