#!/usr/bin/env python
__file__       = "waveformModel.py"
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
import os, sys
import numpy as np
from gwhunter.utils.generalPythonFunc import printf
from gwhunter.waveform.waveform import IMRPhenomPv2FD
import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================

def evaluateModel(modelName, modelTag):
  if modelName == "IMRPhenomPv2":
    None
  else:
    printf("The model evaludation of (%s) is not supported. Exiting ...", __file__, "error") 
    sys.exit()

def generateTS(modelName = "IMRPhenomPv2", modelTag="hpReal"):
  supportedModel = [ "IMRPhenomPv2" ]
  supportedTag   = [ "plus", "cross", "hphp", "hchc", "hphc", "hpPlushcSquared" ]

  freqList = np.linspace(20.,1024.,4017)

  TSVec_File = open(os.getcwd() + "/input/TSVec_" + modelName + "_" + modelTag + ".txt", "w")
  TSParams_File = open(os.getcwd() + "/input/TSParams_" + modelName + "_" + modelTag + ".txt", "r")
  for irow in TSParams_File:
    irow = irow.split()
    for j in range(0, len(irow)):
      irow[j] = float(irow[j])
    iTSVec = vec.vector1D(IMRPhenomPv2FD(freqList, irow[0], irow[1], irow[2], irow[3], irow[4], irow[5], irow[6])[0].real)
    TSVec_File.write(iTSVec.printComponent())
  TSParams_File.close()
  TSVec_File.close()

def main():
  generateTS()

if __name__ == "__main__":
  main()
