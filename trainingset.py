#!/usr/bin/env python
__file__       = "trainingset.py"
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
import gwhunter.utils.general as gu
from gwhunter.waveform.lalwaveform import IMRPhenomPv2FD
import gwhunter.utils.vector as vec
import gwhunter.utils.randomNumber as rand
import gwhunter.utils.dataFile as df

#===============================================================================
#  Main
#===============================================================================

def buildingTrainingset(outputdir, filePath, columnSequence, paramsDict):
  paramsMatrix = []

  paramsMatrix_before = []
  columnIndexFromFile = []
  columnIndexFromRand = []
  
  # Check whether you need to generate a new trainingset.
  # If you dont need, you can just copy the filePath as your trainingset.
  needToGenerateNewTS = False
  for icolumn in columnSequence:
    if paramsDict[icolumn]["method"] != "file":
      needToGenerateNewTS = True
      break

  if needToGenerateNewTS:
    # Obtain array from file or generate from random number
    for icolumn in columnSequence:
      myrand = rand.randomhunter()
      # from file
      if paramsDict[icolumn]["method"] = "file":
        paramsMatrix_before.append(df.datahunter(filePath).getColumn(paramsDict[icolumn]["column"]))
        columnIndexFromFile.append(columnSequence.index(icolumn))
      # from random number
      else:
        paramsMatrix_before_tmp = []
        for i in int(paramsDict[icolumn]["numberOfPoints"]):
          paramsMatrix_before_tmp.append(myrand.getNumber(paramsDict[icolumn]["min"], paramsDict[icolumn]["max"], paramsDict[icolumn]["method"]))
        paramsMatrix_before.append(paramsMatrix_before_tmp)
        columnIndexFromRand.append(columnSequence.index(icolumn))

    # Generate the whole paramsMatrix
    if len(columnIndexFromFile)>0:
      for i in xrange(len(paramsMatrix_before[columnIndexFromFile[0]])):
        paramsMatrix_tmp = []

        paramsMatrix_tmp.append(paramsMatrix_before[i

  else:
    os.system("cp %s %s/trainingset.txt" % (filePath, outputdir))



"""
  Input:
    freqList       : a list to store frequency nodes being evaluated
    columnSequence : a list to indicate the variable name for each column
    paramsMatrix   : a matrix containing parameters needed for that model. Each row corresponds to a set of parameters.
    modelName      : item in [ "IMRPhenomPv2FD" ]
    modelTag       : item in [ "hp", "hc", "hphp", "hphc", "hchc", "hchc", "hpPlushcSquared" ]
  Output:
    vecMatrix      : a list of vec.vector1D (defined in vectorUtils.py)
"""
def evaluateModel(freqList, columnSequence, paramsMatrix, modelName, modelTag):
  vecMatrix = []
  progressBar = gu.progressBar(len(paramsMatrix))
  progressBar.start()
  if modelName == "IMRPhenomPv2FD":
    assert len(columnSequence) >= 7
    nonGRparams = ["dchi0", "dchi1", "dchi2", "dchi3", "dchi4", "dchi5l", "dchi6", "dchi6l", "dchi7",
            "dbeta2", "dbeta3", "dalpha2", "dalpha3", "dalpha4",
            "dsigma2", "dsigma3", "dsigma4"]
    for i in xrange(len(paramsMatrix)):
      progressBar.update(i)
      # Check is there any nonGR parameters, if so, put it in the dictionary
      nonGRdict_input = {}
      for inonGRparams in nonGRparams:
        if inonGRparams in columnSequence:
          nonGRdict_input[inonGRparams] = paramsMatrix[i][columnSequence.index(inonGRparams)]
      # Evaluate the waveform vector
      iVecList  = IMRPhenomPv2FD(freqList,
              mass1     = paramsMatrix[i][columnSequence.index("m1")],
              mass2     = paramsMatrix[i][columnSequence.index("m2")],
              chi1L     = paramsMatrix[i][columnSequence.index("chi1L")],
              chi2L     = paramsMatrix[i][columnSequence.index("chi2L")],
              chip      = paramsMatrix[i][columnSequence.index("chip")],
              thetaJ    = paramsMatrix[i][columnSequence.index("thetaJ")],
              alpha0    = paramsMatrix[i][columnSequence.index("alpha0")],
              nonGRdict = nonGRdict_input)
      # Evaluate modelTag
      vecMatrix.append(evaluateModelTag(iVecList, modelTag))
  else:
    raise ValueError("The model (%s) is not supported")
  progressBar.end()
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
    raise ValueError("The modelTag (%s) is not supported." % modelTag)

