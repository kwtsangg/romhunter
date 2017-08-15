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
import numpy as np
import gwhunter.utils.general as gu
import gwhunter.utils.vector as vec
import gwhunter.utils.number as num
import gwhunter.utils.dataFile as df

import model

#===============================================================================
#  Main
#===============================================================================

def buildingTrainingset(outputdir, filePath, columnSequence, paramsDict):
  paramsMatrix = []
  paramsMatrix_tmp = []
  
  # Generate the deterministic points from the range given
  detParamsMatrix = []
  atLeastOneMethodFile = False
  for icolumnName in columnSequence:
    if paramsDict[icolumnName]["method"] != "file":
      if detParamsMatrix == []:
        detParamsMatrix = num.numberhunter().getNumber(paramsDict[icolumnName]["min"], paramsDict[icolumnName]["max"], paramsDict[icolumnName]["numberOfPoints"], mode = "det")
      else:
        detParams_tmp = num.numberhunter().getNumber(paramsDict[icolumnName]["min"], paramsDict[icolumnName]["max"], paramsDict[icolumnName]["numberOfPoints"], mode = "det")
        detParamsMatrix = df.formExhaustive2DArray([detParamsMatrix, detParams_tmp])
    else:
      atLeastOneMethodFile = True

  # Get the paramsMatrix_tmp, which is the extended-column version of the input file, like [ file row, others ]
  if atLeastOneMethodFile:
    fileParamsMatrix = df.datahunter(filePath).getMatrix(dataFormat = "float")
  else:
    fileParamsMatrix = [[]]
  paramsMatrix_tmp = df.formExhaustive2DArray( [fileParamsMatrix, detParamsMatrix] )

  # Store a dict indicating the index number in paramsMatrix_tmp
  dictColIndex_paramsMatrix_tmp = {}
  additionalCount = 1
  for icolumnName in columnSequence:
    if paramsDict[icolumnName]["method"] == "file":
      dictColIndex_paramsMatrix_tmp[icolumnName] = int(paramsDict[icolumnName]["column"])
    else:
      dictColIndex_paramsMatrix_tmp[icolumnName] = len(fileParamsMatrix[0]) + additionalCount
      additionalCount += 1

  # Generate the final paramsMatrix, by deleting/exchanging columns
  progressBar = gu.progressBar(len(paramsMatrix_tmp))
  progressBar.start()
  for i in xrange(len(paramsMatrix_tmp)):
    progressBar.update(i)
    paramsVec_tmp = []
    paramsVec_tmp_string = ""
    for icolumnName in columnSequence:
      paramsVec_tmp.append(paramsMatrix_tmp[i][dictColIndex_paramsMatrix_tmp[icolumnName]-1])
      paramsVec_tmp_string += str(paramsVec_tmp[-1]) + " "
    paramsMatrix.append(paramsVec_tmp)
  progressBar.end()
  np.savetxt("%s/trainingset.txt" % outputdir, paramsMatrix)
  return paramsMatrix

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
    from gwhunter.waveform.lalwaveform import IMRPhenomPv2FD
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
  elif modelName == "AmpFactor_PointMassLens":
    assert len(columnSequence) >= 2
    for i in xrange(len(paramsMatrix)):
      progressBar.update(i)
      # Evaluate the function vector
      iVecList = model.AmpFactor_PointMassLens(freqList,
              M_Lz = paramsMatrix[i][columnSequence.index("M_Lz")],
              y    = paramsMatrix[i][columnSequence.index("y")])
      vecMatrix.append(evaluateModelTag(iVecList, "bypass"))
  else:
    raise ValueError("The model (%s) is not supported")
  progressBar.end()
  return vecMatrix

"""
  This function is built specific for IMRPhenomPv2. For other model, use modelTag = "bypass"
  Input:
    iVecList  : The list with component [0] to be hplus vector list and component [1] to be hcross vector list
    modelTag  : item in [ "hp", "hc", "hphp", "hphc", "hchc", "hchc", "hpPlushcSquared" ]
  Output:
    a vec.vector1D object (defined in vectorUtils.py) with tag considered.
"""
def evaluateModelTag(iVecList, modelTag = "bypass"):
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
  # bypass
  elif modelTag == "bypass":
    return vec.vector1D(iVecList)
  else:
    raise ValueError("The modelTag (%s) is not supported." % modelTag)

#===============================================================================
#  other
#===============================================================================

def getGrammianMatrix(TSMatrix, weight, outputdir):
  for i in xrange(len(TSMatrix)):
    TSMatrix[i] = TSMatrix[i].unitVector(weight)
  grammianMatrix = []
  progressBar = gu.progressBar(len(TSMatrix))
  progressBar.start()
  for i in xrange(len(TSMatrix)):
    progressBar.update(i)
    grammianMatrix_tmp = []
    for jTSMatrix in TSMatrix:
      grammianMatrix_tmp.append(TSMatrix[i].innerProduct(jTSMatrix, weight))
    grammianMatrix.append(grammianMatrix_tmp)
  progressBar.end()
  np.save("%s/grammianMatrix.npy" % outputdir, grammianMatrix)

