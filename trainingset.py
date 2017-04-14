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
import gwhunter.utils.number as num
import gwhunter.utils.dataFile as df

#===============================================================================
#  Main
#===============================================================================

def buildingTrainingset(outputdir, filePath, columnSequence, paramsDict):
  paramsMatrix = []
  paramsMatrix_tmp = []
  
  # Check whether you need to generate a new trainingset.
  # If you dont need, you can just copy the filePath as your trainingset.
  needToGenerateNewTS = False
  for icolumn in columnSequence:
    if paramsDict[icolumn]["method"] != "file":
      needToGenerateNewTS = True
      break

  if needToGenerateNewTS:
    # Generate the deterministic points from the range given
    detParamsMatrix = []
    for icolumnName in columnSequence:
      if paramsDict[icolumnName]["method"] != "file":
        if detParamsMatrix == []:
          detParamsMatrix = num.numberhunter().getNumber(paramsDict[icolumnName]["min"], paramsDict[icolumnName]["max"], paramsDict[icolumnName]["numberOfPoints"], mode = "det")
        else:
          detParams_tmp = num.numberhunter().getNumber(paramsDict[icolumnName]["min"], paramsDict[icolumnName]["max"], paramsDict[icolumnName]["numberOfPoints"], mode = "det")
          detParamsMatrix = df.formExhaustive2DArray(detParamsMatrix, detParams_tmp)

    # Get the paramsMatrix_tmp, which is the extended-column version of the input file, like [ file row, others ]
    # Here I assumed at least one method is "file", if not, it will waste time to extract the TS
    fileParamsMatrix = df.datahunter(filePath).getMatrix(dataFormat = "float")
    paramsMatrix_tmp = df.formExhaustive2DArray( [fileParamsMatrix, detParamsMatrix] )

    # Store a dict indicating the index number in paramsMatrix_tmp
    dictColIndex_paramsMatrix_tmp = {}
    for icolumnName in columnSequence:
      additionCount = 1
      if paramsDict[icolumnName]["method"] == "file":
        dictColIndex_paramsMatrix_tmp[icolumnName] = int(paramsDict[icolumnName]["column"])
      else:
        dictColIndex_paramsMatrix_tmp[icolumnName] = len(fileParamsMatrix[0]) + additionCount
        additionCount += 1

    # Generate the final paramsMatrix, by deleting/exchanging columns
    trainingsetFile = open("%s/trainingset.txt" % outputdir, "w+")
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
      trainingsetFile.write(paramsVec_tmp_string[:-1] + "\n")
    trainingsetFile.close()
    progressBar.end()
  else:
    os.system("cp %s %s/trainingset.txt" % (filePath, outputdir))
    paramsMatrix = df.datahunter(outputdir+"/trainingset.txt").getMatrix(dataFormat = "float")
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

