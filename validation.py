#!/usr/bin/env python
__file__       = "validation.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Feb-07"

Description=""" To generate random trainingset points and check whether it is bad.
"""

#===============================================================================
#  Module  
#===============================================================================
import os, sys
import numpy as np
import yaml
import time

import gwhunter.utils.number as num
import gwhunter.utils.massParamsConv as mpc
import gwhunter.utils.vector as vec
import gwhunter.utils.general as gu
import gwhunter.utils.dataFile as df

import trainingset as ts

#===============================================================================
#  Function
#===============================================================================
def generateRandomParamsMatrix(columnSequence, randParamsRangeDict, numberOfPoints):
  myrand = num.numberhunter()
  randParamsMatrix = []
  progressBar = gu.progressBar(numberOfPoints)
  progressBar.start()
  for i in xrange(numberOfPoints):
    progressBar.update(i)
    iTSParams_tmp = [0.]*len(columnSequence)
    for j in xrange(len(columnSequence)):
      # mass1 and mass2 may need to be determined together
      if columnSequence[j] in ["m1", "m2"] and randParamsRangeDict[columnSequence[j]]["method"] == "qMc":
        iTSParams_tmp[columnSequence.index("m1")] = -1
        iTSParams_tmp[columnSequence.index("m2")] = -1
        while iTSParams_tmp[columnSequence.index("m1")] < randParamsRangeDict["m1"]["min"] or iTSParams_tmp[columnSequence.index("m2")] < randParamsRangeDict["m2"]["min"]:
          q_tmp  = myrand.getNumber(randParamsRangeDict["q"]["min"], randParamsRangeDict["q"]["max"], method = randParamsRangeDict["q"]["method"])
          Mc_tmp = myrand.getNumber(randParamsRangeDict["Mc"]["min"], randParamsRangeDict["Mc"]["max"], method = randParamsRangeDict["Mc"]["method"])
          m1m2   = mpc.Conv_q_Mc_to_m1m2(q_tmp, Mc_tmp)
          iTSParams_tmp[columnSequence.index("m1")] = m1m2[0]
          iTSParams_tmp[columnSequence.index("m2")] = m1m2[1]
      else:
        iTSParams_tmp[j] = myrand.getNumber(randParamsRangeDict[columnSequence[j]]["min"], randParamsRangeDict[columnSequence[j]]["max"], method = randParamsRangeDict[columnSequence[j]]["method"])
    # HARD CODE TO CHECK FOR KERR BOUND AND chi1L BOUND:
    while iTSParams_tmp[columnSequence.index("chi1L")]**2 + iTSParams_tmp[columnSequence.index("chi2L")]**2 > float(randParamsRangeDict["kerrBound"]) or iTSParams_tmp[columnSequence.index("chi1L")] <= 0.4 - 7.*mpc.Conv_m1m2_to_eta(iTSParams_tmp[columnSequence.index("m1")], iTSParams_tmp[columnSequence.index("m2")]):
      iTSParams_tmp[columnSequence.index("chi1L")] = myrand.getNumber(randParamsRangeDict["chi1L"]["min"], randParamsRangeDict["chi1L"]["max"], method = randParamsRangeDict["chi1L"]["method"])
      iTSParams_tmp[columnSequence.index("chi2L")] = myrand.getNumber(randParamsRangeDict["chi2L"]["min"], randParamsRangeDict["chi2L"]["max"], method = randParamsRangeDict["chi2L"]["method"])
    # END HARD CODE
    randParamsMatrix.append(iTSParams_tmp)
  progressBar.end()
  return randParamsMatrix

def calculateGreedyError2(hVector, RBMatrix, weight):
  ProjNorm2 = 0.
  for i in xrange(len(RBMatrix)):
    Projcoeff = hVector.projectionCoeff(RBMatrix[i], weight)
    ProjNorm2 += Projcoeff.real*Projcoeff.real+Projcoeff.imag*Projcoeff.imag
  return 1. - ProjNorm2

def calculateInterpError2(hVector, EIMNodes, BkfMatrix, weight):
  hVector_k = np.array([ hVector.component[EIMNodes[i]] for i in xrange(len(EIMNodes)) ])
  hInterpVec = vec.vector1D(np.dot(BkfMatrix.T, hVector_k))
  residual = hVector - hInterpVec
  return residual.norm2(weight) 

#===============================================================================
#  Main
#===============================================================================
def main():
  timeGenerateRandomParams_i = time.time()
  gu.printAndWrite(generalStdout_FilePath, "a", "Generating random parameters matrix ...", withTime = True)
  randParamsMatrix = generateRandomParamsMatrix(columnSequence, randParamsRangeDict, numberOfPoints)
  timeGenerateRandomParams = time.time() - timeGenerateRandomParams_i
  gu.printAndWrite(generalStdout_FilePath, "a", "Random parameters matrix is generated succefully in %E seconds!" % timeGenerateRandomParams, withTime = True)

  timeEvaluateWaveform_i = time.time()
  gu.printAndWrite(generalStdout_FilePath, "a", "Evaluating the waveform ...", withTime = True)
  randVecMatrix = ts.evaluateModel(freqList, columnSequence, randParamsMatrix, modelName, modelTag)
  timeEvaluateWaveform = time.time() - timeEvaluateWaveform_i
  gu.printAndWrite(generalStdout_FilePath, "a", "Evaluation of waveform is finished successfully in %E seconds!" % timeEvaluateWaveform, withTime = True)

  timeNormalization_i = time.time()
  gu.printAndWrite(generalStdout_FilePath, "a", "Normalizing the waveform ...", withTime = True)
  for i in xrange(len(randVecMatrix)):
    randVecMatrix[i] = randVecMatrix[i].unitVector(weight)
  timeNormalization = time.time() - timeNormalization_i
  gu.printAndWrite(generalStdout_FilePath, "a", "Normalization of waveform is finished successfully in %E seconds!" % timeNormalization, withTime = True)

  # Initialize variables
  greedyError2 = []
  interpError2 = []
  isBadPoints  = []
  randParams_badPoints = []

  timeValidation_i = time.time()
  gu.printAndWrite(generalStdout_FilePath, "a", "Starting validation ...", withTime = True)
  progressBar = gu.progressBar(len(randVecMatrix))
  progressBar.start()
  for randIndexm1 in xrange(len(randVecMatrix)):
    progressBar.update(randIndexm1)
    timeSweep_i = time.time()

    greedyError2.append(calculateGreedyError2(randVecMatrix[randIndexm1], RBMatrix, weight))
    randParamsMatrix[randIndexm1].append(greedyError2[-1])

    interpError2.append(calculateInterpError2(randVecMatrix[randIndexm1], EIMNodes, BkfMatrix, weight))
    randParamsMatrix[randIndexm1].append(interpError2[-1])

    # Determine whether it is a bad point
    if greedyError2[-1] > toleranceValidation:
      isBadPoints.append(1)
      randParams_badPoints.append(randParamsMatrix[randIndexm1][:-2])
    else:
      isBadPoints.append(0)
    # Print and Save all general information
    timeSweep = time.time() - timeSweep_i
    if randIndexm1 == 0:
      gu.printAndWrite(validationStdout_FilePath, "w+", "randIndex %i | GreedyError2 %E | InterpError2 % E | isBad %i | timeSweep(s) %E" % (randIndexm1+1, greedyError2[-1], interpError2[-1], isBadPoints[-1],timeSweep), withTime = True)
    else:
      gu.printAndWrite(validationStdout_FilePath, "a", "randIndex %i | GreedyError2 %E | InterpError2 % E | isBad %i | timeSweep(s) %E" % (randIndexm1+1, greedyError2[-1], interpError2[-1], isBadPoints[-1],timeSweep), withTime = True)
  progressBar.end()
  
  timeValidation = time.time() - timeValidation_i
  gu.printAndWrite(generalStdout_FilePath, "a", "The validation is finished successfully in %E seconds!" % timeValidation, withTime = True)
  gu.printAndWrite(generalStdout_FilePath, "a", "There are %i bad points out of %i. (%.1f %s)" % (sum(isBadPoints), numberOfPoints, sum(isBadPoints)/float(numberOfPoints)*100., "%"), withTime = True)
  # Save the matrix with random-generated params, greedyError2 and interpError2
  np.savetxt(randParams_FilePath, randParamsMatrix)
  np.savetxt(randParams_badPoints_FilePath, randParams_badPoints)

  # Get the enriched_trainingset.txt
  if len(randParams_badPoints) != 0:
    os.system("cat %s/greedyPoints.txt %s/randParams_badPoints.txt > %s/enriched_trainingset.txt" % (outputdir, outputdir, outputdir))
  else:
    os.system("echo "" > %s/enriched_trainingset.txt" % (outputdir))

#===============================================================================
#  Footer
#===============================================================================
if __name__ == "__main__":
  timeValidationPipeline_i = time.time()
  if len(sys.argv) != 2:
    sys.exit("Please specify the config yaml file.")
  if sys.argv[1][-4:] != "yaml":
    sys.exit("The config file must be in yaml format.")
  with open(sys.argv[1], "r") as f:
    config = yaml.load(f)

  # Reading the yaml file and calculate quantities
    # general
  outputdir = config["general"]["outputdir"]
  modelName = config["general"]["modelName"]
  modelTag  = config["general"]["modelTag"]
  fmin      = float(config["general"]["fmin"])
  fmax      = float(config["general"]["fmax"])
  seglen    = float(config["general"]["seglen"])
  # TODO: allow weight to pass as vec.vector1D  (2017-02-08 02:59:53 CET)
  weight   = 1./seglen
  freqList = np.linspace(fmin,fmax,int((fmax-fmin)*seglen)+1)
  columnSequence            = config["general"]["columnSequence"]

    # validation
  toleranceValidation           = float(config["validation"]["tolerance"])
  numberOfPoints                = int(config["validation"]["numberOfPoints"])
  randParamsRangeDict           = config["validation"]["randParamsRangeDict"]
  randParams_FilePath           = outputdir + "/randParams.txt"
  randParams_badPoints_FilePath = outputdir + "/randParams_badPoints.txt"
  validationStdout_FilePath     = outputdir + "/validationStdout.txt"
  generalStdout_FilePath        = outputdir + "/validationGeneralStdout.txt"

  EIMStdout_FilePath        = config["validation"]["EIMStdout_FilePath"]
  EIMNodes = df.datahunter(EIMStdout_FilePath).getColumn(6, dataFormat = "int")
  Bkf_FilePath              = config["validation"]["Bkf_FilePath"]
  BkfMatrix                 = np.load(Bkf_FilePath)
  RBMatrix_FilePath         = config["validation"]["RBMatrix_FilePath"]

  # Main
  os.system("mkdir -p %s" % outputdir)
  gu.printAndWrite(generalStdout_FilePath, "w+", "Starting validation pipeline ...", withTime = True)

    # Get RBMatrix
  gu.printAndWrite(generalStdout_FilePath, "a", "Getting reduced basis matrix from file ...", withTime = True)
  RBMatrix = []
  RBMatrix_numpy = np.load(RBMatrix_FilePath)
  for i in xrange(len(RBMatrix_numpy)):
    RBMatrix.append(vec.vector1D(RBMatrix_numpy[i]))
  del RBMatrix_numpy

  main()

  # Final information
  timeValidationPipeline = time.time() - timeValidationPipeline_i
  gu.printAndWrite(generalStdout_FilePath, "a", "Validation pipeline is finished successfully in %E seconds!" % timeValidationPipeline, withTime = True)


