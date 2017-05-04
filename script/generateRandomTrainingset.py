#!/usr/bin/env python
__file__       = "generateRandomTrainingset.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-May-04"

Description=""" To generate a trainingset with random params indicated by the input yaml.
"""

#===============================================================================
#  module
#===============================================================================
import sys
import numpy as np
import yaml
import time

import gwhunter.utils.general as gu
import gwhunter.utils.number as num
import gwhunter.utils.massParamsConv as mpc

#===============================================================================
#  function (copied from validation.py)
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

#===============================================================================
#  main
#===============================================================================
if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.exit("Please specify the config yaml file.")
  if sys.argv[1][-4:] != "yaml":
    sys.exit("The config file must be in yaml format.")
  gu.printf("Starting the generation of random parameters trainingset ...", "", withTime = True)
  timeROQ_i = time.time()
  with open(sys.argv[1], "r") as f:
    config = yaml.load(f)

  #Read the config
  numberOfPoints      = int(config["numberOfPoints"])
  outputFilePath      = config["outputFilePath"]
  columnSequence      = config["columnSequence"]
  randParamsRangeDict = config["randParamsRangeDict"]

  randParamsMatrix = generateRandomParamsMatrix(columnSequence, randParamsRangeDict, numberOfPoints)
  np.savetxt(outputFilePath, randParamsMatrix)

  timeROQ_f = time.time() - timeROQ_i
  gu.printf("The generation of random parameters trainingset finished successfully in %E seconds!" % timeROQ_f, "", withTime = True)

