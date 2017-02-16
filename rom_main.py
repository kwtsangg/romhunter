#!/usr/bin/env python
__file__       = "rom_main.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Jan-26"

Description=""" Main pipeline for ROQ.
"""
#===============================================================================
#  Module
#===============================================================================
#import pyximport; pyximport.install()
import os, sys
import ConfigParser
import numpy as np
from datetime import datetime
import time

import gwhunter.utils.vector as vec
import gwhunter.utils.dataFile as df
import gwhunter.utils.general as gu

import greedy
import trainingset as ts
import eim

#===============================================================================
#  Main
#===============================================================================
def main():
  # setup the training set
  gu.printAndWrite(stdout_FilePath, "w+", "Building trainingset ...", withTime = True)
  TSParams = df.datahunter(TSParams_FilePath)
  TSParamsMatrix = TSParams.getMatrix(dataFormat = "float")
  TSMatrix = ts.evaluateModel(freqList, TSParamsMatrix, modelName, modelTag)
  gu.printAndWrite(stdout_FilePath, "a", "trainingset is built successfully!", withTime = True)

  # greedy algorithm
  gu.printAndWrite(stdout_FilePath, "a", "Generating reduced basis ...", withTime = True)
  RBMatrix = greedy.generateRB(TSMatrix, weight, orthoNormalRBVec_FilePath, greedyStdout_FilePath, stdout_FilePath, tolerance, maxRB)
  gu.printAndWrite(stdout_FilePath, "a", "Reduced basis is generated successfully!", withTime = True)

  # eim
  RB_index = df.datahunter(greedyStdout_FilePath).getColumn(5, dataFormat="int")
  for i in xrange(len(RBMatrix)):
    RBMatrix[i] = TSMatrix[RB_index[i]].unitVector()

  gu.printAndWrite(stdout_FilePath, "a", "Generating EIM ...", withTime = True)
  EIM_index = eim.generateEIM(RBMatrix, freqList, EIMStdout_FilePath, stdout_FilePath)
  gu.printAndWrite(stdout_FilePath, "a", "EIM is generated successfully!", withTime = True)

  # interpolation coefficient for basis h(param, F_k). F_k are EIM nodes.
  # ps. You can interpolate the waveform for basis h(param_i, f) or h(param, F_k).
  #     The former is done in eim.generateEIM while the latter is done here.
  gu.printAndWrite(stdout_FilePath, "a", "Generating interpolation list ...", withTime = True)
  eim.generateInterpolationList(RBMatrix, EIM_index, AinvStdout_FilePath, BkfStdout_FilePath)
  gu.printAndWrite(stdout_FilePath, "a", "Interpolation list is generated successfully!", withTime = True)

if __name__ == "__main__":
  timeROQ_i = time.time()
  # Checking existence of config file
  if len(sys.argv) != 2:
    gu.printf("This script takes only one argument which is the path to the config file. Exiting ...", __file__, "error")
    sys.exit()
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[1])

  # Reading the config file and calculate quantities
    # "general"
  outputdir         = config.get("general", "outputdir") + "/"+ datetime.now().strftime("%Y%m%d") + "_output"
  stdout_FilePath   = outputdir + "/stdout.txt"

  TSParams_FilePath = config.get("general", "TSParams_FilePath")
  modelName         = config.get("general", "modelName")
  modelTag          = config.get("general", "modelTag")
  fmin              = config.getfloat("general", "fmin")
  fmax              = config.getfloat("general", "fmax")
  seglen            = config.getint("general", "seglen")
  # TODO: allow weight to pass as vec.vector1D  (2017-02-08 02:59:53 CET)
  weight            = 1./seglen
  freqList          = np.linspace(20,1024,4017)

    # "greedy"
  maxRB     = config.getint("greedy", "maxRB")
  tolerance = config.getfloat("greedy", "tolerance")

  orthoNormalRBVec_FilePath = outputdir + "/orthonormalRBVec.txt"
  greedyStdout_FilePath     = outputdir + "/greedyStdout.txt"

    # "EIM"
  EIMStdout_FilePath        = outputdir + "/EIMStdout.txt"
  BkfStdout_FilePath        = outputdir + "/BkfStdout.txt"
  AinvStdout_FilePath       = outputdir + "/AinvStdout.txt"

  # Create the output directory and Start the main ROQ program
  os.system("mkdir -p %s" % outputdir) 
  main()

  # Final information
  timeROQ_f = time.time() - timeROQ_i
  gu.printAndWrite(stdout_FilePath, "a", "romhunter is finished successfully in %E seconds!" % timeROQ_f, withTime = True)

