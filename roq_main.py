#!/usr/bin/env python
__file__       = "roq_main.py"
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
import pyximport; pyximport.install()
import sys
import ConfigParser
import numpy as np

import gwhunter.utils.vector as vec
import gwhunter.utils.dataFile as df
import gwhunter.utils.general as gu

import greedy as gd
import trainingset as ts
import eim

#===============================================================================
#  Main
#===============================================================================
def main():
  # get input values from the config
  freqList                  = np.linspace(20,1024,4017)
  weight                    = 0.25
  TSParams_FilePath         = "/home/kwtsang/romhunter/input/SmithEtAlBases/GreedyPoints_4s.txt"
  orthoNormalRBVec_FilePath = "/home/kwtsang/romhunter/output/orthonormalRBVec.txt"
  greedyStdout_FilePath     = "/home/kwtsang/romhunter/output/greedyStdout.txt"
  stdout_FilePath           = "/home/kwtsang/romhunter/output/stdout.txt"
  EIMStdout_FilePath        = "/home/kwtsang/romhunter/output/EIMStdout.txt"
  BkfStdout_FilePath        = "/home/kwtsang/romhunter/output/BkfStdout.txt"
  AinvStdout_FilePath       = "/home/kwtsang/romhunter/output/AinvStdout.txt"
  tolerance                 = 1e-12
  maxRB                     = 3000

  # setup the training set
  gu.printAndWrite(stdout_FilePath, "w+", "Building trainingset ...", withTime = True)
  TSParams = df.datahunter(TSParams_FilePath)
  TSParamsMatrix = TSParams.getMatrix(dataFormat = "float")
  TSMatrix = ts.evaluateModel(freqList, TSParamsMatrix, "IMRPhenomPv2FD", "hp")
  gu.printAndWrite(stdout_FilePath, "a", "trainingset is built successfully!", withTime = True)

  # greedy algorithm
  gu.printAndWrite(stdout_FilePath, "a", "Generating reduced basis ...", withTime = True)
  RBMatrix = gd.generateRB(TSMatrix, weight, orthoNormalRBVec_FilePath, greedyStdout_FilePath, stdout_FilePath, tolerance, maxRB)
  gu.printAndWrite(stdout_FilePath, "a", "Reduced basis is generated successfully!", withTime = True)

  # eim
  for i in xrange(len(RBMatrix)):
    RBMatrix[i] = RBMatrix[i].component
#  RB_index = df.datahunter(greedyStdout_FilePath).getColumn(5, dataFormat="int")
#  for i in xrange(len(RBMatrix)):
#    RBMatrix[i] = TSMatrix[RB_index[i]].unitVector().component

  gu.printAndWrite(stdout_FilePath, "a", "Generating EIM ...", withTime = True)
  EIM_index = eim.generateEIM(RBMatrix, freqList, EIMStdout_FilePath, stdout_FilePath)
  gu.printAndWrite(stdout_FilePath, "a", "EIM is generated successfully!", withTime = True)

  # interpolation coefficient for basis h(param, F_k). F_k are EIM nodes.
  # ps. You can interpolate the waveform for basis h(param_i, f) or h(param, F_k).
  #     The former is done in eim.generateEIM while the latter is done here.
  gu.printAndWrite(stdout_FilePath, "a", "Generating interpolation list ...", withTime = True)
  eim.generateInterpolationList(RBMatrix, EIM_index, len(freqList), AinvStdout_FilePath, BkfStdout_FilePath)
  gu.printAndWrite(stdout_FilePath, "a", "Interpolation list is generated successfully!", withTime = True)

  # Final information
  gu.printAndWrite(stdout_FilePath, "a", "romhunter is finished successfully!", withTime = True)

if __name__ == "__main__":
#  if len(sys.argv) != 2:
#    gu.printf("This script takes only one argument which is the path to the config file. Exiting ...", __file__, "error")
#    sys.exit()
#  config = ConfigParser.ConfigParser()
#  config.read(sys.argv[1])
  main()
