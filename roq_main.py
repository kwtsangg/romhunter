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
import os
import numpy as np

import vectorUtils as vec
import greedy as gd
import trainingset as ts

#===============================================================================
#  Main
#===============================================================================
def main():
  # get input values from the config

  # setup the training set
  freqList = np.linspace(20,1024,4017)
  weight = 1.
  weight = 0.25
  TSParams_FilePath = "/home/kwtsang/romhunter/input/SmithEtAlBases/GreedyPoints_4s.txt"
  orthoNormalRBVec_FilePath = "/home/kwtsang/romhunter/output/orthonormalRBVec.txt"
  greedyStdout_FilePath = "/home/kwtsang/romhunter/output/greedyStdout.txt"
  tolerance = 1e-12
  maxRB = 5000

  TSVec_FilePath = "/home/kwtsang/romhunter/input/TS_testing.txt"
  TSMatrix = ts.evaluateModel(freqList, TSParams_FilePath, "IMRPhenomPv2FD", "hp")
#  TSMatrix = ts.getFromFile(TSVec_FilePath)

  # greedy algorithm
  gd.generateRB(TSMatrix, weight, orthoNormalRBVec_FilePath, greedyStdout_FilePath, tolerance, maxRB)

  # eim

if __name__ == "__main__":
  main()
