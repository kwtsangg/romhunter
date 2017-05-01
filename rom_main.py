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
import os, sys
import yaml
import numpy as np
import time

import gwhunter.utils.vector as vec
import gwhunter.utils.dataFile as df
import gwhunter.utils.general as gu
import gwhunter.utils.massParamsConv as mpc
import gwhunter.waveform.lalutils as lalu

import greedy
import trainingset as ts
import eim

#===============================================================================
#  Main
#===============================================================================
def main():
    #===============================#
    # training set 
    #===============================#
  gu.printAndWrite(ROMStdout_FilePath, "w+", "Building the input trainingset file ...", withTime = True)
  TSParamsMatrix = ts.buildingTrainingset(outputdir, TSParams_FilePath, columnSequence, paramsDict)
  gu.printAndWrite(ROMStdout_FilePath, "a", "Evaluating trainingset ...", withTime = True)
  TSMatrix = ts.evaluateModel(freqList, columnSequence, TSParamsMatrix, modelName, modelTag)
  gu.printAndWrite(ROMStdout_FilePath, "a", "Trainingset is evaluated successfully!", withTime = True)

    #===============================#
    # greedy algorithm 
    #===============================#
  gu.printAndWrite(ROMStdout_FilePath, "a", "Generating reduced basis ...", withTime = True)
  RBMatrix = greedy.generateRB(TSMatrix, weight, orthoNormalRB_FilePath, greedyStdout_FilePath, ROMStdout_FilePath, toleranceGreedy, maxRB)
  del TSMatrix
  gu.printAndWrite(ROMStdout_FilePath, "a", "Reduced basis is generated successfully!", withTime = True)

  # Save the greedy points params
  gu.printAndWrite(ROMStdout_FilePath, "a", "Saving the reduced basis ...", withTime = True)
  RBIndex = df.datahunter(greedyStdout_FilePath).getColumn(6, dataFormat="int")
  RBParams = []
  for iRBIndex in RBIndex:
    RBParams.append(TSParamsMatrix[iRBIndex])
  np.savetxt(greedyPoints_FilePath, RBParams)
  del RBIndex, RBParams, TSParamsMatrix

    #===============================#
    # eim 
    #===============================#
  gu.printAndWrite(ROMStdout_FilePath, "a", "Generating EIM ...", withTime = True)
  EIM_index = eim.generateEIM(RBMatrix, freqList, EIMStdout_FilePath, ROMStdout_FilePath)
  gu.printAndWrite(ROMStdout_FilePath, "a", "EIM is generated successfully!", withTime = True)

  # interpolation coefficient for basis h(param, F_k). F_k are EIM nodes.
  # ps. You can interpolate the waveform for basis h(param_i, f) or h(param, F_k).
  #     The former is done in eim.generateEIM while the latter is done here.
  gu.printAndWrite(ROMStdout_FilePath, "a", "Generating interpolation list ...", withTime = True)
  eim.generateInterpolationList(RBMatrix, EIM_index, Vinv_FilePath, Bkf_FilePath)
  gu.printAndWrite(ROMStdout_FilePath, "a", "Interpolation list is generated successfully!", withTime = True)


#===============================================================================
#  Footer
#===============================================================================

if __name__ == "__main__":
  timeROQ_i = time.time()
  if len(sys.argv) != 2:
    sys.exit("Please specify the config yaml file.")
  if sys.argv[1][-4:] != "yaml":
    sys.exit("The config file must be in yaml format.")
  with open(sys.argv[1], "r") as f:
    config = yaml.load(f)

  # Reading the yaml file and calculate quantities
    # "general"
  outputdir          = config["general"]["outputdir"]
  ROMStdout_FilePath = outputdir + "/ROMStdout.txt"
  TSParams_FilePath = config["general"]["TSParams_FilePath"]
  modelName         = config["general"]["modelName"]
  modelTag          = config["general"]["modelTag"]
  fmin              = float(config["general"]["fmin"])
  fmax              = float(config["general"]["fmax"])
  seglen            = float(config["general"]["seglen"])
  bands             = config["general"]["bands"]
  fudgeFactor       = config["general"]["fudge"]
  McMin             = config["general"]["Mc-min"]
  if len(bands) == 0:
    freqList        = np.linspace(fmin,fmax,int((fmax-fmin)*seglen)+1)
    weight          = 1./seglen
  else:
    freq_weight_tmp = lalu.generateMultibandFreqVector(McMin, bands, fudge = fudgeFactor)
    freqList        = freq_weight_tmp[0][:-1]
    weight          = freq_weight_tmp[1]
    del freq_weight_tmp
  np.savetxt(outputdir + "/freq_nodes.txt", freqList)
  np.savetxt(outputdir + "/freq_weights.txt", freq_weight_tmp)
  columnSequence    = config["general"]["columnSequence"]
  paramsDict        = config["general"]["paramsDict"]

    # "greedy"
  maxRB           = int(config["greedy"]["maxRB"])
  toleranceGreedy = float(config["greedy"]["toleranceGreedy"])

  orthoNormalRB_FilePath = outputdir + "/orthonormalRB.npy"
  greedyStdout_FilePath  = outputdir + "/greedyStdout.txt"
  greedyPoints_FilePath  = outputdir + "/greedyPoints.txt"

    # "EIM"
  EIMStdout_FilePath = outputdir + "/EIMStdout.txt"
  Bkf_FilePath       = outputdir + "/Bkf.npy"
  Vinv_FilePath      = outputdir + "/Vinv.npy"

  # Create the output directory and Start the main ROQ program
  os.system("mkdir -p %s" % outputdir) 
  os.system("cp %s %s/config.yaml" % (sys.argv[1], outputdir))
  main()

  # Final information
  timeROQ_f = time.time() - timeROQ_i
  gu.printAndWrite(ROMStdout_FilePath, "a", "romhunter is finished successfully in %E seconds!" % timeROQ_f, withTime = True)

