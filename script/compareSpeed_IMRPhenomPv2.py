#!/usr/bin/env python
__file__       = "compareSpeed_AmpFactor_PointMassLens.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Aug-02"

Description=""" To compare the evaluation speed of the amplification factor from point mass lens by direct evaluation and ROM.
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import time

import gwhunter.utils.vector as vec
from gwhunter.waveform.lalwaveform import IMRPhenomPv2FD

#===============================================================================
#  Main
#===============================================================================

def eval_ROM(pathDir, paramsList, hVec, freqList):
  Bkf         = np.load(pathDir+"/Bkf.npy")
  EIMNodeList = np.loadtxt(pathDir+"/EIMStdout.txt", usecols=(8,))

#  hVector_k = np.array( IMRPhenomPv2FD(np.sort(EIMNodeList),
#                                       mass1  = paramsList[0],
#                                       mass2  = paramsList[1],
#                                       chi1L  = paramsList[2],
#                                       chi2L  = paramsList[3],
#                                       chip   = paramsList[4],
#                                       thetaJ = paramsList[5],
#                                       alpha0 = paramsList[6],
#                                       nonGRdict = {"dchi0": paramsList[7]}
#                                       )[0] )
#  hVector_k_unsorted = []
#  for iEIMNode in EIMNodeList:
#    hVector_k_unsorted.append( hVector_k[list(np.sort(EIMNodeList)).index(iEIMNode)] )
#  hVector_k_unsorted = np.array(hVector_k_unsorted)
#
#  Ih = np.dot(Bkf.T, hVector_k_unsorted)

  # TODO: Becoz the input of IMRPhenomPv2FD has to be strictly increasing, i just take the calculated version. It is stupid in practice but it s ok here. (2017-09-21 06:49:01 CEST)
  hVector_k = [ hVec[ list(freqList).index(iEIMNode) ] for iEIMNode in EIMNodeList ]
  Ih = np.dot(Bkf.T, hVector_k)

  return Ih

def eval_direct(freqList, paramsList):
  hVector   = np.array( IMRPhenomPv2FD(freqList,
                                       mass1  = paramsList[0],
                                       mass2  = paramsList[1],
                                       chi1L  = paramsList[2],
                                       chi2L  = paramsList[3],
                                       chip   = paramsList[4],
                                       thetaJ = paramsList[5],
                                       alpha0 = paramsList[6],
                                       nonGRdict = {"dchi0": paramsList[7]}
                                       )[0] )
#  return hVector/np.linalg.norm(hVector)
  return hVector

def main():
  pathDir = "/home/kwtsang/romhunter/dchi0_8s_pm2p0"
  freqList = np.arange(20,1024+0.125,0.125)
  paramsList = [20., 30., 0.1, 0.1, 0.2, 0.1, 0.2, -0.1]

  print "Calculating direct ..."
  Direct_start_time = time.time()
  h_Direct = eval_direct(freqList, paramsList)
  Direct_end_time = time.time()

  print "Calculating ROM ..."
  ROM_start_time = time.time()
  h_ROM = eval_ROM(pathDir, paramsList, h_Direct, freqList)
  ROM_end_time = time.time()
  
  print "no. of freq. nodes is            : %s" % len(freqList)
  print "Time for ROM    calculation takes: %e" % (ROM_end_time - ROM_start_time)
  print "Time for direct calculation takes: %e" % (Direct_end_time - Direct_start_time)
  print "Empirical time elapsed ratio     : %f" % ((Direct_end_time - Direct_start_time)/(ROM_end_time - ROM_start_time))
#  print "h_ROM is" , AmpFactor_PointMassLens_ROM_evaluated
#  print "h_direct is", AmpFactor_PointMassLens_direct_evaluated
#  print "h_ROM - h_Direct is", (AmpFactor_PointMassLens_ROM_evaluated - AmpFactor_PointMassLens_direct_evaluated)
  print "max(h_ROM - h_Direct) is", np.max((h_ROM - h_Direct))
  print "|h_ROM - h_Direct| is", np.linalg.norm((h_ROM - h_Direct))
  print "|h_ROM - h_Direct|/|h_Direct| is", np.linalg.norm((h_ROM - h_Direct))/np.linalg.norm(h_Direct)

if __name__ == "__main__":
  main()





