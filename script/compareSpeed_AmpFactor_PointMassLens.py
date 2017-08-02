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

import model

import gwhunter.utils.dataFile as df

#===============================================================================
#  Main
#===============================================================================

def AmpFactor_PointMassLens_ROM(freqList, M_Lz, y):
  Bkf         = np.load("/home/kwtsang/romhunter/AmpFactor_PointMassLens/Bkf.npy")
  EIMNodeList = df.datahunter("/home/kwtsang/romhunter/AmpFactor_PointMassLens/EIMStdout.txt").getColumn(9, dataFormat="float")
  hVector_k = np.array([ model.AmpFactor_PointMassLens(EIMNodeList, M_Lz, y) ])
  Ih = np.dot(Bkf.T, hVector_k)

  nodeList = df.datahunter("/home/kwtsang/romhunter/AmpFactor_PointMassLens/freq_nodes.txt").getColumn(1, dataFormat="float")

  result = []
  for ifreq in freqList:
    result.append( Ih[nodeList.index("ifreq")] )
  return np.array([result])

def AmpFactor_PointMassLens_direct(freqList, M_Lz, y):
  return np.array([model.AmpFactor_PointMassLens(freqList, M_Lz, y)])

def main():
  freqList = [50.]
  M_Lz = 1e2
  y    = 3

  ROM_start_time = time.time()
  AmpFactor_PointMassLens_ROM(freqList, M_Lz, y)
  ROM_end_time = time.time()

  Direct_start_time = time.time()
  AmpFactor_PointMassLens_direct(freqList, M_Lz, y)
  Direct_end_time = time.time()
  
  print "Time for ROM    calculation takes: %e" % (ROM_start_time - ROM_end_time)
  print "Time for direct calculation takes: %e" % (Direct_start_time - Direct_end_time)
  print "AmpFactor_PointMassLens_ROM is" , AmpFactor_PointMassLens_ROM
  print "AmpFactor_PointMassLens_direct is", AmpFactor_PointMassLens_direct
  print "ROM - Direct is", (AmpFactor_PointMassLens_ROM - AmpFactor_PointMassLens_direct)







