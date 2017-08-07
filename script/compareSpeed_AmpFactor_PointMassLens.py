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
#  Bkf         = np.load("/home/kwtsang/MountCluster/nikhef_home/romhunter/AmpFactor_PointMassLens/Bkf.npy")
  EIMNodeList = df.datahunter("/home/kwtsang/romhunter/AmpFactor_PointMassLens/EIMStdout.txt").getColumn(9, dataFormat="float")
#  EIMNodeList = df.datahunter("/home/kwtsang/MountCluster/nikhef_home/romhunter/AmpFactor_PointMassLens/EIMStdout.txt").getColumn(9, dataFormat="float")
  hVector_k = np.array( model.AmpFactor_PointMassLens(EIMNodeList, M_Lz, y, False) )
  Ih = np.dot(Bkf.T, hVector_k)

  nodeList = df.datahunter("/home/kwtsang/romhunter/AmpFactor_PointMassLens/freq_nodes.txt").getColumn(1, dataFormat="float")
#  nodeList = df.datahunter("/home/kwtsang/MountCluster/nikhef_home/romhunter/AmpFactor_PointMassLens/freq_nodes.txt").getColumn(1, dataFormat="float")

  result = []
  Msun_time = 4.92567e-6
  xm   = (y + np.sqrt(y*y + 4.))/2.
  phim = (xm - y)*(xm - y)/2. - np.log(xm)

  for ifreq in freqList:
    w  = 8.*np.pi*M_Lz*ifreq*Msun_time
    result.append( np.exp(np.pi*w/4. + 1j*w/2.*(np.log(w/2.)-2.*phim)) * Ih[nodeList.index(ifreq)] )
#    result.append( Ih[nodeList.index(ifreq)] )
  return np.array(result)

def AmpFactor_PointMassLens_direct(freqList, M_Lz, y):
  return np.array( model.AmpFactor_PointMassLens(freqList, M_Lz, y, withExpPrefactor = True) )
#  return np.array( model.AmpFactor_PointMassLens(freqList, M_Lz, y, withExpPrefactor = False) )

def main():
#  freqList = np.arange(20,1024,0.125)
  freqList = [512.]
  M_Lz = 1234
  y    = 2.88

  print "Calculating ROM ..."
  ROM_start_time = time.time()
  AmpFactor_PointMassLens_ROM_evaluated = AmpFactor_PointMassLens_ROM(freqList, M_Lz, y)
  ROM_end_time = time.time()

  print "Calculating direct ..."
  Direct_start_time = time.time()
  AmpFactor_PointMassLens_direct_evaluated = AmpFactor_PointMassLens_direct(freqList, M_Lz, y)
  Direct_end_time = time.time()
  
  print "no. of freq. nodes is            : %s" % len(freqList)
  print "Theoretical time ratio is        : %f" % (len(freqList)/94.)
  print "Time for ROM    calculation takes: %e" % (ROM_end_time - ROM_start_time)
  print "Time for direct calculation takes: %e" % (Direct_end_time - Direct_start_time)
  print "Empirical time elapsed ratio     : %f" % ((Direct_end_time - Direct_start_time)/(ROM_end_time - ROM_start_time))
  print "AmpFactor_PointMassLens_ROM is" , AmpFactor_PointMassLens_ROM_evaluated
  print "AmpFactor_PointMassLens_direct is", AmpFactor_PointMassLens_direct_evaluated
  print "ROM - Direct is", (AmpFactor_PointMassLens_ROM_evaluated - AmpFactor_PointMassLens_direct_evaluated)
  print "max(ROM - Direct) is", np.max((AmpFactor_PointMassLens_ROM_evaluated - AmpFactor_PointMassLens_direct_evaluated))
  print "|ROM - Direct| is", np.linalg.norm((AmpFactor_PointMassLens_ROM_evaluated - AmpFactor_PointMassLens_direct_evaluated))

if __name__ == "__main__":
  main()





