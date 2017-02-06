#!/usr/bin/env python
__file__       = "eim.pyx"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Feb-04"

Description=""" To store the algorithm for finding empirical interpolation node
"""

#===============================================================================
#  Module  
#===============================================================================
import cython
import time
import numpy as np
cimport numpy as np

import gwhunter.utils.general as gu

import trainingset as ts

#===============================================================================
#  Main
#===============================================================================

"""
  Input:
    RBMatrix           : a list of list storing the orthonormalized waveform. Its dimension is [dimRB][dimFreq]
    freqList           : a list of frequency node. Its dimension is [dimFreq]
    EIMStdout_FilePath : an output file path for the standard output of EIM.
    stdout_FilePath    : an output file path for the standard output
"""
cpdef generateEIM(RBMatrix, freqList, EIMStdout_FilePath, stdout_FilePath):
  # Start time
  timeEIM_i = time.time()
  gu.printAndWrite(stdout_FilePath, "a+", "---", withTime = True)
  gu.printAndWrite(stdout_FilePath, "a", "EIM starts.", withTime = True)

  # Preliminary work
  cdef int  dimRB   = len(RBMatrix)
  cdef int  dimEIM
  cdef int  dimFreq = len(freqList)
  cdef list EIM_index = []
  cdef list EIM_freq  = []

  # First EIM is defined by argmax(abs(e_1(f)))
  timeSweep_i = time.time()
  EIM_index.append( np.argmax( np.absolute(RBMatrix[0]) ) )
  EIM_freq.append( freqList[EIM_index[-1]] )
  timeSweep_f = time.time() - timeSweep_i
  gu.printAndWrite( EIMStdout_FilePath, "w+", "dimEIM %i | EIMindex %i | EIMfreq %.4f | timeSweep(s) %E" % (1, EIM_index[-1], EIM_freq[-1], timeSweep_f) )
  
  # Calculate other EIM index
  for dimEIM in xrange(2, dimRB+1):
    timeSweep_i = time.time()
    nextIndex = generateNewEIM(RBMatrix, EIM_index, dimFreq)
    EIM_index.append( nextIndex )
    EIM_freq.append( freqList[EIM_index[-1]] )
    timeSweep_f = time.time() - timeSweep_i
    gu.printAndWrite( EIMStdout_FilePath, "a", "dimEIM %i | EIMindex %i | EIMfreq %.4f | timeSweep(s) %E" % (dimEIM, EIM_index[-1], EIM_freq[-1], timeSweep_f) )
  
  timeEIM_f = time.time() - timeEIM_i
  gu.printAndWrite(stdout_FilePath, "a", "EIM is finished successfully!", withTime = True)
  gu.printAndWrite(stdout_FilePath, "a", "EIM takes %f wall seconds to complete." % timeEIM_f, withTime = True)

"""
  Input:
    RBMatrix : a list of list representing the orthonormalized waveform. Its dimension is [dimRB][dimFreq]
    freqList : a list of frequency node. Its dimension is [dimFreq]
    dimFreq  : len(freqList). Number of frequency nodes used for each waveform (NOT number of empirical interpolation points!) 
  Output:
    next EIM index
  Remark:
    The kth EIM node is the frequency node (f_k) that maximize abs( I_(k-1)[param_k, f] - h(param_k, f) )
    I_(k-1)[param, f] is the interpolation of waveform h[param, f], and defined to be Sum_i=1^k-1 x_ik(param)*h(param_i, f)
    I_0 is defined to be identical to zero
    x_ik is the interpolation coefficient and defined by demanding I_(k-1)[param_k, F_j] = h(param_k, F_j) where j = 1, 2, ..., k-1
    This program is to solve h_ij x_ik = h_kj and obtain the kth frequency node.
"""
cpdef generateNewEIM(RBMatrix, EIM_index, dimFreq):
  # Preliminary work
  cdef int i, j, f

  # Calculating the interpolation coefficient
  cdef int dimEIM  = len(EIM_index)
  h_ij = np.zeros((dimEIM,dimEIM), dtype=np.complex128)
  h_kj = np.zeros(dimEIM, dtype=np.complex128)

  for i in xrange(dimEIM):
    for j in xrange(dimEIM):
      h_ij[i][j] = RBMatrix[i][EIM_index[j]]
  for j in xrange(dimEIM):
    h_kj[j] = RBMatrix[dimEIM][EIM_index[j]]

  x_ik = np.linalg.solve(np.transpose(h_ij), h_kj)

  # Calculating the residual
  h_if = np.zeros((dimEIM,dimFreq), dtype=np.complex128)
  h_kf = np.zeros(dimFreq, dtype=np.complex128)
  for i in xrange(dimEIM):
    for f in xrange(dimFreq):
      h_if[i][f] = RBMatrix[i][f]
  for f in xrange(dimFreq):
    h_kf[f] = RBMatrix[dimEIM][f]
  residual = np.dot(x_ik, h_if) - h_kf

  return np.argmax(np.absolute(residual))

