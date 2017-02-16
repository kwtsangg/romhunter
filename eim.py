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
import time
import numpy as np

import gwhunter.utils.general as gu

import trainingset as ts

#===============================================================================
#  Main
#===============================================================================

"""
  Input:
    RBMatrix           : a list of list storing the (orthonormalized or without) waveform. Its dimension is [dimRB][dimFreq]
    freqList           : a list of frequency node. Its dimension is [dimFreq]
    EIMStdout_FilePath : an output file path for the standard output of EIM.
    stdout_FilePath    : an output file path for the standard output
  Output:
    EIM_index          : a list holding the TSindex of EIM
"""
def generateEIM(RBMatrix, freqList, EIMStdout_FilePath, stdout_FilePath):
  # Start time
  timeEIM_i = time.time()
  gu.printAndWrite(stdout_FilePath, "a+", "---", withTime = True)
  gu.printAndWrite(stdout_FilePath, "a", "EIM starts.", withTime = True)

  # Preliminary work
  dimRB   = len(RBMatrix)
  dimEIM  = 1             # number of empirical interpolants
  EIM_index = []          # index of selected empirical interpolants
  EIM_freq  = []          # frequency of selected empirical interpolants

  # First EIM is defined by argmax(abs(e_1(f)))
  timeSweep_i = time.time()
  EIM_index.append( np.argmax( np.abs(RBMatrix[0].component) ) )
  EIM_freq.append( freqList[EIM_index[-1]] )
  timeSweep_f = time.time() - timeSweep_i
  gu.printAndWrite( EIMStdout_FilePath, "w+", "dimEIM %i | EIMindex %i | EIMfreq %.4f | timeSweep(s) %E" % (1, EIM_index[-1], EIM_freq[-1], timeSweep_f) )
  
  # Calculate other EIM index
  for dimEIM in xrange(2, dimRB+1):
    timeSweep_i = time.time()
    nextIndex = generateNewEIM(RBMatrix, EIM_index)
    EIM_index.append( nextIndex )
    EIM_freq.append( freqList[EIM_index[-1]] )
    timeSweep_f = time.time() - timeSweep_i
    gu.printAndWrite( EIMStdout_FilePath, "a", "dimEIM %i | EIMindex %i | EIMfreq %.4f | timeSweep(s) %E" % (dimEIM, EIM_index[-1], EIM_freq[-1], timeSweep_f) )
  
  # Final information
  timeEIM_f = time.time() - timeEIM_i
  gu.printAndWrite(stdout_FilePath, "a", "EIM is finished successfully!", withTime = True)
  gu.printAndWrite(stdout_FilePath, "a", "EIM takes %E wall seconds to complete." % timeEIM_f, withTime = True)

  return EIM_index

"""
  The kth EIM node is the frequency node (f_k) that maximize abs( I_(k-1)[param_k, f] - h(param_k, f) )
  I_(k-1)[param, f] is the interpolation of waveform h[param, f], and defined to be Sum_i=1^k-1 x_ik(param)*h(param_i, f)
  I_0 is defined to be identical to zero
  x_ik is the interpolation coefficient and defined by demanding I_(k-1)[param_k, F_j] = h(param_k, F_j) where j = 1, 2, ..., k-1

  This program is to solve h_ij x_ik = h_kj and obtain the x_ik
                 and obtain the kth EIM node by argmax(abs(I-h)).

  Input:
    RBMatrix : a list of list representing the orthonormalized waveform. Its dimension is [dimRB][dimFreq]
    freqList : a list of frequency node. Its dimension is [dimFreq]
  Output:
    next EIM index
"""
def generateNewEIM(RBMatrix, EIM_index):
  # Calculating the interpolation coefficient
  dimEIM  = len(EIM_index)

  h_ij = np.array([ [RBMatrix[i].component[EIM_index[j]] for j in xrange(dimEIM)] for i in xrange(dimEIM) ])
  h_kj = np.array([ RBMatrix[dimEIM].component[EIM_index[j]] for j in xrange(dimEIM) ])
  x_ik = np.linalg.solve(np.transpose(h_ij), h_kj)

  h_if = np.array([ RBMatrix[i].component for i in xrange(dimEIM) ])
  h_kf = RBMatrix[dimEIM].component
  residual = np.dot(x_ik, h_if) - h_kf

  return np.argmax(np.absolute(residual))

"""
  After having RB and EIM nodes, the interpolation I[h] of a waveform h(param, f) is given by Sum_k=1^n B_k(f)*h(param, F_k) where B_k(f) is Sum_k=1^n inverse(A)_ik h(param_i, f).
  A is {{ e1(F1) e2(F1) ... eN(F1) },
        { e1(F2) e2(F2) ... eN(F2) },
        { ...    ...    ...  ...   },
        { e1(FN) e2(FN) ... eN(FN) }}
  So A_ij = h(param_j)(EIM_i)

  Input:
    RBMatrix  : The list of lists which stores the RB. Its dimension is [dimRB, dimFreq]
    EIM_index : The list of EIM nodes indices obtained from generateEIM function above
  Output:
    a list of lists of B_kf = Ainv^T * h_if, its dimension is [dimRB, dimFreq]
"""
# TODO: Check the result correct or not (2017-02-07 06:45:19 CET)
def generateInterpolationList(RBMatrix, EIM_index, AinvStdout_FilePath, BkfStdout_FilePath):
  # Preliminary work
  dimRB = len(RBMatrix)

  A_ij = np.array([ [RBMatrix[j].component[EIM_index[i]] for j in xrange(dimRB)] for i in xrange(dimRB) ])
  h_if = np.array([ RBMatrix[i].component for i in xrange(dimRB) ])

  # Calculating A inverse and save it
  Ainv_ij = np.linalg.inv(A_ij)
  np.savetxt(AinvStdout_FilePath, Ainv_ij)

  # Calculating B_k(f) = (Ainv)^T * h(param_i, f) and save it
  B_kf = np.dot(np.transpose(Ainv_ij),h_if)
  np.savetxt(BkfStdout_FilePath, B_kf)

  # Converting the 2D numpy array to a 1D list of lists.
  return [B_kf[i] for i in xrange(dimRB)]

