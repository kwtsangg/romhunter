#!/usr/bin/env python
__file__       = "greedy.pyx"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Jan-26"

Description=""" To perform the greedy algorithm.
"""

#===============================================================================
#  Module
#===============================================================================
import cython
import time
import numpy as np
import multiprocessing as mp

import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================
@cython.cdivision(True) 
cpdef generateRB(
                 TSMatrix,
                 orthoNormalRBVec_FilePath,
                 greedyStdout_FilePath,
                 tolerance = 1e-12,
                 maxRB = 5000,
                 ):
  # Add header to greedyStdout_File
  printAndWrite(greedyStdout_FilePath, "w+", "#1 dimRB #2 TSIndex #3 Error #4 timeSweep(s)")

  # Choose an arbitrary seed choice, here the first vector in TS is chosen.
  RBMatrix = []
  RBMatrix.append(TSMatrix[0].unitVector())

  orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "w+")
  orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
  orthoNormalRBVec_File.close()
  
  # preliminary work
  cdef int i # for looping the TS
  cdef int j # for IMGS
  cdef int dimRB  = 0
  cdef int sizeTS = len(TSMatrix)
  error_dimRB     = [1.]        # to store the max greedy error at each step
  error_dimRB_tmp = [0.]*sizeTS # to store the greedy error for each iTS
  RB_index        = [0]         # to store the index of TS selected to be added to RB

  TSNorm2   = [0.] * sizeTS
  ProjNorm2 = [0.] * sizeTS
  cdef int ii
  for ii in xrange(sizeTS):
    TSNorm2[ii]   = TSMatrix[ii].norm()

  # Initial info printing and saving
  printAndWrite(greedyStdout_FilePath, "a", "%i %i %E %E" % (dimRB, RB_index[-1], error_dimRB[-1], 0.))

  # greedy algorithm
  continueToWork = True
  while continueToWork:
    dimRB += 1
    timeSweep_i = time.time()
    # Use the last reduced basis to update the error vector
    # In fact, it is called modified Gram-Schmidt process
    for i in xrange(sizeTS):
      Projcoeff = TSMatrix[i].innerProduct(RBMatrix[-1])
      ProjNorm2[i] += abs(Projcoeff)**2
      error_dimRB_tmp[i] = TSNorm2[i] - ProjNorm2[i]
    RB_index.append(np.argmax(error_dimRB_tmp))
    error_dimRB.append(error_dimRB_tmp[RB_index[-1]])

    # Decide to iterate further or not
    if error_dimRB[-1] < tolerance:
      print "Because error = %E < tolerance = %E" % (error_dimRB[-1], tolerance)
      print "Greedy algorithm is finished successfully!"
      continueToWork = False
    elif dimRB == maxRB:
      print "Because dimRB = maxRB = %i" % (dimRB)
      print "Greedy algorithm is finished unfortunately!"
      continueToWork = False
    elif dimRB == sizeTS:
      print "Because dimRB = sizeTS = %i" % (dimRB)
      print "Greedy algorithm is finished unfortunately!"
      continueToWork = False
    else:
      # Perform orthonormalization with modified Gram-Schmidt process
      RBVec_add = TSMatrix[RB_index[-1]]
      norm_old = 1.
      continueToMGS = True
      while continueToMGS:
        for j in xrange(dimRB):
          RBVec_add = RBVec_add.rejection(RBMatrix[j])
        norm_new = RBVec_add.norm()
        if norm_new/norm_old < 0.5:
          norm_old = norm_new
        else:
          continueToMGS = False

      RBMatrix.append(RBVec_add.unitVector())

      # Save the orthonormalized resultant basis vector
      orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "a")
      orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
      orthoNormalRBVec_File.close() 

      # Print and Save all general information
      timeSweep_f = time.time() - timeSweep_i
      printAndWrite(greedyStdout_FilePath, "a", "%i %i %E %E" % (dimRB, RB_index[-1], error_dimRB[-1], timeSweep_f))

def printAndWrite(filePath, mode, strToBeSaved):
  print strToBeSaved
  f = open(filePath, mode)
  f.write(strToBeSaved)
  f.close()

