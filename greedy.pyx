#!/usr/bin/env python
__file__       = "greedy.py"
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
import os
import time
import numpy as np
import multiprocessing as mp

import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================
@cython.cdivision(True) 
cpdef greedy(
             TSVec_FilePath,
             orthoNormalRBVec_FilePath = os.getcwd() + "/output/orthonormalRBVec.txt",
             greedyStdout_FilePath     = os.getcwd() + "/output/greedyStdout.txt",
             tolerance = 1e-12,
             maxRB     = 1000
             ):
  # Add header to greedyStdout_File
  printAndWrite(greedyStdout_FilePath, "w+", "#1 dimRB #2 TSIndex #3 Error #4 timeSweep(s)")
  printAndWrite(greedyStdout_FilePath, "a", "%i %i %E %f" % (0, 0, 0., 0.))

  # get trainingset and TSsize
  TSMatrix = []
  TSVec_File = open(TSVec_FilePath, "r")
  for sizeTS, iTSVec in enumerate(TSVec_File):
    TSMatrix.append(vec.vector1D(iTSVec.split()))
  TSVec_File.close()
  sizeTS += 1

  # Choose an arbitrary seed choice, here the first vector in TS is chosen.
  RBMatrix = []
  RBMatrix.append(TSMatrix[0].unitVector())

  orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "w+")
  orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
  orthoNormalRBVec_File.close()
  
  # preliminary work
  cdef int i
  cdef int dimRB = 0
  error_dimRB     = [0.]        # to store the max greedy error at each step
  error_dimRB_tmp = [0.]*sizeTS # to store the greedy error for each iTS
  RB_index        = [0]         # to store the index of TS selected to be added to RB

  # greedy algorithm
  continueToWork = True
  while continueToWork:
    dimRB += 1
    timeSweep_i = time.time()
    # Use the last reduced basis to update the error vector
    # In fact, it is called modified Gram-Schmidt process
    for i in xrange(sizeTS):
      # To avoid the same index being selected twice
      if i in RB_index:
        error_dimRB_tmp[i] = 0.
        continue
      TSMatrix[i] = TSMatrix[i].rejectionUnitVector(RBMatrix[-1])
      error_dimRB_tmp[i] = TSMatrix[i].norm()

    RB_index.append(np.argmax(error_dimRB_tmp))
    error_dimRB.append(error_dimRB_tmp[RB_index[-1]])
    RBMatrix.append(TSMatrix[RB_index[-1]].unitVector())

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
      # p.s. In fact, one needs only normalization only.
      # Save the orthonormalized resultant basis vector
      orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "a")
      orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
      orthoNormalRBVec_File.close() 

      # Print and Save all general information
      timeSweep_f = time.time() - timeSweep_i
      printAndWrite(greedyStdout_FilePath, "a", "%i %i %E %f" % (dimRB, RB_index[-1], error_dimRB[-1], timeSweep_f))

def printAndWrite(filePath, mode, strToBeSaved):
  print strToBeSaved
  f = open(filePath, mode)
  f.write(strToBeSaved)
  f.close()

