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
cimport numpy as np
import multiprocessing as mp

import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================

@cython.cdivision(True) 
cpdef generateRB(
                 TSMatrix,
                 weight,
                 orthoNormalRBVec_FilePath,
                 greedyStdout_FilePath,
                 tolerance = 1e-12,
                 maxRB = 5000,
                 ):

  # Normalize TS and get the size of TS
  cdef int sizeTS = len(TSMatrix)
  TSNorm2   = [0.] * sizeTS
  ProjNorm2 = [0.] * sizeTS
  cdef int j
  for j in xrange(sizeTS):
    TSMatrix[j] = TSMatrix[j].unitVector(weight)
    TSNorm2[j]  = TSMatrix[j].norm2(weight)

  # preliminary work
  cdef int i                    # for looping the TS
  cdef int dimRB  = 1           # to store the number of reduced basis vector
  error_dimRB     = [1.]        # to store the max greedy error at each step
  error_dimRB_tmp = [0.]*sizeTS # to store the greedy error for each iTS
  RB_index        = [0]         # to store the index of TS selected to be added to RB

  # Choose an arbitrary seed choice, here the first vector in TS is chosen.
  RBMatrix = []
  RBMatrix.append(TSMatrix[0].unitVector(weight))

  orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "w+")
  orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
  orthoNormalRBVec_File.close()

  # Initial info printing and saving
  printAndWrite(greedyStdout_FilePath, "w+", "dimRB %i | TSIndex %i | GreedyError2 %E | timeSweep(s) %E" % (dimRB, RB_index[-1], error_dimRB[-1], 0.))

  # greedy algorithm
  continueToWork = True
  while continueToWork:
    dimRB += 1
    timeSweep_i = time.time()
    # Use the last reduced basis to update the error vector
    # In fact, it is called modified Gram-Schmidt process
    for i in xrange(sizeTS):
      # To skip the index already added to be the reduced basis vector for speeding
      if i in RB_index:
        error_dimRB_tmp[i] = 0.
        continue
      # "projectionCoeffOnUnitVector" is used instead of "projectionCoeff" for speeding up becoz all RB are normalized before.
      Projcoeff = TSMatrix[i].projectionCoeffOnUnitVector(RBMatrix[-1], weight)
      ProjNorm2[i] += abs(Projcoeff)**2
      error_dimRB_tmp[i] = TSNorm2[i] - ProjNorm2[i]
    RB_index.append(np.argmax(error_dimRB_tmp))
    error_dimRB.append(error_dimRB_tmp[RB_index[-1]])

    # Decide to iterate further or not
    if error_dimRB[-1] < tolerance:
      printAndWrite(greedyStdout_FilePath, "a", "Because error = %E < tolerance = %E" % (error_dimRB[-1], tolerance))
      printAndWrite(greedyStdout_FilePath, "a", "Greedy algorithm is finished successfully!")
      continueToWork = False
    elif dimRB == maxRB:
      printAndWrite(greedyStdout_FilePath, "a", "Because dimRB = maxRB = %i" % (dimRB))
      printAndWrite(greedyStdout_FilePath, "a", "Greedy algorithm is finished unfortunately!")
      continueToWork = False
    elif dimRB == sizeTS:
      printAndWrite(greedyStdout_FilePath, "a", "Because dimRB = sizeTS = %i" % (dimRB))
      printAndWrite(greedyStdout_FilePath, "a", "Greedy algorithm is finished unfortunately!")
      continueToWork = False
    else:
      # Perform orthonormalization with modified Gram-Schmidt process
      RBVec_add = IMGS(TSMatrix[RB_index[-1]], RBMatrix, weight, dimRB, greedyStdout_FilePath)
      RBMatrix.append(RBVec_add.unitVector(weight))

      # Save the orthonormalized resultant basis vector
      orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "a")
      orthoNormalRBVec_File.write(RBMatrix[-1].printComponent())
      orthoNormalRBVec_File.close() 

      # Print and Save all general information
      timeSweep_f = time.time() - timeSweep_i
      printAndWrite(greedyStdout_FilePath, "a", "dimRB %i | TSIndex %i | GreedyError2 %E | timeSweep(s) %E" % (dimRB, RB_index[-1], error_dimRB[-1], timeSweep_f))

"""
  Iterative modified Gram-Schmidt process, Hoffmann
  Input:
    vectorA               : The vector to be added to the current reduced basis
    RBMatrix              : a list of current reduced basis.
    weight                : a number/vec.vector1D for calculating weighted inner product.
    dimRB                 : len(RBMatrix)
    greedyStdout_FilePath : The file path for saving stdout if any
    orthoCondition        : a number range from 0 (loosest) to 1 (tightest)
    maxCount              : maximum number of iterations
  Output:
    a orthonormal vector to the space spanned by reduced basis.
"""
def IMGS(vectorA, RBMatrix, weight, dimRB, greedyStdout_FilePath, orthoCondition = 0.25, maxCount = 3):
  cdef int j
  cdef double norm_old
  cdef double norm_new
  norm2_old = vectorA.norm2()
  count = 0
  continueToMGS = True
  while continueToMGS:
    count += 1
    for j in xrange(dimRB-1):
      # "rejectionOnUnitVector" is used instead of "rejection" for speeding up becoz all RB are normalized before.
      vectorA = vectorA.rejectionOnUnitVector(RBMatrix[j], weight)
    norm2_new = vectorA.norm2()
    if norm2_new/norm2_old < orthoCondition:
      norm2_old = norm2_new
    elif count >= maxCount:
      continueToMGS = False
    else:
      continueToMGS = False
  return vectorA

def printAndWrite(filePath, mode, strToBeSaved):
  print strToBeSaved
  f = open(filePath, mode)
  f.write(strToBeSaved+"\n")
  f.close()

