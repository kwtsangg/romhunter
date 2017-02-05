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

import gwhunter.utils.vector as vec
import gwhunter.utils.general as gu

#===============================================================================
#  Main
#===============================================================================
"""
  Input:
    TSMatrix                  : a list of vec.vector1D storing the trainingset vector
    weight                    : a number/vec.vector1D storing the weight for calculating innerProduct
    orthoNormalRBVec_FilePath : an output file path for the orthonormalized reduced basis vector
    greedyStdout_FilePath     : an output file path for the standard output of greedy algorithm
    stdout_FilePath           : an output file path for the standard output
    tolerance                 : maximum squared greedy error
    maxRB                     : maximum number of reduced basis vector 
  Output:
    RBMatrix                  : a list of vec.vector1D storing the orthonormalized reduced basis vector (for EIM)
                                It is saved in orthoNormalRBVec_FilePath as well
"""
@cython.cdivision(True) 
cpdef generateRB(
                 TSMatrix,
                 weight,
                 orthoNormalRBVec_FilePath,
                 greedyStdout_FilePath,
                 stdout_FilePath,
                 tolerance = 1e-12,
                 maxRB = 5000,
                 ):
  # Start time and print initial information
  timeGreedy_i = time.time()
  gu.printAndWrite(stdout_FilePath, "a+", "---")
  gu.printAndWrite(stdout_FilePath, "a", "Greedy algorithm starts.")

  # Normalize TS and get the size of TS
  cdef int  sizeTS    = len(TSMatrix)
  cdef list TSNorm2   = [0.] * sizeTS
  cdef list ProjNorm2 = [0.] * sizeTS
  cdef int j
  for j in xrange(sizeTS):
    TSMatrix[j] = TSMatrix[j].unitVector(weight)
    TSNorm2[j]  = TSMatrix[j].norm2(weight)

  # preliminary work
  cdef int i                              # for looping the TS
  cdef complex Projcoeff                  # the intermediate step variable for calculating the ProjNorm2
  cdef int  dimRB           = 1           # the number of reduced basis vector
  cdef list error_dimRB     = [1.]        # the max greedy error at each step
  cdef list error_dimRB_tmp = [0.]*sizeTS # the greedy error for each iTS
  cdef list RB_index        = [0]         # the index of TS selected to be added to RB

  # Choose an arbitrary seed choice, here the first vector in TS is chosen.
  cdef list RBMatrix = []
  RBMatrix.append(TSMatrix[0].unitVector(weight))
  gu.write(orthoNormalRBVec_FilePath, "w+", RBMatrix[-1].printComponent())

  # Initial info printing and saving
  gu.printAndWrite(greedyStdout_FilePath, "w+", "dimRB %i | TSIndex %i | GreedyError2 %E | timeSearch(s) %E | timeOrtho(s) %E | timeSweep(s) %E" % (dimRB, RB_index[-1], error_dimRB[-1], 0., 0., 0.))

  # greedy algorithm
  continueToWork = True
  while continueToWork:
    dimRB += 1
    timeSweep_i = time.time()

    # Use the last reduced basis to update the squared norm of the error vector
    timeSearch_i = time.time()
    for i in xrange(sizeTS):
      # To skip the index already added to be the reduced basis vector for speeding
      if i in RB_index:
        error_dimRB_tmp[i] = 0.
        continue
      # "projectionCoeffOnUnitVector" is used instead of "projectionCoeff" for speeding up becoz all RB are normalized before.
      Projcoeff = TSMatrix[i].projectionCoeffOnUnitVector(RBMatrix[-1], weight)
      ProjNorm2[i] += (Projcoeff*Projcoeff.conjugate()).real
      error_dimRB_tmp[i] = TSNorm2[i] - ProjNorm2[i]
    RB_index.append(np.argmax(error_dimRB_tmp))
    error_dimRB.append(error_dimRB_tmp[RB_index[-1]])
    timeSearch = time.time() - timeSearch_i

    # Perform orthonormalization with iterative modified Gram-Schmidt process
    timeOrtho_i = time.time()
    RBVec_add = IMGS(TSMatrix[RB_index[-1]], RBMatrix, weight, dimRB, RB_index[-1], stdout_FilePath)
    timeOrtho = time.time()-timeOrtho_i
    RBMatrix.append(RBVec_add.unitVector(weight))

    # Save the orthonormalized resultant basis vector
    gu.write(orthoNormalRBVec_FilePath, "a", RBMatrix[-1].printComponent())

    # Print and Save all general information
    timeSweep = time.time() - timeSweep_i
    gu.printAndWrite(greedyStdout_FilePath, "a", "dimRB %i | TSIndex %i | GreedyError2 %E | timeSearch(s) %E | timeOrtho(s) %E | timeSweep(s) %E" % (dimRB, RB_index[-1], error_dimRB[-1], timeSearch, timeOrtho, timeSweep))

    # Decide to iterate further or not
    if error_dimRB[-1] < tolerance:
      gu.printAndWrite(stdout_FilePath, "a", "At dimRB = %i, because error = %E < tolerance = %E" % (dimRB, error_dimRB[-1], tolerance))
      gu.printAndWrite(stdout_FilePath, "a", "Greedy algorithm is finished successfully!")
      continueToWork = False
    elif dimRB == maxRB:
      gu.printAndWrite(stdout_FilePath, "a", "Because dimRB = maxRB = %i" % (dimRB))
      gu.printAndWrite(stdout_FilePath, "a", "Greedy algorithm is finished unfortunately!")
      continueToWork = False
    elif dimRB == sizeTS:
      gu.printAndWrite(stdout_FilePath, "a", "Because dimRB = sizeTS = %i" % (dimRB))
      gu.printAndWrite(stdout_FilePath, "a", "Greedy algorithm is finished unfortunately!")
      continueToWork = False

  # Final part
  timeGreedy = time.time() - timeGreedy_i
  gu.printAndWrite(stdout_FilePath, "a", "The greedy algorithm takes %f wall seconds to complete." % timeGreedy)
  
  return RBMatrix

"""
  Iterative modified Gram-Schmidt process, Hoffmann
  Input:
    vectorA               : The vector (vec.vector1D) to be added to the current reduced basis
    RBMatrix              : a list of current reduced basis.
    weight                : a number/vec.vector1D for calculating weighted inner product.
    dimRB                 : len(RBMatrix)
    RBindex               : The TSindex corresponding to vectorA.
    stdout_FilePath       : The output file path for saving stdout if any
    orthoCondition        : a number range from 0 (loosest) to 1 (tightest)
    maxCount              : maximum number of iterations
  Output:
    a orthonormal vector (vec.vector1D) to the space spanned by reduced basis.
"""
cpdef IMGS(vectorA, RBMatrix, weight, dimRB, TSIndex, stdout_FilePath, orthoCondition = 0.25, maxCount = 3):
  cdef int j
  cdef double norm2_old = vectorA.norm2()
  cdef double norm2_new
  cdef int count = 0
  continueToMGS = True
  while continueToMGS:
    count += 1
    for j in xrange(dimRB-1):
      # "rejectionOnUnitVector" is used instead of "rejection" for speeding up becoz all RB are normalized before.
      vectorA = vectorA.rejectionOnUnitVector(RBMatrix[j], weight)
    norm2_new = vectorA.norm2()
    if norm2_new/norm2_old < orthoCondition:
      if count > maxCount:
        continueToMGS = False
        gu.printAndWrite(stdout_FilePath, "a", "Warning :: The maxCount (%i) for IMGS reached. %ith (TSIndex %i) might not be orthonormalized enough." % (maxCount, dimRB, TSIndex))
      else:
        norm2_old = norm2_new
    else:
      continueToMGS = False
  return vectorA

