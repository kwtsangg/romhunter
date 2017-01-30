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

import vectorUtils as vec

#===============================================================================
#  Main
#===============================================================================

@cython.cdivision(True) 
def greedy(TSVec_FilePath,
           orthoNormalRBVec_FilePath = os.getcwd() + "/output/orthonormalRBVec.txt",
           greedyStdout_FilePath     = os.getcwd() + "/output/greedyStdout.txt",
           tolerance = 1e-12,
           maxRB     = 1000):
  # Add header to greedyStdout_File
  print "#1 dimRB #2 TSIndex #3 Error #4 timeSweep(s)"
  print "%i %i %E %f" % (0, 0, 0., 0.)
  greedyStdout_File = open(greedyStdout_FilePath, "w+")
  greedyStdout_File.write("#1 dimRB #2 TSIndex #3 Error #4 timeSweep\n")
  greedyStdout_File.write("%i %i %E %f\n" % (0, 0, 0., 0.))
  greedyStdout_File.close()

  # Choose an arbitrary seed choice, here the first vector in TS is chosen.
  TSVec_File = open(TSVec_FilePath, "r")
  RBVec_tmp = vec.vector1D(TSVec_File.readline().split())
  TSVec_File.close()

  orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "w+")
  orthoNormalRBVec_File.write(RBVec_tmp.unitVector().printComponent())
  orthoNormalRBVec_File.close()
  
  # preliminary work
  cdef int dimRB = 0
  error_dimRB = [0.] # to store the max greedy error at each step
  RB_index    = [0]  # to store the index of TS selected to be added to RB

  cdef i
#  RBVec_tmp = vec.vector1D([0.])
#  cdef double RBVec_tmp_error       = -1.
#  cdef int    RBVec_tmp_error_index = -1

    # get trainingset size
  TSVec_File = open(TSVec_FilePath, "r")
  for sizeTS, none in enumerate(TSVec_File):
    pass
  TSVec_File.close()
  # greedy algorithm
  continueToWork = True
  while continueToWork:
    dimRB += 1
    timeSweep_i = time.time()
    # Loop over trainingset/parameters space to calculate errors
    error_dimRB_tmp = [] # to store the greedy error for each iTS
    TSVec_File = open(TSVec_FilePath, "r")
    for i, iTS in enumerate(TSVec_File):
      # To avoid the same index is being selected twice
      if i in RB_index:
        continue
      iTSVec      = vec.vector1D(iTS.split())
      # Loop over the reduced basis to get the error vector
      # In fact, it is modified Gram-Schmidt process
      orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "r")
      for jRB in orthoNormalRBVec_File:
        iTSVec = iTSVec.rejectionUnitVector(vec.vector1D(jRB.split()))
      orthoNormalRBVec_File.close()
      # Now iTSVec is the orthogonal vector to RB
      # To avoid confusion, we define iTSVec_rej redundantly.
      iTSVec_rej = iTSVec

      error_dimRB_tmp.append( iTSVec_rej.norm() )
#      print "%i th TS , greedy error = %f." % (i, error_dimRB_tmp[-1])
#      print "              max error = %f." % ( max(error_dimRB_tmp) )
#      print "           iTSVec_rej   = %s." % iTSVec_rej.printComponent()
#      print "       iTSVec_rej.norm()= %f." % iTSVec_rej.norm()
#      print " iTSVec_rej.unitVector()= %s." % iTSVec_rej.unitVector().printComponent()
      if max(error_dimRB_tmp) == error_dimRB_tmp[-1]:
        RBVec_tmp             = iTSVec_rej
        RBVec_tmp_error       = error_dimRB_tmp[-1]
        RBVec_tmp_error_index = i

      # Obtain the maximum error, the corresponding paramaters and the orthonormalized vector
    TSVec_File.close()

    # Decide to iterate further or not
    if RBVec_tmp_error < tolerance:
      print "Because error = %E < tolerance = %E" % (RBVec_tmp_error, tolerance)
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
#    elif RBVec_tmp_error_index in RB_index:
#      print "Because TS (index = %i) is selected twice as RB" % (RBVec_tmp_error_index)
#      print "Greedy algorithm is finished unfortunately!"
#      continueToWork = False
    else:
      # Perform orthonormalization with modified Gram-Schmidt process
      # p.s. In fact, one needs only normalization only.
      # Save the orthonormalized resultant basis vector
      orthoNormalRBVec_File = open(orthoNormalRBVec_FilePath, "a")
      orthoNormalRBVec_File.write(RBVec_tmp.unitVector().printComponent())
      orthoNormalRBVec_File.close() 

      # Save all general information
      error_dimRB.append(RBVec_tmp.norm())
      RB_index.append(RBVec_tmp_error_index)

      timeSweep_f = time.time() - timeSweep_i
      print "%i %i %E %f" % (dimRB, RB_index[-1], error_dimRB[-1], timeSweep_f)
      greedyStdout_File = open(greedyStdout_FilePath, "a")
      greedyStdout_File.write("%i %i %E %f\n" % (dimRB, RB_index[-1], error_dimRB[-1], timeSweep_f))
      greedyStdout_File.close()


