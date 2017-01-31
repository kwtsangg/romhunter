#!/usr/bin/env python
__file__       = "vectorUtils.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2017"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2017-Jan-26"

Description=""" Useful functions for manipulations of vectors.
"""

#===============================================================================
#  Module
#===============================================================================
import cython

cdef extern from "math.h":
  double sqrt(double number)

#===============================================================================
#  Main
#===============================================================================
class vector1D():
  @cython.cdivision(True) 
  def __init__(self, listA):
    self.tolerance = 1e-20
    self.dim       = len(listA)
    self.component = [0.]*self.dim
    cdef int i
    for i in xrange(self.dim):
      self.component[i] = float(listA[i])
      if abs(self.component[i]) < self.tolerance:
        self.component[i] = 0.
  # Operator definition
  # vector addition
  @cython.cdivision(True) 
  def __add__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i] + other.component[i]
    return vector1D(result)
  # vector subtraction
  @cython.cdivision(True) 
  def __sub__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(0, self.dim):
      result[i] = self.component[i] - other.component[i]
    return vector1D(result)

  # General function
  @cython.cdivision(True) 
  def add(self, a):
    return self + vector1D(self.dim*[a])
  @cython.cdivision(True) 
  def sub(self, a):
    return self - vector1D(self.dim*[a])
  @cython.cdivision(True) 
  def scalarMul(self, a):
    result = [0.]*self.dim
    cdef int i
    for i in xrange(self.dim):
      result[i] = self.component[i] * float(a)
    return vector1D(result)
  @cython.cdivision(True) 
  def innerProduct(self, other):
    cdef double result = 0.
    cdef int i
    assert self.dim == other.dim
    for i in xrange(self.dim):
      result += self.component[i] * other.component[i]
    return result
  @cython.cdivision(True) 
  def norm(self):
    return sqrt(self.innerProduct(self))
  @cython.cdivision(True) 
  def unitVector(self):
    if self.norm() < self.tolerance:
      return vector1D([0.]*self.dim)
    return self.scalarMul(1./self.norm())
  @cython.cdivision(True) 
  def projection(self, other):
    return other.scalarMul(self.innerProduct(other)/other.innerProduct(other))
  @cython.cdivision(True) 
  def rejection(self, other):
    return self - self.projection(other)
  @cython.cdivision(True) 
  def projectionUnitVector(self, other):
    return other.scalarMul(self.innerProduct(other))
  @cython.cdivision(True) 
  def rejectionUnitVector(self, other):
    return self - self.projectionUnitVector(other)

  # Other function
  def printComponent(self):
    result = ""
    for i in xrange(self.dim):
      result += str(self.component[i]) + " "
    result = result[:-1] + "\n"
    return result

