#!/usr/bin/env python
__file__       = "vectorUtils.pyx"
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
from gwhunter.utils.generalPythonFunc import printf

cdef extern from "math.h":
  double sqrt(double)

#===============================================================================
#  Main
#===============================================================================

"""
  Input: a list of complex number
"""
cdef class vector1D(object):
  cpdef public double tolerance
  cpdef public int    dim
  cpdef public list   component

  @cython.cdivision(True) 
  def __init__(self, listA):
    self.tolerance = 1e-30
    self.dim       = len(listA)
    self.component = [0.]*self.dim
    cdef int i
    for i in xrange(self.dim):
      self.component[i] = complex(listA[i])

  # Operator definition
  # elementwise addition
  @cython.cdivision(True) 
  def __add__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i] + other.component[i]
    return vector1D(result)
  # elementwise subtraction
  @cython.cdivision(True) 
  def __sub__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i] - other.component[i]
    return vector1D(result)
  # elementwise multiplication
  @cython.cdivision(True) 
  def __mul__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i] * other.component[i]
    return vector1D(result)
  # elementwise division
  @cython.cdivision(True) 
  def __div__(self, other):
    cdef int i
    assert self.dim == other.dim
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i] / other.component[i]
    return vector1D(result)

  # General function
  @cython.cdivision(True) 
  def conj(self):
    cdef int i
    result = [0.]*self.dim
    for i in xrange(self.dim):
      result[i] = self.component[i].conjugate()
    return vector1D(result)
  @cython.cdivision(True) 
  def add(self, a):
    cdef int i
    result = [0.]*self.dim
    a = complex(a)
    for i in xrange(self.dim):
      result[i] = self.component[i] + a
    return vector1D(result)
  @cython.cdivision(True) 
  def sub(self, a):
    cdef int i
    result = [0.]*self.dim
    a = complex(a)
    for i in xrange(self.dim):
      result[i] = self.component[i] - a
    return vector1D(result)
  @cython.cdivision(True) 
  def mul(self, a):
    cdef int i
    result = [0.]*self.dim
    a = complex(a)
    for i in xrange(self.dim):
      result[i] = self.component[i] * a
    return vector1D(result)
  @cython.cdivision(True) 
  def div(self, a):
    cdef int i
    result = [0.]*self.dim
    a = complex(a)
    for i in xrange(self.dim):
      result[i] = self.component[i] / a
    return vector1D(result)

  @cython.cdivision(True) 
  def innerProduct(self, other, weight = None):
    cdef int i
    cdef complex result = 0.j
    assert self.dim == other.dim
    if weight == None:
      for i in xrange(self.dim):
        result += self.component[i].conjugate() * other.component[i]
    elif isinstance(weight, (int, float, complex)):
      for i in xrange(self.dim):
        result += self.component[i].conjugate() * other.component[i]
        result *= weight
    else:
      assert weight.dim == self.dim
      for i in xrange(self.dim):
        result += weight.component[i] * self.component[i].conjugate() * other.component[i]
    return result

  @cython.cdivision(True) 
  def norm(self, weight = None):
    return sqrt(abs(self.innerProduct(self, weight)))
  @cython.cdivision(True) 
  def unitVector(self, weight = None):
    if self.norm(weight) < self.tolerance:
      printf("Cannot normalize an unit vector. Keeping it as null vector ...", __file__, "warning")
      return vector1D([0.]*self.dim)
    return self.div(self.norm(weight))

  @cython.cdivision(True) 
  def projectionCoeff(self, other, weight = None):
    return self.innerProduct(other,weight)/other.innerProduct(other, weight)
  @cython.cdivision(True) 
  def projection(self, other, weight = None):
    return other.mul(self.projectionCoeff(other, weight))
  @cython.cdivision(True) 
  def rejection(self, other, weight = None):
    return self - self.projection(other, weight)

  @cython.cdivision(True) 
  def projectionCoeffOnUnitVector(self, other, weight = None):
    return self.innerProduct(other, weight)
  @cython.cdivision(True) 
  def projectionOnUnitVector(self, other, weight = None):
    return other.mul(self.innerProduct(other, weight))
  @cython.cdivision(True) 
  def rejectionOnUnitVector(self, other, weight = None):
    return self - self.projectionOnUnitVector(other, weight)

  # Other function
  def printComponent(self):
    result = ""
    for i in xrange(self.dim):
      result += str(self.component[i]) + " "
    result = result[:-1] + "\n"
    return result

