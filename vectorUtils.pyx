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
  double sqrt(double number)

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
    self.tolerance = 1e-20
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
  def innerProduct(self, other):
    cdef complex result = 0.
    cdef int i
    assert self.dim == other.dim
    for i in xrange(self.dim):
      result += (self.component[i]).conjugate() * other.component[i]
    return result
  @cython.cdivision(True) 
  def norm(self):
    cdef double result = 0.
    cdef double component = 0.
    cdef int i
    for i in xrange(self.dim):
      component = abs(self.component[i])
      result += component*component
    return sqrt(result)
  @cython.cdivision(True) 
  def unitVector(self):
    if self.norm() < self.tolerance:
      printf("Cannot normalize an unit vector. Keeping it as null vector ...", __file__, "warning")
      return vector1D([0.]*self.dim)
    return self.div(self.norm())

  @cython.cdivision(True) 
  def projectionCoeff(self, other):
    return self.innerProduct(other)/other.innerProduct(other)
  @cython.cdivision(True) 
  def projection(self, other):
    return other.mul(self.projectionCoeff(other))
  @cython.cdivision(True) 
  def rejection(self, other):
    return self - self.projection(other)

  @cython.cdivision(True) 
  def projectionCoeffOnUnitVector(self, other):
    return self.innerProduct(other)
  @cython.cdivision(True) 
  def projectionOnUnitVector(self, other):
    return other.mul(self.innerProduct(other))
  @cython.cdivision(True) 
  def rejectionOnUnitVector(self, other):
    return self - self.projectionOnUnitVector(other)

  # Other function
  def printComponent(self):
    result = ""
    for i in xrange(self.dim):
      result += str(self.component[i]) + " "
    result = result[:-1] + "\n"
    return result

