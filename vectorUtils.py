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
import sys
import numpy as np

from gwhunter.utils.generalPythonFunc import printf

#===============================================================================
#  Main
#===============================================================================
class vector1D():
  def __init__(self, listA):
    self.tolerance = 1e-20
    self.dim       = len(listA)
    self.component = [0.]*self.dim
    for i in range(0, self.dim):
      self.component[i] = np.float128(listA[i])
      if abs(self.component[i]) < self.tolerance:
        self.component[i] = 0.
  # Operator definition
  # vector addition
  def __add__(self, other):
    if self.dim == other.dim:
      result = [0.0]*self.dim
      for i in range(0, self.dim):
        result[i] = self.component[i] + other.component[i]
      return vector1D(result)
    else:
      printf("vector1D: vectors have to have same dim.", __file__, "warning")
      return None
  # vector subtraction
  def __sub__(self, other):
    if self.dim == other.dim:
      result = [0.0]*self.dim
      for i in range(0, self.dim):
        result[i] = self.component[i] - other.component[i]
      return vector1D(result)
    else:
      printf("vector1D: vectors have to have same dim.", __file__, "warning")
      return None

  # General function
  def add(self, a):
    return self + vector1D(self.dim*[a])
  def sub(self, a):
    return self - vector1D(self.dim*[a])
  def scalarMul(self, a):
    result = [0.0]*self.dim
    for i in range(0, self.dim):
      result[i] = self.component[i] * np.float128(a)
    return vector1D(result)
  def innerProduct(self, other):
    if self.dim == other.dim:
      result = 0.0
      for i in range(0,self.dim):
        result += self.component[i] * other.component[i]
      return result
    else:
      printf("vector1D: vectors have to have same dim.", __file__, "warning")
      return None
  def norm(self):
    return np.sqrt(self.innerProduct(self))
  def unitVector(self):
    if self.norm() < self.tolerance:
      return vector1D([0.]*self.dim)
    return self.scalarMul(1./self.norm())
  def projection(self, other):
    return other.scalarMul(self.innerProduct(other)/other.innerProduct(other))
  def rejection(self, other):
    return self - self.projection(other)
      
  # Other function
  def printComponent(self):
    result = ""
    for item in self.component:
      result += str(item) + " "
    result = result[:-1] + "\n"
    return result

