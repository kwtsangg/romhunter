import os, sys
import numpy as np
import gwhunter.utils.dataFile as df


class data:
  def __init__(self, path=os.getcwd()):
    self.invV           = np.load(path+"/Vinv.npy")
    self.basis          = np.load(path+"/orthonormalRB.npy")
#    self.oldFreq        = np.loadtxt(path+"/freq_nodes.txt")
    self.oldFreq        = np.array(df.datahunter(path+"/EIMStdout.txt").getColumn(9, dataFormat = "float"))

    self.oldB           = np.dot(self.invV.T, self.basis)
    self.sorting()
    np.save("Bkf_sorted.npy",       self.newB)
    np.save("EIM_nodes_sorted.npy", self.newFreq)

  def sorting(self):
    oldFreqArgSort      = self.oldFreq.argsort()

    self.newFreq        = self.oldFreq[oldFreqArgSort]
    self.newB           = self.oldB[oldFreqArgSort]

try:
  mydata = data(sys.argv[1])
except:
  mydata = data()
