#!/usr/bin/python
from ann import Parameters, ANN
import input
import sys
import ctypes
import numpy as np
import random

class SampleTester(object):
  def prepare(self, testSet, testPos, numSamples=20, samplePercent=70):
    self.numSamples = numSamples
    self.samplePercent = samplePercent
    
    sampleSize = len(testSet) * self.samplePercent / 100
    samplePos =       testPos * self.samplePercent / 100
    sampleNeg = sampleSize - samplePos
    self.sampleSize = sampleSize

    self.anns = []
    for sampleIndex in range(self.numSamples):
      sampleSet = (random.sample(testSet[:testPos], samplePos) +
                   random.sample(testSet[testPos:], sampleNeg))

      a = ANN()
      a.prepare(sampleSet, samplePos, 1)
      self.anns.append(a)

  def test(self, param):
    n = self.sampleSize * 20 / 100

    lifts = []
    for a in self.anns:
      a.evaluate([param], returnOutputs=False)

      threshold = a.nlargest(n)[0]
      #print "threshold: %.03e" % threshold

      lift = a.lift(n)[0]
      
      lifts.append(lift)
      #print "Lift@20: %.02f" % lift

    avg, std = float(sum(lifts)) / len(lifts), np.std(lifts)
    return avg, std

def main():
  dataSet = input.Input("train3.tsv")
  dataSet = list(dataSet)
  
  _, (testSet, testPos) = input.traintest(dataSet, 30)
  
  print "Test set size: %d, %.02f%%+" % (len(testSet), 100.0 * testPos / len(testSet))

  annfile = sys.argv[1]
  param = eval(open(annfile).read(),
    {"Parameters": Parameters,
     "ctypes": ctypes})

  random.seed(1005108)
  
  tester = SampleTester()
  tester.prepare(testSet, testPos)
  avg, std = tester.test(param)

  print "avg:", avg
  print "std:", std

if __name__ == "__main__":
  main()
