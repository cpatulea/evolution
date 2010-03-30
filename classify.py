#!/usr/bin/python
from ann import Parameters, ANN
import input
import sys
import ctypes
import numpy as np
import random

class SampleTester(object):
  def prepare(self, testSet, rand, numSamples=20, samplePercent=50):
    """Prepares for bootstrap estimation of lift of one ANN.
    
    @param testSet: test set on which to test the ANN
    @type testSet: input.DataSet
    @param rand: source of randomness for bootstrap samples
    @type rand: random.Random
    @param numSamples: number of bootstrap samples
    @type numSamples: int
    @param samplePercent: size of each sample, in percent of the testSet
    @type samplePercent: int
    """
    self.sampleSets = []
    self.anns = []
    for sampleIndex in range(numSamples):
      sampleSet = testSet.sample(samplePercent, rand)
      self.sampleSets.append(sampleSet)

      a = ANN()
      a.prepare(sampleSet, popSize=1)
      self.anns.append(a)

    self.sampleSize = self.sampleSets[0].size

  def showSampleSets(self):
    for sampleIndex, sampleSet in enumerate(self.sampleSets):
      print "Sample %d:" % sampleIndex,
      sampleSet.show()

  def test(self, param):
    """Performs the bootstrap test to estimate one ANN's lift.
    
    @param param: ANN to be tested
    @type param: ann.Parameters
    @return (float, float): lift average and standard deviation over numSamples
    """
    n = self.sampleSize * 20 / 100

    lifts = []
    for a in self.anns:
      a.evaluate([param], returnOutputs=False)

      threshold = a.nlargest(n)[0]
      #print "threshold: %.03e" % threshold

      lift = a.lift(n)[0]
      
      lifts.append(lift)
      #print "Lift@20: %.02f" % lift

    avg, std = np.mean(lifts), np.std(lifts)
    return avg, std

def main(annfile):
  randSample = random.Random(input.SAMPLE_SEED)
  
  inp = input.Input("train3.tsv", randSample)
  
  print "Test set:",
  inp.testSet.show()

  param = eval(open(annfile).read(),
    {"Parameters": Parameters,
     "ctypes": ctypes})

  tester = SampleTester()
  tester.prepare(inp.testSet, randSample)
  tester.showSampleSets()
  
  avg, std = tester.test(param)

  print "Lift:", "avg: %.03f" % avg, "std: %.03f" % std

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Usage: classify.py <file.ann>"
    sys.exit(1)

  main(sys.argv[1])
