#!/usr/bin/python
from ann import Parameters, ANN
import input
import sys
import ctypes

def main():
  dataSet = input.Input("train3.tsv")
  dataSet.remove(7)
  dataSet.remove(9)
  dataSet = list(dataSet)
  
  _, (testSet, testPos) = input.traintest(dataSet, 30)
  
  print "Test set size: %d, %.02f%%+" % (len(testSet), 100.0 * testPos / len(testSet))

  annfile = sys.argv[1]
  param = eval(open(annfile).read(),
    {"Parameters": Parameters,
     "ctypes": ctypes})

  a = ANN()
  a.prepare(testSet, testPos, 1)
  outputs = a.evaluate([param], returnOutputs=True)[0]

  sortedOutputs = sorted(outputs, reverse=True)
  n = len(testSet) * 20 / 100
  threshold = sortedOutputs[n]
  print "threshold (cpu): %.03e" % threshold
  
  thresholdGpu = a.nlargest(n)[0]
  print "threshold (gpu): %.03e" % thresholdGpu

  tp = sum(o > threshold for o in outputs[:testPos])
  print "TP:", tp
  print "FP:", sum(o > threshold for o in outputs[testPos:])
  print "TN:", sum(o <= threshold for o in outputs[testPos:])
  print "FN:", sum(o <= threshold for o in outputs[:testPos])
  
  tpr20 = float(tp) / float(n)
  print "TPR@20 (cpu): %.02f" % tpr20

  tpr = float(testPos) / float(len(testSet))
  print "Lift@20 (cpu): %.02f" % (tpr20 / tpr)
  
  liftGpu = a.lift(n)[0]
  print "Lift@20 (gpu): %.02f" % liftGpu

if __name__ == "__main__":
  main()
