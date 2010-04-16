#!/usr/bin/python
from itertools import islice
import numpy as np
import ctypes
import logging
from bench import timefun
import math

log = logging.getLogger("ann")

class ANN(object):
  NODES_PER_LAYER = 4

  def prepare(self, trainSet, popSize):
    """Prepare for many ANN fitness calculations.
    
    @param trainSet: training set
    @type trainSet: input.DataSet
    @param popSize: number of networks which will be evaluated in each run
    @type popSize: int
    """
    self.trainSet = trainSet
    self.popSize = popSize
    log.debug("training set size: %d", self.trainSet.size)
    log.debug("population size: %d", self.popSize)

    # Store training set in row-major order so all features for a given
    # instance are nearby.
    self.trainSetMat = np.asarray(trainSet.allInstances(), np.float32)
    assert self.trainSetMat.shape[1] == Parameters.ih.size / Parameters.w.size

    # Pre-allocate various large arrays
    self.outputsMat = np.zeros(shape=(self.popSize, self.trainSet.size),
                               dtype=np.float32)
    
  def evaluate(self, params, returnOutputs=False):
    """Evaluate several networks (with given params) on training set.
    
    @param params: network params
    @type params: list of Parameters
    @param returnOutputs: return network output values (debug)
    @type returnOutputs: bool, default False
    
    @return output matrix if returnOutputs=True, else None
    """
    if self.popSize != len(params):
      raise ValueError("Need %d Parameter structures (provided %d)" % (
        self.popSize, len(params)))
    
    for popIndex, p in enumerate(params):
      ih = np.asarray(p.ih)
      ih = ih.reshape((ih.shape[0], 1, ih.shape[1]))
      c = np.asarray(p.c)
      c = c.reshape((c.shape[0], 1, c.shape[1]))
      w = np.asarray(p.w)
      w = w.reshape((w.shape[0], 1))
      ho = np.asarray(p.ho)
      #ho = ho.reshape((1, ho.shape[0]))
      #print "w shape:", w.shape
      #print "ho shape:", ho.shape
      
      #print "trainSetMat shape:", self.trainSetMat.shape
      #print "ih shape:", ih.shape
      #outer = np.multiply(self.trainSetMat, ih)
      outer = self.trainSetMat * ih
      #print "outer shape:", outer.shape
      #print "c shape:", c.shape
      d = outer - c
      d2 = (d * d).sum(axis=2)
      #print d2
      #print "d2 shape:", d2.shape
      
      h = np.exp(-w * d2)
      #print "h shape:", h.shape
      
      o = np.dot(ho, h)
      #print "o shape:", o.shape
      self.outputsMat[popIndex] = o

    if returnOutputs:
      return self.outputsMat

  def nlargest_sort(self, n):
    """Returns the per-individual threshold above which there are n outputs.
    
    @param n: number of outputs which should be above the threshold
    @type params: int

    @return list of thresholds, in order of individuals, which delimit the top
            n output values
    """
    log.debug("enter nlargest with n=%d", n)

    sortedOutputs = np.sort(self.outputsMat, axis=1)
    self.thresholdsMat = sortedOutputs[:, sortedOutputs.shape[1] - n]
    return self.thresholdsMat

  def nlargest_heap(self, n):
    import heapq
    hr = heapq.heapreplace
    
    self.thresholdsMat = np.zeros(shape=(self.outputsMat.shape[0],))
    for popIndex, popOutputs in enumerate(self.outputsMat):
      h = [float("-inf")] * n
      for out in popOutputs:
        if out > h[0]:
          hr(h, out)
      self.thresholdsMat[popIndex] = h[0]

    return self.thresholdsMat
  
  # Having to iterate over the data points in Python makes the heap
  # implementation significantly (~350x) slower in practice.
  nlargest = nlargest_sort

  def lift(self, n):
    """Returns (positive rate within n largest) / (overall positive rate) for
       each individual.
    
    @return list of counts, in order of individuals
    """
    thresholds = self.thresholdsMat.reshape((self.thresholdsMat.shape[0], 1))
    positives = self.outputsMat[:, 0:len(self.trainSet.positives)]
    self.countSums = np.sum(positives > thresholds, axis=1)

    self.nlargestPositiveRate = np.float32(self.countSums) / n
    log.debug("positive rate (n largest outputs): %s", str(self.nlargestPositiveRate))
    
    overallPositiveRate = float(len(self.trainSet.positives)) / float(self.trainSet.size)
    log.debug("positive rate (overall): %.04f", overallPositiveRate)
    
    lifts = self.nlargestPositiveRate / overallPositiveRate
    
    sortedLifts = sorted(enumerate(lifts), key=lambda (i, l): l, reverse=True)
    topIndex, topLift = sortedLifts[0]
    
    topOutputs = self.outputsMat[topIndex]
    
    nans = np.sum(np.isnan(topOutputs))
    neginfs = np.sum(np.isneginf(topOutputs))
    posinfs = np.sum(np.isposinf(topOutputs))
    omin = np.nanmin(topOutputs)
    omax = np.nanmax(topOutputs)
    threshold = self.thresholdsMat[topIndex]
    
    log.debug("The top ANN's outputs are:")
    log.debug(
      "  %.02f%% NaN, %.02f%% -inf, %.02f%% +inf, min %.02e, max %.02e, thresh %.02e",
      100.0 * nans / len(topOutputs),
      100.0 * neginfs / len(topOutputs),
      100.0 * posinfs / len(topOutputs),
      omin, omax, threshold)
    
    return lifts

class Parameters(ctypes.Structure):
  _fields_ = [("ih", ANN.NODES_PER_LAYER * (19 * ctypes.c_float)),
              ("c", ANN.NODES_PER_LAYER * (19 * ctypes.c_float)),
              ("w", ANN.NODES_PER_LAYER * ctypes.c_float),
              ("ho", ANN.NODES_PER_LAYER * ctypes.c_float)]

  def _float_list_str(self, l):
    return ", ".join("%401g" % el for el in l)

  def __str__(self):
    s = []
    s.append("Parameters(\n")
    for i in range(ANN.NODES_PER_LAYER):
      ihl = list(self.ih[i])
      s.append("  ih[%d]=[0: %s,\n" % (i, self._float_list_str(ihl[0:10])))
      s.append("        10: %s]\n" % (self._float_list_str(ihl[10:])))
    for i in range(ANN.NODES_PER_LAYER):
      cl = list(self.c[i])
      s.append("  c[%d] =[0: %s,\n" % (i, self._float_list_str(cl[0:10])))
      s.append("        10: %s]\n" % (self._float_list_str(cl[10:])))
    s.append("  w =%s,\n" % self._float_list_str(list(self.w)))
    s.append("  ho=%s,\n" % self._float_list_str(list(self.ho)))
    s.append(")")
    return "".join(s)

  def _array_repr(self, dims, value):
    if dims == []:
      return "%e" % value

    typestr = "ctypes.c_float"
    for dim in reversed(dims):
      typestr = "(%d*%s)" % (dim, typestr)
    
    valuestr = typestr + "(" + ", ".join(self._array_repr(dims[1:], subvalue) for subvalue in value) + ")"
    return valuestr

  def __repr__(self):
    r = ["Parameters(\n"]
    r.append("  ih=%s,\n" % self._array_repr([ANN.NODES_PER_LAYER, 19], self.ih))
    r.append("  c=%s,\n" % self._array_repr([ANN.NODES_PER_LAYER, 19], self.c))
    r.append("  w=%s,\n" % self._array_repr([ANN.NODES_PER_LAYER], self.w))
    r.append("  ho=%s\n" % self._array_repr([ANN.NODES_PER_LAYER], self.ho))
    r.append(")")
    return "".join(r)

  @staticmethod
  def from_file(annfile):
    return eval(compile(open(annfile).read(), annfile, "eval"),
                {"Parameters": Parameters,
                 "ctypes": ctypes})

def linterp(a, b, p):
  return a + (b - a) * p

def forceOnlyFeatures(params, featList):
  for p in params:
    # we want only feature 0 and the class
    for i in range(ANN.NODES_PER_LAYER):
      for j in range(19):
        if j not in featList:
          p.ih[i][j] = 0
          p.c[i][j] = 0
    p.ho[1] = p.ho[2] = p.ho[3] = 0.0

def outputTypes(label, valueIter):
  nan = neginf = posinf = others = 0
  for out in valueIter:
    if np.isnan(out):
      nan += 1
    elif np.isneginf(out):
      neginf += 1
    elif np.isposinf(out):
      posinf += 1
    else:
      others += 1

  print "%s output types: nan=%d neginf=%d posinf=%d others=%d" % (
    label, nan, neginf, posinf, others)

def showParams(p):
  topLift, top, topIndex = taggedParams[0]
  topThreshold = thresholds[topIndex]
  print "Top: index=%d, lift=%.02f, threshold=%.02e" % (
    topIndex, topLift, topThreshold), top
  
  topOutputs = outputValues[topIndex]
  outputTypes("Top", topOutputs.flat)
  goodOutputs = [o for o in topOutputs if not np.isnan(o)]

  topMin, topMax = min(topOutputs), max(topOutputs)
  topRange = topMax - topMin
  print "Top output stats: min=%.02e max=%.02e" % (topMin, topMax)
  
  print "Top poscount: %d" % a.countSums[topIndex]
  
  cpuCount = sum(o > topThreshold for o in topOutputs[0:len(a.trainSet.positives)])
  print "Top cpucount: %d" % cpuCount

  print "Top posrate: %.02e" % a.nlargestPositiveRate[topIndex]
  print "Top lift: %.02e" % lifts[topIndex]

  print "Train set: %d+ %d-" % (len(a.trainSet.positives), len(a.trainSet.negatives))

  #histRange = (topMin - .1*topRange, topMax + .1*topRange)
  histRange = (.5*topThreshold, 1.5*topThreshold)
  print "Hist range:", histRange
  histPos, binsPos = np.histogram(goodOutputs[0:len(a.trainSet.positives)],
                                  bins=20, range=histRange)
  histNeg, binsNeg = np.histogram(goodOutputs[len(a.trainSet.positives):],
                                  bins=20, range=histRange)
  assert (binsPos == binsNeg).all()
  bins = binsPos
  
  for bl, bh, hp, hn in zip(bins[:-1], bins[1:], histPos, histNeg):
    print "%.02e .. %.02e: %d+ %d-" % (bl, bh, hp, hn)

def main():
  import input
  import random
  logging.basicConfig(level=logging.DEBUG)
  np.set_printoptions(precision=3, edgeitems=3, threshold=20)

  randSample = random.Random(input.SAMPLE_SEED)
  
  a = ANN()
  inp = input.Input("train3.tsv", randSample)
  
  popSize = 10
  timefun(a.prepare, inp.trainSet, popSize)

  params = []

  for paramsIndex in range(popSize):
    p = Parameters()
    for i in range(19):
      p.ih[0][i] = 0.0
      p.ih[1][i] = p.ih[2][i] = p.ih[3][i] = 0.0

      p.c[0][i] = 0.0
      p.c[1][i] = p.c[2][i] = p.c[3][i] = 0.0

    p.ih[0][0] = 1.0
    p.ih[1][11] = 1.0

    p.w[0] = -1e-0
    p.w[1] = -1.0 / (10.0 ** linterp(0, 4, float(paramsIndex) / popSize))
    p.w[2] = p.w[3] = 0.0

    p.ho[0] = 1.0
    p.ho[1] = 0.0 #-1.0
    p.ho[2] = p.ho[3] = 0.0

    params.append(p)

  outputValues = timefun(a.evaluate, params, True)

  n = inp.trainSet.size * 20/100
  thresholds = timefun(a.nlargest, n)

  lifts = timefun(a.lift, n)
  print "Lifts:", lifts

if __name__ == "__main__":
  main()
