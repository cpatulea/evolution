#!/usr/bin/python
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
from pycuda import tools
from itertools import islice
import numpy as np
import ctypes
import logging
from bench import timefun
import math

log = logging.getLogger("ann")

class ANN(object):
  NODES_PER_LAYER = 4
  mod = compiler.SourceModule(open("ann_kernels.cu").read())

  def prepare(self, trainSet, popSize):
    """Prepare for many parallel ANN fitness calculations.
    
                         len(trainSet[0]) x len(trainSet)
                       <--     training instances    -->
                       +--------+--------+--------+----+
       network       p0|blk(0,0)|blk(1,0)|blk(2,0)| .. |
       params        p1|blk(0,1)|blk(1,1)|   ..   | .. |
      (popSize x     p2|blk(0,2)|   ..   |   ..   | .. |
    sizeof(Params))    +--------+--------+--------+----+
     
    @param trainSet: training set
    @type trainSet: input.DataSet
    @param popSize: number of networks which will be evaluated in each run
    @type popSize: int
    """
    self.trainSet = trainSet
    self.popSize = popSize
    log.debug("training set size: %d", self.trainSet.size)
    log.debug("population size: %d", self.popSize)

    # Calculate block/grid size and prepare evaluate() kernel.
    # (avoids Function.__call__ overhead)
    maxBlockDimX = driver.Context.get_device().get_attribute(
      driver.device_attribute.MAX_BLOCK_DIM_X
    )
    self.evaluateBlockDim = (maxBlockDimX, 1, 1)
    log.debug("evaluate kernel block dim: %r", self.evaluateBlockDim)
    
    blockDimX = self.evaluateBlockDim[0]
    self.evaluateGridDim = ((self.trainSet.size + blockDimX - 1) / blockDimX,
                            self.popSize)
    log.debug("evaluate kernel grid dim: %r", self.evaluateGridDim)

    self.evaluateKernel = self.mod.get_function("evaluate")
    self.evaluateKernel.prepare(
      (np.intp, np.uint32, np.intp, np.uint32, np.intp),
      block=self.evaluateBlockDim
    )

    # Calculate block/grid size and prepare nlargest() kernel.
    self.nlargestBlockDim = (1, 1, 1)
    log.debug("nlargest kernel block dim: %r", self.nlargestBlockDim)
    
    self.nlargestGridDim = (1, self.popSize)
    log.debug("nlargest kernel grid dim: %r", self.nlargestGridDim)

    self.nlargestKernel = self.mod.get_function("nlargest")
    self.nlargestKernel.prepare(
      (np.intp, np.uint32, np.uint32, np.uint32, np.intp, np.intp),
      block=self.nlargestBlockDim
    )
    
    # Calculate block/grid size and prepare lift() kernel.
    self.countBlockDim = (maxBlockDimX, 1, 1)
    log.debug("count kernel block dim: %r", self.countBlockDim)
    
    self.countGridDim = (1, self.popSize)
    log.debug("count kernel grid dim: %r", self.countGridDim)
    
    self.countKernel = self.mod.get_function("count")
    self.countKernel.prepare(
      (np.intp, np.uint32, np.uint32, np.uint32, np.intp, np.intp),
      block=self.countBlockDim
    )

    # Heap size in each pass is limited by shared memory per multiprocessor.
    sharedBytesPerBlock = tools.DeviceData().shared_memory
    floatBytes = np.dtype(np.float32).itemsize
    log.debug("max shared memory per block: %d bytes (%d floats)",
      sharedBytesPerBlock, sharedBytesPerBlock / floatBytes)
    
    self.maxHeapFloats = (sharedBytesPerBlock / floatBytes
      * 99/100 # max size allocations fail on a GTX 275
    )
    maxHeapBytes = self.maxHeapFloats * floatBytes

    log.debug("using heap size: %d bytes (%d floats)",
      maxHeapBytes, self.maxHeapFloats)
    self.nlargestKernel.set_shared_size(maxHeapBytes)

    # Store training set in column-major order so that fetches for the same
    # input feature across instances occur at consecutive memory addresses.
    # (avoids "Strided Accesses", see CUDA Best Practices Guide section 3.2.1.4)
    # TODO: Align each feature on 128-byte boundary?
    trainSetMat = np.asmatrix(trainSet.allInstances(), np.float32)
    assert trainSetMat.shape[1] == Parameters.ih.size / Parameters.w.size
    self.trainSetDev = driver.to_device(
      trainSetMat.reshape(tuple(reversed(trainSetMat.shape)), order="F")
    )

    # Pre-allocate various large arrays
    # TODO: mem_alloc_pitch?
    floatBytes = np.dtype(np.float32).itemsize
    self.params = driver.mem_alloc(
      self.popSize * ctypes.sizeof(Parameters) * floatBytes
    )
    self.outputs = driver.mem_alloc(self.popSize * self.trainSet.size * floatBytes)
    
    uintBytes = np.dtype(np.uint32).itemsize
    self.counts = driver.mem_alloc(
      self.popSize * self.countBlockDim[0] * uintBytes
    )

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
    
    paramArrayType = Parameters * len(params)
    driver.memcpy_htod(self.params, paramArrayType(*params))

    # TODO: remove
    driver.memset_d8(self.outputs, 0, self.popSize * self.trainSet.size * 4)
    
    self.evaluateKernel.prepared_call(self.evaluateGridDim,
                                      self.trainSetDev,
                                      self.trainSet.size,
                                      self.params,
                                      self.popSize,
                                      self.outputs)

    driver.Context.synchronize()

    self.outputsMat = driver.from_device(self.outputs,
                                         shape=(self.popSize, self.trainSet.size),
                                         dtype=np.float32)
    
    if returnOutputs:
      return self.outputsMat

  def evaluate_cpu(self, p, inputs):
    out = 0.0
    
    for j in range(ANN.NODES_PER_LAYER):
      d2 = 0.0
      for i in range(19):
        d = inputs[i] * p.ih[j][i] - p.c[j][i]
        d2 += d * d
      
      h = math.exp(-p.w[j] * d2)
      out += h * p.ho[j]
    
    return out

  def nlargest(self, n):
    """Returns the per-individual threshold above which there are n outputs.
    
    @param n: number of outputs which should be above the threshold
    @type params: int

    @return list of thresholds, in order of individuals, which delimit the top
            n output values
    """
    log.debug("enter nlargest with n=%d", n)

    # Find one more output so that we can use strictly-less-than when counting
    # and underestimate lift rather than overestimating it.
    n = n + 1

    passSizes = []
    while n > 0:
      nextSize = min(self.maxHeapFloats, n)
      passSizes.append(nextSize)
      n -= nextSize

    log.debug("pass sizes: %r", passSizes)
    
    thresholdsMat = np.ones(shape=(self.popSize,),
                            dtype=np.float32) * np.inf
    self.thresholds = driver.to_device(thresholdsMat)

    uintBytes = np.dtype(np.uint32).itemsize
    thresholdCounts = np.zeros(shape=(self.popSize,),
                               dtype=np.uint32)
    self.thresholdCounts = driver.to_device(thresholdCounts)

    for passSize in passSizes:
      log.debug("begin pass size %d", passSize)
      self.nlargestKernel.prepared_call(self.nlargestGridDim,
                                        self.outputs,
                                        self.trainSet.size,
                                        self.popSize,
                                        passSize,
                                        self.thresholds,
                                        self.thresholdCounts)

      driver.Context.synchronize()

      if log.isEnabledFor(logging.DEBUG):
        thresholdsMat = driver.from_device_like(self.thresholds, thresholdsMat)
        log.debug("thresholds: %s", str(thresholdsMat))
        
        thresholdCounts = driver.from_device_like(self.thresholdCounts, thresholdCounts)
        log.debug("thresholdCounts: %s", str(thresholdCounts))

    self.thresholdsMat = driver.from_device_like(self.thresholds, thresholdsMat)
    return self.thresholdsMat

  def lift(self, n):
    """Returns (positive rate within n largest) / (overall positive rate) for
       each individual.
    
    @return list of counts, in order of individuals
    """
    self.countKernel.prepared_call(self.countGridDim,
                                   self.outputs,
                                   self.trainSet.size,
                                   len(self.trainSet.positives),
                                   self.popSize,
                                   self.thresholds,
                                   self.counts)
    
    driver.Context.synchronize()

    countsMat = driver.from_device(self.counts,
                                   shape=(self.popSize, self.countBlockDim[0]),
                                   dtype=np.uint32)
    #log.debug("counts %r: %s", countsMat.shape, str(countsMat))
    log.debug("count sum over threads: %s", str(countsMat.sum(axis=1)))
    
    self.countSums = countsMat.sum(axis=1)
    
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

  randSample = random.Random(input.RAND_SAMPLE)
  
  a = ANN()
  inp = input.Input("train3.tsv", randSample)
  
  popSize = 50
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

    p.w[0] = -1e-4
    p.w[1] = -1.0 / (10.0 ** linterp(0, 4, float(paramsIndex) / popSize))
    p.w[2] = p.w[3] = 0.0

    p.ho[0] = 1.0
    p.ho[1] = -1.0
    p.ho[2] = p.ho[3] = 0.0

    params.append(p)

  outputValues = timefun(a.evaluate, params, True)

  n = inp.trainSet.size * 20/100
  thresholds = timefun(a.nlargest, n)

  lifts = timefun(a.lift, n)
  print "Lifts:", lifts

if __name__ == "__main__":
  main()
