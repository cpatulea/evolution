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
import random, math

log = logging.getLogger("ann")

class Parameters(ctypes.Structure):
  _fields_ = [("ih", 4 * (19 * ctypes.c_float)),
              ("c", 4 * (19 * ctypes.c_float)),
              ("w", 4 * ctypes.c_float),
              ("ho", 4 * ctypes.c_float)]

  def _float_list_str(self, l):
    return ", ".join("%4.01g" % el for el in l)

  def __str__(self):
    s = []
    s.append("Parameters(\n")
    for i in range(4):
      ihl = list(self.ih[i])
      s.append("  ih[%d]=[0: %s,\n" % (i, self._float_list_str(ihl[0:10])))
      s.append("        10: %s]\n" % (self._float_list_str(ihl[10:])))
    for i in range(4):
      cl = list(self.c[i])
      s.append("  c[%d] =[0: %s,\n" % (i, self._float_list_str(cl[0:10])))
      s.append("        10: %s]\n" % (self._float_list_str(cl[10:])))
    s.append("  w =%s,\n" % self._float_list_str(list(self.w)))
    s.append("  ho=%s,\n" % self._float_list_str(list(self.ho)))
    s.append(")")
    return "".join(s)
    
class ANN(object):
  mod = compiler.SourceModule(open("ann_kernels.cu").read())

  def _trainSetInClassOrder(self, trainSet):
    for index, instance in enumerate(trainSet):
      if (index < self.trainPositives) != (instance[-1] == 1.0):
        return False
    
    return True

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
    @type trainSet: list of tuples, one tuple per training instance, last
                    tuple element is the class (0=negative 1=positive)
    @param popSize: number of networks which will be evaluated in each run
    @type popSize: int
    """
    self.trainSize = len(trainSet)
    self.popSize = popSize
    log.info("training set size: %d", self.trainSize)
    log.info("population size: %d", self.popSize)

    self.trainPositives = sum(i[-1] == 1.0 for i in trainSet)

    # Ensure training set has all positive instances first.
    if not self._trainSetInClassOrder(trainSet):
      raise ValueError("Training set must have all positive instances first")

    # Calculate block/grid size and prepare evaluate() kernel.
    # (avoids Function.__call__ overhead)
    maxBlockDimX = driver.Context.get_device().get_attribute(
      driver.device_attribute.MAX_BLOCK_DIM_X
    )
    self.evaluateBlockDim = (maxBlockDimX, 1, 1)
    log.info("evaluate kernel block dim: %r", self.evaluateBlockDim)
    
    blockDimX = self.evaluateBlockDim[0]
    self.evaluateGridDim = ((self.trainSize + blockDimX - 1) / blockDimX,
                            self.popSize)
    log.info("evaluate kernel grid dim: %r", self.evaluateGridDim)

    self.evaluateKernel = self.mod.get_function("evaluate")
    self.evaluateKernel.prepare(
      (np.intp, np.uint32, np.intp, np.uint32, np.intp),
      block=self.evaluateBlockDim
    )

    # Calculate block/grid size and prepare nlargest() kernel.
    self.nlargestBlockDim = (1, 1, 1)
    log.info("nlargest kernel block dim: %r", self.nlargestBlockDim)
    
    self.nlargestGridDim = (1, self.popSize)
    log.info("nlargest kernel grid dim: %r", self.nlargestGridDim)

    self.nlargestKernel = self.mod.get_function("nlargest")
    self.nlargestKernel.prepare(
      (np.intp, np.uint32, np.uint32, np.uint32, np.intp, np.intp),
      block=self.nlargestBlockDim
    )
    
    # Calculate block/grid size and prepare lift() kernel.
    self.countBlockDim = (maxBlockDimX, 1, 1)
    log.info("count kernel block dim: %r", self.countBlockDim)
    
    self.countGridDim = (1, self.popSize)
    log.info("count kernel grid dim: %r", self.countGridDim)
    
    self.countKernel = self.mod.get_function("count")
    self.countKernel.prepare(
      (np.intp, np.uint32, np.uint32, np.uint32, np.intp, np.intp),
      block=self.countBlockDim
    )

    # Heap size in each pass is limited by shared memory per multiprocessor.
    sharedBytesPerBlock = tools.DeviceData().shared_memory
    floatBytes = np.dtype(np.float32).itemsize
    log.info("max shared memory per block: %d bytes (%d floats)",
      sharedBytesPerBlock, sharedBytesPerBlock / floatBytes)
    
    self.maxHeapFloats = (sharedBytesPerBlock / floatBytes
      * 99/100 # max size allocations fail on a GTX 275
    )
    maxHeapBytes = self.maxHeapFloats * floatBytes

    log.info("using heap size: %d bytes (%d floats)",
      maxHeapBytes, self.maxHeapFloats)
    self.nlargestKernel.set_shared_size(maxHeapBytes)

    # Store training set in column-major order so that fetches for the same
    # input feature across instances occur at consecutive memory addresses.
    # (avoids "Strided Accesses", see CUDA Best Practices Guide section 3.2.1.4)
    # TODO: Align each feature on 128-byte boundary?
    trainSetMat = np.asmatrix(trainSet, np.float32)
    self.trainSet = driver.to_device(
      trainSetMat.reshape(tuple(reversed(trainSetMat.shape)), order="F")
    )

    # Pre-allocate various large arrays
    # TODO: mem_alloc_pitch?
    floatBytes = np.dtype(np.float32).itemsize
    self.params = driver.mem_alloc(
      self.popSize * ctypes.sizeof(Parameters) * floatBytes
    )
    self.outputs = driver.mem_alloc(self.popSize * self.trainSize * floatBytes)
    
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
    driver.memset_d8(self.outputs, 0, self.popSize * self.trainSize * 4)
    
    self.evaluateKernel.prepared_call(self.evaluateGridDim,
                                      self.trainSet,
                                      self.trainSize,
                                      self.params,
                                      self.popSize,
                                      self.outputs)

    driver.Context.synchronize()

    if returnOutputs:
      return driver.from_device(self.outputs,
                                shape=(self.popSize, self.trainSize),
                                dtype=np.float32)

  def nlargest(self, n):
    """Returns the per-individual threshold above which there are n outputs.
    
    @param n: number of outputs which should be above the threshold
    @type params: int

    @return list of thresholds, in order of individuals, which delimit the top
            n output values
    """
    log.info("enter nlargest with n=%d", n)

    # Find one more output so that we can use strictly-less-than when counting
    # and underestimate lift rather than overestimating it.
    n = n + 1

    passSizes = []
    while n > 0:
      nextSize = min(self.maxHeapFloats, n)
      passSizes.append(nextSize)
      n -= nextSize

    log.info("pass sizes: %r", passSizes)
    
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
                                        self.trainSize,
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

    thresholdsMat = driver.from_device_like(self.thresholds, thresholdsMat)
    return thresholdsMat

  def lift(self, n):
    """Returns (positive rate within n largest) / (overall positive rate) for
       each individual.
    
    @return list of counts, in order of individuals
    """
    self.countKernel.prepared_call(self.countGridDim,
                                   self.outputs,
                                   self.trainSize,
                                   self.trainPositives,
                                   self.popSize,
                                   self.thresholds,
                                   self.counts)
    
    driver.Context.synchronize()

    countsMat = driver.from_device(self.counts,
                                   shape=(self.popSize, self.countBlockDim[0]),
                                   dtype=np.uint32)
    #log.debug("counts %r: %s", countsMat.shape, str(countsMat))
    log.debug("count sum over threads: %s", str(countsMat.sum(axis=1)))
    
    nlargestPositiveRate = np.float32(countsMat.sum(axis=1)) / n
    log.debug("positive rate (n largest outputs): %s", str(nlargestPositiveRate))

    overallPositiveRate = float(self.trainPositives) / float(self.trainSize)
    log.debug("positive rate (overall): %.04f", overallPositiveRate)
    
    return nlargestPositiveRate / overallPositiveRate

def linterp(a, b, p):
  return a + (b - a) * p

if __name__ == "__main__":
  import input
  logging.basicConfig(level=logging.DEBUG)
  np.set_printoptions(precision=3, edgeitems=3, threshold=20)

  a = ANN()
  trainSet = list(input.Input("train3.tsv"))

  popSize = 450
  timefun(a.prepare, trainSet, popSize)

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

  n = len(trainSet) * 20/100
  thresholds = timefun(a.nlargest, n)

  lifts = timefun(a.lift, n)
  print lifts
