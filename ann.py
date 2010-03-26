#!/usr/bin/python
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
from pycuda import tools
import numpy as np
import ctypes
import time
import logging

log = logging.getLogger("ann")

class Parameters(ctypes.Structure):
  _fields_ = [("ih", 4 * (19 * ctypes.c_float)),
              ("c", 4 * (19 * ctypes.c_float)),
              ("w", 4 * ctypes.c_float),
              ("ho", 4 * ctypes.c_float)]

class ANN(object):
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
    @type trainSet: list of tuples, one tuple per training instance
    @param popSize: number of networks which will be evaluated in each run
    @type popSize: int
    """
    self.trainSize = len(trainSet)
    self.popSize = popSize
    log.info("training set size: %d", self.trainSize)
    log.info("population size: %d", self.popSize)

    maxBlockDimX = driver.Context.get_device().get_attribute(
      driver.device_attribute.MAX_BLOCK_DIM_X
    )
    self.evaluateBlockDim = (maxBlockDimX, 1, 1)
    log.info("evaluate kernel block dim: %r", self.evaluateBlockDim)
    
    blockDimX = self.evaluateBlockDim[0]
    self.evaluateGridDim = ((self.trainSize + blockDimX - 1) / blockDimX,
                            self.popSize)
    log.info("evaluate kernel grid dim: %r", self.evaluateGridDim)

    # Avoid Function.__call__ overhead
    self.evaluateKernel = self.mod.get_function("evaluate")
    self.evaluateKernel.prepare(
      (np.intp, np.uint32, np.intp, np.uint32, np.intp),
      block=self.evaluateBlockDim
    )
    
    # Store training set in column-major order so that fetches for the same
    # input feature across instances occur at consecutive memory addresses.
    # (avoids "Strided Accesses", see CUDA Best Practices Guide section 3.2.1.4)
    # TODO: Align each feature on 128-byte boundary?
    trainSetMat = np.asmatrix(trainSet, np.float32)
    self.trainSet = driver.to_device(
      trainSetMat.reshape(tuple(reversed(trainSetMat.shape)), order="F")
    )

    # Pre-allocate various arrays
    # TODO: mem_alloc_pitch?
    floatBytes = np.dtype(np.float32).itemsize
    self.params = driver.mem_alloc(
      self.popSize * ctypes.sizeof(Parameters) * floatBytes
    )
    self.outputs = driver.mem_alloc(self.popSize * self.trainSize * floatBytes)

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
    
    if returnOutputs:
      return driver.from_device(self.outputs,
                                shape=(self.popSize, self.trainSize),
                                dtype=np.float32)
    else: # wait for kernel to finish executing
      driver.Context.synchronize()

def timefun(fun, *args, **kwargs):
  repeat = kwargs["repeat"]

  deltas = []
  for _ in range(repeat):
    start = time.clock()
    ret = fun(*args)
    end = time.clock()
    deltas.append(end - start)
  
  log.info("%s: min %.01f, avg %.01f, max %.01f ms", fun.func_name,
    min(deltas)*1000.0, sum(deltas)/len(deltas)*1000.0, max(deltas)*1000.0
  )
  
  if repeat == 1:
    return ret

if __name__ == "__main__":
  import input
  logging.basicConfig(level=logging.DEBUG)
  np.set_printoptions(precision=3, edgeitems=3, threshold=8)

  a = ANN()
  trainSet = list(input.Input("train.tsv"))

  popSize = 500
  timefun(a.prepare, trainSet, popSize, repeat=1)

  p = Parameters()
  for i in range(19):
    p.ih[0][i] = 1.0
    p.ih[1][i] = p.ih[2][i] = p.ih[3][i] = 0.0

    p.c[0][i] = 0.0
    p.c[1][i] = p.c[2][i] = p.c[3][i] = 0.0

  p.w[0] = 1e-4
  p.w[1] = p.w[2] = p.w[3] = 0.0
  
  p.ho[0] = 1.0
  p.ho[1] = p.ho[2] = p.ho[3] = 0.0

  params = [p] * popSize
  outputValues = timefun(a.evaluate, params, True, repeat=1)
  print outputValues
