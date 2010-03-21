#!/usr/bin/python
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
import numpy as np
import ctypes

class Parameters(ctypes.Structure):
  _fields_ = [("ih", ctypes.c_float * 4 * 19),
              ("c", ctypes.c_float * 4 * 19),
              ("w", ctypes.c_float * 4),
              ("ho", ctypes.c_float * 4)]

class ANN(object):
  mod = compiler.SourceModule(open("ann_kernels.cu").read())

  def prepare(self, trainSet, popSize):
    """Prepare for many parallel ANN fitness calculations.
    
                         len(trainSet[0]) x len(trainSet)
                       <--     training instances    -->
                       +-------------------------------+
       network       p1|block1                         |
       params        p2|block2                         |
      (popSize x     p3|block3                         |
    sizeof(Params))    +-------------------------------+
     
    @param trainSet: training set
    @type trainSet: list of tuples, one tuple per training instance
    @param popSize: number of networks which will be evaluated in each run
    @type popSize: int
    """
    # Avoid Function.__call__ overhead
    self.evaluateKernel = self.mod.get_function("evaluate")
    self.evaluateKernel.prepare(
      (np.intp, np.uint32, np.intp, np.uint32, np.intp),
      block=(popSize, 1, 1)
    )

    # Store training set in column-major order so that fetches for the same
    # input feature across instances occur at consecutive memory addresses.
    # (avoids "Strided Accesses", see CUDA Best Practices Guide section 3.2.1.4)
    # TODO: Align each feature on 128-byte boundary?
    self.trainSize = len(trainSet)
    trainSetMat = np.asmatrix(trainSet, np.float32)
    self.trainSet = driver.to_device(
      trainSetMat.reshape(tuple(reversed(trainSetMat.shape)), order="F")
    )

    # Pre-allocate various arrays
    # TODO: mem_alloc_pitch?
    self.popSize = popSize
    
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
    driver.memset_d8(self.outputs, 0xcc, self.popSize * self.trainSize * 4)
    
    self.evaluateKernel.prepared_call((self.popSize, 1),
                                      self.trainSet,
                                      self.trainSize,
                                      self.params,
                                      self.popSize,
                                      self.outputs)
    
    if returnOutputs:
      return driver.from_device(self.outputs,
                                shape=(self.popSize, self.trainSize),
                                dtype=np.float32)

if __name__ == "__main__":
  a = ANN()
  trainSet = [
    (1, 0, 0, 0, 0),
    (0, 2, 0, 0, 0),
    (0, 0, 3, 0, 0),
    (0, 0, 0, 4, 5)
  ]

  popSize = 10
  a.prepare(trainSet, popSize)

  p = Parameters()
  p.ho = (ctypes.c_float * 4)(*([1] * 4))
  params = [p] * 10
  outputs = a.evaluate(params, returnOutputs=True)
  print outputs
