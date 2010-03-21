#!/usr/bin/python
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
import numpy as np

class ANN(object):
  mod = compiler.SourceModule(open("ann_kernels.cu").read())
  NUM_WEIGHTS = 4

  def prepare(self, trainSet, popSize):
    """Prepare for many parallel ANN fitness calculations.
    
                      len(trainSet[0]) x len(trainSet)
                  <--     training instances    -->
                  +-------------------------------+
     network    w1|block1                         |
     weights    w2|block2                         |
    (popSize x  w3|block3                         |
    NUM_WEIGHTS)  +-------------------------------+
     
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
    self.weights = driver.mem_alloc(self.popSize * self.NUM_WEIGHTS * floatBytes)
    self.outputs = driver.mem_alloc(self.popSize * self.trainSize * floatBytes)

  def evaluate(self, weights, returnOutputs=False):
    """Evaluate several networks (with given weights) on training set.
    
    @param weights: network weights
    @type weights: list of tuples, one tuple per network
    @param returnOutputs: return network output values (debug)
    @type returnOutputs: bool, default False
    
    @return output matrix if returnOutputs=True, else None
    """
    weightsMat = np.asmatrix(weights, np.float32)
    weightsShape = (self.popSize, self.NUM_WEIGHTS)
    if weightsMat.shape != weightsShape:
      raise ValueError("Weights shape should be %r (was %r)" % (
        weightsShape, weightsMat.shape))
    
    driver.memcpy_htod(self.weights, weightsMat)

    # TODO: remove
    driver.memset_d8(self.outputs, 0xcc, self.popSize * self.trainSize * 4)
    
    self.evaluateKernel.prepared_call((self.popSize, 1),
                                      self.trainSet,
                                      self.trainSize,
                                      self.weights,
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

  weights = [(1, 2, 3, 4)] * 10
  outputs = a.evaluate(weights, returnOutputs=True)
  print outputs
