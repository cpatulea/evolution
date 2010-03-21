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
      (np.intp, np.uint32, np.intp, np.uint32, np.uint32, np.intp),
      block=(popSize, 1, 1)
    )

    # Training set does not change across network evaluations
    # TODO: Align each instance on 128-byte boundary?? (ndarray.strides or
    # driver.mem_alloc_pitch)
    self.trainSize = len(trainSet)
    self.trainSet = driver.to_device(np.asmatrix(trainSet, np.float32))

    # Pre-allocate various arrays
    # TODO: mem_alloc_pitch?
    self.popSize = popSize
    
    floatBytes = np.dtype(np.float32).itemsize
    self.weights = driver.mem_alloc(self.popSize * self.NUM_WEIGHTS * floatBytes)
    self.fitness = driver.mem_alloc(self.popSize * self.trainSize * floatBytes)

  def evaluate(self, weights, returnFitness=False):
    """Evaluate several networks (with given weights) on training set.
    
    @param weights: network weights
    @type weights: list of tuples, one tuple per network
    @param returnFitness: return fitness values (debug)
    @type returnFitness: bool, default False
    
    @return fitness matrix if returnFitness=True, else None
    """
    weightsMat = np.asmatrix(weights, np.float32)
    weightsShape = (self.popSize, self.NUM_WEIGHTS)
    if weightsMat.shape != weightsShape:
      raise ValueError("Weights shape should be %r (was %r)" % (
        weightsShape, weightsMat.shape))
    
    driver.memcpy_htod(self.weights, weightsMat)

    # TODO: remove
    driver.memset_d8(self.fitness, 0xcc, self.popSize * self.trainSize * 4)
    
    self.evaluateKernel.prepared_call((self.popSize, 1),
                                      self.trainSet,
                                      self.trainSize,
                                      self.weights,
                                      self.NUM_WEIGHTS,
                                      self.popSize,
                                      self.fitness)
    
    if returnFitness:
      return driver.from_device(self.fitness,
                                shape=(self.popSize, self.trainSize),
                                dtype=np.float32)

if __name__ == "__main__":
  a = ANN()
  trainSet = [
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 1)
  ]
  popSize = 10
  a.prepare(trainSet, popSize)

  weights = [(1, 2, 3, 4)] * 10
  fitness = a.evaluate(weights, returnFitness=True)
  print fitness
