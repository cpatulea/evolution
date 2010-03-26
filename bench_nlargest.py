#!/usr/bin/python
"""Compares performance of nlargest on GPU vs CPU.

Outputs a txt file with the performance numbers:

bench_nlargest.txt
                   increasing training set size ----->
increasing  +----------------------------------------------+
population  |                                              |
   size     |                                              |
    |       |             ... times ...                    |
    |       |                                              |
    |       |                                              |
    v       +----------------------------------------------+

  x 4: GPU average, GPU stddev, CPU average, CPU stddev
"""
import input
import ann
from bench import timefun
from pycuda import driver
import heapq
import logging
import time
import numpy as np

log = logging.getLogger("bench_nlargest")

def generateParams(w0):
  p = ann.Parameters()
  for i in range(19):
    p.ih[0][i] = 1.0
    p.ih[1][i] = p.ih[2][i] = p.ih[3][i] = 0.0

    p.c[0][i] = 0.0
    p.c[1][i] = p.c[2][i] = p.c[3][i] = 0.0

  p.w[0] = w0
  p.w[1] = p.w[2] = p.w[3] = 0.0
  
  p.ho[0] = 1.0
  p.ho[1] = p.ho[2] = p.ho[3] = 0.0
  
  return p

def linterp(a, b, p):
  return a + (b - a) * p

# from http://docs.python.org/dev/_sources/library/itertools.txt
def product(*args, **kwds):
  # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
  # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
  pools = map(tuple, args) * kwds.get('repeat', 1)
  result = [[]]
  for pool in pools:
    result = [x+[y] for x in result for y in pool]
  for prod in result:
    yield tuple(prod)

def nlargest_cpu(ann, n):
  """CPU implementation of nlargest."""
  outputs = driver.from_device(ann.outputs,
                               shape=(ann.popSize, ann.trainSize),
                               dtype=np.float32)

  thresholds = []
  for row in outputs:
    sortedRow = sorted(row, reverse=True)
    thresholds.append(sortedRow[n])

  return thresholds

def main():
  import input
  logging.basicConfig(level=logging.DEBUG)
  np.set_printoptions(precision=3, edgeitems=3, threshold=10)

  # don't clutter the screen with details
  logging.getLogger("ann").setLevel(logging.ERROR)

  fullTrainSet = list(input.Input("train.tsv"))

  popSizes = [10, 100, 1000]
  trainSizes = [90, 900, 9000, 90000]
  benchParams = list(product(enumerate(popSizes), enumerate(trainSizes)))
  
  benchGpuAvg = np.zeros((len(popSizes), len(trainSizes)), np.float32)
  benchGpuStd = np.zeros_like(benchGpuAvg)
  for index, ((popIndex, popSize), (trainIndex, trainSize)) in enumerate(benchParams):
    log.info("benchmarking %.01f%% popSize %d trainSize %d",
      100.0 * index / len(benchParams), popSize, trainSize)
    a = ann.ANN()

    trainSet = fullTrainSet[:trainSize]
    assert len(trainSet) == trainSize

    a.prepare(trainSet, popSize)
    
    params = [generateParams(linterp(1e-4, 1e-5, float(index) / popSize))
              for index in range(popSize)]
    a.evaluate(params)
    
    n = len(trainSet) * 20/100
    avg, std = stats = timefun(a.nlargest, n, repeat=10, stats=True)
    benchGpuAvg[popIndex, trainIndex] = avg
    benchGpuStd[popIndex, trainIndex] = std
    
    outputs = a.evaluate(params, returnOutputs=True)
    stats = timefun(nlargest_cpu, a, n, repeat=10, stats=True)

if __name__ == "__main__":
  main()
