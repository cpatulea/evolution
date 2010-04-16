#!/usr/bin/python -u
"""Benchmark ANN evaluate, nlargest and lift calculation.

This program takes command-line arguments to control the benchmarking.
"""
import optparse
import ann_cpu
import ann
import bench
import input
import itertools
import sys
import random

def initializeData():
  global INPUT_SET, ANN_PARAM
  rand = random.Random(input.SAMPLE_SEED)
  inp = input.Input("train3.tsv", rand)
  INPUT_SET = inp.trainSet
  
  # This set of parameters gives a good variation of network output values for
  # normalized train3.tsv, so it exercises nlargest with representative data.
  p = ann.Parameters()
  for i in range(19):
    p.ih[0][i] = p.ih[1][i] = p.ih[2][i] = p.ih[3][i] = 0.0
    p.c[0][i] = p.c[1][i] = p.c[2][i] = p.c[3][i] = 0.0

  p.ih[0][0] = 1.0

  p.w[0] = -1.0
  p.w[1] = p.w[2] = p.w[3] = 0.0

  p.ho[0] = 1.0
  p.ho[1] = p.ho[2] = p.ho[3] = 0.0
  
  ANN_PARAM = p

def generatePopulation(size):
  return [ANN_PARAM] * size

def _repeatInstances(instances, size):
  repeated = instances * ((size + len(instances) - 1) / len(instances))
  return repeated[0:size]
 
def generateTrainSet(size):
  numPositives = len(INPUT_SET.positives) * size / INPUT_SET.size
  return input.DataSet(
    _repeatInstances(INPUT_SET.positives, numPositives),
    _repeatInstances(INPUT_SET.negatives, size - numPositives)
  )

def main(popSizes, trainSizes):
  initializeData()
  
  sys.stdout.write("%12s%12s%12s%12s\n" % (
    "PopSize", "TrainSize", "CPU (ms)", "GPU (ms)"
  ))
  for trainSize in trainSizes:
    # generateTrainSet is more expensive than generatePopulation
    trainSet = generateTrainSet(trainSize)
    for popSize in popSizes:
      population = generatePopulation(popSize)
      n = trainSet.size * 20 / 100
      
      sys.stdout.write("%12d%12d" % (popSize, trainSize))
      
      def benchImpl(impl):
        a = impl.ANN()
        a.prepare(trainSet, len(population))

        def benchmarkee():
          a.evaluate(population, returnOutputs=False)
          a.nlargest(n)
          a.lift(n)

        avg, std = bench.timefun(benchmarkee, repeat=10, stats=True)
        sys.stdout.write("%12.01f" % (avg * 1000.0))

      benchImpl(ann_cpu)
      benchImpl(ann)
      
      sys.stdout.write("\n")

if __name__ == "__main__":
  #main([1, 10, 100, 1000, 10000], [1000])
  main([100], [100, 1000, 10000, 100000])

