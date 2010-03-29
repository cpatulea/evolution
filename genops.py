#!/usr/bin/python
import random, math, copy, logging
import numpy as np
import md5
import input
import genplot
from ann import ANN, Parameters

log = logging.getLogger("genops")

POPSIZE = 50

"""
Population generation / initializer.  Uses method from Montana and Davis.
@param popAmt: Number of individuals in the population
@type popAmt: integer
@return a list of Parameters objects, representing the population
"""
def generatePop(generation):
    for i in range(POPSIZE):
        nextMember = Parameters()
        for j in range(4):
            for k in range(19):
                nextMember.ih[j][k] = getInitialFloat()
                nextMember.c[j][k] = getInitialFloat()
            nextMember.w[j] = getInitialFloat()
            nextMember.ho[j] = getInitialFloat()
        generation.append(nextMember)

    #return generation

"""
Create a new generation, based on the current generation
@param oldGen: The generation that will be mated and mutated
@type oldGen: A list of Parameters operations
@param fitList: A list of fitnesses of the generation
@type fitList: A list of float values
@return: List of new parameters (IE, a new generation)
"""
def generateGeneration(oldGen):
    newGen = []
    newGen.append(oldGen[0])
    for i in range(POPSIZE - 1):
        if random.choice([0,1]) == 1:
            index = getIndex()
            newGen.append(
                mutate(oldGen[index])
            )
        else:
            index = getIndex()
            parent1 = oldGen[index]
            index = getIndex()
            parent2 = oldGen[index]
            while parent2 == parent1:
                index = getIndex()
                parent2 = oldGen[index]
            newGen.append(mate(parent1, parent2))

    return newGen
        
"""
Mutation operator.  Uses MUTATE NODES operator from Montana and Davis.
@param xman: The parent who will be mutated
@type xman: Parameters (see class Parameters, above)
@return Mutated Parameters object
"""
def mutate(xman):  
    xmanJr = Parameters()

    for i in range(4):
        for j in range(19):
            xmanJr.ih[i][j] = xman.ih[i][j]
            xmanJr.c[i][j] = xman.c[i][j]
        xmanJr.w[i] = xman.w[i]
        xmanJr.ho[i] = xman.ho[i]

    node = random.randint(0,3)
    for i in range(19):
        xmanJr.ih[node][i] += getMutationValue()
        xmanJr.c[node][i] += getMutationValue()
    xmanJr.w[node] += getMutationValue()
    xmanJr.ho[node] += getMutationValue()
    
    return xmanJr

"""
Mate operator.  Uses CROSSOVER WEIGHTS operator from Montana and Davis.
@param parent1: One of the parents who will be mated
@type parent1: Parameters (see class Parameters, above)
@param parent2: One of the parents who will be mated
@type parent2: Parameters (see class Parameters, above)
@return Child Parameters object
"""
def mate(parent1, parent2):
    parentList = [parent1, parent2]
    child = Parameters()

    for i in range(4):
        for j in range(19):
            child.ih[i][j] = parentList[random.randint(0,1)].ih[i][j]
            child.c[i][j] = parentList[random.randint(0,1)].c[i][j]
        child.w[i] = parentList[random.randint(0,1)].w[i]
        child.ho[i] = parentList[random.randint(0,1)].ho[i]

    return child

"""
Function for determining initial parameter values
@param none
@return a random value from (currently) an exponential distribution
"""
def getInitialFloat():
    return random.expovariate(2)*random.choice([-1,1])

def getMutationValue():
    return random.expovariate(12)*random.choice([-1,1])
    #return random.normalvariate(0.0, 7.0)

INDEX_LAMBDA = -math.log(0.92)

def getIndex():
    index = int(random.expovariate(INDEX_LAMBDA))
    while index >= POPSIZE:
        index = int(random.expovariate(INDEX_LAMBDA))
    return index

def logFP(label, buf):
  """Logs a fingerprint of important data in each generation.
  
  Used to be sure that two runs are exactly identical.
  """
  m = md5.new()

  if isinstance(buf, list):
    for el in buf:
      m.update(buffer(el))
  else:
    m.update(buffer(buf))
  
  log.debug("%s FP: %s", label, m.hexdigest())

def main():
  import input
  logging.basicConfig(level=logging.INFO)
  np.set_printoptions(precision=3, edgeitems=3, threshold=20)

  random.seed(1005108)

  a = ANN()
  inp = input.Input("train3.tsv")
  inp.remove(7)
  inp.remove(9)
  trainSet = list(inp)
  
  trainPositives = sum(i[-1] == 1.0 for i in trainSet)
  trainSet = [instance[:-1] + [0, 0] for instance in inp]

  n = len(trainSet) * 20/100

  a.prepare(trainSet, trainPositives, POPSIZE)

  params = []
  generatePop(params)

  for genIndex in range(5):
    print "Generation", genIndex, "starting."
    logFP("Population", params)
    outputValues = a.evaluate(params, returnOutputs=True)
    
    logFP("Outputs", outputValues)
    
    thresholds = a.nlargest(n)
    logFP("Thresholds", thresholds)

    lifts = a.lift(n)
    logFP("Lifts", lifts)

    genplot.addGeneration(lifts, genIndex)

    taggedParams = sorted(zip(lifts, params, range(len(params))),
                          key=lambda (l, p, i): l,
                          reverse=True)
    sortedParams = [p for l, p, i in taggedParams]
    logFP("Sorted pop", sortedParams)

    params = generateGeneration(sortedParams)
    print "Generation", genIndex, "complete."
    
  genplot.plot()

if __name__ == "__main__":
  main()
