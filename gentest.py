#!/usr/bin/python
import genops
import genplot

def fitnessTest(generation):
    fitList = []
    for i in generation:
        fitList.append(i.ih[3][2] + i.c[1][15] - i.w[0] - i.ho[3])

    zippedGen = zip(fitList, generation)
    zippedGen.sort()
    zippedGen.reverse()
    return zippedGen

currGen = genops.generatePop(500)
currFit = []
zippedGen = fitnessTest(currGen)
currFit, currGen = zip(*zippedGen)

genplot.addGeneration(currFit, 0)
for i in range(1,100):
    print "Generation", i
    nextGen = genops.generateGeneration(currGen)
    zippedGen = fitnessTest(nextGen)
    currFit, currGen = zip(*zippedGen)
    genplot.addGeneration(currFit, i)

genplot.plot()
