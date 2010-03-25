import matplotlib.pyplot as plt
import numpy

def plot():
    plt.show()

def addGeneration(fitnessList, genNumber):
    plt.errorbar(genNumber, numpy.average(fitnessList), numpy.std(fitnessList))
    plt.scatter(genNumber, max(fitnessList))
    plt.scatter(genNumber, min(fitnessList))
