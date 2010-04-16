import matplotlib.pyplot as plt
import numpy

def plot():
    plt.show()

def addGeneration(fitnessList, testFit, genNumber):
    plt.scatter(genNumber, max(fitnessList))
    plt.scatter(genNumber, testFit, marker="x")
    plt.scatter(genNumber, min(fitnessList))
