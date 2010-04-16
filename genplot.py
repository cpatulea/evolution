from pygooglechart import XYLineChart, Axis
import cgi
import numpy

minFitnessHistory = []
avgFitnessHistory = []
maxFitnessHistory = []
testFitnessHistory = []

def plot():
    pass

def addGeneration(fitnessList, testFitness, genNumber):
    fmin = min(fitnessList)
    avg = sum(fitnessList) / len(fitnessList)
    fmax = max(fitnessList)
    print "G%d: avg %.02f min %.02f max %.02f test %.02f" % (
        genNumber,
        avg,
        fmin,
        fmax,
        testFitness
    )

    minFitnessHistory.append(fmin)
    avgFitnessHistory.append(avg)
    maxFitnessHistory.append(fmax)
    testFitnessHistory.append(testFitness)
    
    #xMax = len(avgFitnessHistory) * 120/100
    xMax = 120
    
    chart = XYLineChart(640, 400, x_range=(0, xMax), y_range=(0, 6),
      legend=["min", "avg", "max", "test"],
      colours=["66CC00", "FF9933", "FF0000", "0000FF"])
    chart.set_axis_range(Axis.BOTTOM, 0, xMax)
    chart.set_axis_range(Axis.LEFT, 0, 6)
    chart.set_axis_labels(Axis.BOTTOM, ["Generation"])
    chart.set_axis_labels(Axis.LEFT, ["Lift"])
    chart.set_axis_positions(2, [50])
    chart.set_axis_positions(3, [50])
    chart.add_data(range(len(avgFitnessHistory)))
    chart.add_data(minFitnessHistory)
    chart.add_data(range(len(avgFitnessHistory)))
    chart.add_data(avgFitnessHistory)
    chart.add_data(range(len(avgFitnessHistory)))
    chart.add_data(maxFitnessHistory)
    chart.add_data(range(len(testFitnessHistory)))
    chart.add_data(testFitnessHistory)
    
    open("chart.html", "w").write("<img src=\"%s\"/>" % chart.get_url())
