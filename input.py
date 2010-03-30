#!/usr/bin/python
import csv
import string
import random
import array
import md5

SAMPLE_SEED = 7713

class DataSet(object):
  def __init__(self, positives, negatives):
    self.positives = positives
    self.negatives = negatives
    self.size = len(self.allInstances())

  def allInstances(self):
    """Returns a list of the instances in the data set, positives first.
    
    @return list: instances in the data set, positives first
    """
    return self.positives + self.negatives

  def split(self, percent, rand):
    """Sample without replacement; return sample and remaining data sets.
    
    @param percent: size of the sample, in percent of the initial data set
    @type percent: int, from 0 to 100
    @param rand: source of randomness
    @type rand: random.Random
    @return (DataSet, DataSet): (sampleSet, restSet)
    """
    def _sample(instances, percent):
      size = len(instances) * percent / 100
      indexes = set(rand.sample(xrange(len(instances)), size))
      sample = []
      rest = []
      for index, inst in enumerate(instances):
        if index in indexes:
          sample.append(inst)
        else:
          rest.append(inst)
      return sample, rest

    samplePos, restPos = _sample(self.positives, percent)
    sampleNeg, restNeg = _sample(self.negatives, percent)
    
    return DataSet(samplePos, sampleNeg), DataSet(restPos, restNeg)

  def sample(self, percent, rand):
    """Sample without replacement; return only the sample data set.
    
    @param percent: size of the sample, in percent of the initial data set
    @type percent: int, from 0 to 100
    @param rand: source of randomness
    @type rand: random.Random
    @return DataSet: the sample data set
    """
    sample, _ = self.split(percent, rand)
    return sample

  def show(self, verbose=False):
    """Prints a human-friendly overview of the data set on stdout."""
    m = md5.new()
    for instance in self.allInstances():
      m.update(array.array("f", instance))
    fingerprint = m.hexdigest()

    print "%d+ %d- total %d fingerprint %s" % (
      len(self.positives),
      len(self.negatives),
      len(self.allInstances()),
      fingerprint
    )

    if verbose:
      def _showInsts(insts):
        for inst in insts:
          print "[" + ", ".join("%5.02f" % value for value in inst) + "]"

      print "  First 3 positives:"
      _showInsts(self.positives[:3])

      print "  Last 3 positives:"
      _showInsts(self.positives[-3:])

      print "  First 3 negatives:"
      _showInsts(self.negatives[:3])

      print "  Last 3 negatives:"
      _showInsts(self.negatives[-3:])

class Input(object):
  def __init__(self, infile, rand):
    reader = csv.reader(map(string.strip, open(infile)), delimiter="\t")
    self.data = [map(float, row) for row in reader]
    featureCount = len(self.data[0])

    self._initIndexes()

    #self._remove(2)
    #self._remove(5)
    self._remove(7) # dupe of 1
    self._remove(9) # dupe of 0

    numPositives = sum(instance[-1] == 1.0 for instance in self.data)
    
    # Instances are in class order if the first numPositives instances account
    # for all positive instances.
    if numPositives != sum(instance[-1] == 1.0 for instance
                           in self.data[:numPositives]):
      raise ValueError("Data set must have positive instances first")

    # Remove class
    self._remove(featureCount - 1)
    
    # Pad to 19 features
    padding = [0.0] * (19 - len(self.data[0]))
    self.data = [row + padding for row in self.data]
    
    dataSet = DataSet(self.data[:numPositives], self.data[numPositives:])
    self.trainSet, self.testSet = dataSet.split(70, rand)

  def _initIndexes(self):
    # keeps indexes consistent across removal of features
    self._indexes = range(len(self.data[0]))

  def _realIndex(self, featureIndex):
    try:
      return self._indexes.index(featureIndex)
    except ValueError:
      raise ValueError("Feature %d not present (already removed?)" % featureIndex)

  def _remove(self, index):
    realIndex = self._realIndex(index)
    
    for row in self.data:
      del row[realIndex]
    
    del self._indexes[realIndex]

if __name__ == "__main__":
  randSample = random.Random(SAMPLE_SEED)
  inp = Input("train3.tsv", randSample)
  print "Train:",
  inp.trainSet.show()
  print "Test:",
  inp.testSet.show()
  
  sampleSet = inp.testSet.sample(50, randSample)
  print "Sample 0:",
  sampleSet.show()
  
  sampleSet = inp.testSet.sample(50, randSample)
  print "Sample 1:",
  sampleSet.show()
