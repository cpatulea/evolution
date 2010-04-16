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

class BlindSet(object):
  def __init__(self, instances):
    self.instances = instances
    self.size = len(self.instances)

  def allInstances(self):
    return self.instances

  def show(self):
    m = md5.new()
    for instance in self.allInstances():
      m.update(array.array("f", instance))
    fingerprint = m.hexdigest()

    print "total %d fingerprint %s" % (
      len(self.allInstances()),
      fingerprint
    )

class Input(object):
  def __init__(self, infile, rand, blindfile=None):
    reader = csv.reader(map(string.strip, open(infile)), delimiter="\t")
    self.data = list(reader)
    #self._bayes(self.data, 2)
    #self._bayes(self.data, 5)
    self.data = [map(float, row) for row in self.data]

    featureCount = len(self.data[0]) - 1

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
    self._remove(featureCount)
    
    # Pad to 19 features
    padding = [0.0] * (19 - len(self.data[0]))
    self.data = [row + padding for row in self.data]
    
    # Numerically, ANNs behave much more nicely with normalized data
    quantiles = self._estimateQuantiles(self.data)
    
    if blindfile is not None:
      blindreader = csv.reader(map(string.strip, open(blindfile)), delimiter="\t")
      blinddata = [map(float, row) for row in blindreader]
      if len(blinddata[0]) != featureCount:
        raise ValueError("Blind set must have same number of features as "
                         "training set (had %d, expected %d)" % (
                         len(blinddata[0]), featureCount))

      blinddata = [map(row.__getitem__, self._indexes) for row in blinddata]
      blinddata = [row + padding for row in blinddata]

      self._normalize(blinddata, quantiles)

      self.blindSet = BlindSet(blinddata)

    self._normalize(self.data, quantiles)

    dataSet = DataSet(self.data[:numPositives], self.data[numPositives:])
    self.trainSet, self.testSet = dataSet.split(70, rand)

  def _bayes(self, data, featureIndex):
    pos = neg = 0
    p = {}
    n = {}
    for row in data:
      v, Class = row[featureIndex], row[-1]

      if Class == "1":
        pos += 1
        if v not in p:
          p[v] = 0
        p[v] += 1
      else:
        neg += 1
        if v not in n:
          n[v] = 0
        n[v] += 1
    
    llrs = {} # "log likelihood ratios"
    import sys, math
    sys.stderr.write("pos: %d neg: %d ratio: %.03f\n" %
      (pos, neg, float(pos)/float(neg))
    )
    allvalues = set(p.keys() + n.keys())
    priorp = 1.0 / len(allvalues)
    smoothing = 1.0
    for key in sorted(allvalues, key=lambda k: (len(k), k)):
      keypos, keyneg = p.get(key, 0), n.get(key, 0)
      if keyneg == 0:
        pnr = float("inf")
      else:
        pnr = float(keypos) / float(keyneg)
      sys.stderr.write("  %s %d+ %d- pnr: %.03f" % (key, keypos, keyneg, pnr))

      # Laplace smoothing
      
      ppos = (float(keypos) + smoothing * priorp) / (float(pos) + smoothing)
      pneg = (float(keyneg) + smoothing * priorp) / (float(neg) + smoothing)

      # "likelihood ratio"
      lr = ppos / pneg
      llr = math.log(lr)
      llrs[key] = llr
      sys.stderr.write(" lr: %.03f llr: %.03f\n" % (lr, llr))

    for row in data:
      row[featureIndex] = llrs[row[featureIndex]]

  def _estimateQuantiles(self, data):
    """Estimates 10 and 90% quantiles of each feature in data.
    
    @return [(q10, q90)]: list of tuples with quantile 10 and 90, one per
                          feature
    """
    quantiles = []
    
    for featureIndex, _ in enumerate(data[0]):
      featureData = [0] * len(data)
      for index, row in enumerate(data):
        featureData[index] = row[featureIndex]
      
      featureData.sort()
      
      import numpy
      q10 = featureData[len(featureData) * 10 / 100]
      q90 = featureData[len(featureData) * 90 / 100]
      """
      mean, stddev = (numpy.mean(featureData, dtype=numpy.float64),
                      numpy.std(featureData, dtype=numpy.float64))
      q10 = mean - stddev
      q90 = mean + stddev
      """

      quantiles.append((q10, q90))

    return quantiles

  def _normalize(self, data, quantiles):
    """Normalize data by shifting and scaling according to 10/90% quantiles.
    
    Mean of each feature is estimated using (q10 + q90) / 2.
    
    Features are scaled by 2 / (q90 - q10) such that 80% of values are within
    -1 to 1.
    
    Data is normalized in-place.
    """
    shiftscale = []
    for q10, q90 in quantiles:
      shift = (q10 + q90) / 2

      if q10 == q90:
        scale = 1.0
      else:
        scale = 2.0 / (q90 - q10)
      
      shiftscale.append((shift, scale))

    for row in data:
      for featureIndex, _ in enumerate(row):
        shift, scale = shiftscale[featureIndex]
        row[featureIndex] = (row[featureIndex] - shift) * scale

    return data

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
  inp = Input("train3.tsv", randSample, blindfile="test5.tsv")
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

  print "Blind:"
  inp.blindSet.show()
