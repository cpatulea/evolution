#!/usr/bin/python
import csv
import string

class Input(object):
  def __init__(self, infile):
    reader = csv.reader(map(string.strip, open(infile)), delimiter="\t")
    self.data = list(reader)
    
    self._initIndexes()

    #self.remove(2)
    #self.remove(5)
    self.remove(7)
    self.remove(9)

    #self.remove(6) # dupe of 1
    #self.remove(8) # dupe of 0

    #self.nominalToNumeric(3 - 1)
    #self.nominalToNumeric(6 - 1)
    #self.nominalToNumeric(19 - 1, key=int)
    
    self.convertAll(float)

  def _initIndexes(self):
    # keeps indexes consistent across removal of features
    self._indexes = range(len(self.data[0]))

  def _realIndex(self, featureIndex):
    realIndex = self._indexes.index(featureIndex)

    if realIndex == -1:
      raise ValueError("Feature %d not present (removed?)", index)
    
    return realIndex

  def nominalToNumeric(self, index, key=str):
    realIndex = self._realIndex(index)

    values = set(row[realIndex] for row in self.data)
    
    values = dict((v, i) for i, v in enumerate(sorted(values, key=key)))
    
    for row in self.data:
      row[realIndex] = values[row[realIndex]]

  def remove(self, index):
    realIndex = self._realIndex(index)
    
    for row in self.data:
      del row[realIndex]
    
    del self._indexes[realIndex]

  def convertAll(self, converter):
    for row in self.data:
      row[:] = map(converter, row)

  def __iter__(self):
    return iter(self.data)

def traintest(dataSet, testPercent):
  # dataSet:
  # [+++++++++++----------------------------]
  # ^^^^^^^^vvvv^^^^^^^^^^^^^^^^^^^^^vvvvvvv
  # train   test train               test
  # 70%     30%  70%                 30%
  dataPos = sum(i[-1] == 1.0 for i in dataSet)
  dataNeg = len(dataSet) - dataPos

  # Remove class, pad to 19 features.
  padding = [0.0] * (19 - len(dataSet[0]))
  dataSet = [inst[:-1] + [0.0] + padding for inst in dataSet]
  
  testPos = dataPos * testPercent / 100
  testNeg = dataNeg * testPercent / 100
  assert testPos + testNeg == testPercent * len(dataSet) / 100

  trainPercent = 100 - testPercent
  trainPos = dataPos - testPos
  trainNeg = dataNeg - testNeg
  
  #trainSize = trainPos + trainNeg
  #assert trainPos + trainNeg == trainPercent * len(dataSet) / 100
  
  trainSet = dataSet[:trainPos] + dataSet[dataPos:dataPos + trainNeg]
  #assert len(trainSet) == trainSize
  
  # testSet:
  # [++++++++++--------------------------]
  #           ^ testPos
  testSet = dataSet[trainPos - testPos:trainPos] + dataSet[-testNeg:]
  #assert len(testSet) == testSize

  print "Split: train %d (%d+) test %d (%d+)" % (
    len(trainSet), trainPos, len(testSet), testPos)

  return (trainSet, trainPos), (testSet, testPos)

if __name__ == "__main__":
  data = list(Input("train.tsv"))
  print len(data[0]), data[0]
