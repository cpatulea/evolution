#!/usr/bin/python
import csv

class Input(object):
  def __init__(self, infile):
    reader = csv.reader(open(infile), delimiter="\t")
    self.data = list(reader)
    
    self._initIndexes()

    self.remove(8 - 1) # dupe of 2
    self.remove(10 - 1) # dupe of 1

    self.nominalToNumeric(3 - 1)
    self.nominalToNumeric(6 - 1)
    self.nominalToNumeric(19 - 1, key=int)
    
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

if __name__ == "__main__":
  data = list(Input("train.tsv"))
  print len(data[0]), data[0]
