#!/usr/bin/python
import sys
import input
import ann
import random

def main(annfile):
  randSample = random.Random(input.SAMPLE_SEED)
  inp = input.Input("train5.tsv", rand=randSample, blindfile="test5.tsv")
  
  param = ann.Parameters.from_file(annfile)

  a = ann.ANN()
  a.prepare(inp.blindSet, popSize=1)
  outputs = a.evaluate([param], returnOutputs=True)[0]
  
  sortedOutputs = sorted(enumerate(outputs),
                         key=lambda (index, output): output,
                         reverse=True)
  indexToRank = {}
  for rank0, (index, output) in enumerate(sortedOutputs):
    indexToRank[index] = rank0 + 1

  for index, output in enumerate(outputs):
    rank = indexToRank[index]
    print index, output, rank

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Usage: blind.py <file.ann>"
    sys.exit(1)

  main(sys.argv[1])
