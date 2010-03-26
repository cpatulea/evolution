#!/usr/bin/python
import logging
import time
import math

log = logging.getLogger("bench")

def timefun(fun, *args, **kwargs):
  repeat = kwargs.get("repeat", 1)
  stats = kwargs.get("stats", False)

  deltas = []
  for _ in range(repeat):
    start = time.clock()
    ret = fun(*args)
    end = time.clock()
    deltas.append(end - start)
  
  avg = sum(deltas)/len(deltas)
  std = math.sqrt(sum(d*d for d in deltas)/len(deltas) - avg*avg)
  log.info("%s: avg %.01f, std %.01f ms", fun.func_name,
    avg*1000.0, std*1000.0)

  if stats:
    return avg, std
  elif repeat == 1:
    return ret
