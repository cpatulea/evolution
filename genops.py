import ctypes, random

class Parameters(ctypes.Structure):
  _fields_ = [("ih", 4 * (19 * ctypes.c_float)),
              ("c", 4 * (19 * ctypes.c_float)),
              ("w", 4 * ctypes.c_float),
              ("ho", 4 * ctypes.c_float)]

"""
Population generation / initializer.  Uses method from Montana and Davis.
@param popAmt: Number of individuals in the population
@type popAmt: integer
@return a list of Parameters objects, representing the population
"""
def generatePop(popAmt):

    generation = []
    nextMember = Parameters()
    for i in range(popAmt):
        for j in range(4):
            for k in range(19):
                child.ih[j][k] = getInitialFloat()
                child.c[j][k] = getInitialFloat()
            child.w[j] = getInitialFloat()
            child.ho[j] = getInitialFloat()

"""
Mutation operator.  Uses MUTATE NODES operator from Montana and Davis.
@param xman: The parent who will be mutated
@type xman: Parameters (see class Parameters, above)
@return Mutated Parameters object
"""
def mutate(xman):
    
    node = random.randint(0,3)
    for i in range(19):
        xman.ih[node][i] += getInitialFloat()
        xman.c[node][i] += getInitialFloat()
    xman.w[node] += getInitialFloat()
    xman.ho[node] += getInitialFloat()

"""
Mate operator.  Uses CROSSOVER WEIGHTS operator from Montana and Davis.
@param parent1: One of the parents who will be mated
@type parent1: Parameters (see class Parameters, above)
@param parent2: One of the parents who will be mated
@type parent2: Parameters (see class Parameters, above)
@return Child Parameters object
"""
def mate(parent1, parent2):

    parentList = [parent1, parent2]
    child = Parameters()

    for i in range(4):
        for j in range(19):
            child.ih[i][j] = parentList[random.randint(0,1)].ih[i][j]
            child.c[i][j] = parentList[random.randint(0,1)].c[i][j]
        child.w[i] = parentList[random.randint(0,1)].w[i]
        child.ho[i] = parentList[random.randint(0,1)].ho[i]

    return child

"""
Function for determining initial parameter values
@param none
@return a random value from (currently) an exponential distribution
"""
def getInitialFloat():
    return random.expovariate(1)*random.choice([-1,1])
        
