import ctypes, random, math, ctypes

class Parameters(ctypes.Structure):
  _fields_ = [("ih", 4 * (17 * ctypes.c_float)),
              ("c", 4 * (17 * ctypes.c_float)),
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
    for i in range(popAmt):
        nextMember = Parameters()
        for j in range(4):
            for k in range(17):
                nextMember.ih[j][k] = getInitialFloat()
                nextMember.c[j][k] = getInitialFloat()
            nextMember.w[j] = getInitialFloat()
            nextMember.ho[j] = getInitialFloat()
        generation.append(nextMember)

    return generation

"""
Create a new generation, based on the current generation
@param oldGen: The generation that will be mated and mutated
@type oldGen: A list of Parameters operations
@param fitList: A list of fitnesses of the generation
@type fitList: A list of float values
@return: List of new parameters (IE, a new generation)
"""
def generateGeneration(oldGen):
    
    newGen = []
    newGen.append(oldGen[0])
    for i in range(499):
        if random.choice([0,1]) == 1:
            newGen.append(
                mutate(oldGen[int(random.expovariate(-math.log(0.92)))])
            )
        else:
            print "doing some mating"
            parent1 = oldGen[int(random.expovariate(-math.log(0.92)))]
            parent2 = oldGen[int(random.expovariate(-math.log(0.92)))]
            while parent2 == parent1:
                parent2 = oldGen[int(random.expovariate(-math.log(0.92)))]
            newGen.append(mate(parent1, parent2))
            print "mating complete"

    return newGen
        
"""
Mutation operator.  Uses MUTATE NODES operator from Montana and Davis.
@param xman: The parent who will be mutated
@type xman: Parameters (see class Parameters, above)
@return Mutated Parameters object
"""
def mutate(xman):
    
    node = random.randint(0,3)
    for i in range(17):
        xman.ih[node][i] += getInitialFloat()
        xman.c[node][i] += getInitialFloat()
    xman.w[node] += getInitialFloat()
    xman.ho[node] += getInitialFloat()
    
    return xman

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
        for j in range(17):
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
        
