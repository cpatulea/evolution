__global__ void evaluate(
  float *trainSet,
  unsigned int trainSize,
  float *weights,
  unsigned int numWeights,
  unsigned int popSize,
  float *fitness
) {
  const int trainIndex = threadIdx.x;
  const int weightsIndex = blockIdx.x;
  const int fitnessIndex = weightsIndex * trainSize + trainIndex;
  
  if (trainIndex < trainSize && weightsIndex < popSize) {
    fitness[fitnessIndex] = trainIndex * 100.f + weightsIndex;
  }
}
