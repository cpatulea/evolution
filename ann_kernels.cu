#define ARRAY_SIZE(array) sizeof(array) / sizeof((array)[0])

struct Parameters {
  float ih[4][19];  // input->hidden edge weight
  float c[4][19];   // RBF center
  float w[4];       // RBF width
  float ho[4];      // hidden->output edge weight
};

__global__ void evaluate(
  const float *trainSet,
  unsigned int trainSize,
  const Parameters *params,
  unsigned int popSize,
  float *outputs
) {
  const int trainIndex = threadIdx.x;
  const int paramsIndex = blockIdx.x;
  const int outputIndex = paramsIndex * trainSize + trainIndex;
  
  if (trainIndex < trainSize && paramsIndex < popSize) {
    // Read input features.
    float inputs[19];
    for (int i = 0; i < ARRAY_SIZE(inputs); i++) {
      inputs[i] = trainSet[i * trainSize + trainIndex];
    }
    
    // TODO: Read parameters into shared memory.
    const Parameters * const p = &params[paramsIndex];

    // Calculate network output.
    float output = 0.f;
    
    // TODO: Make sure this gets unrolled
    for (int j = 0; j < 4; j++) {
      float d2 = 0.f; // input to hidden node j
      for (int i = 0; i < 19; i++) {
        const float d = inputs[i] * p->ih[j][i] - p->c[j][i];
        d2 += d * 2;
      }
      
      const float h = __expf(-p->w[j] * d2); // Gaussian RBF
      output += h * p->ho[j];
    }
    
    outputs[outputIndex] = output;
  }
}
