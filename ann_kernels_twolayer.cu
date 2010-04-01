#include <cuda.h>
#include <math_constants.h>

#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define NODES_PER_LAYER 4

struct Parameters {
  float ih[NODES_PER_LAYER][19];			// input->hidden edge weight
  float hh[NODES_PER_LAYER][NODES_PER_LAYER];// hidden->hidden edge weight
  float c[NODES_PER_LAYER][19];				// RBF center
  float c2[NODES_PER_LAYER][NODES_PER_LAYER];// RBF center second layer
  float w[NODES_PER_LAYER];					// RBF width
  float w2[NODES_PER_LAYER];				// RBF width second layer
  float ho[NODES_PER_LAYER];				// hidden->output edge weight
};

__global__ void evaluate(
  const float *trainSet,
  unsigned int trainSize,
  const Parameters *params,
  unsigned int popSize,
  float *outputs
) {
  const int trainIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int paramsIndex = blockIdx.y;
  const int outputIndex = paramsIndex * trainSize + trainIndex;
  
  if (trainIndex < trainSize && paramsIndex < popSize) {
    // Read input features.
    float inputs[19];
	float l2inputs[NODES_PER_LAYER] = {0.f};
    for (int i = 0; i < ARRAY_SIZE(inputs); i++) {
      inputs[i] = trainSet[i * trainSize + trainIndex];
    }
    
    // TODO: Read parameters into shared memory.
    const Parameters * const p = &params[paramsIndex];

    // Calculate network output.
	float output = 0.f;
    
    // TODO: Make sure this gets unrolled
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      float d2 = 0.f; // input to hidden node j
      for (int i = 0; i < 19; i++) {
        const float d = inputs[i] * p->ih[j][i] - p->c[j][i];
        d2 += d * d;
      }
      
      const float h = __expf(-p->w[j] * d2); // Gaussian RBF

	  for (int i = 0; i < NODES_PER_LAYER; i++) {
		const float d = h * p->hh[j][i] - p->c2[j][i];
		l2inputs[i] += d * d;
	  }
    }

	for (int j = 0; j < NODES_PER_LAYER; j++) {
	  const float h = __expf(-p->w[j] * l2inputs[j]); // Gaussian RBF
      output += h * p->ho[j];
	}
    
    outputs[outputIndex] = output;
  }
}

#define heapparent(i) (((i) - 1) / 2)
#define heapleft(i)   (2 * ((i) + 1) - 1)
#define heapright(i)  (2 * ((i) + 1))
#define swap(a, b)    {float temp = (a); (a) = (b); (b) = temp; }

__device__ void heapreplace(
  float *heap,
  unsigned int size,
  float value
) {
  if (value < heap[0]) {
    return;
  }
  
  // Replace min (root) with new value.
  heap[0] = value;
  
  // Down-heap.
  int i = 0;
  while (heapleft(i) < size) { // stop before leaf level
    const int left = heapleft(i);
    const int right = heapright(i);
    
    int smallest;
    if (left < size && heap[left] < heap[i]) {
      smallest = left;
    } else {
      smallest = i;
    }
    
    if (right < size && heap[right] < heap[smallest]) {
      smallest = right;
    }
    
    if (smallest != i) {
      swap(heap[smallest], heap[i]);
      i = smallest;
    } else {
      break;
    }
  }
}

__global__ void nlargest(
  const float *outputs,
  unsigned int trainSize,
  unsigned int popSize,
  unsigned int n,
  float *thresholds,
  unsigned int *thresholdCounts
) {
  const int paramsIndex = blockIdx.y;
  
  if (paramsIndex < popSize) {
    float maxValue = thresholds[paramsIndex];
    unsigned int maxCount = thresholdCounts[paramsIndex];
    extern __shared__ float heap[/* n */];
    
    // First n values sink to the bottom of the heap.
    for (int i = 0; i < n; i++) {
      heap[i] = -CUDART_INF_F;
    }
    
    for (int trainIndex = 0; trainIndex < trainSize; trainIndex++) {
      const int outputIndex = paramsIndex * trainSize + trainIndex;
      const float output = outputs[outputIndex];
      if (isnan(output)) {
        continue;
      }
      
      if (output < maxValue) {
        heapreplace(heap, n, output);
      } else if (output == maxValue) {
        if (maxCount == 0) {
          heapreplace(heap, n, output);
        } else {
          maxCount--;
        }
      }
    }
    
    // If maxValue hasn't changed, carry over the number of occurrences from the
    // previous pass.
    if (maxValue == heap[0]) {
      maxCount = thresholdCounts[paramsIndex];
    } else {
      maxValue = heap[0];
      maxCount = 0;
    }
    
    // During the next pass, skip the occurrences of maxValue that were already
    // accounted for in this pass.
    for (int i = 0; i < n; i++) {
      if (maxValue == heap[i]) {
        maxCount++;
      }
    }
    
    thresholds[paramsIndex] = maxValue;
    thresholdCounts[paramsIndex] = maxCount;
  }
}

__global__ void count(
  const float *outputs,
  unsigned int trainSize,
  unsigned int trainPositives,
  unsigned int popSize,
  const float *thresholds,
  unsigned int *counts
) {
  const int paramsIndex = blockIdx.y;
  
  if (paramsIndex < popSize) {
    const float threshold = thresholds[paramsIndex];
    unsigned int count = 0;
    
    const int gridDimX = (trainSize + blockDim.x - 1) / blockDim.x;

    for (int blockX = 0; blockX < gridDimX; blockX++) {
      const int trainIndex = blockX * blockDim.x + threadIdx.x;
      
      if (trainIndex < trainPositives) {
        const float output = outputs[paramsIndex * trainSize + trainIndex];
        
        if (output > threshold) {
          count++;
        }
      }
    }
    
    counts[paramsIndex * blockDim.x + threadIdx.x] = count;
  }
}
