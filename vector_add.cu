#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int numElements = 1 << 20; // 1M elements
    size_t size = numElements * sizeof(float);
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float *)malloc(size);q
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    const int numIterations = 100; // Number of iterations for the compute part
    const int minStreams = 2; // Minimum number of streams
    const int maxStreams = 6; // Maximum number of streams
    const int minThreadsPerBlock = 64; // Minimum number of threads per block
    const int maxThreadsPerBlock = 256; // Maximum number of threads per block
    const int threadBlockStep = 64; // Step size for the number of threads per block

    // Iterate over different numbers of threads per block
    for (int threadsPerBlock = minThreadsPerBlock; threadsPerBlock <= maxThreadsPerBlock; threadsPerBlock += threadBlockStep) {
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Iterate over different numbers of streams
        for (int numStreams = minStreams; numStreams <= maxStreams; ++numStreams) {
            std::vector<cudaStream_t> streams(numStreams);
            std::vector<float*> d_A(numStreams), d_B(numStreams), d_C(numStreams);

            // Allocate and initialize streams and device memory
            for (int i = 0; i < numStreams; ++i) {
                gpuErrchk(cudaStreamCreate(&streams[i]));
                gpuErrchk(cudaMalloc((void **)&d_A[i], size));
                gpuErrchk(cudaMalloc((void **)&d_B[i], size));
                gpuErrchk(cudaMalloc((void **)&d_C[i], size));
                gpuErrchk(cudaMemcpy(d_A[i], h_A, size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_B[i], h_B, size, cudaMemcpyHostToDevice));
            }

            for (int iter = 0; iter < numIterations; ++iter) {
                for (int i = 0; i < numStreams; ++i) {
                    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], numElements);
                }
            }

            // Synchronize streams
            for (int i = 0; i < numStreams; ++i) {
                gpuErrchk(cudaStreamSynchronize(streams[i]));
            }

            // Copy results back to host and verify
            for (int i = 0; i < numStreams; ++i) {
                gpuErrchk(cudaMemcpy(h_C, d_C[i], size, cudaMemcpyDeviceToHost));
                for (int j = 0; j < numElements; ++j) {
                    if (fabs(h_A[j] + h_B[j] - h_C[j]) > 1e-5) {
                        fprintf(stderr, "Result verification failed at element %d!\n", j);
                        exit(EXIT_FAILURE);
                    }
                }
            }

            // Clean up
            for (int i = 0; i < numStreams; ++i) {
                gpuErrchk(cudaStreamDestroy(streams[i]));
                gpuErrchk(cudaFree(d_A[i]));
                gpuErrchk(cudaFree(d_B[i]));
                gpuErrchk(cudaFree(d_C[i]));
            }

            std::cout << "Test PASSED for " << numStreams << " streams with " << threadsPerBlock << " threads per block\n";
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Completed successfully!" << std::endl;

    return 0;
}
