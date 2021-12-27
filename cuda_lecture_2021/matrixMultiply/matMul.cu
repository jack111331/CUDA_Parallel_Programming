// Vector addition: C = 1/A + 1/B, for arbitrarily long vectors
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define INDEX_OF_MAT(i, j, N) ((i) * (N) + (j))

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* h_D;

float* d_A;   // device vectors
float* d_B;
float* d_C;

// Functions
// Allocates an array with random float entries.
void RandomInit(float* data, long n)
{
    for (long i = 0; i < n; ++i) {
	    for (long j = 0; j < n; ++j) {
	        data[INDEX_OF_MAT(i, j, n)] = rand() / (float)RAND_MAX;    	    
	    }
    }
}

void ZeroInit(float *data, long n) {
    for (long i = 0; i < n; ++i) {
	    for (long j = 0; j < n; ++j) {
	        data[INDEX_OF_MAT(i, j, n)] = 0;    	    
	    }
    }	
}

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C, long N)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    long j = blockDim.y * blockIdx.y + threadIdx.y;

    for (long k = 0;k < N; ++k) {
        C[INDEX_OF_MAT(i, j, N)] += A[INDEX_OF_MAT(i, k, N)] * B[INDEX_OF_MAT(k, j, N)];
    }
    
}

float *allocateMatrixMemory(long N) {
	float *matrix = (float *)malloc(N * N * sizeof(float));
	if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
	}
    return matrix;
}

// Host code

int main(void)
{

    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Matrix Multiplication: C = A * B\n");
    int N;

    printf("Enter the size of the vectors: ");
    scanf("%ld",&N);        
    printf("%ld\n",N);        

    // Allocate input matrix h_A and h_B in host memory
    // Allocate device output matrix h_C and host output matrix h_D in host memory

    h_A = allocateMatrixMemory(N);
    h_B = allocateMatrixMemory(N);
    h_C = allocateMatrixMemory(N);
    h_D = allocateMatrixMemory(N);    

    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);
    ZeroInit(h_C, N);
    ZeroInit(h_D, N);

    // Set the sizes of threads and blocks

    int threadsPerBlock[2];
    printf("Enter the number of threads per block: ");
    scanf("%d %d",&threadsPerBlock[0], &threadsPerBlock[1]);
    printf("Threads per block: (%d, %d)\n",threadsPerBlock[0], threadsPerBlock[1]);
    if( threadsPerBlock[0] > 1024 || threadsPerBlock[1] > 1024 || threadsPerBlock[0] * threadsPerBlock[1] > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(0);
    }
    dim3 threadsPerBlockDim(threadsPerBlock[0], threadsPerBlock[1], 1);

    int blocksPerGrid[2] = {(N + threadsPerBlock[0] - 1)/threadsPerBlock[0], (N + threadsPerBlock[1] - 1)/threadsPerBlock[1]};
    printf("The number of blocks is (%d, %d)\n", blocksPerGrid[0], blocksPerGrid[1]);

    dim3 blocksPerGridDim(blocksPerGrid[0], blocksPerGrid[1], 1);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    VecAdd <<< blocksPerGridDim, threadsPerBlockDim >>> (d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n", N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    for (long i = 0; i < N; ++i) {
    	for (long j = 0; j < N; ++j) {
    		for (long k = 0; k < N; ++k) {
		        h_D[INDEX_OF_MAT(i, j, N)] += h_A[INDEX_OF_MAT(i, k, N)] * h_B[INDEX_OF_MAT(k, j, N)];    		
    		}
    	}
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n", N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (long i = 0; i < N; ++i) {
        for (long j = 0;j < N; ++j) {
            diff = abs(h_D[INDEX_OF_MAT(i, j, N)] - h_C[INDEX_OF_MAT(i, j, N)]);
            sum += diff*diff; 
            if(diff > 1.0e-15) { 
                printf("i=%d, j=%d, h_D=%15.10e, h_C=%15.10e \n", i, j, h_D[INDEX_OF_MAT(i, j, N)], h_C[INDEX_OF_MAT(i, j, N)]);
            }        
        }
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);

    cudaDeviceReset();
}
