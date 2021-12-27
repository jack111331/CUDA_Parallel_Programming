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


// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* h_D;
float* d_A;   // device vectors
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, long);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C, long N)
{

    long i = blockDim.x * blockIdx.x + threadIdx.x;

//    if (i < N)                     // only for N < blockDim.x*gridDim.x 
//      C[i] = 1.0/A[i] + 1.0/B[i];

    while (i < N) {
        C[i] = 1.0/A[i] + 1.0/B[i];
        i += blockDim.x * gridDim.x;  // go to the next grid
    }
    
    __syncthreads();

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

    printf("Vector Addition: C = 1/A + 1/B\n");
    int N;

    printf("Enter the size of the vectors: ");
    scanf("%ld",&N);        
    printf("%ld\n",N);        
    long size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Check memory allocations  
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Set the sizes of threads and blocks

    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if( threadsPerBlock > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(0);
    }

//    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("%d\n",blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(0); 
    }
    printf("The number of blocks is %d\n", blocksPerGrid);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    VecAdd <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",3*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
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

    h_D = (float*)malloc(size);       // to compute the reference solution
    for (long i = 0; i < N; ++i) 
        h_D[i] = 1.0/h_A[i] + 1.0/h_B[i];
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (long i = 0; i < N; ++i) {
      diff = abs(h_D[i] - h_C[i]);
      sum += diff*diff; 
      if(diff > 1.0e-15) { 
//        printf("i=%d, h_D=%15.10e, h_C=%15.10e \n", i, h_D[i], h_C[i]);
      }
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);

    cudaDeviceReset();
}


// Allocates an array with random float entries.
void RandomInit(float* data, long n)
{
    for (long i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}



