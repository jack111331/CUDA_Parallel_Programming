// Vector addition: C = 1/A + 1/B 
// using multiple GPUs with OpenMP

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_RandomNumber;   // host vectors
float* h_Integration;

// Functions
void RandomInit(float*, int);
float SumOfDivider(float*, int);
float xOfY(float);
float wOfX(float);


const float C=1.0f;
const float a=0.001f;

// Device code
__global__ void monteCarloIntegration(const float *randomNumber, float* Integration, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;
    float integration = 0.0f;
    if (i < N) {
        float weight = 1.0f;
        float divider = 1.0f;
        float sample;
        for (int j = 0; j < 10; ++j) {
            sample = randomNumber[i*10+j];
            sample = log((-a)/C*sample+1.0)/(-a);
            weight *= C*exp((-a) * sample);
            divider += sample * sample;
        }
        integration += 1.0f / divider / weight;
        i += stride;
    }
    cache[cacheIndex] = integration;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib) {
        cache[cacheIndex] += cache[cacheIndex + ib]; 
      }

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0) {
      Integration[blockIdx.x] = cache[0];
    }
}

// Host code

int main(void)
{
    printf("\n");
    printf("Monte Carlo integration with multiple GPUs \n");
    int numberOfSampling;
    printf("Enter the number of sampling(2^N): ");
    scanf("%d", &numberOfSampling);
    numberOfSampling = 1<<numberOfSampling;
    float integration = 0.0f;
    h_RandomNumber = (float*)malloc(10 * numberOfSampling * sizeof(float));
    // Important sampling using metropolis algorithm
    float sampleOnePoint[15];
    for(int i = 0;i < numberOfSampling;++i) {
    	float weight = 1.0f;
    	RandomInit(&h_RandomNumber[i*10], 10);
    	for(int j = 0;j < 10;++j) {
            sampleOnePoint[j] = xOfY(h_RandomNumber[i*10+j]);
    		weight *= wOfX(sampleOnePoint[j]);
    	}
    	float divider = SumOfDivider(sampleOnePoint, 10);
    	integration += 1.0f / divider / weight;
    }
    integration /= (float) numberOfSampling;
    printf("CPU Important sampling using N=%d, integration result=%f\n", numberOfSampling, integration);

    int N, NGPU, cpu_thread_id=0;
    int *Dev; 
    long mem = 1024*1024*1024;     // 4 Giga for float data type.

    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int)*NGPU);

    int numDev = 0;
    printf("GPU device number: ");
    for(int i = 0; i < NGPU; i++) {
      scanf("%d", &Dev[i]);
      printf("%d ",Dev[i]);
      numDev++;
      if(getchar() == '\n') break;
    }
    printf("\n");
    if(numDev != NGPU) {
      fprintf(stderr,"Should input %d GPU device numbers\n", NGPU);
      exit(1);
    }
// each thread should produce one integration value
    if (numberOfSampling > mem) {
        printf("The size of these 1 vectors cannot be fitted into 4 Gbyte\n");
        exit(1);
    }

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if(threadsPerBlock > 1024) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(1);
    }
    int blocksPerGrid = (numberOfSampling + threadsPerBlock*NGPU - 1) / (threadsPerBlock*NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(1);
    }
    int blockSize = blocksPerGrid * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_Integration = (float*)malloc(blockSize);
    if (! h_Integration) {
	printf("!!! Not enough memory.\n");
	exit(1);
    }
    
    // declare cuda event for timer
    cudaEvent_t start, stop;
//    cudaEventCreate(&start);    // events must be created after devices are set 
//    cudaEventCreate(&stop);

    float Intime,gputime,Outime;

    omp_set_num_threads(NGPU);

    #pragma omp parallel private(cpu_thread_id)
    {
    float *d_RandomNumber, *d_Integration;
	cpu_thread_id = omp_get_thread_num();
	cudaSetDevice(Dev[cpu_thread_id]);
//	cudaSetDevice(cpu_thread_id);

        // start the timer
        if(cpu_thread_id == 0) {
          cudaEventCreate(&start);
          cudaEventCreate(&stop);
          cudaEventRecord(start,0);
        }

	// Allocate vectors in device memory
    cudaMalloc((void**)&d_Integration, blockSize/NGPU);
    cudaMalloc((void**)&d_RandomNumber, 10*numberOfSampling * sizeof(float)/NGPU);

	#pragma omp barrier

        // stop the timer
	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &Intime, start, stop);
          printf("Data input time for GPU: %f (ms) \n",Intime);
	}

        // start the timer
        if(cpu_thread_id == 0) cudaEventRecord(start,0);

        int sm = threadsPerBlock*sizeof(float);// Cache Integration
        monteCarloIntegration<<<blocksPerGrid, threadsPerBlock, sm>>>(d_RandomNumber, d_Integration, numberOfSampling/NGPU);
	cudaDeviceSynchronize();

        // stop the timer

	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &gputime, start, stop);
          printf("Processing time for GPU: %f (ms) \n",gputime);
          printf("GPU Gflops: %f\n",3*N/(1000000.0*gputime));
	}

        // Copy result from device memory to host memory
        // h_C contains the result in host memory

        // start the timer
        if(cpu_thread_id == 0) cudaEventRecord(start,0);

        cudaMemcpy(h_Integration+blocksPerGrid/NGPU*cpu_thread_id, d_Integration, blockSize/NGPU, cudaMemcpyDeviceToHost);
    cudaFree(d_RandomNumber);
    cudaFree(d_Integration);

        // stop the timer

	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &Outime, start, stop);
          printf("Data output time for GPU: %f (ms) \n",Outime);
	}
    } 

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    integration = 0.0f;
    for(int i = 0;i < blockSize;++i) {
        integration += h_Integration[i];
    }
    integration /= (float)numberOfSampling;
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/gputime_tot);

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("GPU Monte Carlo integration: %f\n", integration);
/*
    double sum=0; 
    double diff;
    for (int i = 0; i < N; ++i) {
        diff = abs(h_D[i] - h_C[i]);
        sum += diff*diff; 
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n",sum);

    for (int i=0; i < NGPU; i++) {
	cudaSetDevice(i);
	cudaDeviceReset();
    }
*/

    return 0;
}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

float SumOfDivider(float* data, int n) {
	float divider = 1.0f;
	for(int i = 0;i < n;++i) {
		divider += data[i] * data[i];
	}
	return divider;
}
float xOfY(float y) {
	return log((-a)/C*y+1.0)/(-a);
}
float wOfX(float x) {
	return C*exp((-a) * x);
}
