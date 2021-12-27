// Vector addition: C = 1/A + 1/B 
// using multiple GPUs with OpenMP

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_RandomNumber;   // host vectors
float* h_MetropolisX;   // host vectors
float* h_Integration;
float* h_Double;

// Functions
void RandomInit(float*, int);
float SumOfDivider(float*, int);
float xOfY(float);
float wOfX(float);


const float a=0.001f;
const float C=-a/(exp(-a)-1);

// Device code
__global__ void monteCarloIntegration(const float *randomNumber, float* Integration, float *Double, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;
    float integration = 0.0f;
    float doubleX = 0.0f;
    if (i < N) {
        float divider = 1.0f;
        float sample;
        for (int j = 0; j < 10; ++j) {
            sample = randomNumber[i*10+j];
            divider += sample * sample;
        }
        integration += 1.0f / divider;
        doubleX += (1.0f / divider) * (1.0f / divider);
        i += stride;
    }
    cache[cacheIndex] = integration;   // set the cache value 
    cache[cacheIndex + blockDim.x] = doubleX;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib) {
        cache[cacheIndex] += cache[cacheIndex + ib]; 
        cache[cacheIndex + blockDim.x] += cache[cacheIndex + ib + blockDim.x]; 
      }

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0) {
      Integration[blockIdx.x] = cache[0];
      Double[blockIdx.x] = cache[blockDim.x];
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
    int N = numberOfSampling;
    h_RandomNumber = (float*)malloc(10 * (numberOfSampling+1) * sizeof(float));
    h_MetropolisX = (float*)malloc(10 * numberOfSampling * sizeof(float));
    // Important sampling using metropolis algorithm
    for(int i = 0;i < numberOfSampling+1;++i) {
        RandomInit(&h_RandomNumber[i*10], 10);
    }

    float oldWeight = 1.0f;
    float currentX[15];

    for(int j = 0;j < 10;++j) {
        currentX[j] = h_RandomNumber[j];
        oldWeight *= wOfX(currentX[j]);
    }
    for(int i = 1;i < numberOfSampling+1;++i) {
        float newWeight = 1.0f;
        for(int j = 0;j < 10;++j) {
            newWeight *= wOfX(h_RandomNumber[i*10+j]);
        }
        if(newWeight >= oldWeight) {
            for(int j = 0;j < 10;++j) {
                currentX[j] = h_RandomNumber[i*10+j];
            }
        } else {
            float r = rand() / (float)RAND_MAX;
            if (r < newWeight/oldWeight) {
                for(int j = 0;j < 10;++j) {
                    currentX[j] = h_RandomNumber[i*10+j];
                }                
            }
        }
        for(int j = 0;j < 10;++j) {
            h_MetropolisX[(i-1)*10+j] = currentX[j];
        }                
    }

    int NGPU, cpu_thread_id=0;
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
    h_Integration = (float*)malloc(blockSize*NGPU);
    h_Double = (float*)malloc(blockSize*NGPU);
    if (! h_Integration || !h_Double) {
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
    float *d_RandomNumber, *d_Integration, *d_Double;
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
    cudaMalloc((void**)&d_Integration, blockSize);
    cudaMalloc((void**)&d_RandomNumber, 10*numberOfSampling * sizeof(float)/NGPU);
    cudaMalloc((void**)&d_Double, blockSize);

    cudaMemcpy(d_RandomNumber, h_MetropolisX+10*numberOfSampling/NGPU*cpu_thread_id, 10*numberOfSampling * sizeof(float)/NGPU, cudaMemcpyHostToDevice);

	#pragma omp barrier

        // stop the timer
	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &Intime, start, stop);
//          printf("Data input time for GPU: %f (ms) \n",Intime);
	}

        // start the timer
        if(cpu_thread_id == 0) cudaEventRecord(start,0);

        int sm = 2*threadsPerBlock*sizeof(float);// Cache Integration
        monteCarloIntegration<<<blocksPerGrid, threadsPerBlock, sm>>>(d_RandomNumber, d_Integration, d_Double, numberOfSampling/NGPU);
	cudaDeviceSynchronize();

        // stop the timer

	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &gputime, start, stop);
          //printf("Processing time for GPU: %f (ms) \n",gputime);
          //printf("GPU Gflops: %f\n",17*N/(1000000.0*gputime));
	}

        // Copy result from device memory to host memory
        // h_C contains the result in host memory

        // start the timer
        if(cpu_thread_id == 0) cudaEventRecord(start,0);

        cudaMemcpy(h_Integration+blocksPerGrid*cpu_thread_id, d_Integration, blockSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Double+blocksPerGrid*cpu_thread_id, d_Double, blockSize, cudaMemcpyDeviceToHost);
    cudaFree(d_RandomNumber);
    cudaFree(d_Integration);

        // stop the timer

	if(cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime( &Outime, start, stop);
          //printf("Data output time for GPU: %f (ms) \n",Outime);
	}
    } 

    float gpuIntegration = 0.0f;
    double gpuDouble = 0.0;
    for(int i = 0;i < NGPU*blocksPerGrid;++i) {
        gpuIntegration += h_Integration[i];
        gpuDouble += h_Double[i];
    }
    gpuIntegration /= (float)numberOfSampling;
    gpuDouble /= (float)numberOfSampling;
    //printf("GPU Important sampling using metropolis algorithm.\nN=%d, integration result=%f +- %lf\n", numberOfSampling, gpuIntegration, sqrt((gpuDouble - gpuIntegration * gpuIntegration)/(float) numberOfSampling));
    // check result
    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    //printf("Total time for GPU: %f (ms) \n",gputime_tot);

    oldWeight = 1.0f;
    float cpuIntegration = 0.0f;
    double cpuDouble = 0.0f; 
    // Start the timer
    cudaEventRecord(start,0);
    float oldX[15], newX[15];
    for(int j = 0;j < 10;++j) {
        oldX[j] = h_RandomNumber[j];
        oldX[j] = xOfY(oldX[j]);
        oldWeight *= wOfX(oldX[j]);
    }
    for(int i = 1;i < numberOfSampling+1;++i) {
        float newWeight = 1.0f;
        for(int j = 0;j < 10;++j) {
            newX[j] = h_RandomNumber[i*10+j];
	        newX[j] = xOfY(newX[j]);
            newWeight *= wOfX(newX[j]);
        }
        if(newWeight >= oldWeight) {
            for(int j = 0;j < 10;++j) {
                oldX[j] = newX[j];
            }
            oldWeight = newWeight;
        } else {
            float r = rand() / (float)RAND_MAX;
            if (r < newWeight/oldWeight) {
                for(int j = 0;j < 10;++j) {
                    oldX[j] = newX[j];
                }                
                oldWeight = newWeight;
            }
        }
        float divider = SumOfDivider(oldX, 10);
        float fx = 1.0f / divider / oldWeight;
        cpuIntegration += fx;
        cpuDouble += fx * fx;
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",14*N/(1000000.0*cputime));
    //printf("Speed up of GPU = %f\n", cputime/gputime_tot);


    cpuIntegration /= (float) numberOfSampling;
    cpuDouble /= (float) numberOfSampling;

    printf("CPU Important sampling using metropolis algorithm.\nN=%d, integration result=%f +- %lf\n", numberOfSampling, cpuIntegration, sqrt((cpuDouble - cpuIntegration * cpuIntegration)/(float) numberOfSampling));

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("norm(cpuIntegration - gpuIntegration)=%20.15e\n",(cpuIntegration - gpuIntegration)/cpuIntegration);

    for (int i=0; i < NGPU; i++) {
    cudaSetDevice(i);
    cudaDeviceReset();
    }

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
