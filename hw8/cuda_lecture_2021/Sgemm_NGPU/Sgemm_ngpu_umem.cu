// Sgemm:  C <- alpha*A*B + beta*C


/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <omp.h>


/* CPU implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B, 
                         const float beta, float *C) 
{
  for(int i=0; i < n; ++i) {
    for(int j=0; j < n; ++j) {
      float prod = 0.0f;
      for(int k=0; k < n; ++k) {
        prod += A[i*n + k] * B[k*n + j];    //  A(i,k)*B(k,j)
      }
      C[i*n + j] = alpha*prod + beta*C[i*n + j];
    }
  }
}


/* GPU implementation of a simple version of sgemm with unified memory*/
__global__ void sgemm_umem(int n, float alpha, const float *A, const float *B, 
                          const float beta, float *C, const int NGPU, const int cpu_thread_id) 
{
  int offset = (n/NGPU)*cpu_thread_id;
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;

  for(int j = 0; j < n; ++j) {
    float prod = 0.0f;
    for(int k = 0; k < n; ++k) {
      prod += A[i*n + k] * B[k*n + j];    //  A(i,k)*B(k,j)
    }
    C[i*n + j] = alpha*prod + beta*C[i*n + j];
  }
  __syncthreads();
}


/* Main */
int main(int argc, char **argv)
{
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_A_ref;
    float *h_B_ref;
    float *h_C_ref;
    float alpha;
    float beta;
    int N;
    int cpu_thread_id = 0;
    int NGPU;
    int *Dev;
    float cputime, gputime_tot;

    // the timer
    cudaEvent_t start,stop;
//
    printf("Sgemm: C <- alpha*A*B + beta*C \n");
    printf("Enter the value of alpha: ");
    scanf("%f",&alpha);
    printf("%f\n",alpha);
    printf("Enter the value of beta: ");
    scanf("%f",&beta);
    printf("%f\n",beta);
    printf("Enter the dimension of the matrix: ");
    scanf("%d",&N);
    printf("%d\n",N);
    int n2 = N*N;
//
    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int)*NGPU);
//
    int numDev = 0;
    printf("GPU device numbers: ");
    for(int i=0; i<NGPU; i++) {
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

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if(threadsPerBlock > 1024) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(1);
    }
    int blocksPerGrid = (N + threadsPerBlock*NGPU - 1) / (threadsPerBlock*NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(1);
    }

    int size = n2*sizeof(float);

    // for computing reference solution
    h_A_ref = (float *)malloc(size);
    h_B_ref = (float *)malloc(size);
    h_C_ref = (float *)malloc(size);

    // Allocate Unified Memory -- accessible to CPU and GPU
    cudaMallocManaged((void**)&h_A, size);
    cudaMallocManaged((void**)&h_B, size);
    cudaMallocManaged((void**)&h_C, size);
    if (h_A == NULL) {
	printf("!!! Cannot allocate unified memory: h_A\n");
	exit(1);
    }
    if (h_B == NULL) {
	printf("!!! Cannot allocate unified memory: h_B\n");
	exit(1);
    }
    if (h_C == NULL) {
	printf("!!! Cannot allocate unified memory: h_C\n");
	exit(1);
    }

    /* Fill the matrices with random numbers in (0,1) */
    for(int i = 0; i < n2; i++) {
      h_A[i] = (float)rand() / (float)RAND_MAX;
      h_B[i] = (float)rand() / (float)RAND_MAX;
      h_C[i] = (float)rand() / (float)RAND_MAX;
      h_A_ref[i] = h_A[i];
      h_B_ref[i] = h_B[i];
      h_C_ref[i] = h_C[i];
    }

    float flops =(float)3*N*N*N;

    omp_set_num_threads(NGPU);

    #pragma omp parallel private(cpu_thread_id)
    {
       cpu_thread_id = omp_get_thread_num();
       cudaSetDevice(Dev[cpu_thread_id]);

       // start the timer
       if(cpu_thread_id == 0) {
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
         cudaEventRecord(start,0);
       }
       cudaDeviceSynchronize();

       sgemm_umem<<<blocksPerGrid,threadsPerBlock>>>(N, alpha, h_A, h_B, beta, h_C, NGPU, cpu_thread_id);
       cudaDeviceSynchronize();

       cudaError_t err = cudaGetLastError();
       if (cudaSuccess != err) {
	   fprintf(stderr, "cudaCheckError failed at: sgemm_umem\n");
	   exit(-1);
       }

       // stop the timer
       if(cpu_thread_id == 0) {
         cudaEventRecord(stop,0);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&gputime_tot, start, stop);
         printf("Total GPU Time: %f (ms) \n",gputime_tot);
         printf("%d-GPU Gflops: %f\n",NGPU,flops/(1000000.0*gputime_tot));
       }

       // Destroy timer
       cudaEventDestroy(start);
       cudaEventDestroy(stop);

       cudaDeviceSynchronize();
    }
    
    // create the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    // start the timer
    cudaEventRecord(start,0);

    printf("Computing with CPU \n");
    /* Performs operation using CPU */
    simple_sgemm(N, alpha, h_A_ref, h_B_ref, beta, h_C_ref);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&cputime, start, stop);
    printf("Time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",flops/(1000000.0*cputime));
    printf("Speed up of %d-GPU: %f \n",NGPU, cputime/gputime_tot);

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Check result against reference */

    float error_norm=0;
    float ref_norm=0;
    float diff;
    for(int i = 0; i < n2; ++i) {
      diff = h_C_ref[i] - h_C[i];
      error_norm += diff*diff;
      ref_norm += h_C_ref[i]*h_C_ref[i];
    }
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);
    printf("normalized error_norm = %.5e\n", error_norm/ref_norm);
    printf("\n");

    /* Memory clean up */

    free(h_A_ref);
    free(h_B_ref);
    free(h_C_ref);

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
}

