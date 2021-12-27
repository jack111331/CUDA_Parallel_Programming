// Sgemm:  C <- alpha*A*B + beta*C
// using cublasXT with multiGPUs.


/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublasXt.h>           // header for cublasXt

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, const float alpha, const float *A, const float *B, 
                         const float beta, float *C) 
{
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      float prod = 0;
      for(int k = 0; k < n; ++k) {
        prod += A[k*n + i]*B[j*n + k];    //  A(i,k)*B(k,j)
      }
      C[j*n + i] = alpha*prod + beta*C[j*n + i];
    }
  }
}

/* Main */
int main(int argc, char **argv)
{
//    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_A_ref;
    float *h_B_ref;
    float *h_C_ref;
    float alpha;
    float beta;
    int N;
    int NGPU;
    int *Dev;
    int umem;
//
    cublasXtHandle_t handle;    // cublasXt context     

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
//      int ip=(i+1)%NGPU;      // GPU on the right
//      cudaDeviceEnablePeerAccess(Dev[ip],0);
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

    printf("Enter the option for the unified memory (0/1): ");
    scanf("%d", &umem);
    printf("%d\n", umem);

    cublasXtCreate(&handle);

    /* Select devices for use in CUBLASXT math functions */
    cublasXtDeviceSelect(handle, NGPU, Dev);

    /* Optional: Set a block size for CUBLASXT math functions */
    cublasXtSetBlockDim(handle, threadsPerBlock);

    // for computing reference solution
    h_A_ref = (float *)malloc(n2 * sizeof(h_A[0]));
    h_B_ref = (float *)malloc(n2 * sizeof(h_B[0]));
    h_C_ref = (float *)malloc(n2 * sizeof(h_C[0]));

    if(umem == 0) { 
      /* Allocate host memory for the matrices */
      h_A = (float *)malloc(n2 * sizeof(h_A[0]));
      h_B = (float *)malloc(n2 * sizeof(h_A[0]));
      h_C = (float *)malloc(n2 * sizeof(h_C[0]));
    }
    else if(umem == 1) {
      // Allocate Unified Memory -- accessible to CPU and GPU
      int size = n2*sizeof(float);
      cudaMallocManaged((void**)&h_A, size);
      cudaMallocManaged((void**)&h_B, size);
      cudaMallocManaged((void**)&h_C, size);
    }
    else {
      printf("The selected option for unified memory is incorrect\n");
      exit(0);
    }

    /* Fill the matrices with random numbers in (0,1) */
    for(int i=0; i<n2; i++) {
      h_A[i] = rand() / (float)RAND_MAX;
      h_B[i] = rand() / (float)RAND_MAX;
      h_C[i] = rand() / (float)RAND_MAX;
      h_A_ref[i] = h_A[i];
      h_B_ref[i] = h_B[i];
      h_C_ref[i] = h_C[i];
    }

    // create the timer
    cudaEvent_t start, stop;

    // start the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    
    printf("Calling cublasXTSgemm with %d-GPU\n",NGPU);
    /* Performs operation using cublas */
    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_B, N, &beta, h_C, N);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    float flops = (float) 3*N*N*N;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Time for %d-GPU: %f (ms) \n",NGPU, gputime);
    printf("%d-GPU Gflops: %f\n",NGPU,flops/(1000000.0*gputime));

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // start the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    printf("Computing with CPU\n");
    /* Performs operation using CPU */
    simple_sgemm(N, alpha, h_A_ref, h_B_ref, beta, h_C_ref);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",flops/(1000000.0*cputime));
    printf("Speed up of %d-GPU: %f \n",NGPU, cputime/gputime);

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

    if(umem == 0) {
      free(h_A);
      free(h_B);
      free(h_C);
    }
    else if(umem == 1) {
      cudaFree(h_A);
      cudaFree(h_B);
      cudaFree(h_C);
    }

    /* Shutdown */
    cublasXtDestroy(handle);
}

