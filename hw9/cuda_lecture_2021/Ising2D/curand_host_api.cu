/* This program uses the host CURAND API to generate pseudorandom numbers */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

int main(void)
{
  int i,n;
  float *devData, *hostData;
  curandGenerator_t gen;      

  printf("Enter the number of random numbers to be generated: ");
  scanf("%d",&n);
  printf("%d\n",n);
  printf("\n");

  /* Allocate n floats on host */
  hostData = (float*)malloc(n*sizeof(float));

  /* Allocate n floats on device */
  cudaMalloc((void **)&devData, n*sizeof(float));

  /* Create pseudo-random number generator */

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  /* Set seed */
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  /* Generate n floats on device */
  curandGenerateUniform(gen, devData, n);

  /* Copy device memory to host */
  cudaMemcpy(hostData, devData, n*sizeof(float), cudaMemcpyDeviceToHost);

  /* Show result */
  for(i = 0; i < n; i++) {
    printf("%.5e \n", hostData[i]);
  }
  printf("\n");

  /* Cleanup */
  curandDestroyGenerator(gen);
  cudaFree(devData);
  free(hostData);

}
