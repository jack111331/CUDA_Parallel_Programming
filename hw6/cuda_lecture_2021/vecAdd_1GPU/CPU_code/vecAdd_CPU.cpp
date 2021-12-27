// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// g++ -O3 -o vecAdd_CPU vecAdd_CPU.cpp


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
//using namespace std;

// Set up the clock for timing
clock_t t_start, t_end;

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_D;

// Functions
void RandomInit(float*, int);


// Host code

int main( )
{

    printf("Vector Addition: C = 1/A + 1/B\n");
    int mem = 1024*1024*1024;     // Giga    
    int N;

next:
    printf("Enter the size of the vectors: ");
    scanf("%d",&N);        
    printf("%d\n",N);        
    if( 3*N > mem ) {     // each real number takes 4 bytes
      printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
      goto next;
    }
    long size = N * sizeof(float);


    // Allocate input vectors h_A and h_B in host memory

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);

    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    h_D = (float*)malloc(size);    // the reference solution

    t_start = clock();   // start the clock 

    for (int i = 0; i < N; i++)
        h_D[i] = 1.0/h_A[i] + 1.0/h_B[i];

    t_end = clock();     // stop the clock

    double cputime = 1000*((double)(t_end - t_start))/CLOCKS_PER_SEC;  // milli-second
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));

}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}



