// phi is dirac delta function
// FT of Dirac delta function is 1, distribute (1/2pi)^3 to inverse fourier transform :D
// for each grid of momentum-space in (2^5)^3, insert FT of them
// k is l1-norm 
// to solve poisson, point charge need to be multiplied with 4pi/(k^2)
// cufftMakePlan3d()
// direction CUFFT_INVERSE
// will the cuda ift auto divide 2pi or not(?
// IFFT of cufft should be normalized... multiply 1/(L^3)
// https://math.stackexchange.com/questions/1809871/solve-poisson-equation-using-fft
// https://math.stackexchange.com/questions/877966/fourier-transform-of-poisson-equation/877967
// https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/7_CUDALibraries/simpleCUFFT_2d_MGPU/simpleCUFFT_2d_MGPU.cu
// http://links.uwaterloo.ca/amath353docs/set11.pdf

#include <complex>
#include <cstdio>
#include <cufft.h>

#define M_PI 3.14159265358979323846 // Pi constant with double precision

using namespace std;

// simple test program
int main ()
{
    int realLatticeSize;
    complex<double> *pointCharge;               // point charge at origin, phi
    complex<double> *potential;                    

    int gid;
    printf("Enter the GPU ID (0/1): ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    cudaSetDevice(gid);
    printf("Enter the lattice size: ");
    scanf("%d",&realLatticeSize);
    printf("%d\n", realLatticeSize);
    printf("Would you want to print result? (0/1):");
    int io;
    scanf("%d", &io);
    int fftLatticeSize = 1;
    while (fftLatticeSize < realLatticeSize) {
    	fftLatticeSize <<= 1;
    }
    const double L = 1;

    pointCharge  = (complex<double> *)malloc(sizeof(complex<double>) * fftLatticeSize * fftLatticeSize * fftLatticeSize);
    potential  = (complex<double> *)malloc(sizeof(complex<double>) * fftLatticeSize * fftLatticeSize * fftLatticeSize);

    // generate samples for testing
    double perLattice[4096];
    for (int i = 0; i <= fftLatticeSize/2; ++i) {
        double k = (2 * M_PI * i) / L;
        perLattice[i] = k*k;
    }
    for (int i = fftLatticeSize/2 + 1; i < fftLatticeSize; ++i) {
        double k = (2 * M_PI * (i - fftLatticeSize)) / L;
        perLattice[i] = k*k;
    }
    // all the grid's point charge fourier transform is 1 because point charge is at origin
    for (int i = 0; i < fftLatticeSize; ++i) {
    	for (int j = 0; j < fftLatticeSize; ++j) {
	        for (int k = 0; k < fftLatticeSize; ++k) {
                pointCharge[i * fftLatticeSize * fftLatticeSize + j * fftLatticeSize + k] = complex<double>(0.,0.);
                double k2 = perLattice[i] + perLattice[j] + perLattice[k];
                if (i == 0 && j == 0 && k == 0) {
                    k2 = 1.0;
                }
                pointCharge[i * fftLatticeSize * fftLatticeSize + j * fftLatticeSize + k] = -1 / k2;
   	        }
    	}
    }
//    pointCharge[0] = 0;
    float  gputime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cufftHandle         plan;
    cufftDoubleComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*fftLatticeSize*fftLatticeSize*fftLatticeSize);
    cudaMemcpy(data, pointCharge, sizeof(complex<double>)*fftLatticeSize*fftLatticeSize*fftLatticeSize, cudaMemcpyHostToDevice);

    cudaEventRecord(start,0);
    if (cufftPlan3d(&plan, fftLatticeSize, fftLatticeSize, fftLatticeSize, CUFFT_Z2Z) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed.\n");
		exit(1);
    }
    if (cufftExecZ2Z(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecZ2Z forward failed.\n");
		exit(1);
    }
    cudaMemcpy(potential, data, sizeof(complex<double>)*fftLatticeSize * fftLatticeSize * fftLatticeSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    FILE *poissonSystemFile, *poissonSystemDiagonalFile, *poissonSystemXAxisFile;
    poissonSystemFile = fopen("poissonSystem.dat", "w");
    poissonSystemDiagonalFile = fopen("poissonSystemDiagonal.dat", "w");
    poissonSystemXAxisFile = fopen("poissonSystemXAxis.dat", "w");
	double normalizationFactor = fftLatticeSize*fftLatticeSize*fftLatticeSize;
    for (int i = 0; i < fftLatticeSize; ++i) {
    	for (int j = 0;j < fftLatticeSize; ++j) {
	        for (int k = 0; k < fftLatticeSize; ++k) {
                if(io) {
                    printf("(%d, %d, %d) %+.6f %+.6f\n",
                           i, j, k, potential[i*fftLatticeSize*fftLatticeSize+j*fftLatticeSize+k].real() / normalizationFactor, potential[i*fftLatticeSize*fftLatticeSize+j*fftLatticeSize+k].imag() / normalizationFactor);                
                }
                fprintf(poissonSystemFile, "%.12lf ", potential[i*fftLatticeSize*fftLatticeSize+j*fftLatticeSize+k].real() / normalizationFactor);
			}
            fprintf(poissonSystemFile, "\n");
       }
       fprintf(poissonSystemFile, "\n");
	}
    fclose(poissonSystemFile);

    for (int i = 0; i < fftLatticeSize; ++i) {
        fprintf(poissonSystemDiagonalFile, "%.12lf ", potential[i*fftLatticeSize*fftLatticeSize+i*fftLatticeSize+i].real() / normalizationFactor);
        fprintf(poissonSystemXAxisFile, "%.12lf ", potential[i].real() / normalizationFactor);
    }
    fclose(poissonSystemDiagonalFile);
    fclose(poissonSystemXAxisFile);
    printf("GPU time: %f (ms)\n", gputime);

    cufftDestroy(plan);

    cudaFree(data);
}

// eof
