//  Monte Carlo simulation of Ising model on 2D lattice
//  using Metropolis algorithm
//  using checkerboard (even-odd) update 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include <cuda_runtime.h>


gsl_rng *rng=NULL;    // pointer to gsl_rng random number generator

void exact_2d(double, double, double*, double*);
void rng_MT(float*, int);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

int *spin;            // host spin variables
int **d_spin;          // device spin variables
float *h_rng;         // host random numbers
float **d_rng;         // device random numbers 

__constant__ int x_fw[1000],x_bw[1000];     // declare constant memory for x-axis fw, bw 
__constant__ int y_fw[1000],y_bw[1000];     // declare constant memory for y-axis fw, bw 

__global__ void metro_gmem_odd(int *spin, int *spin_top, int *spin_bottom, int *spin_left, int *spin_right, float *ranf, const float B, const float T)
{
    int    x, y, parity;
    int    i, io;
    int    old_spin, new_spin, spins;
    int    k1, k2, k3, k4;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction
    int Lx = 2*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 
    int Ly = blockDim.y*gridDim.y;     // number of sites in y-axis of the entire lattice

    // next, go over the odd sites 
    // io indicate the actual position in block
    io = threadIdx.x + threadIdx.y*blockDim.x;   

    // x, y indicate the position at the checkboard lattice
    x = (2*io)%Nx;
    y = ((2*io)/Nx)%Nx;
    parity=(x+y+1)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*Lx;
    old_spin = spin[i];
    new_spin = -old_spin;

    if (x == 0) {
	    k3 = spin_left[x_bw[x] + y*Lx];     // left
    } else {
	    k3 = spin[x_bw[x] + y*Lx];     // left
    }
    if (x == Lx-1) {
	    k1 = spin_right[x_fw[x] + y*Lx];     // right    	
    } else {
	    k1 = spin[x_fw[x] + y*Lx];     // right    	
    }
    if (y == 0) {
	    k2 = spin_top[x + y_bw[y]*Lx];     // top
    } else {
	    k2 = spin[x + y_bw[y]*Lx];     // top
    }
    if (y == Ly-1) {
	    k4 = spin_bottom[x + y_fw[y]*Lx];     // bottom
    } else {
	    k4 = spin[x + y_fw[y]*Lx];     // bottom
    }

    spins = k1 + k2 + k3 + k4;
    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }

    __syncthreads();

}


__global__ void metro_gmem_even(int *spin, int *spin_top, int *spin_bottom, int *spin_left, int *spin_right, float *ranf, const float B, const float T)
{
    int    x, y, parity;
    int    i, ie;
    int    old_spin, new_spin, spins;
    int    k1, k2, k3, k4;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction
    int Lx = 2*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 
    int Ly = blockDim.y*gridDim.y;     // number of sites in y-axis of the entire lattice

    // first, go over the even sites 

    ie = threadIdx.x + threadIdx.y*blockDim.x;  
    x = (2*ie)%Nx;
    y = ((2*ie)/Nx)%Nx;
    parity=(x+y)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*Lx;
    old_spin = spin[i];
    new_spin = -old_spin;

    if (x == 0) {
	    k3 = spin_left[x_bw[x] + y*Lx];     // left
    } else {
	    k3 = spin[x_bw[x] + y*Lx];     // left
    }
    if (x == Lx-1) {
	    k1 = spin_right[x_fw[x] + y*Lx];     // right    	
    } else {
	    k1 = spin[x_fw[x] + y*Lx];     // right    	
    }
    if (y == 0) {
	    k2 = spin_top[x + y_bw[y]*Lx];     // top
    } else {
	    k2 = spin[x + y_bw[y]*Lx];     // top
    }
    if (y == Ly-1) {
	    k4 = spin_bottom[x + y_fw[y]*Lx];     // bottom
    } else {
	    k4 = spin[x + y_fw[y]*Lx];     // bottom
    }

    spins = k1 + k2 + k3 + k4;
    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }
    
    __syncthreads();
 
}   

int main(void) {
  int NGx,NGy;   // The partition of the lattice (NGx*NGy=NGPU).
  int NGPU;
  int     *Dev;   // GPU device numbers.
  int Nx,Ny; 		// # of sites in x and y directions respectively
  int Ns; 		// Ns = Nx*Ny, total # of sites
  int Lx, Ly;// lattice size in each GPU.
  int *ffw;      	// forward index
  int *x_ffw;      	// x-axis forward index
  int *y_ffw;      	// y-axis forward index

  int *x_bbw; 	        // x-axis backward index
  int *y_bbw; 	        // y-axis backward index
  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int nb; 		// # of sweeps before saving spin configurations
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2;           // right, top
  int istart; 		// istart = (0: cold start/1: hot start)
  double T; 		// temperature
  double B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >

  float gputime;
  float flops;
  int      cpu_thread_id=0;


  // Input the GPU setting
  printf("  Enter the number of GPUs (NGx, NGy): ");
  scanf("%d %d", &NGx, &NGy);
  printf("%d %d\n", NGx, NGy);
  NGPU = NGx * NGy;
  Dev  = (int *)malloc(sizeof(int)*NGPU);
  for (int i=0; i < NGPU; i++) {
    printf("  * Enter the GPU ID (0/1/...): ");
    scanf("%d",&Dev[i]);
    printf("%d\n", Dev[i]);
  }

  printf("Ising Model on 2D Square Lattice with p.b.c.\n");
  printf("============================================\n");
  printf("Initialize the RNG\n");
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  printf("Enter the seed:\n");
  long seed;
  scanf("%ld",&seed);
  printf("%ld\n",seed); 
  gsl_rng_set(rng,seed);
  printf("The RNG has been initialized\n");
  printf("Enter the number of sites in each dimension (<= 1000)\n");
  scanf("%d",&Nx);
  printf("%d\n",Nx);
  Ny=Nx;
  Ns=Nx*Ny;

  if (Nx % NGx != 0) {
    printf("!!! Invalid partition of lattice: Nx %% NGx != 0\n");
    exit(1);
  }
  if (Ny % NGy != 0) {
    printf("!!! Invalid partition of lattice: Ny %% NGy != 0\n");
    exit(1);
  }
  Lx = Nx / NGx;
  Ly = Ny / NGy;

  ffw = (int*)malloc(Nx*sizeof(int));
  x_ffw = (int*)malloc(Lx*sizeof(int));
  y_ffw = (int*)malloc(Ly*sizeof(int));

  x_bbw = (int*)malloc(Lx*sizeof(int));
  y_bbw = (int*)malloc(Ly*sizeof(int));
  for(int i=0; i<Nx; i++) {
    ffw[i]=(i+1)%Nx;
  }
  for(int i=0; i<Lx; i++) {
    x_ffw[i]=(i+1)%Lx;
    x_bbw[i]=(i-1+Lx)%Lx;
  }
  for(int i=0; i<Ly; i++) {
    y_ffw[i]=(i+1)%Ly;
    y_bbw[i]=(i-1+Ly)%Ly;
  }


  for(int i = 0;i < NGPU;++i) {
    cudaSetDevice(Dev[i]);
	cudaMemcpyToSymbol(x_fw, x_ffw, Lx*sizeof(int));
	cudaMemcpyToSymbol(y_fw, y_ffw, Ly*sizeof(int));
	cudaMemcpyToSymbol(x_bw, x_bbw, Lx*sizeof(int));
	cudaMemcpyToSymbol(y_bw, y_bbw, Ly*sizeof(int));
  }

  spin = (int*)malloc(Ns*sizeof(int));          // host spin variables
  h_rng = (float*)malloc(Ns*sizeof(float));     // host random numbers

  printf("Enter the # of sweeps for thermalization\n");
  scanf("%d",&nt);
  printf("%d\n",nt);
  printf("Enter the # of measurements\n");
  scanf("%d",&nm);
  printf("%d\n",nm);
  printf("Enter the interval between successive measurements\n");
  scanf("%d",&im);
  printf("%d\n",im);
  printf("Enter the display interval\n");
  scanf("%d",&nd);
  printf("%d\n",nd);
  printf("Enter the interval for saving spin configuration\n");
  scanf("%d",&nb);
  printf("%d\n",nb);
  printf("Enter the temperature (in units of J/k)\n");
  scanf("%lf",&T);
  printf("%lf\n",T);
  printf("Enter the external magnetization\n");
  scanf("%lf",&B);
  printf("%lf\n",B);
  printf("Initialize spins configurations :\n");
  printf(" 0: cold start \n");
  printf(" 1: hot start \n");
  scanf("%d",&istart);
  printf("%d\n",istart);
 
  // Set the number of threads (tx,ty) per block

  int tx,ty;
  printf("Enter the number of threads (tx,ty) per block: ");
  printf("For even/odd updating, tx=ty/2 is assumed: ");
  scanf("%d %d",&tx, &ty);
  printf("%d %d\n",tx, ty);

  if(2*tx != ty) exit(0);
  if(tx*ty > 1024) {
    printf("The number of threads per block must be less than 1024 ! \n");
    exit(0);
  }
  dim3 threads(tx,ty);

  // The total number of threads in the grid is equal to (Nx/2)*Ny = Ns/2 

  int bx = Nx/tx/2;
  if(bx*tx*2 != Nx) {
    printf("The block size in x is incorrect\n");
    exit(0);
  }
  int by = Ny/ty;
  if(by*ty != Ny) {
    printf("The block size in y is incorrect\n");
    exit(0);
  }
  if ((bx/NGx > 65535) || (by/NGy > 65535)) {
      printf("!!! The grid size exceeds the limit.\n");
      exit(0);
  }
  if ((bx/NGx%2 == 1) || (by/NGy%2 == 1)) {
      printf("!!! The grid size should be even, or the checkboard scheme will fail.\n");
      exit(0);
  }
  dim3 blocks(bx/NGx,by/NGy);
  printf("The dimension of the grid is (%d, %d)\n",bx/NGx,by/NGy);
  printf("The size of the lattice is (%d, %d)\n",Lx,Ly);

  if(istart == 0) {
    for(int j=0; j<Ns; j++) {       // cold start
      spin[j] = 1;
    }
  }
  else {
    for(int j=0; j<Ns; j++) {     // hot start
      if(gsl_rng_uniform(rng) > 0.5) { 
        spin[j] = 1;
      }
      else {
        spin[j] = -1;
      }
    }
  }

  FILE *output;            
  output = fopen("ising2d_ngpu_gmem.dat","w");
  FILE *output3;
  output3 = fopen("spin_ngpu_gmem.dat","w");   

  // Allocate vectors in device memory

  d_spin = (int **)malloc(NGPU*sizeof(int *));
  d_rng = (float **)malloc(NGPU*sizeof(float *));

  // Initialize GPU connection and copy spin vectors from host memory to device memory
  int enabledAccess[10][10] = {};
  for (int i = 0;i < 10;++i) {
  	enabledAccess[i][i] = 1;
  }
  omp_set_num_threads(NGPU);
  #pragma omp parallel private(cpu_thread_id)
  {
    int cpuid_x, cpuid_y;
    cpu_thread_id = omp_get_thread_num();
    cpuid_x       = cpu_thread_id % NGx;
    cpuid_y       = cpu_thread_id / NGx;
  	// Error code to check return values for CUDA calls
  	cudaError_t err = cudaSuccess;
  	err = cudaSetDevice(Dev[cpu_thread_id]);
  	if(err != cudaSuccess) {
    	printf("!!! Cannot select GPU with device ID = %d\n", Dev[cpu_thread_id]);
    	fflush(stdout);
    	exit(1);
 	}

    int cpuid_r = ((cpuid_x+1)%NGx) + cpuid_y*NGx;         // GPU on the right
    if (enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_r]] == 0) {
	    err = cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
	    enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_r]] = 1;
    }
  	if(err != cudaSuccess) {
    	printf("!!! Cannot enable peer access between GPU with device ID = %d to %d\n", Dev[cpu_thread_id], Dev[cpuid_r]);
    	fflush(stdout);
    	exit(1);
 	}
    int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;     // GPU on the left
    if (enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_l]] == 0) {
	    err = cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
	    enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_l]] = 1;
    }
  	if(err != cudaSuccess) {
    	printf("!!! Cannot enable peer access between GPU with device ID = %d to %d\n", Dev[cpu_thread_id], Dev[cpuid_l]);
    	fflush(stdout);
    	exit(1);
 	}
    int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;         // GPU on the top
    if (enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_t]] == 0) {
	    err = cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
	    enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_t]] = 1;
    }
  	if(err != cudaSuccess) {
    	printf("!!! Cannot enable peer access between GPU with device ID = %d to %d\n", Dev[cpu_thread_id], Dev[cpuid_t]);
    	fflush(stdout);
    	exit(1);
 	}
    int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;     // GPU on the bottom
    if (enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_b]] == 0) {
	    err = cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);
	    enabledAccess[Dev[cpu_thread_id]][Dev[cpuid_b]] = 1;
    }
  	if(err != cudaSuccess) {
    	printf("!!! Cannot enable peer access between GPU with device ID = %d to %d\n", Dev[cpu_thread_id], Dev[cpuid_b]);
    	fflush(stdout);
    	exit(1);
 	}

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_spin[cpu_thread_id], Ns*sizeof(int)/NGPU);         // device spin variables
    cudaMalloc((void**)&d_rng[cpu_thread_id], Ns*sizeof(float)/NGPU);        // device random numbers

    // Copy vectors from the host memory to the device memory

    for (int i=0; i < Ly; i++) {
      int *h, *d;
      h = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
      d = d_spin[cpu_thread_id] + i*Lx;
      cudaMemcpy(d, h, Lx*sizeof(int), cudaMemcpyHostToDevice);
    }

    #pragma omp barrier

  } // OpenMP

  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  Ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, Ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  Ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, Ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  Ns=%d\n", T, B, Ns);
    printf("T=%.5e  B=%.5e  Ns=%d\n", T, B, Ns);
  }
  fprintf(output,"     E           M        \n");
  fprintf(output,"--------------------------\n");

  printf("Thermalizing\n");
  printf("sweeps   < E >     < M >\n");
  printf("---------------------------------\n");
  fflush(stdout);

  te=0.0;                          //  initialize the accumulators
  tm=0.0;
  count=0.0;
  sweeps=nt+nm*im;                 //  total # of sweeps

  // create the timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //start the timer
  cudaEventRecord(start,0);

  for(int swp=0; swp<sweeps; swp++) {
    rng_MT(h_rng, Ns);                                  // generate Ns random numbers 

    #pragma omp parallel private(cpu_thread_id)
    {
      int cpuid_x, cpuid_y;
      cpu_thread_id = omp_get_thread_num();
      cpuid_x       = cpu_thread_id % NGx;
      cpuid_y       = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      // Copy vectors from the host memory to the device memory

      for (int i=0; i < Ly; i++) {
        float *h, *d;
        h = h_rng + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
        d = d_rng[cpu_thread_id] + i*Lx;
        cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
      }
      int *dL_spin, *dR_spin, *dT_spin, *dB_spin, *d0_spin;
      d0_spin = d_spin[cpu_thread_id];
      dL_spin = (cpuid_x == 0)     ? d_spin[NGx-1+cpuid_y*NGx] :  d_spin[cpuid_x-1+cpuid_y*NGx];
      dR_spin = (cpuid_x == NGx-1) ? d_spin[0+cpuid_y*NGx] : d_spin[cpuid_x+1+cpuid_y*NGx];
      dT_spin = (cpuid_y == 0    ) ? d_spin[cpuid_x+(NGy-1)*NGx] : d_spin[cpuid_x+(cpuid_y-1)*NGx];
      dB_spin = (cpuid_y == NGy-1) ? d_spin[cpuid_x+(0)*NGx] : d_spin[cpuid_x+(cpuid_y+1)*NGx];

      metro_gmem_even<<<blocks,threads>>>(d0_spin, dT_spin, dB_spin, dL_spin, dR_spin, d_rng[cpu_thread_id], B, T);
      #pragma omp barrier
      metro_gmem_odd<<<blocks,threads>>>(d0_spin, dT_spin, dB_spin, dL_spin, dR_spin, d_rng[cpu_thread_id], B, T);
      #pragma omp barrier
	  cudaDeviceSynchronize();


    } // OpenMP


    if (swp < nt) {
      // Thermalization
      continue;
    }

    int k; 
    if(swp%im == 0) {

      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x       = cpu_thread_id % NGx;
        cpuid_y       = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);

        // Copy vectors from the host memory to the device memory

        for (int i=0; i < Ly; i++) {
          int *h, *d;
          h = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_spin[cpu_thread_id] + i*Lx;
          cudaMemcpy(h, d, Lx*sizeof(int), cudaMemcpyDeviceToHost);
        }
        #pragma omp barrier

      } // OpenMP
      mag=0.0;
      energy=0.0;
      for(int j=0; j<Ny; j++) {
        for(int i=0; i<Nx; i++) {
          k = i + j*Nx;
          k1 = ffw[i] + j*Nx;
          k2 = i + ffw[j]*Nx;
          mag = mag + spin[k]; // total magnetization;
          energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;
        }
      }
      energy = energy - B*mag;
      te = te + energy;
      tm = tm + mag;
      count = count + 1.0;
      fprintf(output, "%.5e  %.5e\n", energy/(double)Ns, mag/(double)Ns);  // save the raw data 
    }
    if(swp%nd == 0) {
      E = te/(count*(double)(Ns));
      M = tm/(count*(double)(Ns));
      printf("%d  %.5e  %.5e\n", swp, E, M);
    }
    if(swp%nb == 0) {
      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x       = cpu_thread_id % NGx;
        cpuid_y       = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);

        // Copy vectors from the host memory to the device memory

        for (int i=0; i < Ly; i++) {
          int *h, *d;
          h = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_spin[cpu_thread_id] + i*Lx;
          cudaMemcpy(h, d, Lx*sizeof(int), cudaMemcpyDeviceToHost);
        }
        #pragma omp barrier

      } // OpenMP

      fprintf(output3,"swp = %d, spin configuration:\n",swp);
      for(int j=Nx-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(output3,"%d ",spin[i+j*Nx]);
        }
        fprintf(output3,"\n");
      }
      fprintf(output3,"\n");
    }
  }
  fclose(output);      
  fclose(output3);
  printf("---------------------------------\n");
  if(B == 0.0) {
    printf("T=%.5e  B=%.5e  Ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, Ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  Ns=%d\n", T, B, Ns);
  }

  // stop the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);









  cudaEventElapsedTime(&gputime, start, stop);
  printf("Processing time for GPU: %f (ms) \n",gputime);
  flops = 7.0*Nx*Nx*sweeps;
  printf("GPU Gflops: %lf\n",flops/(1000000.0*gputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  gsl_rng_free(rng);

  #pragma omp parallel private(cpu_thread_id)
  {
    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(Dev[cpu_thread_id]);

    cudaFree(d_spin[cpu_thread_id]);
    cudaFree(d_rng[cpu_thread_id]);
  } // OpenMP
  free(d_spin);
  free(d_rng);

  free(spin);
  free(h_rng);

  #pragma omp parallel private(cpu_thread_id)
  {
    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(Dev[cpu_thread_id]);
    cudaDeviceReset();
  } // OpenMP

  return 0;
}
          
          
// Exact solution of 2d Ising model on the infinite lattice

void exact_2d(double T, double B, double *E, double *M)
{
  double x, y;
  double z, Tc, K, K1;
  const double pi = acos(-1.0);
    
  K = 2.0/T;
  if(B == 0.0) {
    Tc = -2.0/log(sqrt(2.0) - 1.0); // critical temperature;
    if(T > Tc) {
      *M = 0.0;
    }
    else if(T < Tc) {
      z = exp(-K);
      *M = pow(1.0 + z*z,0.25)*pow(1.0 - 6.0*z*z + pow(z,4),0.125)/sqrt(1.0 - z*z);
    }
    x = 0.5*pi;
    y = 2.0*sinh(K)/pow(cosh(K),2);
    K1 = ellf(x, y);
    *E = -1.0/tanh(K)*(1. + 2.0/pi*K1*(2.0*pow(tanh(K),2) - 1.0));
  }
  else
    printf("Exact solution is only known for B=0 !\n");
    
  return;
}


/*******
* ellf *      Elliptic integral of the 1st kind 
*******/

double ellf(double phi, double ak)
{
  double ellf;
  double s;

  s=sin(phi);
  ellf=s*rf(pow(cos(phi),2),(1.0-s*ak)*(1.0+s*ak),1.0);

  return ellf;
}

double rf(double x, double y, double z)
{
  double rf,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4;
  ERRTOL=0.08; 
  TINY=1.5e-38; 
  BIG=3.0e37; 
  THIRD=1.0/3.0;
  C1=1.0/24.0; 
  C2=0.1; 
  C3=3.0/44.0; 
  C4=1.0/14.0;
  double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    
  if(min(x,y,z) < 0 || min(x+y,x+z,y+z) < TINY || max(x,y,z) > BIG) {
    printf("invalid arguments in rf\n");
    exit(1);
  }

  xt=x;
  yt=y;
  zt=z;

  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=THIRD*(xt+yt+zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } 
  while (max(abs(delx),abs(dely),abs(delz)) > ERRTOL);

  e2=delx*dely-pow(delz,2);
  e3=delx*dely*delz;
  rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
    
  return rf;
}

double min(double x, double y, double z)
{
  double m;

  m = (x < y) ? x : y;
  m = (m < z) ? m : z;

  return m;
}

double max(double x, double y, double z)
{
  double m;

  m = (x > y) ? x : y;
  m = (m > z) ? m : z;

  return m;
}

void rng_MT(float* data, int n)   // RNG with uniform distribution in (0,1)
{
    for(int i = 0; i < n; i++)
      data[i] = (float) gsl_rng_uniform(rng); 
}

