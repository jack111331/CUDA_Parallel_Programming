//  Monte Carlo simulation of Ising model on 2D lattice
//  using Metropolis algorithm
//  using checkerboard (even-odd) update 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <cooperative_groups.h>

gsl_rng *rng=NULL;    // pointer to gsl_rng random number generator

void exact_2d(double, double, double*, double*);
void rng_MT(float*, int);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

int *spin;            // host spin variables
int *d_spin;          // device spin variables
float *h_rng;         // host random numbers
float *d_rng;         // device random numbers 

__constant__ int fw[1000],bw[1000];     // declare constant memory for fw, bw 

__global__ void metro_gmem(int* spin, int nx, float *ranf, float B, float T)
{
    int    x, y, x0, y0, x1, y1, parity, ith, i;
    int    old_spin, new_spin, spins, gsizeX, gsizeY;
    int    k1, k2, k3, k4;
    float  de; 

    gsizeX = gridDim.x * blockDim.x;
    gsizeY = gridDim.y * blockDim.y;

    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    ith = g.thread_rank();		// thread index of the thread group
    x0  = (2*ith) % (2*gsizeX);
    y0  = (ith / gsizeX) % gsizeY;

    // first, go over the even sites 

    for (y1=y0; y1 < nx; y1 += gsizeY) {
    for (x1=x0; x1 < nx; x1 += (gsizeX*2)) {
        parity=(x1+y1)%2;
        x = x1 + parity;  
        y = y1;
        i = x + y*nx;
        old_spin = spin[i];
        new_spin = -old_spin;
        k1 = fw[x] + y*nx;     // right
        k2 = x + fw[y]*nx;     // top
        k3 = bw[x] + y*nx;     // left
        k4 = x + bw[y]*nx;     // bottom
        spins = spin[k1] + spin[k2] + spin[k3] + spin[k4];
        de = -(new_spin - old_spin)*(spins + B);
        if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
          spin[i] = new_spin;       // accept the new spin;
        }
    }}
    cg::sync(g);

    // next, go over the odd sites in each block

    for (y1=y0; y1 < nx; y1 += gsizeY) {
    for (x1=x0; x1 < nx; x1 += (gsizeX*2)) {
        parity=(x1+y1+1)%2;
        x = x1 + parity;
        y = y1;
        i = x + y*nx;
        old_spin = spin[i];
        new_spin = -old_spin;
        k1 = fw[x] + y*nx;     // right
        k2 = x + fw[y]*nx;     // top
        k3 = bw[x] + y*nx;     // left
        k4 = x + bw[y]*nx;     // bottom
        spins = spin[k1] + spin[k2] + spin[k3] + spin[k4];
        de = -(new_spin - old_spin)*(spins + B);
        if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
          spin[i] = new_spin;       // accept the new spin;
        }
    }}
}

int main(void) {
  int nx,ny; 		// # of sites in x and y directions respectively
  int ns; 		// ns = nx*ny, total # of sites
  int *ffw;      	// forward index
  int *bbw; 	        // backward index
  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int nb; 		// # of sweeps before saving spin configurations
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2;           // right, top
  int istart; 		// istart = (0: cold start/1: hot start)
  float T; 		// temperature
  float B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >

  int gid;              // GPU_ID
  float gputime;
  float flops;

  printf("Enter the GPU ID (0/1): ");
  scanf("%d",&gid);
  printf("%d\n",gid);

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  err = cudaSetDevice(gid);
  if(err != cudaSuccess) {
    printf("!!! Cannot select GPU with device ID = %d\n", gid);
    exit(1);
  }
  printf("Select GPU with device ID = %d\n", gid);
  cudaSetDevice(gid);

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
  scanf("%d",&nx);
  printf("%d\n",nx);
  ny=nx;
  ns=nx*ny;
  ffw = (int*)malloc(nx*sizeof(int));
  bbw = (int*)malloc(nx*sizeof(int));
  for(int i=0; i<nx; i++) {
    ffw[i]=(i+1)%nx;
    bbw[i]=(i-1+nx)%nx;
  }

  cudaMemcpyToSymbol(fw, ffw, nx*sizeof(int));  // copy to constant memory
  cudaMemcpyToSymbol(bw, bbw, nx*sizeof(int));

  spin = (int*)malloc(ns*sizeof(int));          // host spin variables
  h_rng = (float*)malloc(ns*sizeof(float));     // host random numbers

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
  scanf("%f",&T);
  printf("%f\n",T);
  printf("Enter the external magnetization\n");
  scanf("%f",&B);
  printf("%f\n",B);
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

  // The total number of threads in the grid is equal to (nx/2)*ny = ns/2 

  int bx = nx/tx/2;
  if(bx*tx*2 != nx) {
    printf("The block size in x is incorrect\n");
    exit(0);
  }
  int by = ny/ty;
  if(by*ty != ny) {
    printf("The block size in y is incorrect\n");
    exit(0);
  }
  if((bx > 65535)||(by > 65535)) {
    printf("The grid size exceeds the limit ! \n");
    exit(0);
  }
  dim3 blocks(bx,by);
  printf("The dimension of the grid is (%d, %d)\n",bx,by);

/*
 * Additional check for valid grid size for cooperative groups operation.
 */
  cudaDeviceProp prop = { 0 };
  cudaGetDeviceProperties(&prop, gid);

  int numBlocksPerSm = 0;
  int numSm = prop.multiProcessorCount;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm, metro_gmem, tx*ty, 0);
  if (bx*by > numBlocksPerSm * numSm) {
    printf("!!! Too many blocks to fit into cooperative groups operation.\n");
    exit(1);
  }

/*
 * Initialize the simulation.
 */
  if(istart == 0) {
    for(int j=0; j<ns; j++) {		// cold start
      spin[j] = 1;
    }
  }
  else {
    for(int j=0; j<ns; j++) {		// hot start
      if(gsl_rng_uniform(rng) > 0.5) { 
        spin[j] = 1;
      }
      else {
        spin[j] = -1;
      }
    }
  }

  FILE *output;            
  output = fopen("ising2d_1gpu_gmem2.dat","w");
  FILE *output3;
  output3 = fopen("spin_1gpu_gmem2.dat","w");   

  // Allocate vectors in device memory

  cudaMalloc((void**)&d_spin, ns*sizeof(int));         // device spin variables
  cudaMalloc((void**)&d_rng, ns*sizeof(float));        // device random numbers

  // Copy vectors from host memory to device memory

  cudaMemcpy(d_spin, spin, ns*sizeof(int), cudaMemcpyHostToDevice);

  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
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

  for(int swp=0; swp<nt; swp++) {      // thermalization
    rng_MT(h_rng, ns);                 // generate ns random numbers 
    cudaMemcpy(d_rng, h_rng, ns*sizeof(float), cudaMemcpyHostToDevice);
//    metro_gmem<<<blocks,threads>>>(d_spin, nx, d_rng, B, T);

    void *kernelArgs[] = {
        (void*)&d_spin,
        (void*)&nx,
        (void*)&d_rng,
        (void*)&B,
        (void*)&T,
    };
    int err = cudaLaunchCooperativeKernel((void*)metro_gmem, blocks,
			threads, kernelArgs, 0, NULL);
    if (err != cudaSuccess) {
        printf("!!! Cannot launch kernel.\n");
	exit(1);
    }
    cudaDeviceSynchronize();
  }

  for(int swp=nt; swp<sweeps; swp++) {
    rng_MT(h_rng, ns);                 // generate ns random numbers 
    cudaMemcpy(d_rng, h_rng, ns*sizeof(float), cudaMemcpyHostToDevice);
//    metro_gmem<<<blocks,threads>>>(d_spin, nx, d_rng, B, T);

    void *kernelArgs[] = {
        (void*)&d_spin,
        (void*)&nx,
        (void*)&d_rng,
        (void*)&B,
        (void*)&T,
    };
    int err = cudaLaunchCooperativeKernel((void*)metro_gmem, blocks,
			threads, kernelArgs, 0, NULL);
    if (err != cudaSuccess) {
	printf("!!! Cannot launch kernel\n");
	exit(1);
    }
    cudaDeviceSynchronize();

    int k; 
    if(swp%im == 0) {
      cudaMemcpy(spin, d_spin, ns*sizeof(int), cudaMemcpyDeviceToHost);
      mag=0.0;
      energy=0.0;
      for(int j=0; j<ny; j++) {
        for(int i=0; i<nx; i++) {
          k = i + j*nx;
          k1 = ffw[i] + j*nx;
          k2 = i + ffw[j]*nx;
          mag = mag + spin[k]; // total magnetization;
          energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;
        }
      }
      energy = energy - B*mag;
      te = te + energy;
      tm = tm + mag;
      count = count + 1.0;
      fprintf(output, "%.5e  %.5e\n", energy/(double)ns, mag/(double)ns);  // save the raw data 
    }
    if(swp%nd == 0) {
      E = te/(count*(double)(ns));
      M = tm/(count*(double)(ns));
      printf("%d  %.5e  %.5e\n", swp, E, M);
    }
    if(swp%nb == 0) {
      cudaMemcpy(spin, d_spin, ns*sizeof(int), cudaMemcpyDeviceToHost);
      fprintf(output3,"swp = %d, spin configuration:\n",swp);
      for(int j=nx-1;j>-1;j--) {
        for(int i=0; i<nx; i++) {
          fprintf(output3,"%d ",spin[i+j*nx]);
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
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }

  // stop the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&gputime, start, stop);
  printf("Processing time for GPU: %f (ms) \n",gputime);
  flops = 7.0*nx*nx*sweeps;
  printf("GPU Gflops: %lf\n",flops/(1000000.0*gputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  gsl_rng_free(rng);
  cudaFree(d_spin);
  cudaFree(d_rng);

  free(spin);
  free(h_rng);

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

