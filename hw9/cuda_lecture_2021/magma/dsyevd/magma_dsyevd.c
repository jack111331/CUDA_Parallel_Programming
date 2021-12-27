#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"

void out_eigen(char *filename, int n, double *w, double *vec, int pv)
{
   FILE *f;
   int   i, j;

   if((f = fopen(filename, "wt")) == NULL) {
     printf("!!! Cannot output eigenvalues.\n");
     exit(1);
   }
   if(pv == 1) 
     fprintf(f, "Eigenvalues and Eigenvectors:");
   else
     fprintf(f, "Eigenvalues:");
   for(j=0; j < n; j++) {
     fprintf(f, "\n%04d  %17.10E", j+1, w[j]);
     if(pv == 1) {
       for(i=0; i < n; i++) {
         if (i % 10 == 0) fprintf(f,"\n");
         fprintf(f, " %9.2e", vec[i+j*n]);
       }
     }
   }
   fprintf(f, "\n");
   fclose(f);
}

int main(int argc, char **argv)
{
   magma_init();                          // initialize Magma
   double gpu_time, cpu_time;
   magma_int_t n, n2;
   double *a, *r;                         // a, r - nxn matrices on the host
   double *h_work;                        // workspace
   magma_int_t  lwork;                    // h_work size
   magma_int_t *iwork;                    // workspace
   magma_int_t  liwork;                   // iwork size
   double *w1, *w2;                       // w1, w2 - vectors of eigenvalues
   double error , work[1];                // used in difference computations
   magma_int_t ione=1, info;
   double mione = -1.0;
   magma_int_t incr = 1;
   magma_int_t ISEED[4] = {0, 0, 0, 1};   // seed

   int n0;
   int pv;
   printf("Enter the matrix size: ");
   scanf("%d", &n0);
   printf("%d\n", n0);
   printf("Print out eigenvectors (1/0) ? ");
   scanf("%d", &pv);
   printf("%d\n", pv);
   n  = (magma_int_t) n0;
   n2 = n*n;
   fflush(stdout);

   magma_dmalloc_cpu(&w1, n);             // host memory for real
   magma_dmalloc_cpu(&w2, n);             // eigenvalues
   magma_dmalloc_cpu(&a, n2);             // host memory for a
   magma_dmalloc_cpu(&r, n2);             // host memory for r

// Query for workspace sizes
   double aux_work[1];
   magma_int_t aux_iwork[1];
   magma_dsyevd(MagmaVec, MagmaLower, n, r, n, w1, aux_work, -1,
                aux_iwork, -1, &info);
   lwork  = (magma_int_t  )aux_work[0];
   liwork =  aux_iwork[0];
   iwork  = (magma_int_t *)malloc(liwork * sizeof(magma_int_t));
   magma_dmalloc_cpu(&h_work, lwork);      // memory for workspace

// Randomize the matrix a and copy a -> r
   lapackf77_dlarnv(&ione, ISEED, &n2, a);
   lapackf77_dlacpy(MagmaFullStr, &n, &n, a, &n, r, &n);

// compute the eigenvalues and eigenvectors for a real and symmetric nxn matrix;
// Magma version
   gpu_time = magma_sync_wtime(NULL);
   magma_dsyevd(MagmaVec, MagmaLower, n, r, n, w1, h_work, lwork, iwork, liwork, &info);
   gpu_time = magma_sync_wtime(NULL) - gpu_time;
   printf("dsyevd gpu time : %7.5f sec.\n" , gpu_time);     // Magma time
   fflush(stdout);
   out_eigen("eigen_magma.dat", n, w1, r, pv);

// Lapack version
   cpu_time = magma_wtime ();
   lapackf77_dsyevd("V", "L", &n, a, &n, w2, h_work, &lwork, iwork, &liwork, &info);
   cpu_time = magma_wtime() - cpu_time;
   printf("dsyevd cpu time : %7.5f sec.\n", cpu_time);       // Lapack time
   fflush(stdout);
   out_eigen("eigen_lapack.dat", n, w2, a, pv);

// difference in eigenvalues
   blasf77_daxpy(&n, &mione, w1, &incr, w2, &incr);
   error = lapackf77_dlange("M", &n, &ione, w2, &n, work);
   printf("difference in eigenvalues : %e\n", error);
   fflush(stdout);

   free(w1);                                            // free host memory
   free(w2);                                            // free host memory
   free(a);                                             // free host memory
   free(r);                                             // free host memory
   free(h_work);                                        // free host memory
   magma_finalize();                                    // finalize Magma
   return EXIT_SUCCESS;
}
