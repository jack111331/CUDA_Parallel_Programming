#include <stdio.h>

void main() 
{

  int Nx=4;
  int x,y,ie,io;
  int i,parity;
  int sites=Nx*Nx;
  int half_sites = sites/2;
 
  printf("ie  x  y  i\n");
  printf("===========\n");

  for(ie=0; ie<half_sites; ie++) {
    x=(2*ie)%Nx; 
    y=((2*ie)/Nx)%Nx;
    parity = (x+y)%2;
    x = x + parity;
    i = x + y*Nx;
    printf("%2d %2d %2d %2d\n",ie,x,y,i); 
  }
  printf("\n"); 

  printf("io  x  y  i\n");
  printf("===========\n");

  for(io=0; io<half_sites; io++) {
    x=(2*io)%Nx; 
    y=((2*io)/Nx)%Nx;
    parity = (x+y+1)%2;
    x = x + parity;
    i = x + y*Nx;
    printf("%2d %2d %2d %2d\n",io,x,y,i); 
  }
  printf("\n"); 

}
