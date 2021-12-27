***********
* binning *             
***********
!
      PROGRAM binning   ! Binning method to estimate the error of the mean in MC
      Implicit none     
!
!     To compile:
!
!     gfortran -o binning binning.f  
!
      integer nconf,i,j,k,l,m,nbin,N
      real*8  tmp1, tmp2, tmp3, tmp4
      real*8, allocatable :: obs(:)
      real*8, allocatable :: i1(:)
      real*8, allocatable :: i1a(:)
      real*8, allocatable :: di1(:)
      integer, allocatable :: nb(:),binsize(:)
!
      character*50 infile,conf
!
      write(*,*) 'Enter the file name for the observables:'
      read(*,'(A)') infile
      write(*,*) infile
      write(*,*) 'Enter the number of data entries:'
      read(*,*)  nconf
      write(*,*) nconf
      write(*,*) 'Enter the # of bin sizes for binning:'
      read(*,*)  nbin
      write(*,*) nbin
      allocate(binsize(nbin),nb(nbin))
99    write(*,*) 'Enter the bin sizes:'
      read(*,*) (binsize(i),i=1,nbin)
      write(*,*) (binsize(i),i=1,nbin)
!
      do i=1,nbin
        k=binsize(i)
        if(mod(nconf,k).ne.0) then
          goto 99
        end if
        nb(i)=nconf/k      !  number of blocks
      end do  
!
      open(2,file=infile,status='unknown')
!
      allocate(obs(nconf))
!
      Do i=1,nconf
        read(2,*)  obs(i), tmp2, tmp3
!        read(2,*) tmp1, obs(i), tmp3
      End do
      close(2)
!
      allocate(i1a(nbin))
      allocate(di1(nbin))
!
      Do m=1,nbin
!
        N=nb(m)               ! number of blocks 
        allocate(i1(N)) 
!
        do j=1,N
          i1(j)=0.d0         
        end do
!
        k=0                   !  k runs from 1 to nconf=nb*binsize
        do l=1,N
          do j=1,binsize(m) 
            k=k+1
            i1(l)=i1(l)+obs(k)
          end do
          i1(l)=i1(l)/dble(binsize(m))    ! block average
        end do
!
        i1a(m)=0.d0
        di1(m)=0.d0
        do l=1,N
          i1a(m)=i1a(m)+i1(l)
          di1(m)=di1(m)+i1(l)**2
        end do
        i1a(m)=i1a(m)/dble(N)
        di1(m)=di1(m)/dble(N)
        di1(m)=(di1(m)-i1a(m)**2)/dble(max(N-1,1))
        di1(m)=sqrt(di1(m))
!
        deallocate(i1) 
!
      End do
!
      open(10,file='binning.dat',status='unknown')
!
      do m=1,nbin
        write(10,17) binsize(m),nb(m),i1a(m),di1(m)
        write(*,*) binsize(m),nb(m),i1a(m),di1(m)
      end do
!
      close(10)
!
17    format(2(I6,2x),E13.6,1x,E13.6,2x)
!
      END
!      
