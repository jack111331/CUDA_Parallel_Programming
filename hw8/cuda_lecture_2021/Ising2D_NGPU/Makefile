CUDIR     := /usr/local/nvidia

SM    := sm_61
CP    := 61

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g --std=c++11
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_$(CP) -code=$(SM) -Xptxas -v -rdc=true --std=c++11 --compiler-options -fopenmp
LIB     := -lgsl -lgslcblas -lcurand -lcudadevrt
#LIB     := -lgsl -lgslcblas -lcudadevrt

BIN   := ising2d_ngpu_gmem_v2

all: $(BIN)

$(BIN): ising2d_ngpu_gmem_v2.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) ising2d_ngpu_gmem_v2.cu $(LIB)

clean:
	rm -f $(BIN)
