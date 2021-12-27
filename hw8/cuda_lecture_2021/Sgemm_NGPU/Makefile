CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
#CFLAGS  := -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
#CFLAGS  := -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v --compiler-options -fopenmp
LIB     := -lcublas 

BIN    := Sgemm_ngpu_cublas
BIN1   := Sgemm_ngpu_umem

all: $(BIN) $(BIN1)

$(BIN): Sgemm_ngpu_cublas.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) Sgemm_ngpu_cublas.cu $(LIB)

$(BIN1): Sgemm_ngpu_umem.cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) Sgemm_ngpu_umem.cu $(LIB)

clean:
	rm -f $(BIN) $(BIN1)
