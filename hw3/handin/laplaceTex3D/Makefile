CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61,sm_61 -Xptxas -v

BIN     := laplace
BIN_PC := poisson_pc
LIB     :=

all: $(BIN) $(BIN_PC)

$(BIN): laplace.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) laplace.cu $(LIB)

$(BIN_PC): poisson_pc.cu
	$(NVCC) -o $(BIN_PC) $(NVFLAGS) poisson_pc.cu $(LIB)
clean:
	rm -f $(BIN) $(BIN_PC)
