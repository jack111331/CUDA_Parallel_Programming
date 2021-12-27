CUDIR     := /usr/local/nvidia

SM    := sm_61
CP    := 61

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g --std=c++11
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_$(CP) -code=$(SM) -Xptxas -v -rdc=true --std=c++11
LIB     := -lgsl -lgslcblas -lcurand -lcudadevrt
#LIB     := -lgsl -lgslcblas -lcudadevrt

BIN1    := ising2d_cpu
BIN2    := ising2d_cpu_eo 
BIN31   := ising2d_1gpu_gmem_v1 
BIN32   := ising2d_1gpu_gmem_v2 
BIN33   := ising2d_1gpu_gmem_v3 
#BIN4    := curand_host_api 
#BIN5    := curand_device_api 

all: $(BIN1) $(BIN2) $(BIN31) $(BIN32) $(BIN33) $(BIN4) $(BIN5)

$(BIN1): ising2d_cpu.cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) ising2d_cpu.cu $(LIB)

$(BIN2): ising2d_cpu_eo.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) ising2d_cpu_eo.cu $(LIB)

$(BIN31): ising2d_1gpu_gmem_v1.cu
	$(NVCC) -o $(BIN31) $(NVFLAGS) ising2d_1gpu_gmem_v1.cu $(LIB)

$(BIN32): ising2d_1gpu_gmem_v2.cu
	$(NVCC) -o $(BIN32) $(NVFLAGS) ising2d_1gpu_gmem_v2.cu $(LIB)

$(BIN33): ising2d_1gpu_gmem_v3.cu
	$(NVCC) -o $(BIN33) $(NVFLAGS) ising2d_1gpu_gmem_v3.cu $(LIB)

#$(BIN4): curand_host_api.cu
#	$(NVCC) -o $(BIN4) $(NVFLAGS) curand_host_api.cu $(LIB)
#
#$(BIN5): curand_device_api.cu
#	$(NVCC) -o $(BIN5) $(NVFLAGS) curand_device_api.cu $(LIB)

clean:
	rm -f $(BIN1) $(BIN2) $(BIN31) $(BIN32) $(BIN33) $(BIN4) $(BIN5)
