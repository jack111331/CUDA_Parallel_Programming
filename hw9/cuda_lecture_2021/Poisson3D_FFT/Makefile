CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true
LIB     := -lcufft -lcudart

BIN1    := poisson3d_fft

all: $(BIN1)

$(BIN1): poisson3d_fft.cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) $< $(LIB)

clean:
	rm -f $(BIN1) *.o
