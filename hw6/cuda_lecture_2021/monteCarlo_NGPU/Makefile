################################################################################
#
# Build script for project
#
################################################################################

CXX     := icc
CFLAGS  := -O3
NVCC    := nvcc
CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

#CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN_SS     := monteCarlo_ss_ngpu
BIN_IS     := monteCarlo_is_ngpu
BIN_MIS     := monteCarlo_mis_ngpu
OBJ_SS     := monteCarlo_ss_ngpu.o
OBJ_IS     := monteCarlo_is_ngpu.o
OBJ_MIS     := monteCarlo_mis_ngpu.o
LIB     :=

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN_SS) $(BIN_IS) $(BIN_MIS)

$(BIN_SS): $(OBJ_SS)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN_SS) $(CUFLAGS) $(OBJ_SS) $(LIB)

$(BIN_IS): $(OBJ_IS)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN_IS) $(CUFLAGS) $(OBJ_IS) $(LIB)

$(BIN_MIS): $(OBJ_MIS)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN_MIS) $(CUFLAGS) $(OBJ_MIS) $(LIB)


monteCarlo_ss_ngpu.o:  monteCarlo_ss_ngpu.cu
monteCarlo_is_ngpu.o:  monteCarlo_is_ngpu.cu
monteCarlo_mis_ngpu.o:  monteCarlo_mis_ngpu.cu

clean:
	rm -f $(OBJ_SS) $(BIN_SS) $(OBJ_IS) $(BIN_IS) $(OBJ_MIS) $(BIN_MIS)

