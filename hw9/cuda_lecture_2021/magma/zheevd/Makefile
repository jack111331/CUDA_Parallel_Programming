CUDADIR  := /usr/local/nvidia
MKLDIR   := /opt/intel/mkl/lib/intel64
MAGMADIR := /opt/magma

CC       := gcc
CFLAGS   := -O2 -DADD_ -I$(CUDADIR)/include -I$(MAGMADIR)/include

GCCLIB   := -L/opt/gcc-4.9/lib -lgomp
MKLLIB   := -L$(MKLDIR) -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread
MAGMALIB := -L$(MAGMADIR)/lib -lmagma

LIBS     := $(MAGMALIB) $(MKLLIB) $(GCCLIB)
BIN      := magma_zheevd

all: $(BIN)

$(BIN):  magma_zheevd.c
	$(CC) $(CFLAGS) -o $(BIN) $< $(LIBS)

clean:
	rm -f $(BIN)
