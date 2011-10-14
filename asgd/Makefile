C = gcc
DEBUG = -g
CFLAGS = -Wall -std=c99 -O3

# these variables are to compile with OpenBLAS
LIB = -lm -lgfortran -lopenblas
INCDIR = -I/opt/OpenBLAS/include
LIBDIR = -L/opt/OpenBLAS/lib

# these variables are to compile with ACML
#LIB = -lm -lacml
#INCDIR = -I/opt/acml5.0.0/gfortran64_int64/include
#LIBDIR = -I/opt/acml5.0.0/gfortran64_int64/lib

blas_asgd: main.c blas_asgd.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -o blas_asgd main.c blas_asgd.c

