CC = gcc
DEBUG = -g
CFLAGS = -Wall -std=c99 -O3
LIB = -lm -lgfortran -lopenblas
INCDIR = -I/opt/OpenBLAS/include
LIBDIR = -L/opt/OpenBLAS/lib

naive_asgd: main.c naive_asgd.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -o naive_asgd main.c naive_asgd.c

