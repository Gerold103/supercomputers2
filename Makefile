all: clean main.c
	mpicc main.c -g -openmp -o main

clean:
	rm -rf main.dSYM*
	rm -rf main
	rm -rf P_vers*
	rm -rf R_vers*
	rm -rf G_vers*
