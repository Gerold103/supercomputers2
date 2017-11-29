all: clean main.c
	mpicc main.c -O3 -DNDEBUG -g -std=c99 -Wall -openmp -o main

no_omp: clean main.c
	mpicc main.c -O3 -DNDEBUG -g -std=c99 -Wall -Wno-unknown-pragmas -o main

clean:
	rm -rf main.dSYM*
	rm -rf main
	rm -rf P_vers*
	rm -rf R_vers*
	rm -rf G_vers*
