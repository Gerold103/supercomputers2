all: clean main.c cuda_utils.cu
	nvcc -rdc=true -arch=sm_20 -ccbin mpicxx main.c cuda_utils.cu -Xcompiler -std=c99 -Xcompiler -O3 -Xcompiler -DNDEBUG -Xcompiler -g -Xcompiler -Wall -o main

clean:
	rm -rf main.dSYM*
	rm -rf main
