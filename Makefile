all:
	nvcc src/main.cu -include include/matrix/matrices.h -lcublas -o main --std c++17

run:
	./main

