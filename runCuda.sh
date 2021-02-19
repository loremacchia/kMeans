nvcc ./cuda/main.cu -dc 
nvcc main.o -o mainCuda
./mainCuda -n
./mainCuda -r