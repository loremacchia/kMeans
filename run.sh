nvcc ./cuda/main.cu -dc 
nvcc main.o -o main
./main 