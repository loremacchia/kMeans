nvcc ./cuda/main.cu ./cuda/NDimensionalPoint.cu ./cuda/Cluster.cu ./cuda/Clustroid.cu -dc
nvcc main.o NDimensionalPoint.o Cluster.o Clustroid.o -o main
./main 