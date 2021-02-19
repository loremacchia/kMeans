nvcc ./cuda/main.cu -dc 
nvcc main.o -o mainCuda

g++ ./cpp/main.cpp -o mainCpp

g++ -o mainOmp -fopenmp ./omp/main.cpp

for i in {1000..10000000..1000}
    do 
        python3 generateDataset.py "$i"
        for j in {1..20..1}
            do
                ./mainCuda -d
                ./mainCuda -n
                ./mainCuda -cs
                ./mainCuda -cc
                ./mainCpp
                ./mainOmp
            done
    done