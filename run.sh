nvcc ./cuda/main.cu -o mainCuda
nvcc ./cuda/mainSlow.cu -o mainSlow
g++ ./cpp/main.cpp -o mainCpp
g++ -o mainOmp -fopenmp ./omp/main.cpp

for i in {1..7}
    do 
        python3 generateDataset.py $((10 ** $i))
        for j in {1..20..1}
            do
                ./mainCuda
                ./mainSlow
                ./mainCpp
                ./mainOmp
            done
    done