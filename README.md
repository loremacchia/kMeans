# kMeans

## Algorithm
K-Means is a clustering algorithm based on point assignment idea. 
The obtained clustering is:
<img src="/Plots/ScatterPlot.png" width="800">


## Implementation
The code is implemented in different versions: sequential, parallelized with OpenMP and in Cuda.\
To run all those implementations you can run the bash script (run it from the ./kMeans/ base folder):
```
bash run.sh 
```
that will create a dataset of increasing size, and will run each implementation 20 times. \
The Cuda version is implemented with a reduction phase decreasing the number of atomic adds of the algorithm.
If you want to run it on your own you can generate the dataset with
```
python3 generateDataset.py num_samples num_centrois num_dimensions
```
Then run the code you want (check the csv paths in all the implemented files, it can lead to errors):
* Plain c++ code:
```
g++ ./cpp/main.cpp -o mainCpp
./mainCpp
```

* OpenMP code:
```
g++ -o mainOmp -fopenmp ./omp/main.cpp
./mainOmp
```

* Cuda code:
```
nvcc ./cuda/main.cu -o mainCuda
./mainCuda
```

## Results
The performance improvements are significant. In particular the computation times are:
<img src="/Plots/CPPvsCUDAvsOMP.png" width="600">

### Contributors
Contributions are made by Lorenzo Macchiarini and Andrea Leonardo for the course Parallel Computing of the Master Degree in Software Engineering.
