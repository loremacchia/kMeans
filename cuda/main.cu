#include "rapidcsv.h"
#include <math.h> 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>


//Function to make atomicAdd usable for double
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

//Function declarations
double* getDataset();
int getDataLength();
int getDimensions();
double* getFarCentroids(double *points, int pointsLength, int dimensions);
__global__ void assignCluster(int dataLength, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void assignClusterPoints(int dataLength, bool pointsInLocal, bool pointsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void assignClusterCentroids(int dataLength, bool centroidsInConstant, bool centroidsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void assignClusterReduction(int length, int dimensions, double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster);
__global__ void reduceClusters(int length, int dimensions, int numBlocks, double *finalCentroids, int *finalTotPoints, double *blocksCentroids, int *blocksPointPerCluster);
__global__ void computeDivision(double *centroids, int *pointsInCluster, double *newCentroids, double *distanceFromOld);



//Project parameters
const int k = 3;
const int dimensions = 2;
const int threadPerBlock = 128;
const int tileWidth = threadPerBlock*dimensions;
__constant__ double centroidsConst[k*dimensions];


int main(int argc, char const *argv[]) {
    bool normal = false;    
    bool pointsInLocal = false;
    bool pointsInShared = false;
    bool centroidsInConstant = false;
    bool centroidsInShared = false;
    bool divisionParallelized = false;
    bool reduce = false;

    for (int i = 1;  i < 2;  ++i ) {
        const char *arg = argv[i];
        if ( ! strncmp ( arg, "-n", 2 )) {
            normal = true;
        }
        else if ( ! strncmp ( arg, "-pl", 3 ) ) {
            pointsInLocal = true;
        }
        else if ( ! strncmp ( arg, "-ps", 3 ) ) {
            pointsInShared = true;
        }
        else if ( ! strncmp ( arg, "-cc", 3 ) ) {
            centroidsInConstant = true;
        }
        else if ( ! strncmp ( arg, "-cs", 3 ) ) {
            centroidsInShared = true;
        }
        else if ( ! strncmp ( arg, "-d", 3 ) ) {
            divisionParallelized = true;
        }
        else if ( ! strncmp ( arg, "-r", 3 ) ) {
            reduce = true;
        }
        else if ( ! strncmp ( arg, "-h", 3 ) ) {
            normal = true;
            printf("Helper for k-means cuda:\n -n: normal execution \n -pl: points will be put in local memory \n -ps: points will be put in shared memory \n -cc: centroids will be put in constant memory \n -cs: centroids will be put in shared memory \n -d: division will be parallelized \n -r centroids will be evaluated for each block and there will be a reduction\n\n\n");
        }
    }

    double *points = getDataset();
    int dataLength = getDataLength();
    double *centroids = getFarCentroids(points, dataLength, dimensions);
    int numBlocks = (dataLength + threadPerBlock - 1)/threadPerBlock;

    // for (int i = 0; i < dataLength; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         printf("%f ", points[i*dimensions + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         printf("%f ", centroids[i*dimensions + j]);
    //     }
    //     printf("\n");
    // }


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    //Allocate device memory to work with global memory 
    double *points_dev;
    cudaMalloc(&points_dev, dataLength*dimensions*sizeof(double));
    cudaMemcpy(points_dev, points, dataLength*dimensions*sizeof(double), cudaMemcpyHostToDevice);

    double *centroids_dev;
    cudaMalloc(&centroids_dev, k*dimensions*sizeof(double));
    
    double *blocksCentroids;
    int *blocksPointPerCluster;
    int *finalTotPoints = new int[k]; 
    double *finalCentroids = new double[k*dimensions];

    double *newCentroids_dev;
    int *pointsInCluster_dev;
    double *distanceFromOld_dev;

    if(reduce) {
        cudaMalloc(&blocksCentroids, k*dimensions*numBlocks*sizeof(double)); //TODO modificare numero di blocchi
        cudaMalloc(&blocksPointPerCluster, k*numBlocks*sizeof(int)); //TODO modificare numero di blocchi
        cudaMalloc(&finalTotPoints, k*sizeof(int));
        cudaMalloc(&finalCentroids, k*dimensions*sizeof(double));
    }
    else {
        cudaMalloc(&newCentroids_dev, k*dimensions*sizeof(double));
        cudaMalloc(&pointsInCluster_dev, k*sizeof(int));
        cudaMalloc(&distanceFromOld_dev, sizeof(double));
    }

    double distanceFromOld = 0;
    int *pointsInCluster = new int[k]; 
    double *newCentroids = new double[k*dimensions];
    // int iter = 0;
    std::vector<double> times;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float outerTime;
    cudaEventElapsedTime( &outerTime, start, stop );


    do {
        cudaEventRecord(start, 0);
        if(normal || pointsInLocal || pointsInShared || centroidsInShared || divisionParallelized || reduce) {
            cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        }
        else if(centroidsInConstant) {
            cudaMemcpyToSymbol(centroidsConst, centroids, k*dimensions*sizeof(double));
        }

        if(reduce) {
            cudaMemset(finalCentroids, 0, k*dimensions*sizeof(double));
            cudaMemset(finalTotPoints, 0, k*sizeof(int));
            cudaMemset(blocksCentroids, 0, k*dimensions*numBlocks*sizeof(double)); //TODO modificare numero di blocchi
            cudaMemset(blocksPointPerCluster, 0, k*numBlocks*sizeof(int)); //TODO modificare numero di blocchi
        }
        else {
            cudaMemset(newCentroids_dev, 0, k*dimensions*sizeof(double));
            cudaMemset(pointsInCluster_dev, 0, k*sizeof(int));
        }
        
        
        if(normal || divisionParallelized) {
            assignCluster<<<numBlocks, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        else if(pointsInLocal || pointsInShared) {
            assignClusterPoints<<<numBlocks, threadPerBlock>>>(dataLength, pointsInLocal, pointsInShared, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        else if(centroidsInShared || centroidsInConstant) {
            assignClusterCentroids<<<numBlocks, threadPerBlock>>>(dataLength, centroidsInConstant, centroidsInShared, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        else if(reduce) {
            assignClusterReduction<<<numBlocks, threadPerBlock>>>(dataLength, dimensions, points_dev, centroids_dev, blocksCentroids, blocksPointPerCluster);
        }


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        if(reduce) {
            reduceClusters<<<numBlocks, threadPerBlock>>>(dataLength, dimensions, numBlocks, finalCentroids, finalTotPoints, blocksCentroids, blocksPointPerCluster);

            cudaMemcpy(newCentroids, blocksCentroids, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(pointsInCluster, blocksPointPerCluster, k*sizeof(int), cudaMemcpyDeviceToHost);
        }
        else {
            cudaMemcpy(newCentroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(pointsInCluster, pointsInCluster_dev, k*sizeof(int), cudaMemcpyDeviceToHost);
        }

        if(divisionParallelized) {
            cudaMemset(distanceFromOld_dev, 0, sizeof(double));
            computeDivision<<<(k * dimensions + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(centroids_dev, pointsInCluster_dev, newCentroids_dev, distanceFromOld_dev);
            
            cudaMemcpy(&distanceFromOld, distanceFromOld_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(centroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
            for (int j = 0; j < k; j++) {
                for (int x = 0; x < dimensions; x++) {
                    // printf("%f ",centroids[j*dimensions + x]);
                }
                // printf("\n");
            }
        }
        else {
            distanceFromOld = 0;
            for (int j = 0; j < k; j++) {
                // printf("%d ",pointsInCluster[j]);
                for (int x = 0; x < dimensions; x++) {
                    newCentroids[j*dimensions + x] /= pointsInCluster[j];
                    // printf("%f ",newCentroids[j*dimensions + x] );
                }
                // printf("\n");
            }
            for (int j = 0; j < k; j++) {
                for (int x = 0; x < dimensions; x++) {
                    distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
                }
            }
            for (int j = 0; j < k; j++) {
                for (int x = 0; x < dimensions; x++) {
                    centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
                }
            } 
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime( &elapsedTime, start, stop );
    
        times.push_back(elapsedTime);
        printf("%f\n", distanceFromOld);
    } while (distanceFromOld > 0.000001);

    // int *clusterAssign = new int[dataLength];
    // int *clusterAssign_dev;
    // cudaMalloc(&clusterAssign_dev, dataLength*sizeof(int));
    

    // cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemset(newCentroids_dev, 0, k*dimensions*sizeof(double));
    // cudaMemset(pointsInCluster_dev, 0, k*sizeof(int));
    // computeCluster<<<numBlocks, threadPerBlock>>>(dataLength, points_dev, centroids_dev, clusterAssign_dev);
    // cudaMemcpy(clusterAssign, clusterAssign_dev, dataLength*sizeof(int), cudaMemcpyDeviceToHost);
            
    // for(int i = 0; i < dataLength; i++) {
        
    // }

    // } while (++iter < 3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(points_dev);
    cudaFree(centroids_dev);
    if(reduce) {
        cudaFree(blocksCentroids);
        cudaFree(blocksPointPerCluster);
        cudaFree(finalTotPoints);
        cudaFree(finalCentroids);
    }
    else {
        cudaFree(newCentroids_dev);
        cudaFree(pointsInCluster_dev);
        cudaFree(distanceFromOld_dev);
    }

    std::string path;
    if(normal) {
        path = "normal.csv";
    }
    else if(pointsInLocal) {
        path = "pointsLocal.csv";
    }
    else if(pointsInShared) {
        path = "pointsShared.csv";
    }
    else if(centroidsInConstant) {
        path = "centroidsShared.csv";
    }
    else if(centroidsInShared) {
        path = "centroidsShared.csv";
    }
    else if(divisionParallelized) {
        path = "divisionParallelized.csv";
    }
    else if(reduce) {
        path = "reduced.csv";
    }

    std::ofstream myfile;
    myfile.open (path, std::ios::app);
    myfile << dataLength;
    for(auto element : times) {
        outerTime += element;
    }
    printf("%f",outerTime);

    myfile << "," << outerTime;
    for(auto element : times) {
        myfile << "," << element;
    }
    myfile << "\n";
    myfile.close();
    return 0;
}

//Implementazione iniziale
__global__ void assignCluster(int dataLength, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < dataLength) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster

        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from each Cluster
            for (int x = 0; x < dimensions; x++) {
                newDist += fabs(points[idx*dimensions + x] - centroids[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
        }
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
        }
        atomicAdd(&(pointsInCluster[clustId]),1);
    }
}

//Implementazione iniziale
__global__ void computeCluster(int dataLength, double *points, double *centroids, int *clusterAssign){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < dataLength) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster

        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from each Cluster
            for (int x = 0; x < dimensions; x++) {
                newDist += fabs(points[idx*dimensions + x] - centroids[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
        }
        __syncthreads();
        clusterAssign[idx] = clustId;
    }
}


//Each thread copies its point into its local memory or into shared
__global__ void assignClusterPoints(int dataLength, bool pointsInLocal, bool pointsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    double *localPoints = new double[dimensions]; //è in local memory 
    __shared__ double sharedPoints[tileWidth];//Each thread has a copy of its point into the shared memory (tiling)

    if(pointsInLocal) {
        for (int x = 0; x < dimensions; x++) {
            localPoints[x] = points[idx*dimensions + x];
        }
    }
    if(pointsInShared){
        for (int x = 0; x < dimensions; x++) {
            sharedPoints[threadIdx.x*dimensions + x] = points[idx*dimensions + x];
        }
    }
    __syncthreads();
    
    if(idx < dataLength) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster
        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from current Cluster
            for (int x = 0; x < dimensions; x++) {
                if(pointsInLocal)
                    newDist += fabs(localPoints[x] - centroids[j*dimensions + x]);
                if(pointsInShared)
                    newDist += fabs(sharedPoints[threadIdx.x*dimensions + x] - centroids[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
        }
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            if(pointsInLocal)
                atomicAdd(&(newCentroids[clustId*dimensions + x]), localPoints[x]);
            if(pointsInShared)
                atomicAdd(&(newCentroids[clustId*dimensions + x]), sharedPoints[threadIdx.x*dimensions + x]);
        }        
        atomicAdd(&(pointsInCluster[clustId]),1);
    }
}

//here centroids are put in shared or constant memory
__global__ void assignClusterCentroids(int dataLength, bool centroidsInConstant, bool centroidsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    __shared__ double centroidsShared[dimensions*k];
    if(centroidsInShared){
        if(threadIdx.x < dimensions*k) {
            centroidsShared[threadIdx.x] = centroids[threadIdx.x];
        }
    }
    __syncthreads();
    
    if(idx < dataLength) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster

        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from each Cluster
            for (int x = 0; x < dimensions; x++) {
                
                if(centroidsInConstant)
                    newDist += fabs(points[idx*dimensions + x] - centroidsConst[j*dimensions + x]);
                if(centroidsInShared)
                    newDist += fabs(points[idx*dimensions + x] - centroidsShared[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
        }
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
        }
        atomicAdd(&(pointsInCluster[clustId]),1);
    }
}

//newCentroids and pointsInCluster are updated in block's shared memory, then the main function will do a reduction
__global__ void assignClusterReduction(int length, int dim, double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster){
    
    __shared__ double newCentroids[dimensions*k];
    __shared__ int pointsInCluster[k];
    
    if(threadIdx.x < dimensions*k) {
        newCentroids[threadIdx.x] = 0;
    }
    if(threadIdx.x < k) {
        pointsInCluster[threadIdx.x] = 0;
    }
        
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < length) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster

        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from each Cluster
            for (int x = 0; x < dimensions; x++) {
                newDist += fabs(points[idx*dimensions + x] - centroids[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }            
        }
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
        }
        atomicAdd(&(pointsInCluster[clustId]),1);
    }

    if(threadIdx.x < dimensions*k) {
        blocksCentroids[threadIdx.x + blockIdx.x*dimensions*k] = newCentroids[threadIdx.x];
    }
    if(threadIdx.x < k) {
        blocksPointPerCluster[threadIdx.x + blockIdx.x*k] = pointsInCluster[threadIdx.x];
    }
}

//TODO allocare la shared per finalcentroids e final points
//accertarsi che il numero di blocchi e di indici sia multiplo di 2
__global__ void reduceClusters(int length, int dimensions, int numBlocks, double *finalCentroids, int *finalTotPoints, double *blocksCentroids, int *blocksPointPerCluster){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if((blockDim.x * numBlocks / k) % 2 == 0){
        int n = blockDim.x * numBlocks * k;
        do {
            n /= 2;
            if(idx < n) {
                atomicAdd(&(blocksCentroids[idx]), blocksCentroids[idx + n]);
            }
            __syncthreads();
        } while(n > k*dimensions);

        n = blockDim.x * numBlocks/2;
        do {
            if(idx < n) {
                atomicAdd(&(blocksPointPerCluster[idx]), blocksPointPerCluster[idx + n]);
            }
            n /= 2;
            __syncthreads();
        } while(n > k);
    }
    else{
        if(idx < k * dimensions * numBlocks && idx >= k * dimensions){
            atomicAdd(&(finalCentroids[idx%(k*dimensions) + idx/(k*dimensions)]), blocksCentroids[idx]);
        }
        if(idx < k * numBlocks && idx >= k){
            atomicAdd(&(finalTotPoints[idx%(k) + idx/(k)]), blocksPointPerCluster[idx]);
        }
    }
    __syncthreads();
}


__global__ void computeDivision(double *centroids, int *pointsInCluster, double *newCentroids, double *distanceFromOld){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < k * dimensions) {
        newCentroids[idx] /= pointsInCluster[(int)floorf(idx/dimensions)];
        atomicAdd(&(distanceFromOld[0]),fabs(newCentroids[idx] - centroids[idx]));
    }  

}




int getDataLength(){
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    return int(doc.GetRowCount()) - k;
}

int getDimensions(){
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    return doc.GetColumnCount() - 1;
}

double* getDataset() {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    const int rows = int(doc.GetRowCount()) - k;
    const int dimensions = doc.GetColumnCount() - 1;
    double *points = new double[rows*dimensions];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                points[i*dimensions + index] = std::atof(element.c_str());
            }
            index++;
        }
    }
    return points;
}

double* getFarCentroids(double *points, int pointsLength, int dimensions) {
    // Init set of clusters picking a point from the set and the k - 1 points further wrt the chosen point.
    // Those will be the firts clustroids
    double reference[dimensions]; // The first available point is chosen as reference
    for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
        reference[tmpIdx] = points[tmpIdx];
    }
    
    double *distances = new double[k-1]; 
    double *realPoints = new double[k*dimensions]; // Array containing the further points wrt reference
    for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
        realPoints[(k-1)*dimensions + tmpIdx] = reference[tmpIdx];
    }
    
    int maxSize = k - 1; // Maximum size of the array and relative indexes
    // Get the k - 1 points
    for(int i = 0; i < pointsLength; i++){
        double dist = 0;
        for (int x = 0; x < dimensions; x++) {
            dist += fabs(points[i*dimensions + x] - reference[x]);
        }
        if(dist > distances[maxSize - 1]) { // If the distance is higher than the last element of the array
            int index = 0;
            while (dist < distances[index] && index < maxSize) { // Find the right place to put in the point
                index++;
            }
            for (int j = maxSize - 1; j > index; j--) {
                distances[j] = distances[j - 1];
                for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
                    realPoints[j*dimensions + tmpIdx] = realPoints[(j - 1)*dimensions + tmpIdx];
                }
            }
            distances[index] = dist;
            for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
                realPoints[index*dimensions + tmpIdx] = points[i*dimensions + tmpIdx];
            }
        }
    }
    return realPoints;
}