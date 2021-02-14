#include "rapidcsv.h"
#include <math.h> 
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK = 128;
#define TILE_WIDTH = THREADS_PER_BLOCK*3;


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

double* getDataset(int* lenght, int* dim);
double* getFarCentroids(double *points, int pointsLength, int dimensions);
__global__ void assignCluster3(int length, int dimensions, double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster);
__global__ void reduceClusters(int length, int dimensions, int numBlocks, double *finalCentroids, int *finalTotPoints, double *blocksCentroids, int *blocksPointPerCluster);


const int k = 3;
const int dimensions = 2;
const int tileWidth = 128*dimensions; // TODO togliere il 2 e mettere dimensions, togliere 128 e mettere thread per block
__constant__ double centroidsConst[k*dimensions];

int main(int argc, char const *argv[]) {
    int dataLength;
    int dim;
    double *points = getDataset(&dataLength, &dim);
    double *centroids = getFarCentroids(points, dataLength, dimensions);
    int numBlocks = (dataLength + 128 - 1)/128; //TODO deve essere multiplo di 2
    for (int i = 0; i < dataLength; i++) {
        for (int j = 0; j < dimensions; j++) {
            printf("%f ", points[i*dimensions + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dimensions; j++) {
            printf("%f ", centroids[i*dimensions + j]);
        }
        printf("\n");
    }
    
    double *points_dev;
    cudaMalloc(&points_dev, dataLength*dimensions*sizeof(double));
    cudaMemcpy(points_dev, points, dataLength*dimensions*sizeof(double), cudaMemcpyHostToDevice);

    double *centroids_dev;
    cudaMalloc(&centroids_dev, k*dimensions*sizeof(double));
    
    double *blocksCentroids;
    cudaMalloc(&blocksCentroids, k*dimensions*(dataLength + 128 - 1)/128*sizeof(double)); //TODO modificare numero di blocchi

    int *blocksPointPerCluster;
    cudaMalloc(&blocksPointPerCluster, k*(dataLength + 128 - 1)/128*sizeof(int)); //TODO modificare numero di blocchi

    double distanceFromOld = 0;

    int *finalTotPoints = new int[k]; 
    cudaMalloc(&finalTotPoints, k*sizeof(int));

    double *finalCentroids = new double[k*dimensions];
    cudaMalloc(&finalCentroids, k*dimensions*sizeof(double));

    int *pointsInCluster = new int[k]; 
    double *newCentroids = new double[k*dimensions];

    do {
        cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(finalCentroids, 0, k*dimensions*sizeof(double));
        cudaMemset(finalTotPoints, 0, k*sizeof(int));
        cudaMemset(blocksCentroids, 0, k*dimensions*(dataLength + 128 - 1)/128*sizeof(double)); //TODO modificare numero di blocchi
        cudaMemset(blocksPointPerCluster, 0, k*(dataLength + 128 - 1)/128*sizeof(int)); //TODO modificare numero di blocchi
        
        assignCluster3<<<numBlocks, 128>>>(dataLength, dimensions, points_dev, centroids_dev, blocksCentroids, blocksPointPerCluster);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        
        reduceClusters<<<numBlocks, 128>>>(dataLength, dimensions, numBlocks, finalCentroids, finalTotPoints, blocksCentroids, blocksPointPerCluster);

        cudaMemcpy(newCentroids, blocksCentroids, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(pointsInCluster, blocksPointPerCluster, k*sizeof(int), cudaMemcpyDeviceToHost);

        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            printf("%d ",pointsInCluster[j]);
            for (int x = 0; x < dimensions; x++) {
                newCentroids[j*dimensions + x] /= pointsInCluster[j];
                printf("%f ",newCentroids[j*dimensions + x] );
            }
            printf("\n");
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
        printf("%f\n",distanceFromOld);
    } while (distanceFromOld > 0.000001);
    
    cudaFree(points_dev);
    cudaFree(centroids_dev);
    cudaFree(blocksCentroids);
    cudaFree(blocksPointPerCluster);
    cudaFree(finalTotPoints);
    cudaFree(finalCentroids);

    return 0;
}




//newCentroids and pointsInCluster are updated in block's shared memory, then the main function will do a reduction
__global__ void assignCluster3(int length, int dim, double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster){
    
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
            // printf("%f, %d, %d\n",newDist,idx,j);
        }
        // printf("%d - %d\n",idx, clustId);
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            // printf("%d -- %d -- %d -- %f\n", idx, clustId, x, newCentroids[clustId*dimensions + x]);
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
            // printf("%d -- %d -- %d -- %f\n", idx, clustId, x, newCentroids[clustId*dimensions + x]);
        }
        // printf("%d -- %d -- %d\n", idx, clustId, pointsInCluster[clustId]);
        atomicAdd(&(pointsInCluster[clustId]),1);
        // printf("%d -- %d -- %d\n", idx, clustId, pointsInCluster[clustId]);
    }

    if(threadIdx.x < dimensions*k) {
        newCentroids[threadIdx.x] = blocksCentroids[threadIdx.x + blockIdx.x*dimensions*k];
    }
    if(threadIdx.x < k) {
        pointsInCluster[threadIdx.x] = blocksPointPerCluster[threadIdx.x + blockIdx.x*k];
    }
}

//Todo definire numBlocks
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

        n = blockDim.x * numBlocks;
        do {
            n /= 2;
            if(idx < n) {
                atomicAdd(&(blocksPointPerCluster[idx]), blocksPointPerCluster[idx + n]);
            }
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












double* getDataset(int* lenght, int* dim) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    const int rows = int(doc.GetRowCount()) - k;
    const int dimensions = doc.GetColumnCount() - 1;
    *lenght = rows;
    *dim = dimensions;
    printf("%d\n",rows);
    double *points = new double[rows*dimensions];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                // std::cout << element << std::endl;
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