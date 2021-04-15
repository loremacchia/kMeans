#include "rapidcsv.h"
#include <float.h>
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
double* getFarCentroids(double *points, int pointsLength, int dimensions);
__global__ void assignClusterReduction(int length, int dimensions, int numBlocks, double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster);

//Project parameters
const int k = 3;
const int dimensions = 2;
const int threadPerBlock = 1024;


int main(int argc, char const *argv[]) {
    double *points = getDataset();
    int dataLength = getDataLength();
    int numBlocks = (dataLength + threadPerBlock - 1)/threadPerBlock;
    double *centroids = getFarCentroids(points, dataLength, dimensions);


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
    cudaMalloc(&blocksCentroids, k*dimensions*numBlocks*sizeof(double)); 

    int *blocksPointPerCluster;
    cudaMalloc(&blocksPointPerCluster, k*numBlocks*sizeof(int));


    int *pointsInCluster = new int[k]; 
    double *newCentroids = new double[k*dimensions];
    double distanceFromOld;
    int iter = 0;

    do {
        cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(blocksCentroids, 0, k*dimensions*numBlocks*sizeof(double)); 
        cudaMemset(blocksPointPerCluster, 0, k*numBlocks*sizeof(int)); 
        
        assignClusterReduction<<<numBlocks, threadPerBlock>>>(dataLength, dimensions, numBlocks, points_dev, centroids_dev, blocksCentroids, blocksPointPerCluster);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        
        cudaMemcpy(newCentroids, blocksCentroids, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(pointsInCluster, blocksPointPerCluster, k*sizeof(int), cudaMemcpyDeviceToHost);
    
        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                newCentroids[j*dimensions + x] /= pointsInCluster[j];
                distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
                centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
            }
        } 
        iter++;
    } while (distanceFromOld > 0.001);
    
    cudaFree(points_dev);
    cudaFree(centroids_dev);
    cudaFree(blocksCentroids);
    cudaFree(blocksPointPerCluster);
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float outerTime;
    cudaEventElapsedTime( &outerTime, start, stop );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    std::ofstream myfile;
    myfile.open ("./cuda/cuda.csv", std::ios::app);
    myfile << dataLength;
    myfile << "," << outerTime;
    myfile << "\n";
    myfile.close();

    printf("\n\n%f -- iter:%d\n\n\n",outerTime,iter);
    return 0;
}

//newCentroids and pointsInCluster are updated in block's shared memory, then the main function will do a reduction
__global__ void assignClusterReduction(int length, int dim, int numBlocks,double *points, double *centroids, double *blocksCentroids, int *blocksPointPerCluster){
    __shared__ double newCentroids[dimensions*k];
    __shared__ int pointsInCluster[k];
    if(threadIdx.x < dimensions*k) {
        newCentroids[threadIdx.x] = 0;
    }
    if(threadIdx.x < k) {
        pointsInCluster[threadIdx.x] = 0;
    }
    __syncthreads();
        
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < length) {
        double dist = FLT_MAX; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
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
        for (int x = 0; x < dimensions; x++) {
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
        }
        atomicAdd(&(pointsInCluster[clustId]),1);
    }
    __syncthreads();

    if(threadIdx.x < dimensions*k) {
        atomicAdd(&(blocksCentroids[threadIdx.x]), newCentroids[threadIdx.x]);
    }
    if(threadIdx.x < k) {
        atomicAdd(&(blocksPointPerCluster[threadIdx.x]), pointsInCluster[threadIdx.x]);
    }
}


int getDataLength(){
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    return int(doc.GetRowCount()) - k;
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