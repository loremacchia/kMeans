#include "rapidcsv.h"
#include <math.h> 
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
double* getFarCentroids(double *points, int pointsLength, int dimensions);
__global__ void assignCluster(int dataLength, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void assignCluster1(int dataLength, bool pointsInLocal, bool pointsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void assignCluster2(int dataLength, bool centroidsInConstant, bool centroidsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids);
__global__ void computeDivision(double *centroids, int *pointsInCluster, double *newCentroids, double *distanceFromOld);
int getDataLength();
int getDimensions();


//Project parameters
const int k = 3;
const int dimensions = 2;
const int threadPerBlock = 128;
const int tileWidth = threadPerBlock*dimensions;
__constant__ double centroidsConst[k*dimensions];


int main(int argc, char const *argv[]) {
    bool normal, pointsInLocal, pointsInShared, centroidsInConstant, centroidsInShared, divisionParallelized = false;

    for (int i = 1;  i < argc;  ++i ) {
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
        else if ( ! strncmp ( arg, "-h", 3 ) ) {
            printf("Helper for k-means cuda:\n -n: normal execution \n -pl: points will be put in local memory \n -ps: points will be put in shared memory \n -cc: centroids will be put in constant memory \n -cs: centroids will be put in shared memory \n -d: division will be parallelized \n\n\n");
        }
    }


    double *points = getDataset();
    int dataLength = getDataLength();
    double *centroids = getFarCentroids(points, dataLength, dimensions);

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
    
    //Allocate device memory to work with global memory 
    double *points_dev;
    cudaMalloc(&points_dev, dataLength*dimensions*sizeof(double));
    cudaMemcpy(points_dev, points, dataLength*dimensions*sizeof(double), cudaMemcpyHostToDevice);

    double *centroids_dev;
    cudaMalloc(&centroids_dev, k*dimensions*sizeof(double));
    
    double *newCentroids_dev;
    cudaMalloc(&newCentroids_dev, k*dimensions*sizeof(double));

    int *pointsInCluster_dev;
    cudaMalloc(&pointsInCluster_dev, k*sizeof(int));


    double *distanceFromOld_dev;
    cudaMalloc(&distanceFromOld_dev, sizeof(double));

    double distanceFromOld = 0;
    int *pointsInCluster = new int[k]; 
    double *newCentroids = new double[k*dimensions];
    int iter = 0;
    do {
        if(normal || pointsInLocal || pointsInShared || centroidsInShared || divisionParallelized) {
            cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        }
        else if(centroidsInConstant) {
            cudaMemcpyToSymbol(centroidsConst, centroids, k*dimensions*sizeof(double));
        }
        // cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpyToSymbol(centroidsConst, centroids, k*dimensions*sizeof(double));
        
        cudaMemset(newCentroids_dev, 0, k*dimensions*sizeof(double));
        cudaMemset(pointsInCluster_dev, 0, k*sizeof(int));
        if(normal || divisionParallelized) {
            assignCluster<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        else if(pointsInLocal || pointsInShared) {
            assignCluster1<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, pointsInLocal, pointsInShared, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        else if(centroidsInShared || centroidsInConstant) {
            assignCluster2<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, centroidsInConstant, centroidsInShared, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        }
        
        // assignCluster<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        // assignCluster1<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
        // assignCluster2<<<(dataLength + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        cudaMemcpy(newCentroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(pointsInCluster, pointsInCluster_dev, k*sizeof(int), cudaMemcpyDeviceToHost);


        if(divisionParallelized) {
            cudaMemset(distanceFromOld_dev, 0, sizeof(double));
            computeDivision<<<(k * dimensions + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(centroids_dev, pointsInCluster_dev, newCentroids_dev, distanceFromOld_dev);
            
            cudaMemcpy(&distanceFromOld, distanceFromOld_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(centroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
            for (int j = 0; j < k; j++) {
                for (int x = 0; x < dimensions; x++) {
                    printf("%f ",centroids[j*dimensions + x]);
                }
                printf("\n");
            }
        }
        else {
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
        }


        // cudaMemset(distanceFromOld_dev, 0, sizeof(double));
        // computeDivision<<<(k * dimensions + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(centroids_dev, pointsInCluster_dev, newCentroids_dev, distanceFromOld_dev);
        // cudaMemcpy(&distanceFromOld, distanceFromOld_dev, sizeof(double), cudaMemcpyDeviceToHost);

        // cudaMemcpy(centroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);

        // for (int j = 0; j < k; j++) {
        //     for (int x = 0; x < dimensions; x++) {
        //         printf("%f ",centroids[j*dimensions + x]);
        //     }
        //     printf("\n");
        // }


        // distanceFromOld = 0;
        // for (int j = 0; j < k; j++) {
        //     printf("%d ",pointsInCluster[j]);
        //     for (int x = 0; x < dimensions; x++) {
        //         newCentroids[j*dimensions + x] /= pointsInCluster[j];
        //         printf("%f ",newCentroids[j*dimensions + x] );
        //     }
        //     printf("\n");
        // }
        // for (int j = 0; j < k; j++) {
        //     for (int x = 0; x < dimensions; x++) {
        //         distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
        //     }
        // }
        // for (int j = 0; j < k; j++) {
        //     for (int x = 0; x < dimensions; x++) {
        //         centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
        //     }
        // }
        printf("%f\n",distanceFromOld);
    } while (distanceFromOld > 0.000001);
    // } while (++iter < 3);
    
    cudaFree(points_dev);
    cudaFree(centroids_dev);
    cudaFree(newCentroids_dev);
    cudaFree(pointsInCluster_dev);
    cudaFree(distanceFromOld_dev);
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
}

//Each thread copies its point into its local memory or into shared
__global__ void assignCluster1(int dataLength, bool pointsInLocal, bool pointsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
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
    // double *localPoints = new double[dimensions]; //è in local memory 
    // for (int x = 0; x < dimensions; x++) {
    //     localPoints[x] = points[idx*dimensions + x];
    // }

    // __shared__ double sharedPoints[tileWidth];//Each thread has a copy of its point into the shared memory (tiling)
    //     for (int x = 0; x < dimensions; x++) {
    //         sharedPoints[threadIdx.x*dimensions + x] = points[idx*dimensions + x];
    //     }
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
                // newDist += fabs(localPoints[x] - centroids[j*dimensions + x]);
                // newDist += fabs(sharedPoints[threadIdx.x*dimensions + x] - centroids[j*dimensions + x]);
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
            // atomicAdd(&(newCentroids[clustId*dimensions + x]), localPoints[x]);
            // atomicAdd(&(newCentroids[clustId*dimensions + x]), sharedPoints[threadIdx.x*dimensions + x]);
        }        
        atomicAdd(&(pointsInCluster[clustId]),1);
    }
}

//here centroids are put in shared or constant memory
__global__ void assignCluster2(int dataLength, bool centroidsInConstant, bool centroidsInShared, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
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
                // newDist += fabs(points[idx*dimensions + x] - centroidsConst[j*dimensions + x]);
                // newDist += fabs(points[idx*dimensions + x] - centroidsShared[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
            printf("%f, %d, %d\n",newDist,idx,j);
        }
        printf("%d - %d\n",idx, clustId);
        __syncthreads();
        
        for (int x = 0; x < dimensions; x++) {
            printf("%d -- %d -- %d -- %f\n", idx, clustId, x, newCentroids[clustId*dimensions + x]);
            atomicAdd(&(newCentroids[clustId*dimensions + x]), points[idx*dimensions + x]);
            printf("%d -- %d -- %d -- %f\n", idx, clustId, x, newCentroids[clustId*dimensions + x]);
        }
        printf("%d -- %d -- %d\n", idx, clustId, pointsInCluster[clustId]);
        atomicAdd(&(pointsInCluster[clustId]),1);
        printf("%d -- %d -- %d\n", idx, clustId, pointsInCluster[clustId]);
    }
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