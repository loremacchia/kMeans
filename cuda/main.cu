#include "rapidcsv.h"
#include <math.h> 
#include <cuda_runtime.h>

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
__global__ void assignCluster(int length, int dimensions, double *points, double *centroids, int *pointsInCluster, double *newCentroids);

const int k = 3;

int main(int argc, char const *argv[]) {
    int dataLength;
    int dimensions;
    double *points = getDataset(&dataLength, &dimensions);
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
    
    double *points_dev;
    cudaMalloc(&points_dev, dataLength*dimensions*sizeof(double));
    cudaMemcpy(points_dev, points, dataLength*dimensions*sizeof(double), cudaMemcpyHostToDevice);

    double *centroids_dev;
    cudaMalloc(&centroids_dev, k*dimensions*sizeof(double));
    

    double *newCentroids_dev;
    cudaMalloc(&newCentroids_dev, k*dimensions*sizeof(double));

    int *pointsInCluster_dev;
    cudaMalloc(&pointsInCluster_dev, k*sizeof(int));

    double distanceFromOld = 0;
    int *pointsInCluster = new int[k]; 
    double *newCentroids = new double[k*dimensions];

    do {
        cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(newCentroids_dev, 0, k*dimensions*sizeof(double));
        cudaMemset(pointsInCluster_dev, 0, k*sizeof(int));
        
        assignCluster<<<(dataLength +127)/128, 128>>>(dataLength, dimensions, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        cudaMemcpy(newCentroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(pointsInCluster, pointsInCluster_dev, k*sizeof(int), cudaMemcpyDeviceToHost);

        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            printf("%d ",pointsInCluster[j]);
            for (int x = 0; x < dimensions; x++) {
                printf("%f ",newCentroids[j*dimensions + x] );
                newCentroids[j*dimensions + x] /= pointsInCluster[j];
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
    } while (distanceFromOld > 0.0001);
    
    cudaFree(points_dev);
    cudaFree(centroids_dev);
    cudaFree(newCentroids_dev);
    cudaFree(pointsInCluster_dev);
    return 0;
}

//FILE 2
//TODO newCentroids e pointsInCluster devono essere scritti, quindi potrebbe essere buono fare variabili shared
//centroids deve essere solo letto e mai modificato, quindi metterlo in constant memory?
//points sono troppi da tenere tutti in memoria della gpu, dove e come si spezzettano? Metterli in LOCAL memory o REGISTER solamente 1 per ogni thread (o metterli nella shared)
__global__ void assignCluster(int length, int dimensions, double *points, double *centroids, int *pointsInCluster, double *newCentroids){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < length) {
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        int clustId = -1; // Id of the nearest Cluster

        for (int j = 0; j < k; j++) {
            double newDist = 0; //Distance from each Cluster
            for (int x = 0; x < dimensions; x++) {
                newDist += fabsf(points[idx*dimensions + x] - centroids[j*dimensions + x]);
            }
            if(newDist < dist) {
                dist = newDist;
                clustId = j;
            }
            printf("%f, %d, %d\n",newDist,idx,j);
        }
        printf("%d - %d\n",idx, clustId);
        __syncthreads();
        
        //TODO atomic
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