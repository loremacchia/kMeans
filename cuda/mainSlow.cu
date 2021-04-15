#include "rapidcsv.h"
#include <math.h> 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>


// //Function to make atomicAdd usable for double
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



//Project parameters
const int k = 3;
const int dimensions = 2;
const int threadPerBlock = 256;

int main(int argc, char const *argv[]) {

    double *points = getDataset();
    int dataLength = getDataLength();
    for(int completedIteration = 0; completedIteration < 1; completedIteration++) {
        // printf("\n%d\n",completedIteration);
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
        
        double *newCentroids_dev;
        int *pointsInCluster_dev;


        cudaMalloc(&newCentroids_dev, k*dimensions*sizeof(double));
        cudaMalloc(&pointsInCluster_dev, k*sizeof(int));

        double distanceFromOld = 0;
        int *pointsInCluster = new int[k]; 
        double *newCentroids = new double[k*dimensions];
        int iter = 0;

        do {
            cudaMemcpy(centroids_dev, centroids, k*dimensions*sizeof(double), cudaMemcpyHostToDevice);
        
            cudaMemset(newCentroids_dev, 0, k*dimensions*sizeof(double));
            cudaMemset(pointsInCluster_dev, 0, k*sizeof(int));
            
            assignCluster<<<numBlocks, threadPerBlock>>>(dataLength, points_dev, centroids_dev, pointsInCluster_dev, newCentroids_dev);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("Error: %s\n", cudaGetErrorString(err));

            cudaMemcpy(newCentroids, newCentroids_dev, k*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(pointsInCluster, pointsInCluster_dev, k*sizeof(int), cudaMemcpyDeviceToHost);
            

            distanceFromOld = 0;
            for (int j = 0; j < k; j++) {
                //printf("%d ",pointsInCluster[j]);
                for (int x = 0; x < dimensions; x++) {
                    //printf("%f -- %d\n",newCentroids[j*dimensions + x], pointsInCluster[j]);
                    newCentroids[j*dimensions + x] /= pointsInCluster[j];
                    // printf("%f ",newCentroids[j*dimensions + x]);
                }
                //printf("\n");
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
        
            iter++;
            // printf("%f\n", distanceFromOld);
            // printf("%d\n", iter);
        } while (distanceFromOld > 0.001);


        cudaFree(points_dev);
        cudaFree(centroids_dev);
        cudaFree(newCentroids_dev);
        cudaFree(pointsInCluster_dev);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float outerTime;
        cudaEventElapsedTime( &outerTime, start, stop );

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    
        std::ofstream myfile;
        myfile.open ("./cuda/cudaSlow.csv", std::ios::app);
        myfile << dataLength;
        printf("\n\ncudaSlow: %f\n\n\n",outerTime);

        myfile << "," << outerTime;
        myfile << "\n";
        myfile.close();
        
        // printf("%d\n\n\n",iter);
    }
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