#include "NDimensionalPoint.cuh"
#include <math.h> 

__host__ __device__  double* NDimensionalPoint::getPoint() {
    return this->pointVector;
}

int NDimensionalPoint::getDimensions() {
    return this->dimensions;
};

__host__ __device__ void NDimensionalPoint::setClusterId(int id) {
    clusterId = id;
}

__host__ __device__ double NDimensionalPoint::getDistance(NDimensionalPoint p){
    double totalDistance = 0;
    for(int i = 0; i < this->dimensions; i++){
        totalDistance += (pointVector[i] - p.pointVector[i])*(pointVector[i] - p.pointVector[i]);
    }
    totalDistance = sqrt(totalDistance);
    return totalDistance;
}

__host__ __device__ void NDimensionalPoint::print(){
    for(int i = 0; i < this->dimensions; i++){
        printf("%f ", pointVector[i]);
    }
    printf("dim: %d  cluster: %d\n",dimensions, clusterId);
}

