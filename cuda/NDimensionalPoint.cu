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

__host__ __device__ double NDimensionalPoint::getDistance(NDimensionalPoint p){ //TODO __device__ 
    double totalDistance = 0;
    for(int i = 0; i < this->dimensions; i++){
        totalDistance += (pointVector[i] - p.pointVector[i])*(pointVector[i] - p.pointVector[i]);
    }
    totalDistance = sqrt(totalDistance);
    return totalDistance;
}

void NDimensionalPoint::print(){
    for(int i = 0; i < this->dimensions; i++){
        std::cout << pointVector[i] << "  ";
    }
    std::cout << "dim: " << dimensions << "  cluster: " << clusterId << std::endl;
}

