#include "Clustroid.cuh"

__host__ __device__ void Clustroid::addCoordinates(NDimensionalPoint p){ //TODO __device__
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] += p.getPoint()[i];
    }
}

void Clustroid::computeDivision(int div){
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] /= div;
    }
}