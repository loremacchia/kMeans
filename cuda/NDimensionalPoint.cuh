#pragma once
#include <iostream>

class NDimensionalPoint {
protected:
    double *pointVector;
    int dimensions = 2;
    int clusterId = -1;

public:
    //default constructor
    NDimensionalPoint(){
        pointVector = new double[dimensions];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = 0;
        }
    };

    //constructor that performs a deepcopy
    NDimensionalPoint(double *vec, int dim){
        dimensions = dim;
        pointVector = new double[dim];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = vec[i];
        }
    };

    //copy constructor, also here a deepcopy
    NDimensionalPoint(NDimensionalPoint *p){
        dimensions = p->dimensions;
        pointVector = new double[dimensions];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = p->pointVector[i];
        }
        clusterId = p->clusterId;
    };
    
    __host__ __device__ double getDistance(NDimensionalPoint p);
    int getDimensions();
    __host__ __device__  double* getPoint();
    __host__ __device__ void setClusterId(int id);
    void print();
};

