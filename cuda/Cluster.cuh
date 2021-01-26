#pragma once
#include "NDimensionalPoint.cuh"
#include "Clustroid.cuh"

class Cluster {
private:
    Clustroid* centroid;
    Clustroid* newCentroid;
    int pointsNumber = 0;
    int clusterId = -1;
public:
    // Constructor of cluster. In copy constructor of NDimensionalPoint is performed a deep copy because here the centroid 
    // is a different point wrt the dataset's points
    Cluster() {
        centroid = new Clustroid();
        newCentroid = new Clustroid();
    }

    Cluster(int id, NDimensionalPoint* initialCentroid){
        clusterId = id;
        centroid = new Clustroid(initialCentroid);
        newCentroid = new Clustroid();
    };

    ~Cluster(){};
    
    __device__ void addPoint(NDimensionalPoint *point);
    Clustroid* divide();
    void print();
    double newIteration();
    __device__ double getDistance(NDimensionalPoint p);
};
