#include "Cluster.cuh"

__device__ void Cluster::addPoint(NDimensionalPoint* point) {
    pointsNumber++;
    newCentroid->addCoordinates(*point);
    point->setClusterId(clusterId);
}

Clustroid* Cluster::divide() {
    newCentroid->computeDivision(pointsNumber);
    return centroid;
}

void Cluster::print() {
    std::cout << "id: " << clusterId << " elements: " << pointsNumber << std::endl;
    std::cout << "old: ";
    centroid->print();
    std::cout << "new: ";
    newCentroid->print();
    std::cout << std::endl;
}

double Cluster::newIteration() {
    this->divide();
    // this->print();
    double distance = newCentroid->getDistance(*centroid);
    centroid = newCentroid;
    newCentroid = new Clustroid();
    pointsNumber = 0;
    return distance;
}

__device__ double Cluster::getDistance(NDimensionalPoint p){
    return centroid->getDistance(p);
}