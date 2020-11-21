#include "Cluster.h"

void Cluster::addPoint(NDimensionalPoint* point) {
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
    centroid->print();
}

double Cluster::newIteration() {
    this->divide();
    this->print();
    double distance = newCentroid->getDistance(*centroid);
    centroid = newCentroid;
    newCentroid = new Clustroid();
    pointsNumber = 0;
    this->print();
    return distance;
}

double Cluster::getDistance(NDimensionalPoint p){
    return centroid->getDistance(p);
}