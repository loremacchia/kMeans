#include "Cluster.h"

void Cluster::addPoint(NDimensionalPoint* point) {
    pointsNumber++;
    centroid->addCoordinates(*point);
    point->setClusterId(clusterId);
}

Clustroid* Cluster::divide() {
    centroid->computeDivision(pointsNumber);
    return centroid;
}